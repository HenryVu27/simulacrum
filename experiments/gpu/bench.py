"""Speed of the shared learner and the world simulator on this machine (experiment E6 in docs/gpu.md).

Three parts, each written to results/hardware/<tag>/bench.json:

- update: one compiled training update (the learner's loss and optimizer, unchanged) for a sweep of
  core sizes and batch sizes, and for K minds trained at once with jax.vmap. Data sits on the device,
  so this is the ceiling. The clock stops only after the device finishes. Each setting is timed three
  times after one compile (the median is reported); a setting that fails (for example out of GPU
  memory) is recorded with its error instead of stopping the run.
- train: learner.train itself, the current learner on host data, as experiments run it (batches gathered
  on the host and copied in every update). learner.train compiles its update on every call, so two run
  lengths are timed, three times each: the slope is the cost of one update, the intercept the fixed cost
  of a call (mostly compiling).
- e2e: one full mind (human, lawful world, data seed 0, the v1 sizes) through experiment.run_world,
  with its per-stage timings: training, latents, probes, surprise tests, controls.

Run one part at a time with --parts. Do not run anything else on the machine while it runs.

    uv run python experiments/gpu/bench.py mac-cpu
    uv run python experiments/gpu/bench.py pc-gpu --parts update train
"""

import argparse
import json
import os
import platform
import shutil
import time
from pathlib import Path

import jax
import numpy as np
import optax

from umwelt import learner

D = 311  # the human mind's observation size
SINGLE = [(128, 32), (128, 256), (512, 32), (512, 256), (1024, 256)]  # (hidden, batch), one mind
MANY = [1, 8, 16, 32, 64, 128]  # minds at once, current learner (hidden 128, batch 32)


def make_update(cfg):
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(cfg.lr, weight_decay=1e-4))

    def update(p, state, o, a):
        (loss, _), g = jax.value_and_grad(learner.loss_fn, has_aux=True)(p, o, a, cfg.burn_in)
        upd, state = opt.update(g, state, p)
        return optax.apply_updates(p, upd), state, loss

    return opt, update


def _clock(step, carry, min_seconds=3.0):
    """Seconds per call of step(carry) -> carry, after compiling and warming up."""
    for _ in range(3):
        carry = step(carry)
    jax.block_until_ready(carry)
    n, t = 5, 0.0
    while t < min_seconds:
        n *= 2
        t0 = time.perf_counter()
        c = carry
        for _ in range(n):
            c = step(c)
        jax.block_until_ready(c)
        t = time.perf_counter() - t0
    return t / n


def bench_update(K, hidden, batch):
    cfg = learner.LearnerConfig(hidden=hidden, batch=batch)
    opt, update = make_update(cfg)
    keys = jax.random.split(jax.random.PRNGKey(0), K)
    p = jax.vmap(lambda k: learner.init(k, D, cfg))(keys)
    st = jax.vmap(opt.init)(p)
    o = jax.random.normal(jax.random.PRNGKey(1), (K, batch, cfg.window, D))
    a = jax.random.normal(jax.random.PRNGKey(2), (K, batch, cfg.window, 2))
    f = jax.jit(jax.vmap(update))

    def step(c):
        p, st, _ = f(c[0], c[1], o, a)
        return p, st, None

    rec = {"minds": K, "hidden": hidden, "batch": batch}
    try:
        runs = [_clock(step, (p, st, None)) for _ in range(3)]
    except Exception as e:  # noqa: BLE001  (out of memory is a result here, not a crash)
        return rec | {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
    s = float(np.median(runs))
    return rec | {"ms_per_update": s * 1e3, "ms_runs": [r * 1e3 for r in runs], "mind_updates_per_s": K / s,
                  "minutes_per_4000_updates_all_minds": 4000 * s / 60}


def bench_train(short=200, long=1200, repeats=3):
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((256, 256, D), dtype=np.float32).astype(np.float16)
    act = rng.standard_normal((256, 256, 2), dtype=np.float32)
    learner.train(obs, act, learner.LearnerConfig(steps=20), verbose=False)  # warm the process
    secs = {short: [], long: []}
    for _ in range(repeats):
        for n in (short, long):
            t0 = time.perf_counter()
            learner.train(obs, act, learner.LearnerConfig(steps=n), verbose=False)
            secs[n].append(time.perf_counter() - t0)
    a, b = float(np.median(secs[short])), float(np.median(secs[long]))
    per = (b - a) / (long - short)
    fixed = a - short * per
    return {"ms_per_update": per * 1e3, "fixed_seconds_per_call": fixed,
            "seconds": {str(k): v for k, v in secs.items()}, "minutes_per_4000_updates": (fixed + 4000 * per) / 60}


def bench_e2e(tag):
    from umwelt.experiment import RESULTS, run_world

    run_tag = f"bench_{tag}"
    shutil.rmtree(RESULTS / "runs" / run_tag, ignore_errors=True)  # this benchmark's own output only
    t0 = time.perf_counter()
    run_world("lawful", 0, ["human"], learner.LearnerConfig(), tag=run_tag)
    total = time.perf_counter() - t0
    rec = json.loads((RESULTS / "runs" / run_tag / "lawful_s0" / "human.json").read_text())
    return {"seconds_total": total, "timing": rec["timing"], "test_error": rec["test_error"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--parts", nargs="+", default=["update", "train", "e2e"], choices=["update", "train", "e2e"])
    ap.add_argument("--precision", choices=("default", "highest"), default="default")
    ap.add_argument("--out", default="results/hardware")
    args = ap.parse_args()
    if args.precision == "highest":
        jax.config.update("jax_default_matmul_precision", "highest")
    out = os.path.join(args.out, args.tag)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "bench.json")
    res = json.loads(Path(path).read_text()) if os.path.exists(path) else {}
    res["env"] = {"platform": platform.platform(), "backend": jax.default_backend(),
                  "device": jax.devices()[0].device_kind, "jax": jax.__version__,
                  "matmul_precision": args.precision, "xla_flags": os.environ.get("XLA_FLAGS", ""),
                  "allocator": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default")}
    print(res["env"], flush=True)
    if "update" in args.parts:
        res["update"] = []
        for h, b in SINGLE:
            res["update"].append(bench_update(1, h, b))
            print(res["update"][-1], flush=True)
        for K in MANY[1:]:
            res["update"].append(bench_update(K, 128, 32))
            print(res["update"][-1], flush=True)
    if "train" in args.parts:
        res["train"] = bench_train()
        print("train:", res["train"], flush=True)
    if "e2e" in args.parts:
        res["e2e"] = bench_e2e(args.tag)
        print("e2e:", res["e2e"], flush=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
