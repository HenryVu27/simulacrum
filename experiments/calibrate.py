"""Calibration (Amendments 2 and 3): how much does memory cut prediction error in a world?

Trains human and memoryless (data seed 0, learner seed 0, the v1 learner) and reports memory value
M = 1 - err(human) / err(memoryless) on held-out episodes. By the preregistered rule only prediction
error is looked at here: no probes, no surprise tests, and the trained models are not kept.

The breakdown says where the gap comes from: steps in the dark, the first lit step after a dark spell,
steps where an object comes back into view, and everything else. It also traces error through a dark
spell, step by step.

    uv run python experiments/calibrate.py occlusion                         # both minds, 4,000 updates
    uv run python experiments/calibrate.py occlusion --steps 16000 --mind human
    uv run python experiments/calibrate.py occlusion --steps 16000           # summarize once both exist
    uv run python experiments/calibrate.py occlusion --episodes 8192 --mind human   # Amendment 4
"""

import argparse
import json
import time

import numpy as np

from umwelt import learner, presets
from umwelt.data import load
from umwelt.experiment import RESULTS
from umwelt.probes import steps_since_seen

MINDS = ("human", "memoryless")
OUT = RESULTS / "calibration"


def _name(world, steps, n_train):
    # Amendment 2 runs (4,000 updates, 1,024 episodes) keep their plain names.
    return world + ("" if steps == 4000 else f"_u{steps}") + ("" if n_train == 1024 else f"_n{n_train}")


def train_and_score(world, mind, steps, n_train):
    d = load(world, n_train + 256, 256, seed=0)
    tr, te = slice(0, n_train), slice(n_train, n_train + 256)
    p = presets.PRESETS[mind]
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    cache = OUT / "cache" / f"{_name(world, steps, n_train)}_{mind}.npy"
    o = presets.observe(d, p, dtype=np.float16, path=cache)  # memory-mapped: large recordings fit
    a = presets.actions(d, p)
    mu, sd = learner.standardizer(o[tr])
    for i in range(0, len(o), 128):
        o[i : i + 128] = (o[i : i + 128].astype(np.float32) - mu) / sd
    t0 = time.time()
    params, log = learner.train(o[tr], a[tr], learner.LearnerConfig(steps=steps), seed=0, recurrent=p.recurrent,
                                verbose=False, log_every=1000)
    err = learner.batched(learner._errors_jit, params, o[te], a[te])  # (E, T-1): error predicting step t+1
    secs = time.time() - t0
    del o
    cache.unlink()
    name = _name(world, steps, n_train)
    (OUT / "errors").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "errors" / f"{name}_{mind}.npy", err.astype(np.float32))
    (OUT / "errors" / f"{name}_{mind}_log.json").write_text(json.dumps({"seconds": secs, "log": log}))
    print(f"[{name}] {mind} trained in {secs:.0f}s, test error {err[:, 16:].mean():.4f}", flush=True)


def summarize(world, steps, n_train):
    name = _name(world, steps, n_train)
    errs = {m: np.load(OUT / "errors" / f"{name}_{m}.npy") for m in MINDS}
    d = load(world, n_train + 256, 256, seed=0)
    te = slice(n_train, n_train + 256)
    vis = np.asarray(d["gt_vis180"][te]) & np.asarray(d["gt_obj_active"][te])
    lit = np.asarray(d["gt_light"][te]) if "gt_light" in d else np.ones(vis.shape[:2], bool)  # v1 data: always lit
    light, lit_prev = lit[:, 1:], lit[:, :-1]
    since = steps_since_seen(vis)
    returns = (vis[:, 1:] & ~vis[:, :-1] & (since[:, :-1] < 1000) & (since[:, :-1] >= 3)).any(-1)
    masks = {"all": np.ones_like(light, bool), "dark": ~light, "lights_on": light & ~lit_prev,
             "object_returns": returns, "other": light & lit_prev & ~returns}
    burn = np.arange(light.shape[1]) >= 16  # same burn-in as v1 test_error
    out = {"world": world, "updates": steps, "episodes": n_train}
    for key, m in masks.items():
        m = m & burn[None]
        if not m.any():
            continue
        h, z = errs["human"][m].mean(), errs["memoryless"][m].mean()
        out[key] = {"share_of_steps": float(m.mean()), "err_human": float(h), "err_memoryless": float(z),
                     "M": float(1 - h / z)}
    out["M"] = out["all"]["M"]
    if (~lit).any():  # error through a dark spell: k steps into the dark, and k steps after the lights return
        into = np.zeros_like(lit, int)
        after = np.full_like(lit, 999, int)
        for t in range(1, lit.shape[1]):
            into[:, t] = np.where(~lit[:, t], np.where(lit[:, t - 1], 1, into[:, t - 1] + 1), 0)
            after[:, t] = np.where(lit[:, t], np.where(~lit[:, t - 1], 0, np.minimum(after[:, t - 1] + 1, 999)), 999)
        into, after = into[:, 1:], after[:, 1:]  # aligned with the step being predicted
        out["dark_course"] = {m: [float(errs[m][(into == k) & burn[None]].mean()) for k in range(1, 31)] for m in MINDS}
        out["after_lights_on"] = {m: [float(errs[m][(after == k) & burn[None]].mean()) for k in range(0, 11)] for m in MINDS}
    for m in MINDS:
        log = json.loads((OUT / "errors" / f"{name}_{m}_log.json").read_text())
        out[f"seconds_{m}"] = round(log["seconds"])
        out[f"train_log_{m}"] = log["log"]
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(f"[{name}] M = {out['M']:.3f} | " + " ".join(
        f"{k}: M={v['M']:.3f} ({v['share_of_steps']:.0%} of steps)" for k, v in out.items()
        if isinstance(v, dict) and "M" in v), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--mind", choices=MINDS)
    ap.add_argument("--episodes", type=int, default=1024, help="training episodes (Amendment 4: 8192)")
    ap.add_argument("--summarize-only", action="store_true", help="recompute the summary from saved errors")
    args = ap.parse_args()
    for m in [] if args.summarize_only else [args.mind] if args.mind else MINDS:
        train_and_score(args.world, m, args.steps, args.episodes)
    if all((OUT / "errors" / f"{_name(args.world, args.steps, args.episodes)}_{m}.npy").exists() for m in MINDS):
        summarize(args.world, args.steps, args.episodes)
