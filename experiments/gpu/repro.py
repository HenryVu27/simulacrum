"""Cross-machine reproducibility, experiments E1 to E3 in docs/gpu.md.

Each machine runs `make` and writes results/hardware/<tag>/. The senses and training checks (E2, E3)
run on episodes recorded by the reference machine, so they test the sense code and the learner apart
from the world simulator. `compare` runs on the Mac with both folders present.

    uv run python experiments/gpu/repro.py make mac-cpu
    uv run python experiments/gpu/repro.py make pc-gpu --ref results/hardware/mac-cpu/data_dark.npz
    uv run python experiments/gpu/repro.py compare mac-cpu pc-gpu

`--precision highest` makes every float32 matrix multiply use full float32 (on recent NVIDIA GPUs
JAX's default may use TF32, which keeps 10 mantissa bits instead of 23).
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

N_EPISODES, N_TRAIN, N_KEEP = 64, 48, 16  # episodes per world; of these, trained on; observations kept
WORLDS = ("lawful", "dark")
UPDATES = 300


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def env():
    import jax

    dev = jax.devices()[0]
    return {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
            "jax": jax.__version__, "numpy": np.__version__, "backend": jax.default_backend(),
            "device": dev.device_kind, "matmul_precision": str(jax.config.jax_default_matmul_precision),
            "xla_flags": os.environ.get("XLA_FLAGS", "")}


def make(tag, ref, out_root):
    from umwelt import learner, presets
    from umwelt.data import generate

    out = os.path.join(out_root, tag)
    os.makedirs(out, exist_ok=True)
    info = {"tag": tag, "env": env(), "ref": ref, "seconds": {}}
    print(info["env"], flush=True)

    # E1: the world simulator, run here
    for w in WORLDS:
        t0 = time.time()
        d = generate(w, N_EPISODES, 256, seed=0)
        info["seconds"][f"generate_{w}"] = round(time.time() - t0, 2)
        np.savez_compressed(os.path.join(out, f"data_{w}.npz"), **d)
        info[f"data_{w}"] = {k: _sha(v) for k, v in d.items()}

    # E2 and E3 run on the reference machine's episodes (this machine's own, when it is the reference)
    d = dict(np.load(ref or os.path.join(out, "data_dark.npz")))
    info["senses"] = {}
    keep = {}
    for name, p in presets.PRESETS.items():
        o = presets.observe(d, p)
        info["senses"][name] = _sha(o)
        keep[name] = o[:N_KEEP]
    np.savez_compressed(os.path.join(out, "senses_dark.npz"), **keep)

    p = presets.PRESETS["human"]
    o = presets.observe(d, p, dtype=np.float16)  # as experiment.py does: float16 storage, standardized in place
    a = presets.actions(d, p)
    mu, sd = learner.standardizer(o[:N_TRAIN])
    o[:] = (o.astype(np.float32) - mu) / sd
    t0 = time.time()
    params, log = learner.train(o[:N_TRAIN], a[:N_TRAIN], learner.LearnerConfig(steps=UPDATES), seed=0,
                                recurrent=True, log_every=1, verbose=False)
    info["seconds"]["train"] = round(time.time() - t0, 2)
    err = learner.batched(learner._errors_jit, params, o[N_TRAIN:], a[N_TRAIN:])
    flat = {"/".join(k): np.asarray(v) for k, v in _flatten(params)}
    np.savez_compressed(os.path.join(out, "train_dark.npz"), log=np.array(log), test_errors=err, **flat)
    info["train"] = {"final_loss": log[-1][1], "test_error": float(err[:, 16:].mean())}
    with open(os.path.join(out, "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"wrote {out}: {info['seconds']}, test error {info['train']['test_error']:.6f}", flush=True)


def _flatten(tree, prefix=()):
    for k, v in tree.items():
        if isinstance(v, dict):
            yield from _flatten(v, prefix + (k,))
        else:
            yield prefix + (k,), v


def _diff(a, b):
    """How two arrays of the same shape differ."""
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)]}
    ne = a != b
    if a.dtype.kind == "f":
        ne = ne & ~(np.isnan(a) & np.isnan(b))
    r = {"identical": not ne.any(), "frac_differ": float(ne.mean())}
    if ne.any():
        if a.dtype.kind in "fc":
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            r["max_abs"] = float(d.max())
            r["rel_l2"] = float(np.linalg.norm(d) / max(np.linalg.norm(a.astype(np.float64)), 1e-30))
        if a.ndim >= 2:  # first time step (over episodes) where anything differs: does divergence grow?
            t_any = ne.reshape(a.shape[0], a.shape[1], -1).any(-1)
            r["first_step"] = int(np.argmax(t_any.any(0)))
            r["episodes_differing"] = int(t_any.any(1).sum())
    return r


def compare(ref_tag, other_tag, out_root):
    A, B = os.path.join(out_root, ref_tag), os.path.join(out_root, other_tag)
    rep = {"ref": ref_tag, "other": other_tag, "env": {t: json.load(open(os.path.join(p, "info.json")))["env"]
                                                        for t, p in ((ref_tag, A), (other_tag, B))}}
    for w in WORLDS:
        da, db = np.load(os.path.join(A, f"data_{w}.npz")), np.load(os.path.join(B, f"data_{w}.npz"))
        rep[f"E1_{w}"] = {k: _diff(da[k], db[k]) for k in da.files}
    sa, sb = np.load(os.path.join(A, "senses_dark.npz")), np.load(os.path.join(B, "senses_dark.npz"))
    rep["E2"] = {k: _diff(sa[k], sb[k]) for k in sa.files}
    ta, tb = np.load(os.path.join(A, "train_dark.npz")), np.load(os.path.join(B, "train_dark.npz"))
    la, lb = ta["log"][:, 1], tb["log"][:, 1]
    rel = np.abs(la - lb) / np.abs(la)
    rep["E3"] = {
        "loss_rel_diff_at": {int(i): float(rel[i]) for i in (0, 1, 10, 100, len(rel) - 1)},
        "loss_rel_diff_max": float(rel.max()),
        "params": {k: _diff(ta[k], tb[k]) for k in ta.files if k not in ("log", "test_errors")},
        "test_errors": _diff(ta["test_errors"], tb["test_errors"]),
        "test_error": [float(ta["test_errors"][:, 16:].mean()), float(tb["test_errors"][:, 16:].mean())],
    }
    path = os.path.join(out_root, f"compare_{ref_tag}_vs_{other_tag}.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=1)
    _print(rep)
    print(f"wrote {path}")


def _print(rep):
    print(f"== {rep['ref']} vs {rep['other']}")
    for w in WORLDS:
        e = rep[f"E1_{w}"]
        same = [k for k, v in e.items() if v.get("identical")]
        print(f"E1 {w}: {len(same)} of {len(e)} arrays identical")
        for k, v in e.items():
            if not v.get("identical"):
                print(f"   {k}: {v}")
    same = [k for k, v in rep["E2"].items() if v.get("identical")]
    print(f"E2 senses: {len(same)} of {len(rep['E2'])} presets identical")
    for k, v in rep["E2"].items():
        if not v.get("identical"):
            print(f"   {k}: {v}")
    e3 = rep["E3"]
    print(f"E3 loss relative difference at update {e3['loss_rel_diff_at']}, max {e3['loss_rel_diff_max']:.2e}")
    worst = max(e3["params"].items(), key=lambda kv: kv[1].get("rel_l2", 0.0))
    print(f"   weights identical: {all(v.get('identical') for v in e3['params'].values())}; "
          f"largest relative L2 difference {worst[0]} {worst[1].get('rel_l2', 0.0):.2e}")
    print(f"   test error {e3['test_error'][0]:.6f} vs {e3['test_error'][1]:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("make", "compare"))
    ap.add_argument("tag")
    ap.add_argument("other", nargs="?")
    ap.add_argument("--ref", help="reference machine's data_dark.npz, for the senses and training checks")
    ap.add_argument("--precision", choices=("default", "highest"), default="default")
    ap.add_argument("--out", default="results/hardware")
    args = ap.parse_args()
    if args.precision == "highest":
        import jax

        jax.config.update("jax_default_matmul_precision", "highest")
    if args.cmd == "make":
        make(args.tag, args.ref, args.out)
    else:
        compare(args.tag, args.other, args.out)
