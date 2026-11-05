"""Claim 6: the trained oracle's permanence surprise (main/lawful_s0) on the new movies across 20 bystander layouts
(seeds 0-19, the record used 0-4), and on the previous movies with the off-arena marker in several places.
My own forward pass (my_forward.py functions)."""
import json
import pickle
import sys
from pathlib import Path

import jax
import numpy as np
from common import S, old_voe, voe
from umwelt import presets
from umwelt.data import OBS_KEYS

sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("mf_funcs", S / "my_forward.py")
src = (S / "my_forward.py").read_text().split("suite = voe.build_suite()")[0]  # functions only
mf = {}
exec(compile(src, "my_forward_funcs", "exec"), mf)

LIVE = Path("/Users/henryvu/personal/umwelt/results/runs")


def perm_suite(mod, **kw):
    movies, crit = [], []
    for k in mod.INERT:
        for j in (-0.5, 0.0, 0.5):
            m, tc = mod.permanence(k, j, **kw)
            movies.append(m); crit.append(tc)
    T = min(len(mv[0]["oracle"]) for mv in movies)
    d = {f"obs_{k}": np.stack([np.stack([mv[v][k][:T] for v in range(4)]) for mv in movies]).astype(np.float16) for k in OBS_KEYS}
    d = {k: v.reshape(-1, *v.shape[2:]) for k, v in d.items()}
    d["action"] = np.zeros((len(movies) * 4, T, 2), np.float32)
    return d, np.array(crit)


def S_of(run, name, d, crit):
    saved = pickle.load(open(LIVE / run / f"{name}_params.pkl", "rb"))
    p = presets.PRESETS[name]
    params = jax.tree.map(np.asarray, saved["params"])
    o = ((mf["my_observe"](d, p) - saved["mu"]) / saved["sd"]).astype(np.float32).astype(np.float64)
    err = mf["errors_pixel"](params, o, np.zeros(o.shape[:2] + (2,)))
    return mf["my_surprise"](err, crit)


for run in ("main/lawful_s0", "v2/dark_s0"):
    print(f"== {run} oracle, trained, permanence")
    vals = []
    for seed in range(20):
        s, ci = S_of(run, "oracle", *perm_suite(voe, layout_seed=seed))
        vals.append(s)
        print(f"  new movies, layout seed {seed:2d}: S {s:+.4f} [{ci[0]:+.3f}, {ci[1]:+.3f}]", flush=True)
    print(f"  seeds 0-4: {min(vals[:5]):.4f} to {max(vals[:5]):.4f}; seeds 0-19: {min(vals):.4f} to {max(vals):.4f}, sd {np.std(vals):.4f}")
    old = old_voe()
    base = old.HIDDEN_WALL.copy()
    for label, w in (("on boundary y=0 (pre first fix)", [0.5, 0.0, 1.5, 0.0]), ("y=-1 (as committed at d6c6763)", [0.5, -1.0, 1.5, -1.0]),
                     ("y=-5", [0.5, -5.0, 1.5, -5.0]), ("top y=11", [0.5, 11.0, 1.5, 11.0]), ("far (-50,-50)", [-50.0, -50.0, -49.0, -50.0])):
        old.HIDDEN_WALL = np.array(w, np.float32)
        s, ci = S_of(run, "oracle", *perm_suite(old))
        print(f"  previous movies, marker {label:32}: S {s:+.4f} [{ci[0]:+.3f}, {ci[1]:+.3f}]", flush=True)
    old.HIDDEN_WALL = base
