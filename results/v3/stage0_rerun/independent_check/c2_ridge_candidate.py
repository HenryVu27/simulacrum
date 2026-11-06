"""Recompute the ridge, 1,024-episode candidate's criterion (a) on the Mac from the Mac's raw observations
(results/v3/stage0_rerun/raw_obs.npy, left behind by the Mac's failed probes run) and compare with the GPU's scores.json.
usage: c2_ridge_candidate.py FIXTURES(comma list: raw,b24,b369) SEEDS(comma list)"""
import json, sys, time, pickle
from pathlib import Path
import numpy as np
REPO = Path("/Users/henryvu/personal/simulacrum")
OUTD = Path(__file__).parent / "out"; OUTD.mkdir(exist_ok=True)
from umwelt import probes
from umwelt.data import load

fixtures = sys.argv[1].split(","); seeds = [int(s) for s in sys.argv[2].split(",")]
LAGS = {"raw": (0,), "b24": (0, 2, 4), "b369": (0, 3, 6, 9)}
NAME = {"raw": "raw", "b24": "buffer 2,4", "b369": "buffer 3,6,9"}
raw = np.load(REPO / "results/v3/stage0_rerun/raw_obs.npy", mmap_mode="r")
print("raw_obs", raw.shape, raw.dtype)
d = load("dark", 8448, seed=3)
NF, TE0 = 1024, 8192
ttr, tte = np.arange(10, 256, 8), np.arange(10, 256, 4)
def gt(sl): return {k: np.asarray(v[sl]) for k, v in d.items() if k.startswith("gt_")}
Ttr, Tte = probes.targets(gt(slice(0, NF))), probes.targets(gt(slice(TE0, TE0 + 256)))
def R(T, times): return {k: T[k][:, times].reshape(-1, *T[k].shape[2:]) for k in ("ego", "present", "vis", "loc")}
Rtr, Rte = R(Ttr, ttr), R(Tte, tte)
m2 = lambda m: np.repeat(m, 2, 1)
ytr, yte = Rtr["ego"].reshape(len(Rtr["ego"]), -1), Rte["ego"].reshape(len(Rte["ego"]), -1)
groups = np.repeat(np.arange(NF), len(ttr))
ep = np.repeat(np.arange(256), len(tte))
stored = json.loads((REPO / "results/v3/stage0_rerun/scores.json").read_text())["ridge"]["seeds"]
for f in fixtures:
    lags = LAGS[f]
    Xtr = np.concatenate([raw[:NF][:, ttr - L].astype(np.float32) for L in lags], -1).reshape(-1, 311 * len(lags))
    Xte = np.concatenate([raw[TE0:TE0 + 256][:, tte - L].astype(np.float32) for L in lags], -1).reshape(-1, 311 * len(lags))
    for s in seeds:
        t0 = time.time()
        pool = probes.RidgeProbe(Xtr, ytr, m2(Rtr["present"]), "reg", 256, s, groups, 60_000)
        seen = probes.RidgeProbe(Xtr, ytr, m2(Rtr["present"] & Rtr["vis"]), "reg", 256, s + 8, groups, 60_000)
        loc = probes.RidgeProbe(Xtr, Rtr["loc"], np.ones_like(Rtr["loc"], bool), "reg", 256, s + 4, groups, 60_000)
        pp, ps, pl = pool(Xte), seen(Xte), loc(Xte)
        pres, vis = Rte["present"], Rte["vis"]
        mine = {"object_now": probes._r2_cols(pp, yte, m2(pres), m2(pres)),
                "object_hidden": probes._r2_cols(pp, yte, m2(pres & ~vis), m2(pres)),
                "object_seen": probes._r2_cols(ps, yte, m2(pres & vis), m2(pres)),
                "self_location": probes._r2_cols(pl, Rte["loc"], np.ones_like(Rte["loc"], bool), np.ones_like(Rte["loc"], bool))}
        st = stored[f"{NAME[f]}, seed {s}"]
        print(f"{NAME[f]:13} seed {s} ({time.time()-t0:.0f}s): " + "  ".join(f"{k} mac {v:+.6f} gpu {st[k]:+.6f} d {v-st[k]:+.1e}" for k, v in mine.items()), flush=True)
        np.savez(OUTD / f"c2_{f}_s{s}.npz", pp=pp, ps=ps, pl=pl)
np.savez(OUTD / "c2_targets.npz", yte=yte, pres=Rte["present"], vis=Rte["vis"], loc=Rte["loc"], ep=ep)
