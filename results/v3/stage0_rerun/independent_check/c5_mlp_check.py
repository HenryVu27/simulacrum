"""Spot check of the first MLP candidate (early stopping, 1,024 episodes) on the Mac: raw and buffer 2,4, probe seed 0,
all three readouts, compared with the GPU log. CPU and GPU arithmetic differ, so agreement is expected only roughly."""
import json, sys, time
from pathlib import Path
import numpy as np
REPO = Path("/Users/henryvu/personal/simulacrum")
from umwelt import probes
from umwelt.data import load
raw = np.load(REPO / "results/v3/stage0_rerun/raw_obs.npy", mmap_mode="r")
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
stored = json.loads((REPO / "results/v3/stage0_rerun/scores.json").read_text())["MLP, early stopping"]["seeds"]
for name, lags in (("raw", (0,)), ("buffer 2,4", (0, 2, 4))):
    Xtr = np.concatenate([raw[:NF][:, ttr - L].astype(np.float32) for L in lags], -1).reshape(len(ytr), -1)
    Xte = np.concatenate([raw[TE0:TE0 + 256][:, tte - L].astype(np.float32) for L in lags], -1).reshape(len(yte), -1)
    t0 = time.time()
    pool = probes.Probe(Xtr, ytr, m2(Rtr["present"]), "reg", 256, 0, groups, 60_000, early_stop=True)
    seen = probes.Probe(Xtr, ytr, m2(Rtr["present"] & Rtr["vis"]), "reg", 256, 8, groups, 60_000, early_stop=True)
    loc = probes.Probe(Xtr, Rtr["loc"], np.ones_like(Rtr["loc"], bool), "reg", 256, 4, groups, 60_000, early_stop=True)
    pp, ps, pl = pool(Xte), seen(Xte), loc(Xte)
    pres, vis = Rte["present"], Rte["vis"]; one = np.ones_like(Rte["loc"], bool)
    mine = {"object_now": probes._r2_cols(pp, yte, m2(pres), m2(pres)), "object_hidden": probes._r2_cols(pp, yte, m2(pres & ~vis), m2(pres)),
            "object_seen": probes._r2_cols(ps, yte, m2(pres & vis), m2(pres)), "self_location": probes._r2_cols(pl, Rte["loc"], one, one)}
    st = stored[f"{name}, seed 0"]
    print(f"{name:11} ({time.time()-t0:.0f}s) best steps mac {pool.best_step},{seen.best_step},{loc.best_step} gpu {st['best_steps']}: "
          + "  ".join(f"{k} mac {v:+.4f} gpu {st[k]:+.4f}" for k, v in mine.items()), flush=True)
