"""Criteria b, c, d for ridge at 1,024 episodes, recomputed on the Mac: engineered fixtures (full, last seen, updated) and
the tracking mind, with a permanence-skill curve written here (own bins and baseline), compared with scores.json."""
import json, pickle, time, importlib.util
from pathlib import Path
import numpy as np
REPO = Path("/Users/henryvu/personal/simulacrum")
spec = importlib.util.spec_from_file_location("s0", REPO / "experiments/stage0.py"); s0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s0)
from umwelt import probes, learner, presets
from umwelt.data import load
d = load("dark", 8448, seed=3)
NF, TE0 = 1024, 8192
ttr, tte = np.arange(10, 256, 8), np.arange(10, 256, 4)
gtf = {k: np.asarray(v[:8192]) for k, v in d.items() if k.startswith("gt_")}
gte = {k: np.asarray(v[TE0:TE0 + 256]) for k, v in d.items() if k.startswith("gt_")}
Ttr, Tte = probes.targets({k: v[:NF] for k, v in gtf.items()}), probes.targets(gte)
Ttr["dark"] = np.repeat(~gtf["gt_light"][:NF, :, None], 8, -1); Tte["dark"] = np.repeat(~gte["gt_light"][..., None], 8, -1)
def R(T, times): return {k: T[k][:, times].reshape(-1, *T[k].shape[2:]) for k in ("ego", "present", "vis", "since", "dark")}
Rtr, Rte = R(Ttr, ttr), R(Tte, tte)
m2 = lambda m: np.repeat(m, 2, 1)
ytr, yte = Rtr["ego"].reshape(len(Rtr["ego"]), -1), Rte["ego"].reshape(len(Rte["ego"]), -1)
groups = np.repeat(np.arange(NF), len(ttr))
# own baseline: per-kind, per-coordinate mean of ego over present fitting rows
base = np.zeros(16)
for k in range(8):
    p = Rtr["present"][:, k]
    if p.any(): base[2*k:2*k+2] = Rtr["ego"][p, k].mean(0)
def r2(pred, m, ref):
    sc, n = [], []
    for c in range(16):
        if m[:, c].sum() < 30 or ref[:, c].sum() < 30: continue
        v = yte[ref[:, c], c].astype(np.float64).var()
        sc.append(1 - np.mean((pred[m[:, c], c] - yte[m[:, c], c]) ** 2) / v); n.append(m[:, c].sum())
    return np.average(sc, weights=n)
def curve(pred):
    pres, vis, since, dark = Rte["present"], Rte["vis"], Rte["since"], Rte["dark"]
    const = np.broadcast_to(base, pred.shape)
    out = {}
    for name, lo, hi in (("in view", None, None), ("1-5 steps", 1, 5), ("6-20 steps", 6, 20), ("21-60 steps", 21, 60), ("over 60 steps", 61, 999), ("never seen", 1000, 1000)):
        m = pres & vis if lo is None else pres & ~vis & (since >= lo) & (since <= hi)
        a, b = r2(pred, m2(m), m2(pres)), r2(const, m2(m), m2(pres))
        out[name] = (a - b) / (1 - b)
    return out
stored = json.loads((REPO / "results/v3/stage0_rerun/scores.json").read_text())["ridge"]["curves"]
cur = {}
t0 = time.time()
for which in ("full", "last seen, updated"):
    A, B = s0.embed(s0.engineered(gtf, which, ttr), s0.engineered(gte, which, tte))
    pred = probes.RidgeProbe(A[:NF].reshape(-1, A.shape[-1]), ytr, m2(Rtr["present"]), "reg", 256, 0, groups, 60_000)(B.reshape(-1, B.shape[-1]))
    cur[which] = curve(pred)
print(f"engineered fixtures done in {time.time()-t0:.0f}s", flush=True)
raw = np.load(REPO / "results/v3/stage0_rerun/raw_obs.npy", mmap_mode="r")
Xr = raw[:NF][:, ttr].astype(np.float32).reshape(-1, 311); Xr_te = raw[TE0:TE0+256][:, tte].astype(np.float32).reshape(-1, 311)
cur["raw"] = curve(probes.RidgeProbe(Xr, ytr, m2(Rtr["present"]), "reg", 256, 0, groups, 60_000)(Xr_te))
with open(REPO / "results/v3/stage0_rerun/track_params.pkl", "rb") as f: sv = pickle.load(f)
acts = presets.actions(d, presets.PRESETS["human"])
def lat(sl, times):
    out = []
    for i in range(sl.start, sl.stop, 256):
        j = min(i + 256, sl.stop)
        out.append(learner.batched(learner._latents_jit, sv["params"], (raw[i:j].astype(np.float32) - sv["mu"]) / sv["sd"], acts[i:j])[:, times])
    return np.concatenate(out)
t0 = time.time()
Ht, Ht_te = lat(slice(0, NF), ttr), lat(slice(TE0, TE0 + 256), tte)
print(f"tracking-mind latents in {time.time()-t0:.0f}s", flush=True)
cur["track"] = curve(probes.RidgeProbe(Ht.reshape(-1, 128), ytr, m2(Rtr["present"]), "reg", 256, 0, groups, 60_000)(Ht_te.reshape(-1, 128)))
for f, c in cur.items():
    print(f"{f:20} " + "  ".join(f"{k} {v:+.4f} (stored {stored[f][k]:+.4f})" for k, v in c.items()))
print("b. last seen, updated - raw > 0.30 at 1-5 and 6-20:", [round(cur['last seen, updated'][k] - cur['raw'][k], 3) for k in ("1-5 steps", "6-20 steps")])
print("c. full > 0.80 at every delay: min", round(min(cur["full"].values()), 4))
print("d. track - raw at 1-5:", round(cur["track"]["1-5 steps"] - cur["raw"]["1-5 steps"], 4))
