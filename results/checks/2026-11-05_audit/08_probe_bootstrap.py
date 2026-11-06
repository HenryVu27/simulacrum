"""Audit: P1 and STATUS v2 claim 4 with episode-bootstrap intervals, at the stored probe budget (1,024 episodes)
and a 4x budget (4,096 episodes, no row cap). v2 confirmatory recording (dark, data seed 1), both learner seeds."""
import pickle, sys, importlib.util
import numpy as np
sys.path.insert(0, "/Users/henryvu/personal/umwelt/src")
from umwelt import learner, presets, probes
from umwelt.data import load
import jax

class Probe(probes.Probe):
    def __init__(self, X, Y, M, hidden, seed, groups, cap):
        rng = np.random.default_rng(seed)
        keep = M.any(1); eps = np.unique(groups); val_eps = rng.choice(eps, max(len(eps) // 5, 1), replace=False)
        is_val = np.isin(groups, val_eps)
        tr, va = np.flatnonzero(keep & ~is_val), np.flatnonzero(keep & is_val)
        if len(tr) > cap: tr = rng.choice(tr, cap, replace=False)
        if len(va) > 3000: va = rng.choice(va, 3000, replace=False)
        idx = np.concatenate([tr, va]); self.mu, self.sd = X[idx].mean(0), X[idx].std(0) + 1e-6; self.task = "reg"
        Y = Y.astype(np.float32); m = M[idx]
        self.ym = (Y[idx] * m).sum(0) / np.maximum(m.sum(0), 1)
        self.ys = np.sqrt(((Y[idx] - self.ym) ** 2 * m).sum(0) / np.maximum(m.sum(0), 1)) + 1e-6
        Y = (Y - self.ym) / self.ys; Z = (X - self.mu) / self.sd; Mf = M.astype(np.float32)
        self.p = probes._fit(jax.random.PRNGKey(seed), Z[tr], Y[tr], Mf[tr], Z[va], Y[va], Mf[va], "reg", hidden)

N_TRAIN, ST = 8192, 8
d = load("dark", N_TRAIN + 256, seed=1)
gt = {k: v for k, v in d.items() if k.startswith("gt_")}
te = slice(N_TRAIN, N_TRAIN + 256)
Tte = probes.targets({k: np.asarray(v[te]) for k, v in gt.items()})
Rv = {k: probes._rows(Tte, k, 4) for k in ("ego", "present", "vis", "since")}
Rv["dark"] = np.repeat(~np.asarray(gt["gt_light"][te])[..., None], 8, -1)[:, probes.PAST::4].reshape(-1, 8)
ep_of_row = np.repeat(np.arange(256), len(range(probes.PAST, 256, 4)))
TF = {}
for n in (1024, 4096):
    T = probes.targets({k: np.asarray(v[:n]) for k, v in gt.items()})
    TF[n] = {k: probes._rows(T, k, ST) for k in ("ego", "present")}
Y = Rv["ego"].reshape(len(ep_of_row), -1)
pres, vis, since, dk = Rv["present"], Rv["vis"], Rv["since"], Rv["dark"]
BINS = {"1-5": pres & ~vis & (since >= 1) & (since <= 5), "6-20": pres & ~vis & (since >= 6) & (since <= 20),
        "1-5 dark": pres & ~vis & (since >= 1) & (since <= 5) & dk}

def skill(pred, const, rows):
    out = {}
    P, C, Yr, pr = pred[rows], const[rows], Y[rows], np.repeat(pres[rows], 2, 1)
    for b, m in BINS.items():
        mm = np.repeat(m[rows], 2, 1)
        sc, ba, w = [], [], []
        for c in range(Y.shape[1]):
            if mm[:, c].sum() < 30 or pr[:, c].sum() < 30: continue
            var = Yr[pr[:, c], c].var() + 1e-9
            sc.append(1 - np.mean((P[mm[:, c], c] - Yr[mm[:, c], c]) ** 2) / var)
            ba.append(1 - np.mean((C[mm[:, c], c] - Yr[mm[:, c], c]) ** 2) / var)
            w.append(mm[:, c].sum())
        s, bb = np.average(sc, weights=w), np.average(ba, weights=w)
        out[b] = (s - bb) / (1 - bb)
    return out

def rep_rows(kind, sl, stride, params=None, mu=None, sd=None, preset=None):
    rows = []
    for i in range(sl.start, sl.stop, 256):
        part = {k: v[i:min(i + 256, sl.stop)] for k, v in d.items() if k.startswith("obs_") or k in ("action", "gt_light")}
        o = (presets.observe(part, preset) - mu) / sd
        if kind == "net":
            H = learner.batched(learner._latents_jit, params, o, presets.actions(part, preset))
        else:
            H = o
        rows.append(H[:, probes.PAST::stride].reshape(-1, H.shape[-1]).astype(np.float32))
    return np.concatenate(rows)

preds = {}
R = "/Users/henryvu/personal/umwelt/results/runs/v2/"
reps = [("human s0", "net", "dark_s0", "human"), ("human s1", "net", "dark_s1", "human"),
        ("memoryless s0", "net", "dark_s0", "memoryless"), ("memoryless s1", "net", "dark_s1", "memoryless"),
        ("human raw senses", "raw", "dark_s0", "human")]
for label, kind, run, mind in reps:
    sv = pickle.load(open(R + f"{run}/{mind}_params.pkl", "rb")); pr = presets.PRESETS[mind]
    Xev = rep_rows(kind, te, 4, sv["params"], sv["mu"], sv["sd"], pr)
    X4 = rep_rows(kind, slice(0, 4096), ST, sv["params"], sv["mu"], sv["sd"], pr)
    for n in (1024, 4096):
        R_ = TF[n]; Xf = X4[: n * len(range(probes.PAST, 256, ST))]
        groups = np.repeat(np.arange(n), len(range(probes.PAST, 256, ST)))
        p = Probe(Xf, R_["ego"].reshape(len(Xf), -1), np.repeat(R_["present"], 2, 1), 256, 0, groups, 60_000 if n == 1024 else 10**9)
        mean = np.array([[R_["ego"][R_["present"][:, k], k, j].mean() if R_["present"][:, k].any() else 0.0 for j in range(2)] for k in range(8)])
        preds[(label, n)] = (p(Xev), np.broadcast_to(mean.reshape(1, -1), Y.shape))
        print(f"{label:18s} probe {n} eps: " + " ".join(f"{b}={v:+.3f}" for b, v in skill(*preds[(label, n)], np.arange(len(Y))).items()), flush=True)

rng = np.random.default_rng(0)
rows_of = [np.flatnonzero(ep_of_row == e) for e in range(256)]
for n in (1024, 4096):
    boots = {k: [] for k in ("human-raw", "human-memoryless", "human s0 - human s1")}
    for _ in range(400):
        eps = rng.integers(0, 256, 256); rows = np.concatenate([rows_of[e] for e in eps])
        sk = {lab: skill(*preds[(lab, n)], rows) for lab, *_ in reps}
        h = {b: (sk["human s0"][b] + sk["human s1"][b]) / 2 for b in BINS}
        z = {b: (sk["memoryless s0"][b] + sk["memoryless s1"][b]) / 2 for b in BINS}
        boots["human-raw"].append([h[b] - sk["human raw senses"][b] for b in BINS])
        boots["human-memoryless"].append([h[b] - z[b] for b in BINS])
        boots["human s0 - human s1"].append([sk["human s0"][b] - sk["human s1"][b] for b in BINS])
    full = {lab: skill(*preds[(lab, n)], np.arange(len(Y))) for lab, *_ in reps}
    h = {b: (full["human s0"][b] + full["human s1"][b]) / 2 for b in BINS}; z = {b: (full["memoryless s0"][b] + full["memoryless s1"][b]) / 2 for b in BINS}
    point = {"human-raw": [h[b] - full["human raw senses"][b] for b in BINS], "human-memoryless": [h[b] - z[b] for b in BINS],
             "human s0 - human s1": [full["human s0"][b] - full["human s1"][b] for b in BINS]}
    for k, v in boots.items():
        lo, hi = np.percentile(np.array(v), [2.5, 97.5], axis=0)
        print(f"[probe {n} eps] {k:20s} " + " ".join(f"{b}: {point[k][i]:+.3f} [{lo[i]:+.3f}, {hi[i]:+.3f}]" for i, b in enumerate(BINS)), flush=True)
