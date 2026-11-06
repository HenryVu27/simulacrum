"""Audit: is the probe the bottleneck for detecting memory?
v2 confirmatory recording (dark, data seed 1, 8,192 training episodes), learner seed 0 networks.
(1) Same decay-curve skill as experiments/posthoc_permanence.py, probe fit on 1,024 training episodes (as stored)
    versus 4,096 (the 60,000-row cap lifted; otherwise the same probe code).
(2) Positive control: raw human senses now, versus raw senses now plus explicit copies from 2 and 4 steps ago
    (a perfect short memory handed to the probe). If the probe cannot turn that into skill at 1-5 steps since
    seen, it cannot detect a memory a network might have."""
import pickle, sys
import numpy as np
sys.path.insert(0, "/Users/henryvu/personal/umwelt/src")
import jax
from umwelt import learner, presets, probes
from umwelt.data import load

class Probe(probes.Probe):
    def __init__(self, X, Y, M, task, hidden, seed, groups, cap=60_000, steps=1500):
        rng = np.random.default_rng(seed)
        keep = M.any(1)
        eps = np.unique(groups); val_eps = rng.choice(eps, max(len(eps) // 5, 1), replace=False)
        is_val = np.isin(groups, val_eps)
        tr, va = np.flatnonzero(keep & ~is_val), np.flatnonzero(keep & is_val)
        if len(tr) > cap: tr = rng.choice(tr, cap, replace=False)
        if len(va) > 3000: va = rng.choice(va, 3000, replace=False)
        idx = np.concatenate([tr, va])
        self.mu, self.sd = X[idx].mean(0), X[idx].std(0) + 1e-6
        self.task = task
        Y = Y.astype(np.float32); m = M[idx]
        self.ym = (Y[idx] * m).sum(0) / np.maximum(m.sum(0), 1)
        self.ys = np.sqrt(((Y[idx] - self.ym) ** 2 * m).sum(0) / np.maximum(m.sum(0), 1)) + 1e-6
        Y = (Y - self.ym) / self.ys
        Z = (X - self.mu) / self.sd
        Mf = M.astype(np.float32)
        self.p = probes._fit(jax.random.PRNGKey(seed), Z[tr], Y[tr], Mf[tr], Z[va], Y[va], Mf[va], task, hidden, steps=steps)

N_TRAIN, ST = 8192, 8
d = load("dark", N_TRAIN + 256, seed=1)
gt = {k: v for k, v in d.items() if k.startswith("gt_")}
te = slice(N_TRAIN, N_TRAIN + 256)
Tte = probes.targets({k: np.asarray(v[te]) for k, v in gt.items()})
dark_te = np.repeat(~np.asarray(gt["gt_light"][te])[..., None], 8, -1)
Rv = {k: probes._rows(Tte, k, 4) for k in ("ego", "present", "vis", "since")}
Rv["dark"] = dark_te[:, probes.PAST::4].reshape(-1, 8)
m2 = lambda m: np.repeat(m, 2, axis=1)

def fit_targets(n_ep):
    T = probes.targets({k: np.asarray(v[:n_ep]) for k, v in gt.items()})
    return {k: probes._rows(T, k, ST) for k in ("ego", "present")}
TF = {n: fit_targets(n) for n in (1024, 4096)}

def score(Xfit, n_ep, Xev, cap, label):
    R = TF[n_ep]
    groups = np.repeat(np.arange(n_ep), len(range(probes.PAST, 256, ST)))
    p = Probe(Xfit, R["ego"].reshape(len(Xfit), -1), m2(R["present"]), "reg", 256, 0, groups, cap=cap)
    pred, Y = p(Xev), Rv["ego"].reshape(len(Xev), -1)
    mean = np.array([[R["ego"][R["present"][:, k], k, j].mean() if R["present"][:, k].any() else 0.0 for j in range(2)] for k in range(8)])
    const = np.broadcast_to(mean.reshape(1, -1), Y.shape)
    pres, vis, since, dk = Rv["present"], Rv["vis"], Rv["since"], Rv["dark"]
    out = {}
    for name, m in (("in view", pres & vis), ("1-5", pres & ~vis & (since >= 1) & (since <= 5)), ("6-20", pres & ~vis & (since >= 6) & (since <= 20)),
                    ("1-5 dark", pres & ~vis & (since >= 1) & (since <= 5) & dk)):
        s, b = probes._r2_cols(pred, Y, m2(m), m2(pres)), probes._r2_cols(const, Y, m2(m), m2(pres))
        out[name] = (s - b) / (1 - b)
    print(f"{label:58s} " + " ".join(f"{k}={v:+.3f}" for k, v in out.items()), flush=True)

def latents_rows(params, mu, sd, preset, sl, stride):
    rows = []
    for i in range(sl.start, sl.stop, 256):
        part = {k: v[i:min(i + 256, sl.stop)] for k, v in d.items() if k.startswith("obs_") or k in ("action", "gt_light")}
        o = (presets.observe(part, preset) - mu) / sd
        a = presets.actions(part, preset)
        H = learner.batched(learner._latents_jit, params, o, a)
        rows.append(H[:, probes.PAST::stride].reshape(-1, H.shape[-1]))
    return np.concatenate(rows)

R = "/Users/henryvu/personal/umwelt/results/runs/v2/dark_s0/"
for mind in ("human", "memoryless"):
    sv = pickle.load(open(R + f"{mind}_params.pkl", "rb"))
    pr = presets.PRESETS[mind]
    Xev = latents_rows(sv["params"], sv["mu"], sv["sd"], pr, te, 4)
    X4 = latents_rows(sv["params"], sv["mu"], sv["sd"], pr, slice(0, 4096), ST)
    X1 = X4[: 1024 * len(range(probes.PAST, 256, ST))]
    score(X1, 1024, Xev, 60_000, f"{mind} seed 0, probe on 1,024 episodes (as stored)")
    score(X4, 4096, Xev, 10**9, f"{mind} seed 0, probe on 4,096 episodes, no row cap")

# raw senses and the positive control (explicit 2- and 4-step memory)
sv = pickle.load(open(R + "human_params.pkl", "rb"))
pr = presets.PRESETS["human"]
def raw_rows(sl, stride, lags):
    rows = []
    for i in range(sl.start, sl.stop, 256):
        part = {k: v[i:min(i + 256, sl.stop)] for k, v in d.items() if k.startswith("obs_") or k in ("action", "gt_light")}
        o = (presets.observe(part, pr) - sv["mu"]) / sv["sd"]
        ts = np.arange(probes.PAST, 256, stride)
        rows.append(np.concatenate([o[:, ts - L] for L in lags], -1).reshape(-1, o.shape[-1] * len(lags)).astype(np.float32))
    return np.concatenate(rows)
for lags, label in (((0,), "raw senses now"), ((0, 2, 4), "raw senses now + copies 2 and 4 steps back")):
    Xev = raw_rows(te, 4, lags)
    X1 = raw_rows(slice(0, 1024), ST, lags)
    score(X1, 1024, Xev, 60_000, f"human {label}, probe on 1,024 episodes")
    if len(lags) > 1:
        X4 = raw_rows(slice(0, 4096), ST, lags)
        score(X4, 4096, Xev, 10**9, f"human {label}, probe on 4,096 episodes")
