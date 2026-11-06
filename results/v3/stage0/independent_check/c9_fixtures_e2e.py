"""Check 9: end-to-end reproduction of the engineered and tracking fixtures' stored scores (ridge and MLP at 1,024),
with own scoring; then the engineered criteria under a literal reading of 'noise of the hidden states' scale'
(noise sd equal to each unit's sd, instead of 0.1 of it)."""
import json, pickle
import numpy as np
from common import *  # noqa
from umwelt import learner

d, gt, T = get_data()
raw = get_raw(d)
n_fit = 1024
fit_all, fit, te = slice(0, N_TRAIN), slice(0, n_fit), slice(N_TRAIN, N_TRAIN + N_TEST)
Rtr, Rte = rows_of(T, fit, 8), rows_of(T, te, 4)
groups = np.repeat(np.arange(n_fit), len(TIMES_TR))
stored = json.loads((REPO / "results/v3/stage0/scores.json").read_text())
gtf = {k: np.asarray(v[fit_all]) for k, v in gt.items()}
gte = {k: np.asarray(v[te]) for k, v in gt.items()}

def embed(train, test, noise, seed=0):
    mu, sd = train.reshape(-1, train.shape[-1]).mean(0), train.reshape(-1, train.shape[-1]).std(0) + 1e-6
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1.0 / np.sqrt(train.shape[-1]), (train.shape[-1], 128)).astype(np.float32)
    out = []
    for x in (train, test):
        z = ((x - mu) / sd) @ W
        out.append((z + rng.normal(0.0, noise, z.shape) * z.reshape(-1, 128).std(0)).astype(np.float32))
    return out

eng = {w: (s0.engineered(gtf, w, TIMES_TR), s0.engineered(gte, w, TIMES_TE)) for w in ("full", "last seen", "last seen, updated")}
saved = pickle.load(open(REPO / "results/v3/stage0/track_params.pkl", "rb"))
acts = presets.actions(d, presets.PRESETS["human"])
def track_rows(eps, times):
    H = []
    for i in range(eps.start, eps.stop, 256):
        j = min(i + 256, eps.stop)
        x = (np.asarray(raw[i:j]).astype(np.float32) - saved["mu"]) / saved["sd"]
        H.append(learner.batched(learner._latents_jit, saved["params"], x, acts[i:j])[:, times])
    return np.concatenate(H)
trk = (track_rows(fit, TIMES_TR), track_rows(te, TIMES_TE))
rawf = (lagged(raw, fit, TIMES_TR), lagged(raw, te, TIMES_TE))

def run(cls, fx):
    a, b = fx
    a, b = a[:n_fit].reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    p = cls(a, Rtr["ego"].reshape(len(a), -1), m2(Rtr["present"]), "reg", 256, 0, groups, 60_000)
    q = cls(a, Rtr["loc"], np.ones_like(Rtr["loc"], bool), "reg", 256, 4, groups, 60_000)
    return my_scores(p(b), Rtr, Rte, q(b))

KEYS = ("object_seen", "object_hidden", "self_location", "in view", "1-5 steps", "6-20 steps", "21-60 steps", "over 60 steps", "never seen")
for cname, cls in (("ridge", probes.RidgeProbe), ("battery", probes.Probe)):
    res = {"raw": run(cls, rawf), "track": run(cls, trk)}
    for w in eng:
        res[w] = run(cls, tuple(embed(*eng[w], 0.1)))
    for f, r in res.items():
        dev = max(abs(r[k] - stored[cname][f][k]) for k in KEYS)
        print(f"{cname:8s} {f:20s} max |recomputed - stored| over {len(KEYS)} scores: {dev:.2e}", flush=True)
    for noise in (1.0,):
        alt = {w: run(cls, tuple(embed(*eng[w], noise))) for w in ("full", "last seen, updated")}
        c2 = {b: alt["last seen, updated"][b] - res["raw"][b] for b in ("1-5 steps", "6-20 steps")}
        c3 = {n: alt["full"][n] for n, _ in s0.BINS}
        print(f"{cname:8s} noise sd = {noise} x unit sd: last seen, updated minus raw {c2} (needs > 0.30); full by bin "
              + ", ".join(f"{k} {v:.3f}" for k, v in c3.items()) + " (needs > 0.80)", flush=True)
