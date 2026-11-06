"""Check 8: how large is (buffer - raw) object_seen against probe-seed noise and test-episode sampling?
Paired bootstrap over the 256 held-out episodes (1,000 resamples), probe seeds 0, 1, 2. Usage: c8_paired.py CLS N_FIT"""
import sys, time
import numpy as np
from common import *  # noqa

cls = {"mlp": probes.Probe, "ridge": probes.RidgeProbe}[sys.argv[1]]
n_fit = int(sys.argv[2])
seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2]
LAGS = {"raw": (0,), "buffer 2,4": (0, 2, 4), "buffer 3,6,9": (0, 3, 6, 9)}
d, gt, T = get_data()
raw = get_raw(d)
fit, te = slice(0, n_fit), slice(N_TRAIN, N_TRAIN + N_TEST)
Rtr, Rte = rows_of(T, fit, 8), rows_of(T, te, 4)
groups = np.repeat(np.arange(n_fit), len(TIMES_TR))
nr = len(TIMES_TE)
Y = Rte["ego"].reshape(-1, 16)
pres, vis = m2(Rte["present"]), m2(Rte["vis"])
seen = pres & vis
ep = np.repeat(np.arange(N_TEST), nr)

def per_episode(pred):
    """(256, 16) sums so a bootstrap over episodes can rebuild _r2_cols for object_seen."""
    se = np.zeros((N_TEST, 16)); ns = np.zeros((N_TEST, 16)); n = np.zeros((N_TEST, 16)); s1 = np.zeros((N_TEST, 16)); s2 = np.zeros((N_TEST, 16))
    for arr, val in ((se, (pred - Y) ** 2 * seen), (ns, seen), (n, pres), (s1, Y * pres), (s2, Y ** 2 * pres)):
        np.add.at(arr, ep, val)
    return se, ns, n, s1, s2

def r2_from(sums, w):
    se, ns, n, s1, s2 = (np.tensordot(w, a, 1) for a in sums)
    var = s2 / n - (s1 / n) ** 2
    ok = (ns >= 30) & (n >= 30)
    sc = 1 - (se / np.maximum(ns, 1)) / (var + 1e-9)
    return np.average(sc[ok], weights=ns[ok])

rng = np.random.default_rng(7)
W = np.stack([np.bincount(rng.integers(0, N_TEST, N_TEST), minlength=N_TEST) for _ in range(1000)])
for seed in seeds:
    sums, point = {}, {}
    for fname, lags in LAGS.items():
        t0 = time.time()
        Xtr = lagged(raw, fit, TIMES_TR, lags).reshape(-1, 311 * len(lags))
        Xte = lagged(raw, te, TIMES_TE, lags).reshape(-1, 311 * len(lags))
        p = cls(Xtr, Rtr["ego"].reshape(len(Xtr), -1), m2(Rtr["present"]), "reg", 256, seed, groups, 60_000 * n_fit // 1024)
        pred = p(Xte)
        sums[fname] = per_episode(pred)
        point[fname] = r2(pred, Y, seen, pres)
        del Xtr, Xte
    for b in ("buffer 2,4", "buffer 3,6,9"):
        diffs = np.array([r2_from(sums[b], w) - r2_from(sums["raw"], w) for w in W])
        print(f"{sys.argv[1]} {n_fit} seed {seed}: object_seen raw {point['raw']:+.4f}, {b} {point[b]:+.4f}, diff {point[b] - point['raw']:+.4f} "
              f"[paired episode bootstrap 95%: {np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}]", flush=True)
