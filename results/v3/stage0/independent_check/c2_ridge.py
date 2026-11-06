"""Check 2: RidgeProbe against an independent ridge (sklearn, centred, unpenalized intercept), its penalty choice,
and reproduction of the stored ridge scores at 1,024 fitting episodes with independent scoring code."""
import json, sys, time
import numpy as np
from sklearn.linear_model import Ridge
from common import *  # noqa

n_fit = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
d, gt, T = get_data()
raw = get_raw(d)
fit, te = slice(0, n_fit), slice(N_TRAIN, N_TRAIN + N_TEST)
Rtr, Rte = rows_of(T, fit, 8), rows_of(T, te, 4)
groups = np.repeat(np.arange(n_fit), len(TIMES_TR))
stored = json.loads((REPO / "results/v3/stage0/scores.json").read_text())
cname = {1024: "ridge", 4096: "ridge, 4,096 episodes", 8192: "ridge, 8,192 episodes"}[n_fit]
max_rows = 60_000 * n_fit // 1024

for fname, lags in (("raw", (0,)), ("buffer 2,4", (0, 2, 4)), ("buffer 3,6,9", (0, 3, 6, 9))):
    t0 = time.time()
    Xtr, Xte = lagged(raw, fit, TIMES_TR, lags).reshape(-1, 311 * len(lags)), lagged(raw, te, TIMES_TE, lags).reshape(-1, 311 * len(lags))
    Ytr = Rtr["ego"].reshape(len(Xtr), -1); Mtr = m2(Rtr["present"])
    p = probes.RidgeProbe(Xtr, Ytr, Mtr, "reg", 256, 0, groups, max_rows)
    pred = p(Xte)
    # independent: same split rule (seeded the same way), sklearn per kind, choose alpha on val episodes
    rng = np.random.default_rng(0)
    keep = Mtr.any(1)
    val_eps = rng.choice(np.unique(groups), max(n_fit // 5, 1), replace=False)
    is_val = np.isin(groups, val_eps)
    tr, va = np.flatnonzero(keep & ~is_val), np.flatnonzero(keep & is_val)
    assert len(tr) <= max_rows, "row cap binds"
    idx = np.r_[tr, va]
    mu, sd = Xtr[idx].mean(0), Xtr[idx].std(0) + 1e-6
    Z, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    mine = np.zeros_like(pred)
    chosen, curves = [], []
    for k in range(8):
        a, b = tr[Mtr[tr, 2 * k]], va[Mtr[va, 2 * k]]
        if len(a) < 10:
            continue
        errs = []
        for lam in probes.RidgeProbe.LAMBDAS:
            r = Ridge(alpha=lam * len(a), fit_intercept=True, solver="cholesky").fit(Z[a], Ytr[a][:, 2 * k: 2 * k + 2])
            errs.append(((r.predict(Z[b]) - Ytr[b][:, 2 * k: 2 * k + 2]) ** 2).mean())
        best = int(np.argmin(errs))
        r = Ridge(alpha=probes.RidgeProbe.LAMBDAS[best] * len(a), fit_intercept=True, solver="cholesky").fit(Z[a], Ytr[a][:, 2 * k: 2 * k + 2])
        mine[:, 2 * k: 2 * k + 2] = r.predict(Zte)
        chosen.append(probes.RidgeProbe.LAMBDAS[best]); curves.append(np.round(np.array(errs) / min(errs), 4).tolist())
    print(f"{fname}: max |RidgeProbe - sklearn| prediction {np.abs(pred - mine).max():.2e}; chosen lambda per kind {chosen}")
    for k, c in enumerate(curves):
        print(f"   kind {k} val err / min over grid {c}")
    s = my_scores(pred, Rtr, Rte)
    st = stored[cname][fname]
    print("   recomputed vs stored: " + ", ".join(f"{k} {s[k]:+.5f}/{st[k]:+.5f}" for k in ("object_seen", "object_hidden", "1-5 steps", "6-20 steps", "in view")),
          f"| object_now {s['object_now']:+.5f} ({time.time() - t0:.0f} s)", flush=True)
