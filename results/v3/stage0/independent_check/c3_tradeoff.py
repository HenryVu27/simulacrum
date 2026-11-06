"""Check 3: is the ridge object_seen drop a trade-off of one readout shared by seen and hidden rows, or a loss?
Own ridge (float64 normal equations, unpenalized intercept, lambda * n grid as RidgeProbe, chosen on held-out
episodes). Per kind: Gram matrices of seen and hidden training rows, so pooled / seen-only / hidden-only fits
share them. Reports test R^2 (seen, hidden, now) and in-sample R^2 on the fitting rows (no estimation variance)."""
import sys, time
import numpy as np
from common import *  # noqa

LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
n_fit = int(sys.argv[1])
fixt = sys.argv[2].split(";") if len(sys.argv) > 2 else ["raw", "buffer 2,4", "buffer 3,6,9"]
LAGS = {"raw": (0,), "buffer 2,4": (0, 2, 4), "buffer 3,6,9": (0, 3, 6, 9)}
d, gt, T = get_data()
raw = get_raw(d)
fit, te = slice(0, n_fit), slice(N_TRAIN, N_TRAIN + N_TEST)
Rtr, Rte = rows_of(T, fit, 8), rows_of(T, te, 4)
groups = np.repeat(np.arange(n_fit), len(TIMES_TR))
rng = np.random.default_rng(0)
val_eps = rng.choice(np.unique(groups), max(n_fit // 5, 1), replace=False)
is_val = np.isin(groups, val_eps)

def gram(Z, Y, rows):
    D = Z.shape[1] + 1
    G, R = np.zeros((D, D)), np.zeros((D, Y.shape[1]))
    for i in range(0, len(rows), 20000):
        r = rows[i:i + 20000]
        A = np.c_[Z[r].astype(np.float64), np.ones(len(r))]
        G += A.T @ A; R += A.T @ Y[r]
    return G, R, len(rows)

def solve(G, R, n, lam):
    reg = lam * n * np.eye(len(G)); reg[-1, -1] = 0
    return np.linalg.solve(G + reg, R)

def r2_on(P, Y, rows_by_kind, refvar):
    num, w = [], []
    for k, rows in rows_by_kind.items():
        if len(rows) < 30:
            continue
        for j in range(2):
            num.append(1 - ((P[k][rows, j] - Y[rows, 2 * k + j]) ** 2).mean() / refvar[2 * k + j]); w.append(len(rows))
    return float(np.average(num, weights=w))

for fname in fixt:
    t0 = time.time()
    lags = LAGS[fname]
    Xtr = lagged(raw, fit, TIMES_TR, lags).reshape(-1, 311 * len(lags))
    Xte = lagged(raw, te, TIMES_TE, lags).reshape(-1, 311 * len(lags)).astype(np.float64)
    pres, vis = Rtr["present"], Rtr["vis"]
    keep = pres.any(1)
    tr_all, va_all = np.flatnonzero(keep & ~is_val), np.flatnonzero(keep & is_val)
    idx = np.r_[tr_all, va_all]
    mu, sd = Xtr[idx].astype(np.float64).mean(0), Xtr[idx].astype(np.float64).std(0) + 1e-6
    Xtr -= mu.astype(np.float32); Xtr /= sd.astype(np.float32); Ztr = Xtr; Zte = ((Xte - mu) / sd)
    del Xte
    Ytr, Yte = Rtr["ego"].reshape(len(Ztr), -1).astype(np.float64), Rte["ego"].reshape(len(Zte), -1).astype(np.float64)
    pte, vte = Rte["present"], Rte["vis"]
    refte = np.array([Yte[pte[:, k], 2 * k + j].var() for k in range(8) for j in range(2)])
    reftr = np.array([Ytr[pres[:, k] & ~is_val, 2 * k + j].var() for k in range(8) for j in range(2)])
    Ate = np.c_[Zte, np.ones(len(Zte))]
    preds = {v: {} for v in ("pooled", "seen-only", "hidden-only")}
    insample = {v: {} for v in ("pooled",)}
    lam_used = {v: [] for v in preds}
    for k in range(8):
        s_tr = tr_all[pres[tr_all, k] & vis[tr_all, k]]; h_tr = tr_all[pres[tr_all, k] & ~vis[tr_all, k]]
        s_va = va_all[pres[va_all, k] & vis[va_all, k]]; h_va = va_all[pres[va_all, k] & ~vis[va_all, k]]
        cols = [2 * k, 2 * k + 1]
        Gs, Rs, ns = gram(Ztr, Ytr[:, cols], s_tr)
        Gh, Rh, nh = gram(Ztr, Ytr[:, cols], h_tr)
        for v, (G, R, n, va) in {"pooled": (Gs + Gh, Rs + Rh, ns + nh, np.r_[s_va, h_va]), "seen-only": (Gs, Rs, ns, s_va),
                                 "hidden-only": (Gh, Rh, nh, h_va)}.items():
            Ava = np.c_[Ztr[va].astype(np.float64), np.ones(len(va))]
            best = min(((((Ava @ solve(G, R, n, lam)) - Ytr[np.ix_(va, cols)]) ** 2).mean(), lam) for lam in LAMS)
            W = solve(G, R, n, best[1])
            lam_used[v].append(best[1])
            preds[v][k] = Ate @ W
            if v == "pooled":
                insample[v][k] = Ztr @ W[:-1].astype(np.float32) + W[-1].astype(np.float32)
    seen_te = {k: np.flatnonzero(pte[:, k] & vte[:, k]) for k in range(8)}
    hid_te = {k: np.flatnonzero(pte[:, k] & ~vte[:, k]) for k in range(8)}
    now_te = {k: np.flatnonzero(pte[:, k]) for k in range(8)}
    seen_tr = {k: tr_all[pres[tr_all, k] & vis[tr_all, k]] for k in range(8)}
    hid_tr = {k: tr_all[pres[tr_all, k] & ~vis[tr_all, k]] for k in range(8)}
    now_tr = {k: tr_all[pres[tr_all, k]] for k in range(8)}
    P = preds["pooled"]
    print(f"[{n_fit} eps] {fname:13s} TEST pooled: seen {r2_on(P, Yte, seen_te, refte):+.4f} hidden {r2_on(P, Yte, hid_te, refte):+.4f} "
          f"now {r2_on(P, Yte, now_te, refte):+.4f} | seen-only probe: seen {r2_on(preds['seen-only'], Yte, seen_te, refte):+.4f} | "
          f"hidden-only probe: hidden {r2_on(preds['hidden-only'], Yte, hid_te, refte):+.4f}", flush=True)
    I = insample["pooled"]
    print(f"[{n_fit} eps] {fname:13s} IN-SAMPLE pooled (fitting rows): seen {r2_on(I, Ytr, seen_tr, reftr):+.4f} hidden "
          f"{r2_on(I, Ytr, hid_tr, reftr):+.4f} now {r2_on(I, Ytr, now_tr, reftr):+.4f}   lambdas {lam_used} ({time.time() - t0:.0f} s)", flush=True)
