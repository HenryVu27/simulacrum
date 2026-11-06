"""Check 4: is the battery's MLP 'an MLP with early stopping'? Trace the held-out loss over its 1,500-step budget
(same code path as probes._fit, plus a record of the step with the best held-out loss), then train the same MLP
with real early stopping (up to 40,000 steps, stop after 4,000 without improvement) and score raw vs buffers."""
import sys, time
from functools import partial
import jax, jax.numpy as jnp, numpy as np, optax
from common import *  # noqa
from umwelt.probes import _init, _apply, _loss

n_fit = int(sys.argv[1])
which = sys.argv[2].split(";") if len(sys.argv) > 2 else ["raw", "buffer 2,4", "buffer 3,6,9"]
LAGS = {"raw": (0,), "buffer 2,4": (0, 2, 4), "buffer 3,6,9": (0, 3, 6, 9)}
CHUNK = 500

@partial(jax.jit, static_argnames=("n",))
def chunk(carry, key, X, Y, M, Xv, Yv, Mv, n):
    opt = optax.adamw(1e-3, weight_decay=1e-4)
    def body(c, k):
        p, st, best, bl, bstep, step = c
        i = jax.random.randint(k, (1024,), 0, X.shape[0])
        g = jax.grad(_loss)(p, X[i], Y[i], M[i], "reg")
        upd, st = opt.update(g, st, p)
        p = optax.apply_updates(p, upd)
        vl = _loss(p, Xv, Yv, Mv, "reg")
        better = vl < bl
        best = jax.tree.map(lambda a, b: jnp.where(better, a, b), p, best)
        return (p, st, best, jnp.minimum(vl, bl), jnp.where(better, step, bstep), step + 1), vl
    return jax.lax.scan(body, carry, jax.random.split(key, n))

def fit(Xtr, Ytr, Mtr, groups, seed=0, max_steps=1500, patience=None):
    """Probe.__init__ + _fit, reproduced, with a held-out loss trace. max_steps=1500 and patience=None is the battery."""
    rng = np.random.default_rng(seed)
    keep = Mtr.any(1)
    val_eps = rng.choice(np.unique(groups), max(len(np.unique(groups)) // 5, 1), replace=False)
    is_val = np.isin(groups, val_eps)
    tr, va = np.flatnonzero(keep & ~is_val), np.flatnonzero(keep & is_val)
    cap = 60_000 * n_fit // 1024
    if len(tr) > cap:
        tr = rng.choice(tr, cap, replace=False)
    if len(va) > 3000:
        va = rng.choice(va, 3000, replace=False)
    idx = np.r_[tr, va]
    mu, sd = Xtr[idx].mean(0), Xtr[idx].std(0) + 1e-6
    m = Mtr[idx]
    ym = (Ytr[idx] * m).sum(0) / np.maximum(m.sum(0), 1)
    ys = np.sqrt(((Ytr[idx] - ym) ** 2 * m).sum(0) / np.maximum(m.sum(0), 1)) + 1e-6
    Ys = (Ytr - ym) / ys
    Z = (Xtr - mu) / sd
    Mf = Mtr.astype(np.float32)
    key = jax.random.PRNGKey(seed)
    p = _init(key, Z.shape[1], Ys.shape[1], 256)
    opt = optax.adamw(1e-3, weight_decay=1e-4)
    carry = (p, opt.init(p), p, jnp.inf, 0, 0)
    X, Y, M = jnp.asarray(Z[tr]), jnp.asarray(Ys[tr]), jnp.asarray(Mf[tr])
    Xv, Yv, Mv = jnp.asarray(Z[va]), jnp.asarray(Ys[va]), jnp.asarray(Mf[va])
    if patience is None:  # exactly the battery's key schedule: one scan of 1,500 split keys
        carry, trace = chunk(carry, key, X, Y, M, Xv, Yv, Mv, n=max_steps)
        trace = np.asarray(trace)
    else:
        trace = []
        k = jax.random.PRNGKey(seed + 1000)
        while True:
            k, sub = jax.random.split(k)
            carry, tr_ = chunk(carry, sub, X, Y, M, Xv, Yv, Mv, n=CHUNK)
            trace.append(np.asarray(tr_))
            step, bstep = int(carry[5]), int(carry[4])
            if step >= max_steps or step - bstep > patience:
                break
        trace = np.concatenate(trace)
    best = carry[2]
    predict = lambda Xn: np.asarray(_apply(best, jnp.asarray((Xn - mu) / sd))) * ys + ym
    return predict, trace, int(carry[4]), len(tr)

d, gt, T = get_data()
raw = get_raw(d)
fit_, te = slice(0, n_fit), slice(N_TRAIN, N_TRAIN + N_TEST)
Rtr, Rte = rows_of(T, fit_, 8), rows_of(T, te, 4)
groups = np.repeat(np.arange(n_fit), len(TIMES_TR))
for fname in which:
    lags = LAGS[fname]
    Xtr = lagged(raw, fit_, TIMES_TR, lags).reshape(-1, 311 * len(lags))
    Xte = lagged(raw, te, TIMES_TE, lags).reshape(-1, 311 * len(lags))
    Ytr, Mtr = Rtr["ego"].reshape(len(Xtr), -1), m2(Rtr["present"])
    t0 = time.time()
    pr, trace, bstep, ntr = fit(Xtr, Ytr, Mtr, groups)
    ref = probes.Probe(Xtr, Ytr, Mtr, "reg", 256, 0, groups, 60_000 * n_fit // 1024)
    agree = np.abs(pr(Xte) - ref(Xte)).max()
    s = my_scores(pr(Xte), Rtr, Rte)
    q = [trace[i] for i in (99, 299, 749, 1199, 1499)]
    print(f"[{n_fit}] {fname:13s} BATTERY (1,500 steps): best held-out step {bstep}, held-out loss at 100/300/750/1200/1500 "
          + "/".join(f"{x:.4f}" for x in q) + f"; min {trace.min():.4f} | seen {s['object_seen']:+.4f} hidden {s['object_hidden']:+.4f} "
          f"now {s['object_now']:+.4f} 1-5 {s['1-5 steps']:+.4f} | max |diff| vs probes.Probe {agree:.1e} | train rows {ntr} ({time.time() - t0:.0f} s)", flush=True)
    t0 = time.time()
    pr, trace, bstep, _ = fit(Xtr, Ytr, Mtr, groups, max_steps=40_000, patience=4_000)
    s = my_scores(pr(Xte), Rtr, Rte)
    print(f"[{n_fit}] {fname:13s} EARLY STOPPING (patience 4,000): ran {len(trace)} steps, best step {bstep}, held-out loss min "
          f"{trace.min():.4f} | seen {s['object_seen']:+.4f} hidden {s['object_hidden']:+.4f} now {s['object_now']:+.4f} "
          f"1-5 {s['1-5 steps']:+.4f} 6-20 {s['6-20 steps']:+.4f} ({time.time() - t0:.0f} s)", flush=True)
