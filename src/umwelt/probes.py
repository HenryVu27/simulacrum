"""The probe battery: which concepts can be read out of a mind's hidden state.

Probes are fit on training episodes and scored on held-out episodes. Each concept family is one
masked multi-output probe (kinds of object act as identities), either linear or a small MLP
(multilayer perceptron). The same battery runs on trained minds, untrained copies (floor), the
oracle (ceiling) and raw current observations (what the senses give without memory or learning).

Concepts:
  object_seen      where each kind of object is relative to the agent, while it is in view
  object_hidden    the same readout, scored while the object is out of view: object permanence
  object_now       the same readout over all moments
  object_past      where each object was 10 steps ago: how far into the past the mind lives
  object_allo_*    where each object is in the room (not relative to the agent), in view and hidden;
                   exploratory, added after the pilot to separate memory from self-localization
  number           how many objects exist
  self_motion      own displacement and turn over the last 5 steps
  self_location    own position in the room
  cause            for a moving inert object, whether the agent or another object set it moving
  elapsed          how long since an object was last in view
  relations        distance from each object to its nearest neighbour, independent of the agent

Regression scores are R^2 against the variance of the target over all relevant test samples, so
the seen and hidden subsets are on one scale. Classification scores are chance-normalized
(balanced) accuracy: 0 is chance, 1 is perfect.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from umwelt import kinds
from umwelt.world import ARENA

PAST = 10
MOTION = 5
CONCEPTS = ("object_seen", "object_hidden", "object_now", "object_past", "object_allo_seen", "object_allo_hidden",
            "number", "self_motion", "self_location", "cause", "elapsed", "relations")
K = kinds.N_KINDS


def _wrap_np(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _ego(vec, th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([c * vec[..., 0] + s * vec[..., 1], -s * vec[..., 0] + c * vec[..., 1]], -1)


def steps_since_seen(vis):
    """vis (E, T, N) bool -> steps since last in view, capped at 1000 if never seen."""
    E, T, N = vis.shape
    out = np.full((E, T, N), 1000, np.int32)
    last = np.full((E, N), -10_000)
    for t in range(T):
        last = np.where(vis[:, t], t, last)
        out[:, t] = np.minimum(t - last, 1000)
    return out


def _by_kind(x, kind, fill=0):
    """Re-index slot arrays (E, T, N, ...) by object kind -> (E, T, K, ...)."""
    E, T, N = x.shape[:3]
    out = np.full((E, T, K) + x.shape[3:], fill, dtype=x.dtype)
    e = np.arange(E)[:, None]
    for s in range(N):
        out[e, :, kind[:, s : s + 1]] = x[:, :, s][:, None]
    return out


def targets(g):
    """Ground-truth targets per (episode, time), indexed by object kind."""
    E, T, N = g["gt_obj_active"].shape
    th, apos = g["gt_agent_th"], g["gt_agent_pos"]
    kind = g["gt_obj_kind"][:, 0].astype(int)
    rel = g["gt_obj_pos"] - apos[:, :, None]
    ego = _ego(rel, th[:, :, None]) / ARENA * 4
    ego_past = _ego(np.roll(g["gt_obj_pos"], PAST, axis=1) - apos[:, :, None], th[:, :, None]) / ARENA * 4
    active = g["gt_obj_active"]
    d = np.linalg.norm(g["gt_obj_pos"][:, :, :, None] - g["gt_obj_pos"][:, :, None, :], axis=-1)
    both = active[:, :, :, None] & active[:, :, None, :] & ~np.eye(N, dtype=bool)
    nearest = np.where(both, d, np.inf).min(-1)
    nearest = np.where(np.isfinite(nearest), nearest / ARENA * 4, 0.0)
    disp = apos - np.roll(apos, MOTION, axis=1)
    dth = _wrap_np(th - np.roll(th, MOTION, axis=1))
    t = np.arange(T)[None, :].repeat(E, 0)
    return dict(
        present=_by_kind(active, kind, False) & (t >= PAST)[:, :, None],
        vis=_by_kind(g["gt_vis180"], kind, False),
        since=_by_kind(steps_since_seen(g["gt_vis180"]), kind, 1000),
        ego=_by_kind(ego.astype(np.float32), kind),
        ego_past=_by_kind(ego_past.astype(np.float32), kind),
        allo=_by_kind((g["gt_obj_pos"] / ARENA * 4).astype(np.float32), kind),
        nearest=_by_kind(nearest.astype(np.float32), kind),
        cause=_by_kind(g["gt_cause"].astype(np.int8), kind),
        n=active.sum(-1),
        motion=np.concatenate([_ego(disp, th) / ARENA * 20, np.sin(dth)[..., None], np.cos(dth)[..., None]], -1).astype(np.float32),
        loc=(apos / ARENA).astype(np.float32),
        t=t,
    )


# ---------------------------------------------------------------- probe trainer


def _init(key, d_in, d_out, hidden):
    if hidden == 0:
        return [{"w": jnp.zeros((d_in, d_out)), "b": jnp.zeros(d_out)}]
    k1, k2 = jax.random.split(key)
    return [
        {"w": jax.random.normal(k1, (d_in, hidden)) / jnp.sqrt(d_in), "b": jnp.zeros(hidden)},
        {"w": jax.random.normal(k2, (hidden, d_out)) / jnp.sqrt(hidden), "b": jnp.zeros(d_out)},
    ]


def _apply(p, x):
    for layer in p[:-1]:
        x = jax.nn.gelu(x @ layer["w"] + layer["b"])
    return x @ p[-1]["w"] + p[-1]["b"]


def _loss(p, x, y, m, task):
    out = _apply(p, x)
    if task == "reg":
        return jnp.sum(m * (out - y) ** 2) / (jnp.sum(m) + 1e-6)
    if task == "bin":
        ce = optax.sigmoid_binary_cross_entropy(out, y)
        return jnp.sum(m * ce) / (jnp.sum(m) + 1e-6)
    ce = optax.softmax_cross_entropy_with_integer_labels(out, y[:, 0].astype(jnp.int32))
    return jnp.sum(m[:, 0] * ce) / (jnp.sum(m[:, 0]) + 1e-6)


@partial(jax.jit, static_argnames=("task", "hidden", "steps", "batch"))
def _fit(key, X, Y, M, Xv, Yv, Mv, task, hidden, steps=1500, batch=1024):
    d_out = Y.shape[1] if task != "multi" else 8
    p = _init(key, X.shape[1], d_out, hidden)
    opt = optax.adamw(3e-3 if hidden == 0 else 1e-3, weight_decay=1e-4)
    st = opt.init(p)

    def body(carry, k):
        p, st, best, best_loss = carry
        i = jax.random.randint(k, (batch,), 0, X.shape[0])
        g = jax.grad(_loss)(p, X[i], Y[i], M[i], task)
        upd, st = opt.update(g, st, p)
        p = optax.apply_updates(p, upd)
        vl = _loss(p, Xv, Yv, Mv, task)
        better = vl < best_loss
        best = jax.tree.map(lambda a, b: jnp.where(better, a, b), p, best)
        return (p, st, best, jnp.minimum(vl, best_loss)), None

    (p, st, best, _), _ = jax.lax.scan(body, (p, st, p, jnp.inf), jax.random.split(key, steps))
    return best


def _prep(H, stride):
    return H[:, PAST::stride].reshape(-1, H.shape[-1]).astype(np.float32)


def _rows(T, key, stride):
    return T[key][:, PAST::stride].reshape(-1, *T[key].shape[2:])


class Probe:
    """A masked multi-output probe. task: reg (standardized MSE), bin (per-column), multi (one label).

    Validation rows come from held-out whole episodes (groups), so early stopping catches a probe
    that memorizes episodes instead of reading a concept.
    """

    def __init__(self, X, Y, M, task, hidden, seed, groups, max_rows=60_000):
        rng = np.random.default_rng(seed)
        keep = M.any(1) if M.ndim == 2 else M
        eps = np.unique(groups)
        val_eps = rng.choice(eps, max(len(eps) // 5, 1), replace=False)
        is_val = np.isin(groups, val_eps)
        tr = np.flatnonzero(keep & ~is_val)
        va = np.flatnonzero(keep & is_val)
        if len(tr) > max_rows:
            tr = rng.choice(tr, max_rows, replace=False)
        if len(va) > 3_000:
            va = rng.choice(va, 3_000, replace=False)
        idx = np.concatenate([tr, va])
        self.mu, self.sd = X[idx].mean(0), X[idx].std(0) + 1e-6
        self.task = task
        Y = Y.astype(np.float32)
        if task == "reg":
            m = M[idx]
            self.ym = (Y[idx] * m).sum(0) / np.maximum(m.sum(0), 1)
            self.ys = np.sqrt(((Y[idx] - self.ym) ** 2 * m).sum(0) / np.maximum(m.sum(0), 1)) + 1e-6
            Y = (Y - self.ym) / self.ys
        Z = (X - self.mu) / self.sd
        Mf = M.astype(np.float32).reshape(len(M), -1)
        if task == "bin":
            # Weight classes equally per column, so the probe is judged on balanced accuracy.
            pos = (Y[idx] * Mf[idx]).sum(0) / np.maximum(Mf[idx].sum(0), 1)
            pos = np.clip(pos, 0.02, 0.98)
            Mf = Mf * (Y / (2 * pos) + (1 - Y) / (2 * (1 - pos)))
        self.p = _fit(jax.random.PRNGKey(seed), Z[tr], Y[tr], Mf[tr], Z[va], Y[va], Mf[va], task, hidden)

    def __call__(self, X):
        out = np.asarray(_apply(self.p, jnp.asarray((X - self.mu) / self.sd)))
        return out * self.ys + self.ym if self.task == "reg" else out


class RidgeProbe:
    """A masked multi-output ridge regression (v3 Stage 0, Amendment 5: a candidate probe that cannot lose
    information by failing to fit). Each column is fit on its own masked training rows in closed form, once per
    penalty in a fixed grid; the fit whose error on held-out episodes (the same split rule as Probe) is lowest is
    kept. Regression only."""

    LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)

    def __init__(self, X, Y, M, task, hidden, seed, groups, max_rows=60_000):
        assert task == "reg"
        rng = np.random.default_rng(seed)
        keep = M.any(1)
        eps = np.unique(groups)
        val_eps = rng.choice(eps, max(len(eps) // 5, 1), replace=False)
        is_val = np.isin(groups, val_eps)
        tr, va = np.flatnonzero(keep & ~is_val), np.flatnonzero(keep & is_val)
        if len(tr) > max_rows:
            tr = rng.choice(tr, max_rows, replace=False)
        idx = np.concatenate([tr, va])
        self.mu, self.sd = X[idx].mean(0), X[idx].std(0) + 1e-6
        Z = ((X - self.mu) / self.sd).astype(np.float64)
        Y = Y.astype(np.float64)
        D = Z.shape[1]
        self.W = np.zeros((D + 1, Y.shape[1]))
        groups_of = {}
        for c in range(Y.shape[1]):  # columns sharing a mask share one factorization
            groups_of.setdefault(M[:, c].tobytes(), []).append(c)
        for cols in groups_of.values():
            m = M[:, cols[0]]
            a, b = tr[m[tr]], va[m[va]]
            if len(a) < 10:
                continue
            A = np.c_[Z[a], np.ones(len(a))]
            G, R = A.T @ A, A.T @ Y[np.ix_(a, cols)]
            best = None
            for lam in self.LAMBDAS:
                reg = lam * len(a) * np.eye(D + 1)
                reg[-1, -1] = 0.0  # no penalty on the intercept
                W = np.linalg.solve(G + reg, R)
                err = np.mean((np.c_[Z[b], np.ones(len(b))] @ W - Y[np.ix_(b, cols)]) ** 2) if len(b) else 0.0
                if best is None or err < best[0]:
                    best = (err, W)
            self.W[:, cols] = best[1]

    def __call__(self, X):
        Z = (X - self.mu) / self.sd
        return np.c_[Z, np.ones(len(Z))] @ self.W


def _r2_cols(pred, Y, M, ref_mask):
    """R^2 per column on samples in M, against the variance over ref_mask. Count-weighted mean."""
    scores, counts = [], []
    for c in range(Y.shape[1]):
        m, ref = M[:, c], ref_mask[:, c]
        if m.sum() < 30 or ref.sum() < 30:
            continue
        var = Y[ref, c].var() + 1e-9
        scores.append(1 - np.mean((pred[m, c] - Y[m, c]) ** 2) / var)
        counts.append(m.sum())
    return float(np.average(scores, weights=counts)) if scores else float("nan")


def run_battery(Htr, Ttr, Hte, Tte, seed=0, hidden=256, stride_tr=8, stride_te=4):
    """H*: (E, T, D) hidden states (or raw observations). T*: targets(). Returns concept -> score."""
    Xtr, Xte = _prep(Htr, stride_tr), _prep(Hte, stride_te)
    groups = np.repeat(np.arange(Htr.shape[0]), len(range(PAST, Htr.shape[1], stride_tr)))
    R = {k: (_rows(Ttr, k, stride_tr), _rows(Tte, k, stride_te)) for k in Ttr}
    out = {}

    def xy2(a):  # (n, K, 2) -> (n, 2K)
        return a.reshape(len(a), -1)

    def mask2(m):  # (n, K) -> (n, 2K)
        return np.repeat(m, 2, axis=1)

    pres_tr, pres_te = R["present"]
    probe = Probe(Xtr, xy2(R["ego"][0]), mask2(pres_tr), "reg", hidden, seed, groups)
    pred, Y = probe(Xte), xy2(R["ego"][1])
    ref = mask2(pres_te)
    out["object_seen"] = _r2_cols(pred, Y, mask2(pres_te & R["vis"][1]), ref)
    out["object_hidden"] = _r2_cols(pred, Y, mask2(pres_te & ~R["vis"][1]), ref)
    out["object_now"] = _r2_cols(pred, Y, ref, ref)

    probe = Probe(Xtr, xy2(R["allo"][0]), mask2(pres_tr), "reg", hidden, seed + 7, groups)
    pred, Y = probe(Xte), xy2(R["allo"][1])
    out["object_allo_seen"] = _r2_cols(pred, Y, mask2(pres_te & R["vis"][1]), ref)
    out["object_allo_hidden"] = _r2_cols(pred, Y, mask2(pres_te & ~R["vis"][1]), ref)

    probe = Probe(Xtr, xy2(R["ego_past"][0]), mask2(pres_tr), "reg", hidden, seed + 1, groups)
    out["object_past"] = _r2_cols(probe(Xte), xy2(R["ego_past"][1]), ref, ref)

    el_tr = pres_tr & ~R["vis"][0] & (R["since"][0] < 200)
    el_te = pres_te & ~R["vis"][1] & (R["since"][1] < 200)
    probe = Probe(Xtr, np.log1p(R["since"][0]).astype(np.float32), el_tr, "reg", hidden, seed + 2, groups)
    out["elapsed"] = _r2_cols(probe(Xte), np.log1p(R["since"][1]).astype(np.float32), el_te, el_te)

    rel_tr = pres_tr & (R["n"][0] >= 2)[:, None]
    rel_te = pres_te & (R["n"][1] >= 2)[:, None]
    probe = Probe(Xtr, R["nearest"][0], rel_tr, "reg", hidden, seed + 3, groups)
    out["relations"] = _r2_cols(probe(Xte), R["nearest"][1], rel_te, rel_te)

    for name, key in (("self_motion", "motion"), ("self_location", "loc")):
        ytr, yte = R[key]
        mtr, mte = np.ones_like(ytr, bool), np.ones_like(yte, bool)
        probe = Probe(Xtr, ytr, mtr, "reg", hidden, seed + 4, groups)
        out[name] = _r2_cols(probe(Xte), yte, mte, mte)

    ntr, nte = R["n"]
    probe = Probe(Xtr, ntr[:, None].astype(np.float32), np.ones((len(ntr), 1), bool), "multi", hidden, seed + 5, groups)
    pred = probe(Xte).argmax(-1)
    acc = np.mean(pred == nte)
    chance = np.bincount(nte).max() / len(nte)
    out["number"] = float((acc - chance) / (1 - chance))

    inert = ~kinds.ANIMATE[None, :]
    c_tr, c_te = R["cause"]
    m_tr = np.isin(c_tr, (1, 2)) & pres_tr & inert
    m_te = np.isin(c_te, (1, 2)) & pres_te & inert
    if m_tr.sum() < 200 or m_te.sum() < 50:  # e.g. a ghost world, where nothing is ever pushed
        out["cause"] = float("nan")
        return out
    probe = Probe(Xtr, (c_tr == 1).astype(np.float32), m_tr, "bin", hidden, seed + 6, groups)
    logit = probe(Xte)
    scores, counts = [], []
    for k in range(K):
        m = m_te[:, k]
        y = c_te[m, k] == 1
        if m.sum() < 30 or y.all() or (~y).all():
            continue
        p = logit[m, k] > 0
        bacc = 0.5 * (np.mean(p[y]) + np.mean(~p[~y]))
        scores.append((bacc - 0.5) / 0.5)
        counts.append(m.sum())
    out["cause"] = float(np.average(scores, weights=counts)) if scores else float("nan")
    return out
