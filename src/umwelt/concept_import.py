"""Concept import: look inside each mind for features that no human-named variable explains.

A sparse autoencoder (SAE) splits a mind's hidden states into many sparsely active features. Each
feature is then explained, by ridge regression on held-out episodes, from two sources:
  perceptual  the mind's own current senses
  conceptual  named world variables (positions, visibility, elapsed time, causes, counts, self-motion)
A feature that neither explains, yet changes slowly over time (high autocorrelation), is a candidate
unnamed concept: something the mind tracks that our vocabulary does not name. Those are the ones to
inspect and, later, try to teach.
"""

import json
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from umwelt import presets, probes
from umwelt.data import load
from umwelt.experiment import RESULTS


def train_sae(X, n_features=512, l1=3e-3, steps=3000, seed=0):
    key = jax.random.PRNGKey(seed)
    d = X.shape[1]
    k1, k2 = jax.random.split(key)
    p = {"we": jax.random.normal(k1, (d, n_features)) / jnp.sqrt(d), "be": jnp.zeros(n_features),
         "wd": jax.random.normal(k2, (n_features, d)) / jnp.sqrt(n_features), "bd": jnp.zeros(d)}
    opt = optax.adam(1e-3)
    st = opt.init(p)
    Xj = jnp.asarray(X)

    def loss(p, x):
        f = jax.nn.relu(x @ p["we"] + p["be"])
        wd = p["wd"] / (jnp.linalg.norm(p["wd"], axis=1, keepdims=True) + 1e-6)
        rec = f @ wd + p["bd"]
        return jnp.mean((rec - x) ** 2) + l1 * jnp.mean(jnp.sum(f, -1))

    @jax.jit
    def step(p, st, k):
        i = jax.random.randint(k, (2048,), 0, Xj.shape[0])
        g = jax.grad(loss)(p, Xj[i])
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st

    for i in range(steps):
        key, k = jax.random.split(key)
        p, st = step(p, st, k)
    return jax.device_get(p)


def features(p, X):
    return np.maximum(X @ p["we"] + p["be"], 0.0)


def _named(T):
    """Named world variables per (episode, time), flattened to columns."""
    cols = [T["loc"], T["motion"], T["n"][..., None].astype(np.float32)]
    E, Tn, K = T["present"].shape
    cols.append(T["present"].astype(np.float32))
    cols.append((T["ego"] * T["present"][..., None]).reshape(E, Tn, -1))
    cols.append((T["allo"] * T["present"][..., None]).reshape(E, Tn, -1))
    cols.append(T["vis"].astype(np.float32))
    cols.append(np.log1p(np.minimum(T["since"], 300)).astype(np.float32))
    cols.append(np.stack([(T["cause"] == c).any(-1) for c in (1, 2, 3)], -1).astype(np.float32))
    return np.concatenate(cols, -1)


def _explained(Ztr, Ytr, Zte, Yte, lam=1.0):
    """R^2 of each feature column Y from predictors Z, fit on train, scored on test."""
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    A, B = (Ztr - mu) / sd, (Zte - mu) / sd
    ym = Ytr.mean(0)
    W = np.linalg.solve(A.T @ A + lam * len(A) * 1e-3 * np.eye(A.shape[1]), A.T @ (Ytr - ym))
    pred = B @ W + ym
    return 1 - ((pred - Yte) ** 2).sum(0) / (((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-9)


def analyse(mind, world="lawful", seed=0, tag="main", stride=2):
    run = RESULTS / "runs" / tag / f"{world}_s{seed}"
    with open(run / f"{mind}_latents.pkl", "rb") as f:
        _, H = pickle.load(f)  # held-out episodes 1024..1279
    H = H.astype(np.float32)
    d = load(world, 1280)
    te = slice(1024, 1280)
    gt = {k: np.asarray(v[te]) for k, v in d.items() if k.startswith("gt_")}
    T = probes.targets(gt)
    with open(run / f"{mind}_params.pkl", "rb") as f:
        mu, sd = (lambda r: (r["mu"], r["sd"]))(pickle.load(f))
    obs = (presets.observe({k: v[te] for k, v in d.items()}, presets.PRESETS[mind]) - mu) / sd

    E = H.shape[0]
    half = E // 2  # first half of held-out episodes fits the explainers, second half scores them
    sel = (slice(None), slice(probes.PAST, None, stride))
    X = H[sel].reshape(-1, H.shape[-1])
    hm, hs = X.mean(0), X.std(0) + 1e-6
    sae = train_sae((X - hm) / hs)
    F_all = features(sae, (H - hm) / hs)  # (E, T, n_features)
    active = (F_all[sel] > 0).mean((0, 1))
    keep = np.flatnonzero(active > 0.005)
    F = F_all[..., keep]
    lag = 5
    autocorr = np.array([np.corrcoef(F[:, :-lag, j].ravel(), F[:, lag:, j].ravel())[0, 1] for j in range(F.shape[-1])])
    Fs = F[sel]
    N, O = _named(T)[sel], obs[sel]
    r2_named = _explained(N[:half].reshape(-1, N.shape[-1]), Fs[:half].reshape(-1, F.shape[-1]),
                          N[half:].reshape(-1, N.shape[-1]), Fs[half:].reshape(-1, F.shape[-1]))
    r2_sense = _explained(O[:half].reshape(-1, O.shape[-1]), Fs[:half].reshape(-1, F.shape[-1]),
                          O[half:].reshape(-1, O.shape[-1]), Fs[half:].reshape(-1, F.shape[-1]))
    unnamed = (r2_named < 0.3) & (r2_sense < 0.3) & (autocorr > 0.8)
    out = {
        "mind": mind, "world": world, "live_features": int(len(keep)),
        "perceptual": int(np.sum(r2_sense >= 0.5)), "conceptual": int(np.sum(r2_named >= 0.5)),
        "unnamed_candidates": int(unnamed.sum()),
        "candidates": [{"feature": int(keep[j]), "active": float(active[keep[j]]), "autocorr": float(autocorr[j]),
                        "r2_named": float(r2_named[j]), "r2_senses": float(r2_sense[j])}
                       for j in np.flatnonzero(unnamed)[np.argsort(-autocorr[unnamed])][:10]],
    }
    return out


def build(minds=("human", "dog", "bat", "smell_only", "chimera", "blind", "hearing_only", "vision_only"), tag="main"):
    out_dir = RESULTS / "concepts"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in minds:
        if not (RESULTS / "runs" / tag / "lawful_s0" / f"{m}_latents.pkl").exists():
            continue
        r = analyse(m, tag=tag)
        rows.append(r)
        print(f"{m}: {r['live_features']} live features, {r['perceptual']} perceptual, {r['conceptual']} conceptual, "
              f"{r['unnamed_candidates']} unnamed candidates", flush=True)
    (out_dir / "concept_import.json").write_text(json.dumps(rows, indent=1))
    lines = ["# Concept import, first pass", "",
             "Sparse autoencoder features of each mind's hidden state (lawful world, held-out episodes). "
             "Perceptual: R^2 >= 0.5 from current senses. Conceptual: R^2 >= 0.5 from named world variables. "
             "Unnamed candidate: R^2 < 0.3 from both, and 5-step autocorrelation > 0.8.", "",
             "| mind | live features | perceptual | conceptual | unnamed candidates |", "|---|---|---|---|---|"]
    lines += [f"| {r['mind']} | {r['live_features']} | {r['perceptual']} | {r['conceptual']} | {r['unnamed_candidates']} |" for r in rows]
    (out_dir / "CONCEPT_IMPORT.md").write_text("\n".join(lines) + "\n")
    return rows
