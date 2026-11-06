"""Audit of translation (H5) and the oracle's untranslatability, lawful seed 0.
(1) Reproduce stored translation R^2 from stored latents with the repo's translate.matrices.
(2) H5 control: the same matrices on UNTRAINED networks (the runner's untrained init, PRNGKey(10_000 + seed)).
    If random networks show the same asymmetry, H5 reflects which inputs are a superset of which, not learning.
(3) The oracle: how much of its hidden-state variance is constant within an episode (wall layout and which
    kinds are present)? Then human<->oracle R^2 on within-episode (demeaned) states only."""
import json, pickle, sys
import numpy as np
sys.path.insert(0, "/Users/henryvu/personal/umwelt/src")
import jax
from umwelt import learner, presets, translate
from umwelt.data import load
R = "/Users/henryvu/personal/umwelt/results/runs/main/lawful_s0/"
minds = ["oracle", "human", "blind", "vision_only", "dog", "smell_only"]
lat = {m: pickle.load(open(R + f"{m}_latents.pkl", "rb")) for m in minds}
tr = {m: v[0].astype(np.float32) for m, v in lat.items()}
te = {m: v[1].astype(np.float32) for m, v in lat.items()}
st = json.load(open(R + "translation.json"))
names, cka, T = translate.matrices(tr, te)
S = np.array(st["translation_r2"]); SC = np.array(st["cka"])
print("[reproduce] stored vs recomputed translation R^2 (subset of minds; the stored run used all 14, so the random row draw differs only through the rng stream)")
for a in minds:
    for b in minds:
        if a != b:
            i, j = names.index(a), names.index(b); I, J = st["names"].index(a), st["names"].index(b)
            print(f"   {a:11s}->{b:11s} stored {S[I,J]:+.3f} recomputed {T[i,j]:+.3f} | CKA stored {SC[I,J]:.3f} recomputed {cka[i,j]:.3f}")

# (2) untrained networks
d = load("lawful", 1280)
cfg = learner.LearnerConfig()
def untrained(m):
    p = presets.PRESETS[m]
    out = []
    for sl in (slice(0, 128), slice(1024, 1280)):
        part = {k: v[sl] for k, v in d.items() if k.startswith("obs_") or k == "action"}
        o_all = presets.observe({k: v[:1024] for k, v in d.items() if k.startswith("obs_") or k == "action"}, p) if False else None
        o = presets.observe(part, p)
        out.append(o)
    # standardize with the training-set statistics, as the runner does
    otr = presets.observe({k: v[:1024] for k, v in d.items() if k.startswith("obs_") or k == "action"}, p)
    mu, sd = learner.standardizer(otr)
    p0 = learner.init(jax.random.PRNGKey(10_000 + 0), otr.shape[-1], cfg, p.recurrent)
    H = [learner.batched(learner._latents_jit, p0, (o - mu) / sd, presets.actions({"action": d["action"][sl]}, p))
         for o, sl in zip(out, (slice(0, 128), slice(1024, 1280)))]
    return H
U = {m: untrained(m) for m in minds if m != "oracle"}
n2, _, TU = translate.matrices({m: v[0] for m, v in U.items()}, {m: v[1] for m, v in U.items()})
print("\n[H5 on untrained networks] pair: trained a->b vs b->a | untrained a->b vs b->a")
for a, b in (("human", "blind"), ("human", "vision_only"), ("dog", "smell_only")):
    i, j = names.index(a), names.index(b); k, l = n2.index(a), n2.index(b)
    print(f"   {a}->{b}: trained {T[i,j]:.2f} vs {T[j,i]:.2f} | untrained {TU[k,l]:.2f} vs {TU[l,k]:.2f}")

# (3) oracle variance decomposition and within-episode translation
for m in ("oracle", "human", "vision_only"):
    H = te[m]
    tot = H.reshape(-1, H.shape[-1]).var(0).sum()
    between = H.mean(1).var(0).sum()
    print(f"\n[variance] {m}: share of hidden-state variance that is between episodes (constant within an episode) = {between / tot:.3f}")
def within(x):  # remove each episode's mean
    return x - x.mean(1, keepdims=True)
n3, _, TW = translate.matrices({m: within(tr[m]) for m in ("oracle", "human", "vision_only")}, {m: within(te[m]) for m in ("oracle", "human", "vision_only")})
for a, b in (("human", "oracle"), ("oracle", "human"), ("vision_only", "oracle"), ("oracle", "vision_only")):
    i, j = names.index(a), names.index(b); k, l = n3.index(a), n3.index(b)
    print(f"   {a}->{b}: R^2 as stored {T[i,j]:+.3f}; within-episode states only {TW[k,l]:+.3f}")
