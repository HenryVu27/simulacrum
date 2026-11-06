"""P4 and H6 interval parts recomputed on 200 fresh draws (40..239) instead of the stored 20 (0..19)."""
import sys, json
from pathlib import Path
import numpy as np
T, snap = Path(sys.argv[1]), Path(sys.argv[2])
def S(t):
    P = t[..., 0].sum(-1) + t[..., 1].sum(-1)
    return (t[..., 2].sum(-1) + t[..., 3].sum(-1) - P) / P
def code(tot, B=1000, seed=0):
    k, n = tot.shape[:2]; rng = np.random.default_rng(seed)
    I = rng.integers(0, n, (B, n)); out = []
    for b in range(B):
        out.append(S(tot[rng.integers(0, k, k)].mean(0)[I[b]]))
    return np.percentile(out, [2.5, 97.5])
def quad(tot, B=4000, seed=0):
    m = tot.mean(0); n = len(m); rng = np.random.default_rng(seed)
    return np.percentile(S(m[rng.integers(0, n, (B, n))]), [2.5, 97.5])
def load(run, mind, part):
    return np.load(T / f"{run.replace('/', '__')}__{mind}__{part}.npz")["trained/permanence/t64"]
res = {}
for run in ("main/lawful_s0", "main/lawful_s1", "v2/dark_s0", "v2/dark_s1"):
    for mind in ("human", "memoryless", "vision_only"):
        f = load(run, mind, "40_240"); a = load(run, mind, "0_20")
        stored = json.loads((snap / "results/runs" / run / f"{mind}.json").read_text())["voe"]["permanence"]
        res[(run, mind)] = S(f.mean(0))
        print(f"{run:15} {mind:12} stored(0-19) S {stored['S']:+.5f} {np.round(stored['ci'], 5).tolist()} | fresh 200 draws S {S(f.mean(0)):+.5f} "
              f"code-scheme {np.round(code(f), 5).tolist()} quadruplets-only {np.round(quad(f), 5).tolist()}")
s2 = np.mean([res[("v2/dark_s0", "human")], res[("v2/dark_s1", "human")]]); s1 = np.mean([res[("main/lawful_s0", "human")], res[("main/lawful_s1", "human")]])
print(f"P4 third part on fresh draws: human mean S dark {s2:+.5f} vs v1 lawful {s1:+.5f} -> {'holds' if s2 > s1 else 'fails'} (stored: 0.0008 vs 0.0007, holds)")
