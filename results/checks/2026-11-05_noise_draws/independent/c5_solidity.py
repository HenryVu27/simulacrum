"""Claim 5 from my own totals: ghost-trained bat and chimera solidity S over 20 draws, and lawful (mean of s0, s1)
minus ghost per mind; plus how much of each contrast is decided by the draws."""
import sys
from pathlib import Path
import numpy as np
T = Path(sys.argv[1])
def S(t):
    P = t[..., 0].sum(-1) + t[..., 1].sum(-1)
    return (t[..., 2].sum(-1) + t[..., 3].sum(-1) - P) / P
def tot(run, mind):
    return np.load(T / f"{run.replace('/', '__')}__{mind}__0_20.npz")["trained/solidity/t64"]
minds = sorted(p.stem.split("__")[2] for p in T.glob("main__ghost_s0__*__0_20.npz"))
rng = np.random.default_rng(0)
for m in ("bat", "chimera"):
    g = tot("main/ghost_s0", m); k, n = g.shape[:2]
    b = [S(g[rng.integers(0, k, k)].mean(0)[rng.integers(0, n, n)]) for _ in range(4000)]
    print(f"ghost {m}: S {S(g.mean(0)):+.4f} interval {np.round(np.percentile(b, [2.5, 97.5]), 4)}")
worst = 0
for m in minds:
    L = [tot(f"main/lawful_s{s}", m) for s in (0, 1)]; G = tot("main/ghost_s0", m)
    d = np.mean([S(x.mean(0)) for x in L]) - S(G.mean(0))
    k = len(G)
    dd = [np.mean([S(x[j].mean(0)) for x in L]) - S(G[j].mean(0)) for j in (rng.integers(0, k, k) for _ in range(1000))] if k > 1 else [d]
    worst = max(worst, abs(d))
    print(f"  {m:15} lawful - ghost {d:+.4f}   draw-only SD of this contrast {np.std(dd):.4f}")
print(f"max |lawful - ghost| = {worst:.4f}")
