"""P4's third part (human mean S, dark vs v1 lawful) and H6's share: how much of each is decided by which 20 draws?"""
import sys, json
from pathlib import Path
import numpy as np
T = Path(sys.argv[1]); snap = Path(sys.argv[2])
def S(t):
    P = t[..., 0].sum(-1) + t[..., 1].sum(-1)
    return (t[..., 2].sum(-1) + t[..., 3].sum(-1) - P) / P
def tot(run, mind, net="trained", test="permanence"):
    return np.load(T / f"{run.replace('/', '__')}__{mind}__0_20.npz")[f"{net}/{test}/t64"]
# P4: human, dark s0 s1 vs lawful s0 s1
d = {r: tot(r, "human") for r in ("v2/dark_s0", "v2/dark_s1", "main/lawful_s0", "main/lawful_s1")}
s2 = np.mean([S(d[r].mean(0)) for r in ("v2/dark_s0", "v2/dark_s1")]); s1 = np.mean([S(d[r].mean(0)) for r in ("main/lawful_s0", "main/lawful_s1")])
print(f"P4 third part: human mean S dark {s2:.5f} vs v1 lawful {s1:.5f}; difference {s2 - s1:+.5f}")
perdraw = np.mean([S(d[r]) for r in ("v2/dark_s0", "v2/dark_s1")], 0) - np.mean([S(d[r]) for r in ("main/lawful_s0", "main/lawful_s1")], 0)
print(f"  per-draw difference: mean {perdraw.mean():+.5f}, SD over draws {perdraw.std(ddof=1):.5f}, SE of 20-draw mean {perdraw.std(ddof=1) / np.sqrt(20):.5f}")
rng = np.random.default_rng(0)
sub = [np.mean([S(d[r][j].mean(0)) for r in ("v2/dark_s0", "v2/dark_s1")]) - np.mean([S(d[r][j].mean(0)) for r in ("main/lawful_s0", "main/lawful_s1")])
       for j in (rng.integers(0, 20, 20) for _ in range(2000))]
print(f"  draw-bootstrap of the 20 draws: P(dark > lawful) = {np.mean(np.array(sub) > 0):.2f}")
# per-draw: in how many single draws is dark > lawful?
print(f"  single draws with dark mean > lawful mean: {(perdraw > 0).sum()}/20")
# H6 share with my totals
minds = ["bat", "blind", "chimera", "deafblind", "dog", "hearing_only", "human", "memoryless", "passive", "passive_novest", "smell_only", "touch_only", "vision_only"]
diffs = {}
for m in minds:
    L = [tot(r, m) for r in ("main/lawful_s0", "main/lawful_s1")]; I = tot("main/impermanent_s0", m)
    diffs[m] = (S(I.mean(0)) - np.mean([S(x.mean(0)) for x in L]), I, L)
share = np.mean([v[0] < 0 for v in diffs.values()])
print(f"H6 share (my totals, 20 draws): {share:.2f} = {sum(v[0] < 0 for v in diffs.values())}/13")
for m, (dd, I, L) in diffs.items():
    print(f"  {m:15} impermanent - lawful {dd:+.5f}")
shares = []
for _ in range(2000):
    j = rng.integers(0, 20, 20)
    shares.append(np.mean([S(I[j].mean(0)) - np.mean([S(x[j].mean(0)) for x in L]) < 0 for _, I, L in diffs.values()]))
shares = np.array(shares)
print(f"  draw-bootstrap of H6 share: median {np.median(shares):.2f}, 2.5-97.5% {np.percentile(shares, [2.5, 97.5])}, P(>0.5) {np.mean(shares > 0.5):.2f}")
