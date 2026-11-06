"""The live-tree calibration claim (vision_only permanence, lawful s0, 10 sets of 20 draws: SD 0.00052, half-width
0.00102). Reproduce with draws 0..199 and with 40..239, and compare with the precise SD from single draws."""
import sys
from pathlib import Path
import numpy as np
T = Path(sys.argv[1])
def S(t):
    P = t[..., 0].sum(-1) + t[..., 1].sum(-1)
    return (t[..., 2].sum(-1) + t[..., 3].sum(-1) - P) / P
def s_of(t):
    possible, impossible = t[:, 0] + t[:, 1], t[:, 2] + t[:, 3]
    return (impossible.sum() - possible.sum()) / possible.sum()
def code(tot, n_boot=500):
    k, n = tot.shape[:2]; rng = np.random.default_rng(0); m = tot.mean(0); b = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n); b.append(s_of(tot[rng.integers(0, k, k)].mean(0)[i]))
    return s_of(m), np.percentile(b, [2.5, 97.5])
a = np.load(T / "main__lawful_s0__vision_only__0_20.npz")["trained/permanence/t32"]
b = np.load(T / "main__lawful_s0__vision_only__20_40.npz")["trained/permanence/t32"]
c = np.load(T / "main__lawful_s0__vision_only__40_240.npz")["trained/permanence/t32"]
allD = np.concatenate([a, b, c])  # draws 0..239
for lo in (0, 40):
    sets = allD[lo:lo + 200].reshape(10, 20, *allD.shape[1:])
    res = [code(t) for t in sets]
    s20 = np.array([r[0] for r in res]); half = np.mean([(r[1][1] - r[1][0]) / 2 for r in res])
    print(f"draws {lo}..{lo + 199}: SD of S20 over 10 sets {s20.std(ddof=1):.5f} (ddof=0: {s20.std():.5f}); mean half-width {half:.5f}; ratio {half / s20.std(ddof=1):.2f}")
s1 = S(allD.astype(np.float64))
print(f"precise: SD of single-draw S over 240 draws {s1.std(ddof=1):.5f} -> SD of a 20-draw mean {s1.std(ddof=1) / np.sqrt(20):.5f}; half-width / that = {half / (s1.std(ddof=1) / np.sqrt(20)):.2f} (1.96 if the interval only covered the draws)")
rng = np.random.default_rng(1)
sds = [S(allD[rng.permutation(240)[:200]].reshape(10, 20, *allD.shape[1:]).mean(1).astype(np.float64)).std(ddof=1) for _ in range(2000)]
print(f"spread of a 10-set SD estimate (random partitions of the 240 draws): 2.5-97.5% {np.round(np.percentile(sds, [2.5, 97.5]), 5).tolist()}")
