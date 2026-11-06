"""Per-draw S for human / vision_only / memoryless permanence (lawful s0): are draws 0..19 unusual, or is the
distribution heavy-tailed? Also: how likely is a 20-draw window to be as far from the 220-draw mean as 0..19 is?"""
import sys
from pathlib import Path
import numpy as np
from scipy import stats
T = Path(sys.argv[1])
def S(t):
    P = t[..., 0].sum(-1) + t[..., 1].sum(-1)
    return (t[..., 2].sum(-1) + t[..., 3].sum(-1) - P) / P
for mind in ("human", "vision_only", "memoryless", "smell_only"):
    parts = [np.load(T / f"main__lawful_s0__{mind}__{p}.npz")["trained/permanence/t64"] for p in ("0_20", "20_40", "40_240")]
    s = S(np.concatenate(parts))  # draws 0..239
    first = s[:20]; rest = s[40:]
    z = (first.mean() - rest.mean()) / (rest.std(ddof=1) * np.sqrt(1 / 20 + 1 / 200))
    print(f"{mind:12} single-draw S over 240 draws: mean {s.mean():+.5f} sd {s.std(ddof=1):.5f} skew {stats.skew(s):+.2f} excess kurtosis {stats.kurtosis(s):+.2f}; "
          f"draws 0..19 mean {first.mean():+.5f} (z {z:+.2f}); Welch t-test 0..19 vs 40..239 p {stats.ttest_ind(first, rest, equal_var=False).pvalue:.3f}; "
          f"Mann-Whitney p {stats.mannwhitneyu(first, rest).pvalue:.3f}")
