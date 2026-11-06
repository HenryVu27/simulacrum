"""Did the old one-draw interval cover the sensor noise? For each selected score and each of 200 fresh draws, does the
one-draw interval (quadruplets resampled, voe.surprise as before) contain the 200-draw S (noise averaged, same quadruplets)?"""
import sys
from pathlib import Path
import numpy as np
T = Path(sys.argv[1])
def S(t):
    P = t[..., 0].sum(-1) + t[..., 1].sum(-1)
    return (t[..., 2].sum(-1) + t[..., 3].sum(-1) - P) / P
cov = []
for f in sorted(T.glob("*__40_240.npz")):
    Z = np.load(f)
    for net in ("trained", "untrained"):
        if f"{net}/permanence/t64" not in Z: continue
        for test in ("permanence", "solidity", "causality"):
            tot = Z[f"{net}/{test}/t64"]; s200 = S(tot.mean(0)); n = tot.shape[1]
            rng = np.random.default_rng(0); I = rng.integers(0, n, (500, n))
            c = [(lambda b: b[0] <= s200 <= b[1])(np.percentile(S(t[I]), [2.5, 97.5])) for t in tot]
            cov.append(np.mean(c))
cov = np.array(cov)
print(f"{len(cov)} scores x 200 fresh draws: the one-draw interval contains the noise-averaged S (same quadruplets) in {100 * cov.mean():.1f}% (per score min {100 * cov.min():.1f}%, median {100 * np.median(cov):.1f}%)")
