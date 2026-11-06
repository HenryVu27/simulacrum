"""Summarise c3_layouts.json: SD of S across 6 bystander layouts (20 draws each) vs the interval's implied SD."""
import sys, json
import numpy as np
rows = json.load(open(sys.argv[1]))
ratios = []; flips = 0
for r in rows:
    S = np.array(r["S"]); ci = np.array(r["ci"]); half = ((ci[:, 1] - ci[:, 0]) / 2 / 1.96).mean()
    ratio = S.std(ddof=1) / half
    ex = [(c[0] > 0 or c[1] < 0) for c in ci]
    flips += len(set(ex)) > 1
    ratios.append((ratio, r["mind"], r["net"], r["test"], S.std(ddof=1), half, sum(ex)))
ratios.sort(reverse=True)
arr = np.array([x[0] for x in ratios])
print(f"{len(rows)} scores (lawful s0, noisy minds, trained and untrained, 3 tests), 6 layout seeds x 20 draws each")
print(f"SD of S across layouts / interval's implied SD (half-width/1.96, mean over layouts): median {np.median(arr):.2f}, "
      f"90th pct {np.percentile(arr, 90):.2f}, max {arr.max():.2f}; above 1 in {np.sum(arr > 1)} scores")
print(f"'excludes 0' differs between layouts in {flips}/{len(rows)} scores")
print("largest ratios:")
for x in ratios[:8]:
    print(f"  {x[1]:15} {x[2]:9} {x[3]:10} SD across layouts {x[4]:.5f}, interval SD {x[5]:.5f}, ratio {x[0]:.2f}, layouts excluding 0: {x[6]}/6")
