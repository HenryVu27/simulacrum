"""What would the verdict-relevant intervals be without the double-counted noise (resample quadruplets only, on the
20-draw mean)? Same rng and n_boot as voe.surprise. Counts 'excludes 0' changes over all stored noisy scores and
recomputes the interval parts of H6, P4 and V3."""
import sys, json
from pathlib import Path
import numpy as np
T, snap = Path(sys.argv[1]), Path(sys.argv[2])
def s_of(t):
    possible, impossible = t[:, 0] + t[:, 1], t[:, 2] + t[:, 3]
    return (impossible.sum() - possible.sum()) / possible.sum()
def inst_only(tot, n_boot=500):
    k, n = tot.shape[:2]
    rng = np.random.default_rng(0); m = tot.mean(0)
    b = [s_of(m[rng.integers(0, n, n)]) for _ in range(n_boot)]
    return float(s_of(m)), [float(x) for x in np.percentile(b, [2.5, 97.5])]
ex = lambda ci: ci[0] > 0 or ci[1] < 0
alt = {}; changed = []; n = 0
for f in sorted(T.glob("*__0_20.npz")):
    a, b, mind, _ = f.stem.split("__"); run = f"{a}/{b}"
    Z = np.load(f)
    rec = json.loads((snap / "results/runs" / run / f"{mind}.json").read_text())
    for net, key in (("trained", "voe"), ("untrained", "voe_untrained")):
        if f"{net}/permanence/t32" not in Z: continue
        for test in ("permanence", "solidity", "causality"):
            t = Z[f"{net}/{test}/t32"]
            if len(t) == 1: continue
            s, ci = inst_only(t); n += 1
            st = rec[key][test]
            alt[(run, mind, key, test)] = (s, ci, st["ci"])
            if ex(ci) != ex(st["ci"]):
                changed.append((run, mind, net, test, round(s, 4), [round(x, 4) for x in st["ci"]], [round(x, 4) for x in ci]))
w = [ (v[1][1]-v[1][0])/(v[2][1]-v[2][0]) for v in alt.values()]
print(f"{n} noisy scores; quadruplets-only interval width / stored width: median {np.median(w):.2f} (min {min(w):.2f}, max {max(w):.2f})")
print(f"'excludes 0' would change in {len(changed)} scores:")
for c in changed: print("  ", c)
def get(run, mind, key="voe", test="permanence"): return alt.get((run, mind, key, test))
print("\nverdict parts with quadruplets-only intervals:")
for run in ("main/lawful_s0", "main/lawful_s1", "v2/dark_s0", "v2/dark_s1"):
    for m in ("human", "vision_only", "memoryless"):
        s, ci, st = get(run, m)
        print(f"  {run:15} {m:12} permanence S {s:+.4f} quad-only {np.round(ci, 4).tolist()} stored {np.round(st, 4).tolist()}")
