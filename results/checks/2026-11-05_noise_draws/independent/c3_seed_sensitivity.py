"""Monte Carlo error of the 500-sample bootstrap: rerun the code's scheme with bootstrap seeds 1..20 on the same
20-draw totals and count stored 'excludes 0' decisions that change with the seed."""
import sys, json
from pathlib import Path
import numpy as np
T, snap = Path(sys.argv[1]), Path(sys.argv[2])
def s_of(t):
    possible, impossible = t[:, 0] + t[:, 1], t[:, 2] + t[:, 3]
    return (impossible.sum() - possible.sum()) / possible.sum()
def code(tot, seed, n_boot=500):
    k, n = tot.shape[:2]; rng = np.random.default_rng(seed); m = tot.mean(0); b = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n); b.append(s_of(tot[rng.integers(0, k, k)].mean(0)[i]))
    return np.percentile(b, [2.5, 97.5])
ex = lambda ci: ci[0] > 0 or ci[1] < 0
frac = []; flagged = []
for f in sorted(T.glob("*__0_20.npz")):
    a, b, mind, _ = f.stem.split("__"); run = f"{a}/{b}"
    if run not in ("main/lawful_s0", "main/lawful_s1", "v2/dark_s0", "v2/dark_s1", "main/impermanent_s0", "main/ghost_s0"): continue
    Z = np.load(f); rec = json.loads((snap / "results/runs" / run / f"{mind}.json").read_text())
    for net, key in (("trained", "voe"), ("untrained", "voe_untrained")):
        if f"{net}/permanence/t32" not in Z: continue
        for test in ("permanence", "solidity", "causality"):
            t = Z[f"{net}/{test}/t32"]
            if len(t) == 1: continue
            st = ex(rec[key][test]["ci"])
            d = np.mean([ex(code(t, s)) != st for s in range(1, 21)])
            frac.append(d)
            if d >= 0.25: flagged.append((run, mind, net, test, rec[key][test]["S"], rec[key][test]["ci"], d))
frac = np.array(frac)
print(f"{len(frac)} noisy scores (6 runs): 'excludes 0' differs from the stored decision for some bootstrap seed in {np.mean(frac > 0) * 100:.1f}% of scores; "
      f"for >=25% of seeds in {np.mean(frac >= 0.25) * 100:.1f}% ({len(flagged)} scores)")
for x in flagged: print("  ", x[0], x[1], x[2], x[3], f"S {x[4]:+.4f} ci {np.round(x[5], 4).tolist()} differs for {x[6]:.0%} of seeds")
