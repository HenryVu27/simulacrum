"""Is H6's 'memoryless null' (main, lawful s0 and s1) decided by the double-counted noise? Quadruplets-only intervals on
the 20-draw mean: percentile (n_boot 4000, 20 seeds) and a t-interval (t_17 x bootstrap SE), against the stored ones."""
import sys, json
from pathlib import Path
import numpy as np
from scipy import stats
T, snap = Path(sys.argv[1]), Path(sys.argv[2])
def S(t):
    P = t[..., 0].sum(-1) + t[..., 1].sum(-1)
    return (t[..., 2].sum(-1) + t[..., 3].sum(-1) - P) / P
for run in ("main/lawful_s0", "main/lawful_s1", "v2/dark_s0", "v2/dark_s1"):
    for mind in ("memoryless", "human", "vision_only"):
        tot = np.load(T / f"{run.replace('/', '__')}__{mind}__0_20.npz")["trained/permanence/t64"]
        m = tot.mean(0); n = len(m); s = S(m)
        los, his, ses = [], [], []
        for seed in range(20):
            rng = np.random.default_rng(100 + seed); b = S(m[rng.integers(0, n, (4000, n))])
            lo, hi = np.percentile(b, [2.5, 97.5]); los.append(lo); his.append(hi); ses.append(b.std(ddof=1))
        se = np.mean(ses); tq = stats.t.ppf(0.975, n - 1)
        st = json.loads((snap / "results/runs" / run / f"{mind}.json").read_text())["voe"]["permanence"]["ci"]
        print(f"{run:15} {mind:12} S {s:+.5f} | quad-only pct lower bound over 20 seeds [{min(los):+.5f}, {max(los):+.5f}], upper [{min(his):+.5f}, {max(his):+.5f}] "
              f"| t17 interval [{s - tq * se:+.5f}, {s + tq * se:+.5f}] | stored [{st[0]:+.5f}, {st[1]:+.5f}]")
