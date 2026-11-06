"""Recompute experiments/noise_draws_check.py's numbers from my own totals (draws 0..39, lawful s0 and dark s0),
and compare row by row with the committed check.json."""
import sys, json, itertools
from pathlib import Path
import numpy as np
T, snap = Path(sys.argv[1]), Path(sys.argv[2])
def s_of(t):
    possible, impossible = t[:, 0] + t[:, 1], t[:, 2] + t[:, 3]
    return (impossible.sum() - possible.sum()) / possible.sum()
def replica(tot, n_boot=500):
    k, n = tot.shape[:2]
    rng = np.random.default_rng(0)
    mean = tot[0] if k == 1 else tot.mean(0)
    b = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        t = mean if k == 1 else tot[rng.integers(0, k, k)].mean(0)
        b.append(s_of(t[i]))
    return float(s_of(mean)), [float(x) for x in np.percentile(b, [2.5, 97.5])]
ex = lambda ci: ci[0] > 0 or ci[1] < 0
committed = {(r["run"], r["mind"], r["net"], r["test"]): r for r in json.load(open(snap / "results/checks/2026-11-05_noise_draws/check.json"))}
rows = []
for run in ("main/lawful_s0", "v2/dark_s0"):
    for f in sorted(T.glob(f"{run.replace('/', '__')}__*__20_40.npz")):
        mind = f.stem.split("__")[2]
        A_ = np.load(T / f"{run.replace('/', '__')}__{mind}__0_20.npz"); B_ = np.load(f)
        for net in ("trained", "untrained"):
            if f"{net}/permanence/t32" not in A_: continue
            for test in ("permanence", "solidity", "causality"):
                a32, b32 = A_[f"{net}/{test}/t32"], B_[f"{net}/{test}/t32"]
                if len(a32) == 1: continue  # no sensor noise (oracle)
                single = [replica(a32[k:k + 1]) for k in range(20)]
                rows.append(dict(key=(run, mind, net, test), single=single, A=replica(a32), B=replica(b32)))
print(len(rows), "scores; committed rows", len(committed), "; keys equal:", {r["key"] for r in rows} == set(committed))
# row-by-row agreement with committed check.json (float32 errors from the same code path: expect equality or ~1e-7)
dmax = 0.0
for r in rows:
    c = committed[r["key"]]
    for (s, ci), (s2, ci2) in zip(r["single"], c["single"]):
        dmax = max(dmax, abs(s - s2), abs(ci[0] - ci2[0]), abs(ci[1] - ci2[1]))
    for x in ("A", "B"):
        dmax = max(dmax, abs(r[x][0] - c[x][0]), abs(r[x][1][0] - c[x][1][0]), abs(r[x][1][1] - c[x][1][1]))
print(f"max |mine - committed check.json| over all S and interval ends: {dmax:.2e}")
sd = np.array([np.std([s for s, _ in r["single"]], ddof=1) for r in rows])
print(f"single draw: SD of S over draws 0..19: median {np.median(sd):.4f}, max {sd.max():.4f}  (commit: 'moved a score by a median 0.0024 (up to 0.031)')")
rng_ = np.array([np.ptp([s for s, _ in r["single"]]) for r in rows]); dev0 = np.array([abs(r["single"][0][0] - np.mean([s for s, _ in r["single"]])) for r in rows])
print(f"  for scale: range over 20 draws median {np.median(rng_):.4f} max {rng_.max():.4f}; |draw 0 - mean of 20| median {np.median(dev0):.4f} max {dev0.max():.4f}")
fl0 = np.mean([np.mean([ex(ci) != ex(r["single"][0][1]) for _, ci in r["single"][1:]]) for r in rows])
flp = np.mean([np.mean([ex(a[1]) != ex(b[1]) for a, b in itertools.combinations(r["single"], 2)]) for r in rows])
print(f"'excludes 0' differs from draw 0's in {100 * fl0:.1f}% of (score, draw) pairs; across all pairs of draws {100 * flp:.1f}%")
ag = np.mean([ex(r["A"][1]) == ex(r["B"][1]) for r in rows])
print(f"set A (0..19) vs set B (20..39): 'excludes 0' agrees in {100 * ag:.1f}% ({sum(ex(r['A'][1]) != ex(r['B'][1]) for r in rows)} disagree)")
nex = sum(ex(r["A"][1]) for r in rows); print(f"  A excludes 0 in {nex}/{len(rows)} scores; chance agreement if exclusion were random at that rate: {100 * ((nex / len(rows)) ** 2 + (1 - nex / len(rows)) ** 2):.1f}%")
single_ex = np.mean([np.mean([ex(ci) for _, ci in r["single"]]) for r in rows])
print(f"  one draw: interval excludes 0 in {100 * single_ex:.1f}% of (score, draw)")
json.dump([dict(key=list(r["key"]), **{k: r[k] for k in ("single", "A", "B")}) for r in rows], open(T.parent / "c4_rows.json", "w"))
