"""Compare my loop's totals against the stored scores.
(a) old statistic on draw 0 (bootstrap replica, instances only) == *_before_noise_draws, and == a223c3c's voe fields
(b) new statistic on draws 0..19 (bootstrap replica, instances then draws) == voe / voe_untrained
(c) independent float64 point estimate (mask totals, mean over draws) vs stored S
(d) independent float64 vectorised bootstrap with another seed: stored CI ends vs mine (Monte Carlo agreement only)"""
import sys, json, glob
from pathlib import Path
import numpy as np

snap_new, snap_old, tdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
TESTS = ("permanence", "solidity", "causality")


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
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(s_of(mean)), [float(lo), float(hi)]


def point(tot):
    return float(s_of(tot[0] if len(tot) == 1 else tot.mean(0)))


def indep64(t64, B=4000, seed=12345):
    k, n = t64.shape[:2]
    rng = np.random.default_rng(seed)
    m = t64.mean(0)
    S = lambda t: (t[..., 2].sum(-1) + t[..., 3].sum(-1)) / (t[..., 0].sum(-1) + t[..., 1].sum(-1)) - 1
    I = rng.integers(0, n, (B, n))
    J = rng.integers(0, k, (B, k))
    tb = t64[J].mean(1)  # (B, n, 4)
    tb = np.take_along_axis(tb, I[:, :, None], 1)
    return S(m), np.percentile(S(tb), [2.5, 97.5])


rows = []
bad = []
for f in sorted(tdir.glob("*__0_20.npz")):
    run_a, run_b, mind, _ = f.stem.split("__")
    run = f"{run_a}/{run_b}"
    Z = np.load(f)
    rec = json.loads((snap_new / "results/runs" / run / f"{mind}.json").read_text())
    oldf = snap_old / "results/runs" / run / f"{mind}.json"
    old = json.loads(oldf.read_text()) if oldf.exists() else None
    nets = [n for n in ("trained", "untrained") if f"{n}/permanence/t32" in Z]
    for net in nets:
        key = "voe" if net == "trained" else "voe_untrained"
        for test in TESTS:
            t32, t64 = Z[f"{net}/{test}/t32"], Z[f"{net}/{test}/t64"]
            c32 = Z[f"copy/{test}/t32"]
            stored_new, stored_before = rec[key][test], rec[f"{key}_before_noise_draws"][test]
            # (a)
            s0, ci0 = replica(t32[:1])
            a_ok = (s0 == stored_before["S"] and ci0 == list(stored_before["ci"]) and point(c32[:1]) == stored_before["S_copy"])
            a_old = None if old is None else (old[key][test] == stored_before)
            # (b)
            s, ci = replica(t32)
            b_ok = (s == stored_new["S"] and ci == list(stored_new["ci"]) and point(c32) == stored_new["S_copy"])
            # (c), (d)
            s64, ci64 = indep64(t64)
            rows.append(dict(run=run, mind=mind, net=net, test=test, k=len(t32), a=a_ok, a_old=a_old, b=b_ok,
                             dS64=abs(s64 - stored_new["S"]), dlo=ci64[0] - stored_new["ci"][0], dhi=ci64[1] - stored_new["ci"][1],
                             w=stored_new["ci"][1] - stored_new["ci"][0], w_before=stored_before["ci"][1] - stored_before["ci"][0],
                             S=stored_new["S"], S_before=stored_before["S"]))
            if not (a_ok and b_ok and a_old in (None, True)):
                bad.append((run, mind, net, test, a_ok, a_old, b_ok, s, stored_new["S"], ci, stored_new["ci"]))

import collections
print("scores compared:", len(rows), "minds:", len({(r['run'], r['mind']) for r in rows}))
print("(a) draw 0 + old statistic == *_before_noise_draws exactly:", sum(r["a"] for r in rows), "/", len(rows))
print("    *_before_noise_draws == a223c3c stored voe fields:", sum(r["a_old"] is True for r in rows), "/", sum(r["a_old"] is not None for r in rows),
      "(no a223c3c file:", sorted({r['run'] for r in rows if r['a_old'] is None}), ")")
print("(b) draws 0..19 + new statistic == stored voe exactly:", sum(r["b"] for r in rows), "/", len(rows))
print("    by k:", collections.Counter((r["k"], r["b"]) for r in rows))
d = np.array([r["dS64"] for r in rows]); print(f"(c) |S_float64 - S_stored| max {d.max():.2e}, median {np.median(d):.2e}")
w = np.array([r["w"] for r in rows]); dl = np.abs([r["dlo"] for r in rows]); dh = np.abs([r["dhi"] for r in rows])
print(f"(d) other-seed 4000-sample bootstrap vs stored 500-sample: |d lo|/width median {np.median(dl / w):.3f} max {np.max(dl / w):.3f}; |d hi|/width median {np.median(dh / w):.3f} max {np.max(dh / w):.3f}")
for b in bad[:20]:
    print("MISMATCH", b)
print("mismatches:", len(bad))
json.dump(rows, open(tdir.parent / "c2_rows.json", "w"))
