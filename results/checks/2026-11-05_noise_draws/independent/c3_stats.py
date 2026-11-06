"""Claim 3: is S over the mean of 20 draws, with the bootstrap over quadruplets and draws, a sound estimate and interval?

Uses per-draw totals from c2_loop.py (draws 0..19 and fresh draws 40..239 for selected minds).
For each selected score:
  sd1    : SD over single draws of S (one draw, as before)
  sd20   : SD over 10 disjoint sets of 20 fresh draws of S20 (draw-to-draw variability of the new estimate, instances fixed)
  code   : half-width/1.96 of the code's interval, per set (mean over sets)
  inst   : bootstrap SD resampling quadruplets only (on the 20-draw mean)
  drawbs : bootstrap SD resampling draws only (all quadruplets)
  both   : bootstrap SD resampling both (the code's scheme)
  resid  : linearised residual-noise part of 'inst' (within-quadruplet variance across draws / 20)
  cover  : share of the 10 sets whose code interval contains the 200-draw S (instances-fixed target)
"""
import sys, glob, json
from pathlib import Path
import numpy as np

tdir = Path(sys.argv[1])


def S(t):  # t (..., n, 4)
    P = t[..., 0].sum(-1) + t[..., 1].sum(-1)
    return (t[..., 2].sum(-1) + t[..., 3].sum(-1) - P) / P


def code_interval(tot, rng, B=500):  # exactly the code's scheme, float64
    k, n = tot.shape[:2]
    m = tot.mean(0)
    b = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        t = tot[rng.integers(0, k, k)].mean(0)
        b.append(S(t[i]))
    return S(m), np.percentile(b, [2.5, 97.5]), np.std(b, ddof=1)


def boot_sds(tot, rng, B=2000):
    k, n = tot.shape[:2]
    m = tot.mean(0)
    I = rng.integers(0, n, (B, n)); J = rng.integers(0, k, (B, k))
    inst = S(m[I]).std(ddof=1)
    drw = S(tot[J].mean(1)).std(ddof=1)
    tb = np.take_along_axis(tot[J].mean(1), I[:, :, None], 1)
    both = S(tb).std(ddof=1)
    return inst, drw, both


def resid_part(tot):
    """Linearised variance of S from the noise left in the k-draw mean: D_i = I_i - (1+S) P_i; var(S) ~ sum var(D_i)/(sum P)^2."""
    k, n = tot.shape[:2]
    m = tot.mean(0)
    s = S(m)
    D = (tot[..., 2] + tot[..., 3]) - (1 + s) * (tot[..., 0] + tot[..., 1])  # (k, n)
    P = (m[:, 0] + m[:, 1]).sum()
    within = D.var(0, ddof=1)  # per quadruplet, across draws
    between_total = (n / (n - 1)) * ((D.mean(0) - D.mean(0).mean()) ** 2).sum() * (n / (n - 1)) / n  # not used
    return np.sqrt(within.sum() / k) / P, np.sqrt(n * D.mean(0).var(ddof=1)) / P


rows = []
for f in sorted(tdir.glob("*__40_240.npz")):
    run_a, run_b, mind, _ = f.stem.split("__")
    Zf = np.load(f)
    nets = [x for x in ("trained", "untrained") if f"{x}/permanence/t64" in Zf]
    for net in nets:
        for test in ("permanence", "solidity", "causality"):
            fresh = Zf[f"{net}/{test}/t64"]  # (200, n, 4)
            K = 20
            sets = fresh.reshape(10, K, *fresh.shape[1:])
            s_all = S(fresh.mean(0))
            s1 = S(fresh)
            rng = np.random.default_rng(7)
            per = [code_interval(t, rng) for t in sets]
            s20 = np.array([p[0] for p in per]); ci = np.array([p[1] for p in per]); bsd = np.array([p[2] for p in per])
            inst, drw, both = np.mean([boot_sds(t, rng) for t in sets[:3]], 0)
            res, betw_plus = np.mean([resid_part(t) for t in sets], 0)
            rows.append(dict(run=f"{run_a}/{run_b}", mind=mind, net=net, test=test, S200=float(s_all),
                             sd1=float(s1.std(ddof=1)), sd1_over_sqrt20=float(s1.std(ddof=1) / np.sqrt(20)),
                             sd20=float(s20.std(ddof=1)), code_half_over_196=float(((ci[:, 1] - ci[:, 0]) / 2 / 1.96).mean()),
                             code_bootsd=float(bsd.mean()), inst=float(inst), drawbs=float(drw), both=float(both), resid=float(res),
                             cover_fixed=float(np.mean((ci[:, 0] <= s_all) & (s_all <= ci[:, 1]))),
                             excl0=[bool(c[0] > 0 or c[1] < 0) for c in ci], s20=s20.tolist()))
            r = rows[-1]
            print(f"{r['run']:15} {mind:12} {net:9} {test:10} S200 {s_all:+.4f} | sd1 {r['sd1']:.4f} sd1/sqrt20 {r['sd1_over_sqrt20']:.4f} "
                  f"sd20(10 sets) {r['sd20']:.4f} | code half/1.96 {r['code_half_over_196']:.4f} = boot sd {r['code_bootsd']:.4f}: "
                  f"inst {inst:.4f} (of which resid noise {res:.4f}) draws {drw:.4f} both {both:.4f} | "
                  f"code/sd20 {r['code_half_over_196'] / r['sd20']:.1f} | sets excl0 {sum(r['excl0'])}/10 cover {r['cover_fixed']:.1f}", flush=True)
json.dump(rows, open(tdir.parent / "c3_rows.json", "w"))
