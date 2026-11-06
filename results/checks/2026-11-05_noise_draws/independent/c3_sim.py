"""Coverage simulation for the instance part (claim 3).

For a score with 200 fresh draws: the 'population' is the 18 observed quadruplets, each with its noise-averaged
totals mu_i (mean of 200 draws) and its pool of 200 noise residuals. True S = S(mu over the 18). One simulated
experiment: 18 quadruplets drawn with replacement, each seen under 20 residuals drawn from its own pool.
Intervals compared (all percentile, n_boot=500):
  code   : resample quadruplets and draws (voe.surprise)
  inst   : resample quadruplets only, on the 20-draw mean
  inst1  : one draw only, resample quadruplets (the old statistic)
Reported: coverage of the true S, mean width, bias of S_hat, and the same for the old one-draw estimate."""
import sys, json
from pathlib import Path
import numpy as np

T = Path(sys.argv[1]); R = int(sys.argv[2]) if len(sys.argv) > 2 else 600


def S(t):
    P = t[..., 0].sum(-1) + t[..., 1].sum(-1)
    return (t[..., 2].sum(-1) + t[..., 3].sum(-1) - P) / P


def boot(tot, rng, both=True, B=500):
    k, n = tot.shape[:2]
    I = rng.integers(0, n, (B, n))
    if both and k > 1:
        J = rng.integers(0, k, (B, k))
        tb = np.take_along_axis(tot[J].mean(1), I[:, :, None], 1)
    else:
        tb = tot.mean(0)[I]
    return np.percentile(S(tb), [2.5, 97.5])


out = []
for f in sorted(T.glob("*__40_240.npz")):
    run_a, run_b, mind, _ = f.stem.split("__")
    Z = np.load(f)
    for net in ("trained", "untrained"):
        if f"{net}/permanence/t64" not in Z: continue
        for test in ("permanence", "solidity", "causality"):
            tot = Z[f"{net}/{test}/t64"]  # (200, n, 4)
            K, n = tot.shape[:2]
            mu = tot.mean(0); res = tot - mu
            s_true = S(mu)
            rng = np.random.default_rng(2026)
            cov = {"code": [], "inst": [], "inst1": []}; wid = {k: [] for k in cov}; sh, sh1 = [], []
            for _ in range(R):
                idx = rng.integers(0, n, n)
                dr = rng.integers(0, K, (20, n))
                sim = mu[idx][None] + res[dr, idx[None, :]]  # (20, n, 4)
                sh.append(S(sim.mean(0))); sh1.append(S(sim[0]))
                for name, (x, both) in {"code": (sim, True), "inst": (sim, False), "inst1": (sim[:1], False)}.items():
                    lo, hi = boot(x, rng, both)
                    cov[name].append(lo <= s_true <= hi); wid[name].append(hi - lo)
            sh, sh1 = np.array(sh), np.array(sh1)
            row = dict(run=f"{run_a}/{run_b}", mind=mind, net=net, test=test, s_true=float(s_true),
                       bias20=float(sh.mean() - s_true), sd20=float(sh.std()), bias1=float(sh1.mean() - s_true), sd1=float(sh1.std()),
                       **{f"cov_{k}": float(np.mean(v)) for k, v in cov.items()}, **{f"w_{k}": float(np.mean(v)) for k, v in wid.items()})
            out.append(row)
            print(f"{row['run']:15} {mind:12} {net:9} {test:10} S {s_true:+.4f} bias20 {row['bias20']:+.5f} (sd {row['sd20']:.4f}) | coverage code {row['cov_code']:.3f} "
                  f"inst-only {row['cov_inst']:.3f} one-draw {row['cov_inst1']:.3f} | width code {row['w_code']:.4f} inst-only {row['w_inst']:.4f} one-draw {row['w_inst1']:.4f} "
                  f"| ideal width 3.92*sd20 {3.92 * row['sd20']:.4f}", flush=True)
json.dump(out, open(T.parent / "c3_sim_rows.json", "w"))
c = np.array([[r["cov_code"], r["cov_inst"], r["cov_inst1"]] for r in out])
print(f"\nmean coverage over {len(out)} scores: code {c[:, 0].mean():.3f}, inst-only {c[:, 1].mean():.3f}, one-draw {c[:, 2].mean():.3f}; "
      f"min code {c[:, 0].min():.3f}, min inst-only {c[:, 1].min():.3f}")
