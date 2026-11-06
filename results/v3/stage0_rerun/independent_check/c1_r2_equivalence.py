"""Check rr.sums / rr.r2_from against probes._r2_cols, at full weight and under explicit episode resamples."""
import importlib.util, sys
from pathlib import Path
import numpy as np
REPO = Path("/Users/henryvu/personal/simulacrum")
spec = importlib.util.spec_from_file_location("rr", REPO / "experiments/stage0_rerun.py")
rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
from umwelt import probes

rng = np.random.default_rng(123)
n_ep, per = 256, 62
n = n_ep * per
C = 16
ep = np.repeat(np.arange(n_ep), per)
Y = rng.normal(size=(n, C)).astype(np.float32) * rng.uniform(0.5, 3, C) + rng.normal(size=C)
pred = (Y + rng.normal(scale=0.8, size=(n, C))).astype(np.float64)
ref = rng.random((n, C // 2)) < np.linspace(0.05, 0.9, C // 2)
ref = np.repeat(ref, 2, 1)
m = ref & (rng.random((n, C)) < 0.3)
# make one column rare so the >=30 rule matters under resampling
m[:, 0] = False; m[rng.choice(n, 33, replace=False), 0] = True; m[:, 0] &= True
ref[:, 0] |= m[:, 0]
sm = rr.sums(pred, Y, m, ref, ep, n_ep)
a = rr.r2_from(sm, np.ones(n_ep)); b = probes._r2_cols(pred, Y, m, ref)
print(f"full weight: r2_from {a:.12f}  _r2_cols {b:.12f}  diff {a-b:.2e}")
worst = 0
for k in range(200):
    idx = rng.integers(0, n_ep, n_ep)
    w = np.bincount(idx, minlength=n_ep)
    rows = np.concatenate([np.flatnonzero(ep == e) for e in idx])
    a = rr.r2_from(sm, w); b = probes._r2_cols(pred[rows], Y[rows], m[rows], ref[rows])
    worst = max(worst, abs(a - b))
print(f"200 explicit resamples: max |r2_from - _r2_cols on resampled rows| = {worst:.2e}")
# the bootstrap weight matrix as the scripts build it
W = np.stack([np.bincount(np.random.default_rng(b).integers(0, 256, 256), minlength=256) for b in range(1000)])
print("W rows sum to 256:", bool((W.sum(1) == 256).all()), "distinct rows:", len({r.tobytes() for r in W}))
