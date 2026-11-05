"""Solidity: does the object reach the wall exactly at the critical frame tc, or sit still at the wall first?"""
import numpy as np
from common import Capture, voe
from umwelt import kinds
for k in voe.INERT:
    with Capture(voe) as cap:
        (_, tc) = voe.solidity(k, 0.0)
    r = float(kinds.RADIUS[k])
    x = {v: c["pos"][:, 0, 0] for v, c in zip(voe.VARIANTS, cap.calls)}
    xb = x["P1"]
    at_wall = np.flatnonzero(np.isclose(xb, 6 - r, atol=1e-6))
    step = np.diff(xb)
    print(f"kind {k} r {r:.2f}: tc {tc}, frames at the wall {at_wall.tolist()}, x[tc-2..tc+1] {np.round(xb[tc-2:tc+2], 4).tolist()}, "
          f"step before {step[tc-2]:.4f}, step at tc-1->tc {step[tc-1]:.4f}, step after {step[tc]:.4f}, "
          f"impulse at {np.flatnonzero(cap.calls[0]['impulse'][:, 0]).tolist()} = {cap.calls[0]['impulse'][tc, 0]:.3f}")
