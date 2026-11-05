"""In the passing movies (I1 wall in path, P2 wall slid away), after contact: in which frames does any of the human's
rays meet the object, and where is it then? Tests the comment that the screen covers the sight lines to the path
beyond the wall's line."""
import numpy as np
from common import BOUNDARY, Capture, raycast64, voe
from umwelt import kinds
from umwelt.senses import RAY_REL

HUMAN = np.abs(np.rad2deg(RAY_REL)) <= 90
for k in voe.INERT:
    for j in (0.0,):
        with Capture(voe) as cap:
            (p1, p2, i1, i2), tc = voe.solidity(k, j)
        out = []
        for v, c in (("I1", cap.calls[2]), ("P2", cap.calls[1])):
            W = np.r_[BOUNDARY, c["internal"].astype(np.float64)]
            seen = []
            for t in range(tc, len(c["pos"])):
                idx, _, _ = raycast64(c["agent"], c["th"], W, c["pos"][t], kinds.RADIUS[c["kind"]].astype(float), c["active"][t])
                if ((idx == 0) & HUMAN).any():
                    seen.append(round(float(c["pos"][t, 0, 0]), 2))
            out.append(f"{v}: seen at x = {seen}")
        print(f"kind {k} r {kinds.RADIUS[k]:.2f}: " + " | ".join(out))
