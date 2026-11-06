"""Claim 3: previous solidity movies, wall present == wall absent for every sense but the oracle; new movies, the
human's eyes (|ray| <= 90 deg) tell wall-in-path from wall-slid-away in every frame before contact."""
import numpy as np
from common import BOUNDARY, Capture, old_voe, raycast64, voe
from umwelt import kinds
from umwelt.senses import RAY_REL
from umwelt.world import N_SLOTS

HUMAN = np.abs(np.rad2deg(RAY_REL)) <= 90
print("human rays:", HUMAN.sum())
old = old_voe()
print("== previous movies (28996ac)")
hits_wall = 0
for k in old.INERT:
    for j in (-0.5, 0.0, 0.5):
        with Capture(old) as cap:
            (p1, p2, i1, i2), tc = old.solidity(k, j)
        diff = {}
        for name, a, b in (("P1 vs I2", p1, i2), ("I1 vs P2", i1, p2)):
            diff[name] = [kk for kk in a if kk != "oracle" and not np.array_equal(a[kk], b[kk])]
        # my caster: does any ray of any frame meet the wall (internal slot 0 of the wall-present movies)?
        for c in (cap.calls[0], cap.calls[2]):
            W = np.r_[BOUNDARY, c["internal"].astype(np.float64)]
            for t in range(len(c["pos"])):
                pos_t = np.where(c["active"][t][:, None], c["pos"][t], -100.0)
                idx, _, _ = raycast64(c["agent"], c["th"], W, pos_t, kinds.RADIUS[c["kind"]].astype(float), c["active"][t])
                hits_wall += int((idx == N_SLOTS + 4).sum())
        print(f"  kind {k} j {j:+.1f}: senses (not oracle) that differ: {diff}; agent at {cap.calls[0]['agent']}, wall {cap.calls[0]['internal'][0]}")
print(f"  ray-frames meeting the wall in the previous wall-present movies (my caster): {hits_wall}")

print("== new movies (6290064): human rays that differ, frames 0..tc-1, and where those rays land")
allmin = []
for k in voe.INERT:
    for j in (-0.5, 0.0, 0.5):
        with Capture(voe) as cap:
            (p1, p2, i1, i2), tc = voe.solidity(k, j)
        res = []
        for name, a, b, ca in (("P1-I2", p1, i2, cap.calls[0]), ("I1-P2", i1, p2, cap.calls[2])):
            d = ((a["vision"][:tc] != b["vision"][:tc]).any(-1) & HUMAN).sum(-1)
            # in float16 (what the suite stores) too
            d16 = ((a["vision"][:tc].astype(np.float16) != b["vision"][:tc].astype(np.float16)).any(-1) & HUMAN).sum(-1)
            # attribute: rays that differ hit the event wall in one movie or the other (my caster)
            cb = cap.calls[3] if name == "P1-I2" else cap.calls[1]
            att = 0
            for t in range(tc):
                ids = []
                for c in (ca, cb):
                    W = np.r_[BOUNDARY, c["internal"].astype(np.float64)]
                    pos_t = np.where(c["active"][t][:, None], c["pos"][t], -100.0)
                    ids.append(raycast64(c["agent"], c["th"], W, pos_t, kinds.RADIUS[c["kind"]].astype(float), c["active"][t])[0])
                dif = (a["vision"][t] != b["vision"][t]).any(-1) & HUMAN
                att += int((dif & ~((ids[0] == N_SLOTS + 4) | (ids[1] == N_SLOTS + 4))).sum())
            res.append(f"{name}: min {d.min()} max {d.max()} (float16 min {d16.min()}), not on the wall {att}")
            allmin.append(d.min())
        print(f"  kind {k} j {j:+.1f} tc {tc}: " + "; ".join(res))
print("  smallest number of differing human rays in any frame before contact:", min(allmin))
