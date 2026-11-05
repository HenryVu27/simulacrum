"""Which minds' (noise-free) inputs tell wall-in-path (P1) from wall-slid-away (I2) before contact, new movies."""
import numpy as np
from common import voe
from umwelt import presets
from balance import noise_free

for name in ("human", "vision_only", "dog", "bat", "chimera", "blind", "smell_only", "touch_only", "hearing_only", "oracle"):
    p = noise_free(presets.PRESETS[name])
    mins = []
    for k in voe.INERT:
        for j in (-0.5, 0.0, 0.5):
            (p1, p2, i1, i2), tc = voe.solidity(k, j)
            d = {f"obs_{kk}": np.stack([p1[kk], i2[kk]]).astype(np.float16) for kk in p1}
            d["action"] = np.zeros((2, len(p1["oracle"]), 2), np.float32)
            o = presets.observe(d, p)
            mins.append(int((o[0, :tc] != o[1, :tc]).sum(-1).min()))
    print(f"{name:13} input dims that differ, fewest in any frame before contact over 18 quadruplets: {min(mins)} (max of per-quad minima {max(mins)})", flush=True)
