import numpy as np
from common import Capture, quads, voe
for key, make in quads():
    with Capture(voe) as cap:
        make()
    for i, w in enumerate(cap.calls[0]["internal"].astype(np.float64)):
        mid = (w[:2] + w[2:]) / 2
        half = np.linalg.norm(w[2:] - w[:2]) / 2
        if (np.abs(mid - 2.5) < 0.01).any() or (np.abs(mid - 7.5) < 0.01).any() or half < 0.751 or half > 1.499:
            print(key, "slot", i, "centre", mid, "half", round(half, 4), "wall", w)
