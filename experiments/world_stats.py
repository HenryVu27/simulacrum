"""Describe how often objects leave view and come back in each world, before any mind is trained.

A reappearance is an object coming back into the 180 degree front view after being out of it. The
gap is how long it was gone. Memory can only pay for prediction if reappearances are common and
the gaps are long enough that the present moment does not already say what will come back.

    uv run python experiments/world_stats.py lawful lookback dark occluders occlusion
"""

import sys

import numpy as np

from umwelt.data import generate
from umwelt.probes import steps_since_seen


def stats(world, n=64, T=256):
    d = generate(world, n, T, seed=5, chunk=64)
    vis = d["gt_vis180"] & d["gt_obj_active"]
    act = d["gt_obj_active"]
    since = steps_since_seen(vis)
    back = vis[:, 1:] & ~vis[:, :-1] & (since[:, :-1] < 1000)  # came back into view after being seen
    gaps = since[:, :-1][back]
    om = np.abs(np.diff(np.unwrap(d["gt_agent_th"], axis=1), axis=1))
    speed = np.linalg.norm(d["gt_obj_vel"], axis=-1)[act]
    light = d["gt_light"]
    return {
        "in view": vis.sum() / act.sum(),
        "dark": 1 - light.mean(),
        "returns/ep": back.sum() / n,
        "gap>=5/ep": (gaps >= 5).sum() / n,
        "gap>=20/ep": (gaps >= 20).sum() / n,
        "median gap": float(np.median(gaps)) if len(gaps) else np.nan,
        "turn deg/step": float(np.rad2deg(om.mean())),
        "turn>20deg": float((np.rad2deg(om) > 20).mean()),
        "obj speed": float(speed.mean()),
    }


if __name__ == "__main__":
    rows = {w: stats(w) for w in sys.argv[1:]}
    cols = list(next(iter(rows.values())))
    print(f"{'world':10s} " + " ".join(f"{c:>13s}" for c in cols))
    for w, r in rows.items():
        print(f"{w:10s} " + " ".join(f"{r[c]:13.2f}" for c in cols))
