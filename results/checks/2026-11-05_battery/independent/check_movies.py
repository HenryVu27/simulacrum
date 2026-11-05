"""Claims 1 and 2, independently: every movie's internal walls, and the bystanders-hide-nothing check against my
own float64 ray caster, with two different 'absent' places for the bystanders. Also physics of every movie."""
import numpy as np

from common import (BOUNDARY, BYSTANDER_SLOTS, EVENT_SLOTS, FAR2, OFF, Capture, point_seg, quads, raycast64, voe)
from umwelt import kinds
from umwelt.world import N_SLOTS

NONVIS = ["hearing", "smell", "touch", "proprio", "compass", "ray_tex"]
summary = dict(n=0, range_bad=[], order_bad=[], still_bad=[], shared_bad=[], hide_bad=[], nonvis_bad=[],
               oracle_bad=[], changed_not_bystander=[], obj_or_event_changed=0, rays_changed=0, min_margin=np.inf,
               phys=[])
minhalf, maxhalf, minc, maxc = 9, 0, 9, 0
for key, make in quads():
    test = key[0]
    with Capture(voe) as cap:
        make()
    calls = cap.calls
    assert len(calls) == 4
    summary["n"] += 1
    # ---- claim 1: walls in training range, still, shared
    for v, c in zip(voe.VARIANTS, calls):
        w = c["internal"].astype(np.float64)
        mid, half = (w[:, :2] + w[:, 2:]) / 2, np.linalg.norm(w[:, 2:] - w[:, :2], axis=-1) / 2
        minhalf, maxhalf = min(minhalf, half.min()), max(maxhalf, half.max())
        minc, maxc = min(minc, mid.min()), max(maxc, mid.max())
        if not (((2.5 <= mid) & (mid <= 7.5)).all() and ((0.75 <= half) & (half <= 1.5)).all()):
            summary["range_bad"].append((key, v, mid.tolist(), half.tolist()))
        # world.reset: first end = centre - half*(cos a, sin a), a in [0, pi): first end has y <= second's
        d = w[:, 2:] - w[:, :2]
        ang = np.arctan2(d[:, 1], d[:, 0])
        if not ((ang >= 0) & (ang < np.pi)).all():
            summary["order_bad"].append((key, v, np.rad2deg(ang).round(2).tolist()))
        ow = c["obs"]["oracle"][:, -12:].reshape(-1, 3, 4) * 10.0
        if not (np.abs(ow - c["internal"][None]).max() < 1e-5 and (ow == ow[0]).all()):
            summary["still_bad"].append((key, v))
    nonevent = [i for i in range(3) if i not in EVENT_SLOTS[test] or test == "permanence" or i == 1]
    shared = [i for i in range(3) if not (test == "solidity" and i == 0)]
    for c in calls[1:]:
        if not np.array_equal(c["internal"][shared], calls[0]["internal"][shared]):
            summary["shared_bad"].append(key)
    # ---- claim 2: re-render with bystanders moved away, two different places
    bys = BYSTANDER_SLOTS[test]
    ev = EVENT_SLOTS[test]
    for absent in (OFF, FAR2):
        for v, c in zip(voe.VARIANTS, calls):
            internal_wo = c["internal"].copy()
            internal_wo[bys] = absent
            out_wo = voe._render.__wrapped__(c) if hasattr(voe._render, "__wrapped__") else None
            from common import ORIG_RENDER
            out_wo = ORIG_RENDER(c["agent"], c["th"], internal_wo, c["kind"], c["pos"].astype(np.float32), c["active"], c["impulse"])
            a, b = c["obs"], out_wo
            for k in NONVIS:
                if not np.array_equal(a[k], b[k]):
                    summary["nonvis_bad"].append((key, v, k, float(np.abs(a[k] - b[k]).max())))
            keep = np.ones(a["oracle"].shape[1], bool)
            for i in bys:
                keep[a["oracle"].shape[1] - 12 + 4 * i: a["oracle"].shape[1] - 12 + 4 * i + 4] = False
            if not np.array_equal(a["oracle"][:, keep], b["oracle"][:, keep]):
                summary["oracle_bad"].append((key, v))
            # my ray caster on both scenes, every frame
            r_obj = kinds.RADIUS[c["kind"]].astype(np.float64)
            changed = (a["vision"] != b["vision"]).any(-1)  # (T, R)
            summary["rays_changed"] += int(changed.sum())
            for t in range(len(c["pos"])):
                pos_t = np.where(c["active"][t][:, None], c["pos"][t], -100.0)
                W_with = np.r_[BOUNDARY, c["internal"].astype(np.float64)]
                W_wo = np.r_[BOUNDARY, internal_wo.astype(np.float64)]
                i_wo, t_wo, _ = raycast64(c["agent"], c["th"], W_wo, pos_t, r_obj, c["active"][t])
                i_w, t_w, tab_w = raycast64(c["agent"], c["th"], W_with, pos_t, r_obj, c["active"][t])
                cat_wo = np.where(i_wo < N_SLOTS, "obj", np.where(i_wo < N_SLOTS + 4, "boundary",
                                  np.where(np.isin(i_wo - N_SLOTS - 4, ev), "event", "bys")))
                important = (cat_wo == "obj") | (cat_wo == "event")
                summary["obj_or_event_changed"] += int((changed[t] & important).sum())
                # every changed ray must be a ray my caster says now meets a bystander
                bys_cols = [N_SLOTS + 4 + i for i in bys]
                now_bys = np.isin(i_w, bys_cols)
                if (changed[t] & ~now_bys).any():
                    summary["changed_not_bystander"].append((key, v, t, int((changed[t] & ~now_bys).sum())))
                # margin: for important rays, how much farther is the nearest bystander crossing (relative)
                tb = tab_w[:, bys_cols].min(1)
                m = (tb[important] - t_wo[important])
                if m.size:
                    summary["min_margin"] = min(summary["min_margin"], float(m.min()))
    # ---- physics of every movie
    for v, c in zip(voe.VARIANTS, calls):
        r = kinds.RADIUS[c["kind"]]
        W = c["internal"].astype(np.float64)
        for s in range(N_SLOTS):
            act = c["active"][:, s]
            if not act.any():
                continue
            P = c["pos"][act, s]
            for i in range(3):
                dmin = point_seg(P, W[i, :2], W[i, 2:]).min()
                summary["phys"].append((key, v, s, i, float(dmin), float(r[s])))
            for bw in BOUNDARY:
                if point_seg(P, bw[:2], bw[2:]).min() < r[s]:
                    summary["phys"].append((key, v, s, "boundary", float(point_seg(P, bw[:2], bw[2:]).min()), float(r[s])))
        dagent = min(point_seg(c["agent"][None], W[i, :2], W[i, 2:])[0] for i in range(3))
        summary["phys"].append((key, v, "agent", "min wall dist", float(dagent), 0.3))
    print(key, "done", flush=True)

print("\n==== CLAIM 1")
print("quadruplets", summary["n"], "| centre range seen", round(minc, 4), round(maxc, 4), "| half range seen", round(minhalf, 4), round(maxhalf, 4))
print("walls out of range:", summary["range_bad"][:5], len(summary["range_bad"]))
print("walls with end order unlike world.reset (angle not in [0, pi)):", summary["order_bad"][:5], len(summary["order_bad"]))
print("walls moving or oracle readback mismatch:", summary["still_bad"][:5])
print("quadruplets whose non-event walls differ between movies:", summary["shared_bad"])
print("\n==== CLAIM 2 (both absent places, 54 quads x 4 movies)")
print("non-vision sense changes:", summary["nonvis_bad"][:10], len(summary["nonvis_bad"]))
print("oracle non-bystander inputs changed:", summary["oracle_bad"][:5], len(summary["oracle_bad"]))
print("ray-frames changed in total:", summary["rays_changed"])
print("changed rays that (by my float64 caster) met an object or the event's wall without bystanders:", summary["obj_or_event_changed"])
print("changed rays my caster does not attribute to a bystander:", summary["changed_not_bystander"][:10], len(summary["changed_not_bystander"]))
print("smallest distance behind an object/event hit at which a ray meets a bystander:", summary["min_margin"])
import pickle
pickle.dump(summary["phys"], open("phys.pkl", "wb"))
