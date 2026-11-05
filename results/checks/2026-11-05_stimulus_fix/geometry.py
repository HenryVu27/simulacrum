"""Geometry of the hidden wall, in float64, for every agent pose used by the surprise movies."""
import pickle, sys
import numpy as np
from umwelt.senses import RAY_REL, MAX_RANGE, AGENT_R, EAR_OFFSET
from umwelt.world import ARENA
from umwelt import voe

old_hw = np.array([0.5, 0.0, 1.5, 0.0]); new_hw = np.asarray(voe.HIDDEN_WALL, np.float64)
bound = np.array([[0, 0, ARENA, 0], [ARENA, 0, ARENA, ARENA], [ARENA, ARENA, 0, ARENA], [0, ARENA, 0, 0]], float)
poses = {f"permanence j={j}": ((2.0, 5.0 + j), 0.0) for j in (-0.5, 0.0, 0.5)}
poses.update({"solidity": ((6.0, 1.0), np.pi / 2), "causality": ((6.0, 2.0), np.pi / 2)})

def hit(a, u, w):
    p1, q = w[:2], w[2:] - w[:2]
    den = u[0] * q[1] - u[1] * q[0]
    if abs(den) < 1e-12: return np.inf
    pmo = p1 - a
    t = (pmo[0] * q[1] - pmo[1] * q[0]) / den
    s = (pmo[0] * u[1] - pmo[1] * u[0]) / den
    return t if (t > 1e-3 and 0 <= s <= 1) else np.inf

def seg_dist(p, w):
    p1, q = w[:2], w[2:] - w[:2]
    t = np.clip(np.dot(p - p1, q) / np.dot(q, q), 0, 1)
    return np.linalg.norm(p1 + t * q - p)

for name, (a, th) in poses.items():
    a = np.array(a)
    for label, hw in (("old", old_hw), ("new", new_hw)):
        ties, reach, margins, dograys = 0, 0, [], []
        for i, rel in enumerate(RAY_REL.astype(np.float64)):
            u = np.array([np.cos(th + rel), np.sin(th + rel)])
            tb = min(hit(a, u, w) for w in bound)
            th_ = hit(a, u, hw)
            if np.isfinite(th_):
                reach += 1
                margins.append(th_ - tb)
                if abs(th_ - tb) < 1e-9:
                    ties += 1
                    dograys.append(round(np.rad2deg(rel), 1))
        print(f"{name:18} {label}: rays whose line meets the marker {reach}, exact ties with the boundary {ties} "
              f"(ray angles {dograys}), min margin t_marker - t_boundary {min(margins) if margins else None}, "
              f"touch distance {seg_dist(a, hw):.3f} (touch fires below {AGENT_R + 0.05})")
