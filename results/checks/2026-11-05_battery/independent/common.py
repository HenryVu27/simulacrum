"""Shared helpers for the independent check. Imports umwelt from the c35cc6e snapshot (PYTHONPATH)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent
import umwelt
assert str(S) in umwelt.__file__, umwelt.__file__  # the committed snapshot, not the live repo
from umwelt import kinds, voe
from umwelt.senses import RAY_REL
from umwelt.world import boundary_walls

OFF = np.array([0.5, -1.0, 1.5, -1.0], np.float32)
FAR2 = np.array([20.0, 20.0, 21.0, 20.0], np.float32)  # a second "absent" place, beyond the top-right corner
BOUNDARY = np.asarray(boundary_walls(), np.float64)
ORIG_RENDER = voe._render


def old_voe():
    spec = importlib.util.spec_from_file_location("voe_d6c6763", S / "repo_d67" / "src" / "umwelt" / "voe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Capture:
    """Record every _render call of a voe module (args and output)."""

    def __init__(self, mod):
        self.mod, self.calls = mod, []
        self.orig = mod._render

    def __enter__(self):
        def cap(agent_pos, agent_th, internal, kind, pos, active, impulse):
            out = self.orig(agent_pos, agent_th, internal, kind, pos, active, impulse)
            self.calls.append(dict(agent=np.array(agent_pos, np.float64), th=float(agent_th),
                                   internal=np.array(internal, np.float32).reshape(-1, 4), kind=np.array(kind),
                                   pos=np.array(pos, np.float64), active=np.array(active, bool),
                                   impulse=np.array(impulse, np.float64), obs=out))
            return out
        self.mod._render = cap
        return self

    def __exit__(self, *a):
        self.mod._render = self.orig


def quads(mod=voe, jitters=(-0.5, 0.0, 0.5)):
    for k in mod.INERT:
        for j in jitters:
            yield ("permanence", k, j), (lambda k=k, j=j: mod.permanence(k, j))
    for k in mod.INERT:
        for j in jitters:
            yield ("solidity", k, j), (lambda k=k, j=j: mod.solidity(k, j))
    for a, b in zip(mod.INERT, mod.INERT[1:] + mod.INERT[:1]):
        for j in jitters:
            yield ("causality", a, b, j), (lambda a=a, b=b, j=j: mod.causality(a, b, j))


BYSTANDER_SLOTS = {"permanence": [1, 2], "solidity": [2], "causality": [0, 1, 2]}
EVENT_SLOTS = {"permanence": [0], "solidity": [0, 1], "causality": []}


def raycast64(agent, th, walls, obj_pos, obj_r, active):
    """My own float64 ray caster. walls (W,4) incl. boundary. Returns (R,) index of nearest thing
    (0..N-1 objects, N.. walls) and (R,) distance, and the full (R, N+W) distance table."""
    ang = th + RAY_REL.astype(np.float64)
    u = np.stack([np.cos(ang), np.sin(ang)], -1)
    f = obj_pos - agent
    b = u @ f.T
    c2 = (f * f).sum(-1) - obj_r ** 2
    disc = b * b - c2[None]
    t_obj = b - np.sqrt(np.maximum(disc, 0))
    t_obj = np.where((disc >= 0) & (t_obj > 1e-3) & active[None], t_obj, np.inf)
    p1, q = walls[:, :2], walls[:, 2:] - walls[:, :2]
    pmo = p1 - agent
    cr = lambda a, c: a[..., 0] * c[..., 1] - a[..., 1] * c[..., 0]
    den = cr(u[:, None, :], q[None])
    safe = np.where(np.abs(den) > 1e-12, den, 1.0)
    t_w = cr(pmo, q)[None] / safe
    s_w = cr(pmo[None], u[:, None, :]) / safe
    t_w = np.where((np.abs(den) > 1e-12) & (t_w > 1e-3) & (s_w >= 0) & (s_w <= 1), t_w, np.inf)
    t_all = np.concatenate([t_obj, t_w], 1)
    return t_all.argmin(1), t_all.min(1), t_all


def point_seg(p, a, b):
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / max((ab * ab).sum(), 1e-12), 0, 1)
    return np.linalg.norm(p - a - t[..., None] * ab, axis=-1)
