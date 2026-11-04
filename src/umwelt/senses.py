"""The superset of senses, computed from the shared world state.

Every sense here is a plain function from world state to numbers. What matters is what
information each one carries and loses (range, occlusion, lag, persistence), not biology.
Presets pick and degrade subsets of these channels to make particular minds.
"""

import jax
import jax.numpy as jnp
import numpy as np

from umwelt import kinds
from umwelt.world import (
    AGENT_R,
    ANIMATE,
    COLOR,
    N_SLOTS,
    N_WALLS,
    RADIUS,
    SPECTRUM,
    TEMPERATURE,
    TEXTURE,
    Contacts,
    State,
    oracle_vector,
    sample_scent,
    wrap,
)

N_RAYS = 128
RAY_REL = (-np.pi + (np.arange(N_RAYS) + 0.5) * 2 * np.pi / N_RAYS).astype(np.float32)
MAX_RANGE = 12.0
N_TOUCH = 12
TOUCH_REL = (np.arange(N_TOUCH) * 2 * np.pi / N_TOUCH).astype(np.float32)
EAR_OFFSET = 0.3
NOSTRIL_ANGLE = 0.45
NOSTRIL_DIST = 0.35

WALL_COLOR = jnp.concatenate([jnp.asarray(kinds.BOUNDARY_COLOR), jnp.tile(jnp.asarray(kinds.INTERNAL_COLOR), (N_WALLS - 4, 1))])


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def raycast(s: State):
    """Cast rays all around the agent. Returns color, depth, texture and which object each ray hit."""
    ang = s.agent_th + RAY_REL
    u = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)  # (R, 2)
    r = RADIUS[s.obj_kind]
    f = s.obj_pos - s.agent_pos  # (N, 2)
    b = u @ f.T  # (R, N)
    c2 = jnp.sum(f * f, -1) - r * r
    disc = b * b - c2[None]
    t_obj = b - jnp.sqrt(jnp.maximum(disc, 0.0))
    t_obj = jnp.where((disc >= 0) & (t_obj > 1e-3) & s.obj_active[None], t_obj, jnp.inf)

    p1 = s.walls[:, :2]
    q = s.walls[:, 2:] - p1
    pmo = p1 - s.agent_pos
    denom = _cross(u[:, None, :], q[None])  # (R, W)
    safe = jnp.where(jnp.abs(denom) > 1e-8, denom, 1.0)
    t_w = _cross(pmo, q)[None] / safe
    s_w = _cross(pmo[None], u[:, None, :]) / safe
    t_w = jnp.where((jnp.abs(denom) > 1e-8) & (t_w > 1e-3) & (s_w >= 0) & (s_w <= 1), t_w, jnp.inf)

    t_all = jnp.concatenate([t_obj, t_w], 1)
    idx = jnp.argmin(t_all, 1)
    tmin = jnp.min(t_all, 1)
    hit = tmin < MAX_RANGE
    color_tab = jnp.concatenate([COLOR[s.obj_kind], WALL_COLOR], 0)
    tex_tab = jnp.concatenate([TEXTURE[s.obj_kind], jnp.full((N_WALLS,), kinds.WALL_TEXTURE)], 0)
    color = jnp.where(hit[:, None], color_tab[idx], 0.0)
    depth = jnp.where(hit, tmin / MAX_RANGE, 1.0)
    tex = jnp.where(hit, tex_tab[idx], 0.0)
    hit_obj = jnp.where(hit & (idx < N_SLOTS), idx, -1)
    return color, depth, tex, hit_obj


def _segments_cross(a1, a2, b1, b2):
    d1 = _cross(b2 - b1, a1 - b1)
    d2 = _cross(b2 - b1, a2 - b1)
    d3 = _cross(a2 - a1, b1 - a1)
    d4 = _cross(a2 - a1, b2 - a1)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def hearing(s: State, c: Contacts):
    """Two ears, four frequency bands. Sound comes from collisions and from animate things humming."""
    left = jnp.array([jnp.cos(s.agent_th + jnp.pi / 2), jnp.sin(s.agent_th + jnp.pi / 2)])
    ears = jnp.stack([s.agent_pos + EAR_OFFSET * left, s.agent_pos - EAR_OFFSET * left])
    ear_dir = jnp.stack([left, -left])
    amp = c.obj_impulse + 0.15 * (ANIMATE[s.obj_kind] & s.obj_active)
    spec = SPECTRUM[s.obj_kind]  # (N, B)
    d = s.obj_pos[None] - ears[:, None]  # (2, N, 2)
    att = 1.0 / (0.5 + jnp.sum(d * d, -1))
    walls = s.walls[4:]
    blocked = _segments_cross(
        ears[:, None, None], s.obj_pos[None, :, None], walls[None, None, :, :2], walls[None, None, :, 2:]
    ).any(-1)
    occ = jnp.where(blocked, 0.3, 1.0)
    src_dir = s.obj_pos - s.agent_pos
    src_dir = src_dir / (jnp.linalg.norm(src_dir, axis=-1, keepdims=True) + 1e-6)
    shadow = 0.7 + 0.3 * (ear_dir @ src_dir.T)  # (2, N)
    gain = amp[None] * att * occ * shadow * s.obj_active[None]
    level = gain @ spec  # (2, B)
    level = level + c.agent_wall_impulse * jnp.asarray(kinds.WALL_SPECTRUM)[None]
    return jnp.log1p(10.0 * level).reshape(-1)


def smell(s: State):
    """Raw scent concentration at two nostrils, four scent channels. Scent lingers, so it carries the past."""
    ang = s.agent_th + jnp.array([NOSTRIL_ANGLE, -NOSTRIL_ANGLE])
    pts = s.agent_pos + NOSTRIL_DIST * jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)
    return sample_scent(s.scent, pts).reshape(-1)


def touch(s: State, c: Contacts):
    """Twelve pressure sensors around the body. Each reports pressure, texture and temperature."""
    r = RADIUS[s.obj_kind]
    d = s.obj_pos - s.agent_pos
    dist = jnp.linalg.norm(d, axis=-1)
    on = s.obj_active & (dist < AGENT_R + r + 0.05)
    ang = wrap(jnp.arctan2(d[:, 1], d[:, 0]) - s.agent_th)
    press = on * (1.0 + c.agent_obj_impulse)
    tex = TEXTURE[s.obj_kind]
    temp = TEMPERATURE[s.obj_kind]

    p1, p2 = s.walls[:, :2], s.walls[:, 2:]
    q = p2 - p1
    t = jnp.clip(jnp.sum((s.agent_pos - p1) * q, -1) / jnp.sum(q * q, -1), 0.0, 1.0)
    dw = p1 + t[:, None] * q - s.agent_pos
    wdist = jnp.linalg.norm(dw, axis=-1)
    won = wdist < AGENT_R + 0.05
    wang = wrap(jnp.arctan2(dw[:, 1], dw[:, 0]) - s.agent_th)
    wpress = won * (1.0 + c.agent_wall_impulse)

    angs = jnp.concatenate([ang, wang])
    ps = jnp.concatenate([press, wpress])
    texs = jnp.concatenate([tex, jnp.full((N_WALLS,), kinds.WALL_TEXTURE)])
    temps = jnp.concatenate([temp, jnp.full((N_WALLS,), kinds.WALL_TEMPERATURE)])
    w = jnp.maximum(jnp.cos(angs[None] - TOUCH_REL[:, None]), 0.0) ** 8 * ps[None]  # (S, contacts)
    pressure = jnp.sum(w, 1)
    denom = pressure + 1e-6
    return jnp.stack([pressure, (w @ texs) / denom, (w @ temps) / denom], -1).reshape(-1)


def proprio(s: State):
    """Vestibular and kinesthetic sense: own forward speed, sideways speed, turning rate."""
    h = jnp.array([jnp.cos(s.agent_th), jnp.sin(s.agent_th)])
    left = jnp.array([-h[1], h[0]])
    return jnp.array([s.agent_vel @ h, s.agent_vel @ left, s.agent_om / 2.0])


def sense_all(s: State, c: Contacts):
    color, depth, tex, hit_obj = raycast(s)
    front = np.abs(RAY_REL) <= np.pi / 2
    vis180 = jnp.any((hit_obj[None, :] == jnp.arange(N_SLOTS)[:, None]) & front[None], 1)
    return dict(
        vision=jnp.concatenate([color, depth[:, None]], -1),  # (R, 4)
        ray_tex=tex,  # (R,)
        hearing=hearing(s, c),  # (8,)
        smell=smell(s),  # (8,)
        touch=touch(s, c),  # (36,)
        proprio=proprio(s),  # (3,)
        compass=jnp.array([jnp.sin(s.agent_th), jnp.cos(s.agent_th)]),
        oracle=oracle_vector(s),
        vis180=vis180,
    )
