"""Flatland: a 2D physics world in JAX whose rules can be broken on purpose.

One world state is shared by every mind. Minds differ only in which senses read it
(see senses.py and presets.py), so any difference in what they learn comes from the body.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from umwelt import kinds

ARENA = 10.0
N_SLOTS = 6
N_INTERNAL_WALLS = 3
N_WALLS = 4 + N_INTERNAL_WALLS
GRID = 32
DT = 0.1
AGENT_R = 0.3
AGENT_MASS = 2.0
RESTITUTION = 0.8
FAR = -100.0  # where inactive objects are parked

COLOR = jnp.asarray(kinds.COLOR)
SCENT = jnp.asarray(kinds.SCENT)
SPECTRUM = jnp.asarray(kinds.SPECTRUM)
TEXTURE = jnp.asarray(kinds.TEXTURE)
TEMPERATURE = jnp.asarray(kinds.TEMPERATURE)
RADIUS = jnp.asarray(kinds.RADIUS)
MASS = jnp.asarray(kinds.MASS)
ANIMATE = jnp.asarray(kinds.ANIMATE)


@dataclass(frozen=True)
class WorldRules:
    name: str = "lawful"
    solid: bool = True  # objects collide with internal walls, each other and the agent
    teleport_p: float = 0.0  # per-step chance that an object jumps to a random place
    nonlocal_push: bool = False  # pushing one object moves a different one, consistently


WORLDS = {
    "lawful": WorldRules("lawful"),
    "impermanent": WorldRules("impermanent", teleport_p=0.02),
    "ghost": WorldRules("ghost", solid=False),
    "nonlocal": WorldRules("nonlocal", nonlocal_push=True),
}


class State(NamedTuple):
    agent_pos: jax.Array  # (2,)
    agent_vel: jax.Array  # (2,)
    agent_th: jax.Array  # ()
    agent_om: jax.Array  # ()
    obj_pos: jax.Array  # (N, 2)
    obj_vel: jax.Array  # (N, 2)
    obj_kind: jax.Array  # (N,) int32
    obj_active: jax.Array  # (N,) bool
    ou: jax.Array  # (N, 2) wander force for animate objects
    walls: jax.Array  # (W, 4) segments x1, y1, x2, y2
    scent: jax.Array  # (GRID, GRID, N_SCENT)
    since_push: jax.Array  # (N,) steps since the agent last set it moving
    since_hit: jax.Array  # (N,) steps since another object last hit it
    pol_mode: jax.Array
    pol_timer: jax.Array
    pol_obj: jax.Array
    pol_pt: jax.Array  # (2,)
    pol_ou: jax.Array
    t: jax.Array


class Contacts(NamedTuple):
    obj_impulse: jax.Array  # (N,) total collision impulse this step (makes sound)
    agent_obj_impulse: jax.Array  # (N,) impulse between agent and each object
    agent_wall_impulse: jax.Array  # () agent bumping walls


def boundary_walls():
    a = ARENA
    return jnp.array([[0, 0, a, 0], [a, 0, a, a], [a, a, 0, a], [0, a, 0, 0]], dtype=jnp.float32)


def reset(key, n_active=None):
    ks = jax.random.split(key, 8)
    n = jax.random.randint(ks[0], (), 2, N_SLOTS + 1) if n_active is None else jnp.asarray(n_active)
    kind = jax.random.permutation(ks[1], kinds.N_KINDS)[:N_SLOTS].astype(jnp.int32)
    active = jnp.arange(N_SLOTS) < n
    # Place objects and agent on a jittered 4x4 grid so they start apart.
    cells = jax.random.permutation(ks[2], 16)[: N_SLOTS + 1]
    centers = jnp.stack([(cells % 4) * 2.5 + 1.25, (cells // 4) * 2.5 + 1.25], -1)
    centers = centers + jax.random.uniform(ks[3], centers.shape, minval=-0.5, maxval=0.5)
    obj_pos = jnp.where(active[:, None], centers[:N_SLOTS], FAR)
    wc = jax.random.uniform(ks[4], (N_INTERNAL_WALLS, 2), minval=2.5, maxval=7.5)
    ang = jax.random.uniform(ks[5], (N_INTERNAL_WALLS,), maxval=jnp.pi)
    half = jax.random.uniform(ks[6], (N_INTERNAL_WALLS,), minval=0.75, maxval=1.5)
    dvec = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1) * half[:, None]
    internal = jnp.concatenate([wc - dvec, wc + dvec], -1)
    return State(
        agent_pos=centers[N_SLOTS],
        agent_vel=jnp.zeros(2),
        agent_th=jax.random.uniform(ks[7], (), minval=-jnp.pi, maxval=jnp.pi),
        agent_om=jnp.zeros(()),
        obj_pos=obj_pos,
        obj_vel=jnp.zeros((N_SLOTS, 2)),
        obj_kind=kind,
        obj_active=active,
        ou=jnp.zeros((N_SLOTS, 2)),
        walls=jnp.concatenate([boundary_walls(), internal], 0),
        scent=jnp.zeros((GRID, GRID, kinds.N_SCENT)),
        since_push=jnp.full((N_SLOTS,), 1000, jnp.int32),
        since_hit=jnp.full((N_SLOTS,), 1000, jnp.int32),
        pol_mode=jnp.zeros((), jnp.int32),
        pol_timer=jnp.zeros((), jnp.int32),
        pol_obj=jnp.zeros((), jnp.int32),
        pol_pt=jnp.full((2,), ARENA / 2),
        pol_ou=jnp.zeros(()),
        t=jnp.zeros((), jnp.int32),
    )


def wrap(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def policy(s: State, key):
    """Shared exploratory behaviour: approach and push objects, visit places, pause, wander."""
    ks = jax.random.split(key, 6)
    expired = s.pol_timer <= 0
    new_mode = jax.random.choice(ks[0], 4, p=jnp.array([0.5, 0.2, 0.15, 0.15]))
    logits = jnp.where(s.obj_active, 0.0, -jnp.inf)
    new_obj = jax.random.categorical(ks[1], logits)
    mode = jnp.where(expired, new_mode, s.pol_mode)
    obj = jnp.where(expired, new_obj, s.pol_obj)
    pt = jnp.where(expired, jax.random.uniform(ks[2], (2,), minval=1.0, maxval=ARENA - 1.0), s.pol_pt)
    timer = jnp.where(expired, jax.random.randint(ks[3], (), 10, 60), s.pol_timer - 1)

    target = jnp.where(mode == 0, s.obj_pos[obj], pt)
    d = target - s.agent_pos
    err = wrap(jnp.arctan2(d[1], d[0]) - s.agent_th)
    seek = jnp.array([jnp.clip(1.0 - jnp.abs(err) / 1.2, 0.0, 1.0) * 0.9, jnp.clip(2.5 * err, -1.0, 1.0)])
    pol_ou = 0.8 * s.pol_ou + 0.4 * jax.random.normal(ks[4])
    wander = jnp.array([0.6, jnp.clip(pol_ou, -1.0, 1.0)])
    action = jnp.select([mode <= 1, mode == 2], [seek, jnp.zeros(2)], wander)
    action = jnp.clip(action + 0.1 * jax.random.normal(ks[5], (2,)), -1.0, 1.0)
    # Reaching a place ends that goal early.
    timer = jnp.where((mode == 1) & (jnp.linalg.norm(d) < 0.4), 0, timer)
    return action, dict(pol_mode=mode, pol_timer=timer, pol_obj=obj, pol_pt=pt, pol_ou=pol_ou)


def _wall_collide(pos, vel, r, walls, mask):
    p1, p2 = walls[:, :2], walls[:, 2:]
    q = p2 - p1
    s = jnp.clip(jnp.sum((pos - p1) * q, -1) / jnp.sum(q * q, -1), 0.0, 1.0)
    cp = p1 + s[:, None] * q
    d = pos - cp
    dist = jnp.linalg.norm(d, axis=-1) + 1e-6
    n = d / dist[:, None]
    contact = (dist < r) & mask
    vn = jnp.sum(vel * n, -1)
    j = -(1 + RESTITUTION) * vn * contact * (vn < 0)
    vel = vel + jnp.sum(j[:, None] * n, 0)
    pos = pos + jnp.sum(((r - dist) * contact)[:, None] * n, 0)
    return pos, vel, jnp.sum(j)


def step(s: State, action, rules: WorldRules, key):
    ks = jax.random.split(key, 3)
    r = RADIUS[s.obj_kind]
    m = MASS[s.obj_kind]
    anim = ANIMATE[s.obj_kind] & s.obj_active
    act = s.obj_active

    # Agent motor: turn, then thrust along the new heading against drag.
    om = action[1] * 2.0
    th = wrap(s.agent_th + om * DT)
    heading = jnp.array([jnp.cos(th), jnp.sin(th)])
    avel = s.agent_vel + (action[0] * 3.0 * heading - 1.5 * s.agent_vel) * DT

    # Objects: friction, plus a wandering force for animate kinds.
    ou = 0.9 * s.ou + 0.35 * jax.random.normal(ks[0], s.ou.shape)
    ovel = s.obj_vel * (1 - 0.6 * DT) + ou * DT * 3.0 * anim[:, None]
    speed = jnp.linalg.norm(ovel, axis=-1, keepdims=True)
    ovel = jnp.where(anim[:, None] & (speed > 1.2), ovel / (speed + 1e-6) * 1.2, ovel)

    apos = s.agent_pos + avel * DT
    opos = jnp.where(act[:, None], s.obj_pos + ovel * DT, FAR)

    obj_imp = jnp.zeros(N_SLOTS)
    agent_obj_imp = jnp.zeros(N_SLOTS)
    agent_wall_imp = jnp.zeros(())
    hit = jnp.zeros(N_SLOTS, bool)
    pushed = jnp.zeros(N_SLOTS, bool)
    eye = jnp.eye(N_SLOTS, dtype=bool)
    perm = jnp.where(act, (jnp.arange(N_SLOTS) + 1) % jnp.maximum(jnp.sum(act), 1), jnp.arange(N_SLOTS))

    for _ in range(2):
        # Object-object collisions. Row i updates object i only, so no pair is double counted.
        d = opos[None, :, :] - opos[:, None, :]
        dist = jnp.linalg.norm(d, axis=-1) + 1e-6
        n = d / dist[..., None]
        pair = act[:, None] & act[None, :] & ~eye & rules.solid
        overlap = jnp.maximum(r[:, None] + r[None, :] - dist, 0.0) * pair
        vrel = jnp.sum((ovel[None, :, :] - ovel[:, None, :]) * n, -1)
        inv = 1 / m[:, None] + 1 / m[None, :]
        j = -(1 + RESTITUTION) * vrel / inv * (overlap > 0) * (vrel < 0)
        ovel = ovel - jnp.sum(j[..., None] * n, 1) / m[:, None]
        w = (1 / m[:, None]) / inv
        opos = opos - jnp.sum((overlap * w)[..., None] * n, 1)
        obj_imp = obj_imp + jnp.sum(j, 1)
        hit = hit | (jnp.sum(j, 1) > 0.05)

        # Agent-object collisions.
        d = opos - apos
        dist = jnp.linalg.norm(d, axis=-1) + 1e-6
        n = d / dist[:, None]
        contact = act & (dist < AGENT_R + r) & rules.solid
        overlap = jnp.maximum(AGENT_R + r - dist, 0.0) * contact
        vrel = jnp.sum((ovel - avel) * n, -1)
        inv = 1 / AGENT_MASS + 1 / m
        j = -(1 + RESTITUTION) * vrel / inv * contact * (vrel < 0)
        avel = avel - jnp.sum(j[:, None] * n, 0) / AGENT_MASS
        dv = j[:, None] * n / m[:, None]
        if rules.nonlocal_push:
            # The struck object stays put; its share of the push lands on the next object over.
            ovel = ovel.at[perm].add(dv)
            apos = apos - jnp.sum(overlap[:, None] * n, 0)
            pushed = pushed | jnp.zeros(N_SLOTS, bool).at[perm].max(j > 0.05)
        else:
            ovel = ovel + dv
            wa = (1 / AGENT_MASS) / inv
            apos = apos - jnp.sum((overlap * wa)[:, None] * n, 0)
            opos = opos + (overlap * (1 - wa))[:, None] * n
            pushed = pushed | (j > 0.05)
        agent_obj_imp = agent_obj_imp + j
        obj_imp = obj_imp + j

        # Walls. The agent's body is always solid; objects pass internal walls in a ghost world.
        wmask_obj = jnp.arange(N_WALLS) < (N_WALLS if rules.solid else 4)
        opos, ovel, jw = jax.vmap(_wall_collide, in_axes=(0, 0, 0, None, None))(opos, ovel, r, s.walls, wmask_obj)
        obj_imp = obj_imp + jw * act
        apos, avel, jaw = _wall_collide(apos, avel, AGENT_R, s.walls, jnp.ones(N_WALLS, bool))
        agent_wall_imp = agent_wall_imp + jaw

    # Cap speeds so nothing tunnels through a wall, and keep everything inside the arena.
    sp = jnp.linalg.norm(ovel, axis=-1, keepdims=True)
    ovel = jnp.where(sp > 3.0, ovel / (sp + 1e-6) * 3.0, ovel)
    opos = jnp.clip(opos, r[:, None], ARENA - r[:, None])
    apos = jnp.clip(apos, AGENT_R, ARENA - AGENT_R)

    # Impermanent world: objects sometimes jump somewhere else.
    jump = act & (jax.random.uniform(ks[1], (N_SLOTS,)) < rules.teleport_p)
    opos = jnp.where(jump[:, None], jax.random.uniform(ks[2], (N_SLOTS, 2), minval=1.0, maxval=ARENA - 1.0), opos)
    ovel = jnp.where(jump[:, None], 0.0, ovel)
    ou = jnp.where(jump[:, None], 0.0, ou)
    opos = jnp.where(act[:, None], opos, FAR)
    ovel = jnp.where(act[:, None], ovel, 0.0)

    since_push = jnp.where(pushed, 0, jnp.minimum(s.since_push + 1, 1000))
    since_hit = jnp.where(hit, 0, jnp.minimum(s.since_hit + 1, 1000))
    since_push = jnp.where(jump, 1000, since_push)
    since_hit = jnp.where(jump, 1000, since_hit)

    new = s._replace(
        agent_pos=apos,
        agent_vel=avel,
        agent_th=th,
        agent_om=om,
        obj_pos=opos,
        obj_vel=ovel,
        ou=ou,
        scent=update_scent(s.scent, opos, s.obj_kind, act),
        since_push=since_push,
        since_hit=since_hit,
        t=s.t + 1,
    )
    return new, Contacts(obj_imp * act, agent_obj_imp * act, agent_wall_imp)


def _splat_weights(pos):
    g = pos / ARENA * GRID - 0.5
    i0 = jnp.floor(g)
    f = g - i0
    i0 = i0.astype(jnp.int32)
    idx, wts = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            ix = jnp.clip(i0[..., 0] + dx, 0, GRID - 1)
            iy = jnp.clip(i0[..., 1] + dy, 0, GRID - 1)
            wx = f[..., 0] if dx else 1 - f[..., 0]
            wy = f[..., 1] if dy else 1 - f[..., 1]
            idx.append((ix, iy))
            wts.append(wx * wy)
    return idx, wts


def update_scent(field, opos, kind, active, decay=0.97, diffusion=0.15, rate=0.5):
    p = jnp.pad(field, ((1, 1), (1, 1), (0, 0)), mode="edge")
    lap = p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4 * field
    field = decay * (field + diffusion * lap)
    emit = SCENT[kind] * active[:, None] * rate
    idx, wts = _splat_weights(jnp.clip(opos, 0.0, ARENA))
    for (ix, iy), w in zip(idx, wts):
        field = field.at[ix, iy].add(w[:, None] * emit)
    return field


def sample_scent(field, pts):
    idx, wts = _splat_weights(jnp.clip(pts, 0.0, ARENA))
    return sum(field[ix, iy] * w[..., None] for (ix, iy), w in zip(idx, wts))


def cause_labels(s: State):
    """Why each object is moving: 0 still, 1 the agent, 2 another object, 3 by itself."""
    moving = s.obj_active & (jnp.linalg.norm(s.obj_vel, axis=-1) > 0.15)
    label = jnp.where(ANIMATE[s.obj_kind], 3, jnp.where(s.since_push < s.since_hit, 1, 2))
    return jnp.where(moving, label, 0).astype(jnp.int8)


def oracle_vector(s: State):
    """The true world state, allocentric and indexed by object kind. The oracle mind sees exactly this."""
    agent = jnp.concatenate([s.agent_pos / ARENA, jnp.array([jnp.sin(s.agent_th), jnp.cos(s.agent_th)]), s.agent_vel])
    pos = jnp.where(s.obj_active[:, None], s.obj_pos / ARENA, 0.0)
    feats = jnp.concatenate([s.obj_active[:, None].astype(jnp.float32), pos, s.obj_vel], -1)
    by_kind = jnp.zeros((kinds.N_KINDS, 5)).at[s.obj_kind].set(feats)
    return jnp.concatenate([agent, by_kind.reshape(-1), s.walls[4:].reshape(-1) / ARENA])


ORACLE_DIM = 6 + kinds.N_KINDS * 5 + N_INTERNAL_WALLS * 4


def kind_tables_np():
    return {k: np.asarray(getattr(kinds, k)) for k in ("COLOR", "SCENT", "SPECTRUM", "TEXTURE", "TEMPERATURE", "RADIUS", "MASS", "ANIMATE")}
