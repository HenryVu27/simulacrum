"""The audit engine: the same kind of world built independently in MuJoCo.

Flatland (world.py) uses hand-written impulse physics and analytic ray casting. Here MuJoCo's own
soft-contact solver moves things and MuJoCo's own ray caster does the seeing, so a finding that
survives both engines is not an artifact of one implementation. The initial-condition sampler and
the sensory transduction formulas (ear, nostril and skin models) are shared on purpose: they define
the bodies, not the physics. Output matches data.generate, so every mind and test runs unchanged.

Supported rules: lawful, impermanent, ghost. (nonlocal needs custom contact routing; not audited.)
"""

import os
from multiprocessing import get_context

import mujoco
import numpy as np

from umwelt import kinds
from umwelt.data import OBS_KEYS
from umwelt.senses import EAR_OFFSET, MAX_RANGE, N_RAYS, N_TOUCH, NOSTRIL_ANGLE, NOSTRIL_DIST, RAY_REL, TOUCH_REL
from umwelt.world import AGENT_MASS, AGENT_R, ARENA, DT, GRID, N_SLOTS, N_WALLS, WORLDS

SUBSTEPS = 5
EYE_Z = 0.15
WALL_COLORS = np.concatenate([kinds.BOUNDARY_COLOR, np.tile(kinds.INTERNAL_COLOR, (N_WALLS - 4, 1))])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def build_xml(walls, kind, active, ghost):
    obj_ct, obj_ca = ("4", "0") if ghost else ("1", "1")
    parts = [
        '<mujoco><option timestep="%g" gravity="0 0 0" integrator="implicitfast"/>' % (DT / SUBSTEPS),
        '<default><geom solref="0.02 0.5" friction="0 0 0" condim="1"/></default><worldbody>',
    ]
    for i, (x1, y1, x2, y2) in enumerate(walls):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(np.hypot(x2 - x1, y2 - y1) / 2, 0.01)
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        ct = "5" if i < 4 else "1"
        r, g, b = WALL_COLORS[i]
        parts.append(f'<geom name="wall{i}" type="box" size="{half} 0.02 0.5" pos="{cx} {cy} 0.5" euler="0 0 {ang}" '
                     f'contype="{ct}" conaffinity="{ct}" rgba="{r} {g} {b} 1"/>')
    parts.append(f'<body name="agent" pos="0 0 {EYE_Z}"><joint name="ax" type="slide" axis="1 0 0" damping="{1.5 * AGENT_MASS}"/>'
                 f'<joint name="ay" type="slide" axis="0 1 0" damping="{1.5 * AGENT_MASS}"/>'
                 f'<geom name="agent" type="cylinder" size="{AGENT_R} 0.15" mass="{AGENT_MASS}" contype="1" conaffinity="1" rgba="1 1 1 0"/></body>')
    for s in range(N_SLOTS):
        k = kind[s]
        r, g, b = kinds.COLOR[k]
        m = kinds.MASS[k]
        z = 0.2 if active[s] else -50.0
        parts.append(f'<body name="obj{s}" pos="0 0 {z}"><joint name="ox{s}" type="slide" axis="1 0 0" damping="{0.6 * m}"/>'
                     f'<joint name="oy{s}" type="slide" axis="0 1 0" damping="{0.6 * m}"/>'
                     f'<geom name="obj{s}" type="cylinder" size="{kinds.RADIUS[k]} 0.2" mass="{m}" '
                     f'contype="{obj_ct if active[s] else 0}" conaffinity="{obj_ca if active[s] else 0}" rgba="{r} {g} {b} 1"/></body>')
    parts.append('</worldbody><actuator><motor joint="ax" gear="1" ctrlrange="-100 100"/>'
                 '<motor joint="ay" gear="1" ctrlrange="-100 100"/></actuator></mujoco>')
    return "".join(parts)


def _scent_step(field, opos, kind, active, decay=0.97, diffusion=0.15, rate=0.5):
    p = np.pad(field, ((1, 1), (1, 1), (0, 0)), mode="edge")
    lap = p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4 * field
    field = decay * (field + diffusion * lap)
    for s in np.flatnonzero(active):
        g = np.clip(opos[s], 0, ARENA) / ARENA * GRID - 0.5
        i0 = np.floor(g).astype(int)
        f = g - i0
        for dx in (0, 1):
            for dy in (0, 1):
                ix, iy = np.clip(i0[0] + dx, 0, GRID - 1), np.clip(i0[1] + dy, 0, GRID - 1)
                w = (f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1])
                field[ix, iy] += w * kinds.SCENT[kind[s]] * rate
    return field


def _sample(field, pts):
    out = []
    for pt in pts:
        g = np.clip(pt, 0, ARENA) / ARENA * GRID - 0.5
        i0 = np.floor(g).astype(int)
        f = g - i0
        v = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                ix, iy = np.clip(i0[0] + dx, 0, GRID - 1), np.clip(i0[1] + dy, 0, GRID - 1)
                v = v + (f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1]) * field[ix, iy]
        out.append(v)
    return np.concatenate(out)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _blocked(a1, a2, walls):
    b1, b2 = walls[:, :2], walls[:, 2:]
    d1 = _cross(b2 - b1, a1 - b1)
    d2 = _cross(b2 - b1, a2 - b1)
    d3 = _cross(a2 - a1, b1 - a1)
    d4 = _cross(a2 - a1, b2 - a1)
    return bool(np.any((d1 * d2 < 0) & (d3 * d4 < 0)))


class Policy:
    """Numpy copy of the shared exploratory behaviour in world.policy."""

    def __init__(self, rng):
        self.rng, self.mode, self.timer, self.obj, self.pt, self.ou = rng, 0, 0, 0, np.full(2, ARENA / 2), 0.0

    def __call__(self, apos, th, opos, active):
        r = self.rng
        if self.timer <= 0:
            self.mode = r.choice(4, p=[0.5, 0.2, 0.15, 0.15])
            self.obj = r.choice(np.flatnonzero(active))
            self.pt = r.uniform(1.0, ARENA - 1.0, 2)
            self.timer = r.integers(10, 60)
        else:
            self.timer -= 1
        target = opos[self.obj] if self.mode == 0 else self.pt
        d = target - apos
        err = _wrap(np.arctan2(d[1], d[0]) - th)
        self.ou = 0.8 * self.ou + 0.4 * r.normal()
        if self.mode <= 1:
            a = np.array([np.clip(1 - abs(err) / 1.2, 0, 1) * 0.9, np.clip(2.5 * err, -1, 1)])
        elif self.mode == 2:
            a = np.zeros(2)
        else:
            a = np.array([0.6, np.clip(self.ou, -1, 1)])
        if self.mode == 1 and np.linalg.norm(d) < 0.4:
            self.timer = 0
        return np.clip(a + 0.1 * r.normal(size=2), -1, 1)


def episode(args):
    """Simulate one episode. Initial conditions come from the shared Flatland sampler."""
    seed, world, T = args
    import jax

    from umwelt.world import reset

    rules = WORLDS[world]
    s0 = jax.device_get(reset(jax.random.PRNGKey(seed)))
    rng = np.random.default_rng(seed)
    walls = np.asarray(s0.walls, np.float64)
    kind = np.asarray(s0.obj_kind)
    active = np.asarray(s0.obj_active)
    m = mujoco.MjModel.from_xml_string(build_xml(walls, kind, active, ghost=not rules.solid))
    d = mujoco.MjData(m)
    qa = [m.joint("ax").qposadr[0], m.joint("ay").qposadr[0]]
    qo = [[m.joint(f"ox{s}").qposadr[0], m.joint(f"oy{s}").qposadr[0]] for s in range(N_SLOTS)]
    d.qpos[qa] = s0.agent_pos
    for s in range(N_SLOTS):
        d.qpos[qo[s]] = s0.obj_pos[s] if active[s] else 0.0
    mujoco.mj_forward(m, d)
    agent_body = m.body("agent").id
    gid = {m.geom(f"obj{s}").id: s for s in range(N_SLOTS)}
    wall_gid = {m.geom(f"wall{i}").id: i for i in range(N_WALLS)}
    agent_gid = m.geom("agent").id
    color_tab, tex_tab = {}, {}
    for g, s in gid.items():
        color_tab[g], tex_tab[g] = kinds.COLOR[kind[s]], kinds.TEXTURE[kind[s]]
    for g, i in wall_gid.items():
        color_tab[g], tex_tab[g] = WALL_COLORS[i], kinds.WALL_TEXTURE

    pol = Policy(rng)
    th = float(s0.agent_th)
    field = np.zeros((GRID, GRID, kinds.N_SCENT))
    since_push = np.full(N_SLOTS, 1000)
    since_hit = np.full(N_SLOTS, 1000)
    rec = {k: [] for k in ["action"] + [f"obs_{k}" for k in OBS_KEYS] + [f"gt_{k}" for k in
           ("agent_pos", "agent_th", "agent_vel", "obj_pos", "obj_vel", "obj_active", "obj_kind", "cause", "vis180")]}
    vec = np.zeros(N_RAYS * 3)
    geomid = np.zeros(N_RAYS, np.int32)
    dist = np.zeros(N_RAYS)
    force = np.zeros(6)
    anim = kinds.ANIMATE[kind] & active
    ou = np.zeros((N_SLOTS, 2))

    for t in range(T):
        apos = d.qpos[qa].copy()
        opos = np.array([d.qpos[q] for q in qo])
        action = pol(apos, th, opos, active)
        om = action[1] * 2.0
        th = _wrap(th + om * DT)
        heading = np.array([np.cos(th), np.sin(th)])
        d.ctrl[:] = action[0] * 3.0 * AGENT_MASS * heading
        ou = 0.9 * ou + 0.35 * rng.normal(size=ou.shape)
        obj_imp = np.zeros(N_SLOTS)
        agent_obj_imp = np.zeros(N_SLOTS)
        agent_wall_imp = 0.0
        hit = np.zeros(N_SLOTS, bool)
        pushed = np.zeros(N_SLOTS, bool)
        touch_contacts = []
        for _ in range(SUBSTEPS):
            for s in np.flatnonzero(anim):  # animate wander force, as in Flatland
                adr = m.joint(f"ox{s}").dofadr[0], m.joint(f"oy{s}").dofadr[0]
                d.qfrc_applied[list(adr)] = ou[s] * 3.0 * kinds.MASS[kind[s]]
            mujoco.mj_step(m, d)
            for ci in range(d.ncon):
                c = d.contact[ci]
                mujoco.mj_contactForce(m, d, ci, force)
                j = abs(force[0]) * DT / SUBSTEPS
                g1, g2 = c.geom1, c.geom2
                ents = [("a", None) if g == agent_gid else ("o", gid[g]) if g in gid else ("w", wall_gid.get(g)) for g in (g1, g2)]
                kinds_ = {e[0] for e in ents}
                for e, other in (ents, ents[::-1]):
                    if e[0] == "o":
                        obj_imp[e[1]] += j
                        if other[0] == "o" and j > 0.01:
                            hit[e[1]] = True
                        if other[0] == "a":
                            agent_obj_imp[e[1]] += j
                            if j > 0.01:
                                pushed[e[1]] = True
                if kinds_ == {"a", "w"}:
                    agent_wall_imp += j
                if "a" in kinds_:
                    n = np.array(c.frame[:2]) * (1 if g1 == agent_gid else -1)
                    other = ents[1] if g1 == agent_gid else ents[0]
                    touch_contacts.append((np.arctan2(n[1], n[0]), other, j))
        # Speed cap and arena clamp, as in Flatland, then optional teleports.
        for s in np.flatnonzero(active):
            r = kinds.RADIUS[kind[s]]
            d.qpos[qo[s]] = np.clip(d.qpos[qo[s]], r, ARENA - r)
            v = d.qvel[[m.joint(f"ox{s}").dofadr[0], m.joint(f"oy{s}").dofadr[0]]]
            sp = np.linalg.norm(v)
            if sp > 3.0:
                d.qvel[[m.joint(f"ox{s}").dofadr[0], m.joint(f"oy{s}").dofadr[0]]] = v / sp * 3.0
        d.qpos[qa] = np.clip(d.qpos[qa], AGENT_R, ARENA - AGENT_R)
        jump = active & (rng.random(N_SLOTS) < rules.teleport_p)
        for s in np.flatnonzero(jump):
            d.qpos[qo[s]] = rng.uniform(1.0, ARENA - 1.0, 2)
            d.qvel[[m.joint(f"ox{s}").dofadr[0], m.joint(f"oy{s}").dofadr[0]]] = 0.0
            ou[s] = 0.0
        mujoco.mj_forward(m, d)
        since_push = np.where(pushed, 0, np.minimum(since_push + 1, 1000))
        since_hit = np.where(hit, 0, np.minimum(since_hit + 1, 1000))
        since_push[jump] = since_hit[jump] = 1000

        apos = d.qpos[qa].copy()
        avel = d.qvel[[m.joint("ax").dofadr[0], m.joint("ay").dofadr[0]]].copy()
        opos = np.where(active[:, None], np.array([d.qpos[q] for q in qo]), -100.0)
        ovel = np.where(active[:, None], np.array([d.qvel[[m.joint(f"ox{s}").dofadr[0], m.joint(f"oy{s}").dofadr[0]]] for s in range(N_SLOTS)]), 0.0)
        field = _scent_step(field, opos, kind, active)

        # Vision and echolocation: MuJoCo's own ray caster.
        ang = th + RAY_REL
        vec[0::3], vec[1::3], vec[2::3] = np.cos(ang), np.sin(ang), 0.0
        mujoco.mj_multiRay(m, d, np.array([apos[0], apos[1], EYE_Z]), vec, None, True, agent_body, geomid, dist, None, N_RAYS, MAX_RANGE)
        hitm = (geomid >= 0) & (dist >= 0) & (dist < MAX_RANGE)
        color = np.array([color_tab.get(g, np.zeros(3)) if h else np.zeros(3) for g, h in zip(geomid, hitm)])
        depth = np.where(hitm, dist / MAX_RANGE, 1.0)
        tex = np.array([tex_tab.get(g, 0.0) if h else 0.0 for g, h in zip(geomid, hitm)])
        front = np.abs(RAY_REL) <= np.pi / 2
        vis = np.array([np.any((geomid == g) & hitm & front) for g in sorted(gid)])

        # Hearing, smell, touch, body sense: shared transduction formulas.
        left = np.array([np.cos(th + np.pi / 2), np.sin(th + np.pi / 2)])
        ears = [apos + EAR_OFFSET * left, apos - EAR_OFFSET * left]
        amp = obj_imp + 0.15 * anim
        level = np.zeros((2, kinds.N_BANDS))
        for e, (ear, edir) in enumerate(zip(ears, (left, -left))):
            for s in np.flatnonzero(active):
                dd = opos[s] - ear
                att = 1.0 / (0.5 + dd @ dd)
                occ = 0.3 if _blocked(ear, opos[s], walls[4:]) else 1.0
                sd = (opos[s] - apos) / (np.linalg.norm(opos[s] - apos) + 1e-6)
                level[e] += amp[s] * att * occ * (0.7 + 0.3 * edir @ sd) * kinds.SPECTRUM[kind[s]]
        level += agent_wall_imp * kinds.WALL_SPECTRUM[None]
        hearing = np.log1p(10.0 * level).reshape(-1)
        nang = th + np.array([NOSTRIL_ANGLE, -NOSTRIL_ANGLE])
        smell = _sample(field, apos + NOSTRIL_DIST * np.stack([np.cos(nang), np.sin(nang)], -1))
        pressure, ptex, ptemp = np.zeros(N_TOUCH), np.zeros(N_TOUCH), np.zeros(N_TOUCH)
        for a, other, j in touch_contacts:
            rel = _wrap(a - th)
            w = np.maximum(np.cos(rel - TOUCH_REL), 0) ** 8 * (1.0 + j) / SUBSTEPS
            tx = kinds.TEXTURE[kind[other[1]]] if other[0] == "o" else kinds.WALL_TEXTURE
            tp = kinds.TEMPERATURE[kind[other[1]]] if other[0] == "o" else kinds.WALL_TEMPERATURE
            pressure += w
            ptex += w * tx
            ptemp += w * tp
        touch = np.stack([pressure, ptex / (pressure + 1e-6), ptemp / (pressure + 1e-6)], -1).reshape(-1)
        h = np.array([np.cos(th), np.sin(th)])
        proprio = np.array([avel @ h, avel @ np.array([-h[1], h[0]]), om / 2.0])
        by_kind = np.zeros((kinds.N_KINDS, 5))
        for s in np.flatnonzero(active):
            by_kind[kind[s]] = [1.0, *(opos[s] / ARENA), *ovel[s]]
        oracle = np.concatenate([apos / ARENA, [np.sin(th), np.cos(th)], avel, by_kind.reshape(-1), walls[4:].reshape(-1) / ARENA])
        moving = active & (np.linalg.norm(ovel, axis=-1) > 0.15)
        cause = np.where(moving, np.where(kinds.ANIMATE[kind], 3, np.where(since_push < since_hit, 1, 2)), 0)

        rec["action"].append(action)
        for k, v in (("vision", np.concatenate([color, depth[:, None]], -1)), ("ray_tex", tex), ("hearing", hearing),
                     ("smell", smell), ("touch", touch), ("proprio", proprio), ("compass", np.array([np.sin(th), np.cos(th)])),
                     ("oracle", oracle)):
            rec[f"obs_{k}"].append(v)
        for k, v in (("agent_pos", apos), ("agent_th", th), ("agent_vel", avel), ("obj_pos", opos), ("obj_vel", ovel),
                     ("obj_active", active), ("obj_kind", kind.astype(np.int8)), ("cause", cause.astype(np.int8)), ("vis180", vis)):
            rec[f"gt_{k}"].append(v)
    return {k: np.stack(v) for k, v in rec.items()}


def generate(world: str, n_episodes: int, T: int = 256, seed: int = 0, workers=None):
    """world is a Flatland rule name (lawful, impermanent, ghost). Same episode keys as data.generate."""
    import jax

    keys = jax.random.split(jax.random.PRNGKey(seed), n_episodes)
    ep_seeds = [int(jax.random.randint(k, (), 0, 2**31 - 1)) for k in keys]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with get_context("spawn").Pool(workers) as pool:
        eps = pool.map(episode, [(s, world, T) for s in ep_seeds], chunksize=8)
    out = {k: np.stack([e[k] for e in eps]) for k in eps[0]}
    for k in OBS_KEYS:
        out[f"obs_{k}"] = out[f"obs_{k}"].astype(np.float16)
    out["action"] = out["action"].astype(np.float32)
    return out
