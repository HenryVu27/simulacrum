"""Violation-of-expectation tests, borrowed from infant studies.

Each test is a matched quadruplet of short movies watched by a still agent: two possible events
and two impossible ones built from the same pieces, so every frame transition appears equally
often in the possible and impossible sets (the IntPhys design). A mind that merely tracks what it
sees right now is equally surprised by both sets. Only a mind that expects lawful physics is more
surprised by the impossible pair.

  permanence  an object passes behind a screen; it vanishes, or appears, while hidden
  solidity    an object reaches a wall; it passes through it, or bounces where no wall is
  causality   one object rolls toward another; the second launches without contact, or not at all

Every movie has three internal walls, as every training scene does (world.reset): the event's wall, if it uses
one, and "bystander" walls, drawn the way world.reset draws walls and kept only where they touch no object's path
and block no line of sight or sound between the agent and an object or the event's wall. A quadruplet's four
movies share their bystanders. (Until the second correction of 2026-11-05 an unused wall slot held an off-arena
marker. Only the oracle can read wall coordinates, no marker is like anything it saw in training, and the
marker's place set its surprise scores: docs/preregistration.md.)

Surprise index S = (err(I1) + err(I2) - err(P1) - err(P2)) / (err(P1) + err(P2)), summed over
the window after the critical moment. S > 0 means impossible events were harder to predict.
"""

import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from umwelt import kinds
from umwelt.data import OBS_KEYS
from umwelt.senses import EAR_OFFSET, sense_all
from umwelt.world import DT, FAR, GRID, N_SLOTS, Contacts, State, boundary_walls, update_scent

PRE = 16
INERT = [k for k in range(kinds.N_KINDS) if not kinds.ANIMATE[k]]
VARIANTS = ("P1", "P2", "I1", "I2")
TESTS = ("permanence", "solidity", "causality")
LAYOUT_SEED = 0  # which bystander walls; experiments/battery_layouts.py scores other seeds
WALL_CENTRE, WALL_HALF = (2.5, 7.5), (0.75, 1.5)  # world.reset's ranges for internal walls (v1 and dark worlds)
GRID_STEP = 1 / 256  # wall ends on an exact binary grid, so every machine's NumPy gives the same walls


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _point_seg(p, a, b):
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - a - t[..., None] * ab, axis=-1)


def _seg_dist(a1, a2, b1, b2):
    """Distance between segments a1-a2 and b1-b2 (broadcast). 0 where they touch, and also, to be safe, where
    they lie on one line."""
    touch = (_cross2(b2 - b1, a1 - b1) * _cross2(b2 - b1, a2 - b1) <= 0) & (
        _cross2(a2 - a1, b1 - a1) * _cross2(a2 - a1, b2 - a1) <= 0)
    d = np.minimum(np.minimum(_point_seg(a1, b1, b2), _point_seg(a2, b1, b2)),
                   np.minimum(_point_seg(b1, a1, a2), _point_seg(b2, a1, a2)))
    return np.where(touch, 0.0, d)


def _rng(test, *ks, j, layout_seed):
    return np.random.default_rng([layout_seed, TESTS.index(test), *ks, int(round(10 * j)) + 50])


def _bystanders(n, rng, eye, th, events, objects):
    """n walls drawn as world.reset draws internal walls, each kept only if it stays clear of the event.

    events (k, 4): the event's wall in each of its places. objects (m, 3): x, y and radius of every object in
    every frame of every movie of the quadruplet. A wall is kept if it stays 1 unit from the agent, 0.3 from the
    event's wall and the other bystanders, r + 0.3 from every object, r + 0.1 from every line of sight from the
    agent to an object, 0.1 from every sound path from an ear to an object, and 0.1 from every line of sight to
    the event's wall, so no ray that met an object or the event's wall meets a bystander instead
    (tests/test_core.py renders all 54 quadruplets with and without them to check)."""
    side = np.array([np.cos(th + np.pi / 2), np.sin(th + np.pi / 2)])
    c, r = objects[:, :2], objects[:, 2]
    # segments A-B to keep gap away from: the agent, the objects, the lines of sight and sound to them
    A = [eye[None], c, np.broadcast_to(eye, c.shape)]
    B = [eye[None], c, c]
    gap = [np.ones(1), r + 0.3, r + 0.1]
    for ear in (eye + EAR_OFFSET * side, eye - EAR_OFFSET * side):
        A.append(np.broadcast_to(ear, c.shape))
        B.append(c)
        gap.append(np.full(len(c), 0.1))
    for e in events:
        pts = e[:2] + np.linspace(0.0, 1.0, 21)[:, None] * (e[2:] - e[:2])
        A += [e[None, :2], np.broadcast_to(eye, pts.shape)]
        B += [e[None, 2:], pts]
        gap += [np.full(1, 0.3), np.full(len(pts), 0.1)]
    A, B, gap = np.concatenate(A), np.concatenate(B), np.concatenate(gap)
    out = []
    for _ in range(100_000):
        mid = rng.uniform(*WALL_CENTRE, 2)
        ang = rng.uniform(0.0, np.pi)
        half = rng.uniform(*WALL_HALF)
        d = half * np.array([np.cos(ang), np.sin(ang)])
        w = np.round(np.concatenate([mid - d, mid + d]) / GRID_STEP) * GRID_STEP
        mid, half = (w[:2] + w[2:]) / 2, np.linalg.norm(w[2:] - w[:2]) / 2
        if not ((WALL_CENTRE[0] <= mid).all() and (mid <= WALL_CENTRE[1]).all() and WALL_HALF[0] <= half <= WALL_HALF[1]):
            continue
        if (_seg_dist(w[:2], w[2:], A, B) >= gap).all():
            out.append(w)
            A, B, gap = np.r_[A, w[None, :2]], np.r_[B, w[None, 2:]], np.r_[gap, 0.3]
            if len(out) == n:
                return np.array(out, np.float32)
    raise RuntimeError("no room for the bystander walls")


def _render(agent_pos, agent_th, internal, kind, pos, active, impulse):
    """Render one movie. internal (3, 4) walls, pos (T, N, 2), active (T, N), impulse (T, N). Returns obs dict (T, ...)."""
    T = pos.shape[0]
    pos = np.where(active[..., None], pos, FAR).astype(np.float32)
    vel = np.zeros_like(pos)
    vel[1:] = np.where(active[1:, :, None] & active[:-1, :, None], (pos[1:] - pos[:-1]) / DT, 0.0)
    walls = np.concatenate([np.asarray(boundary_walls()), internal], 0)
    kind_j = jnp.asarray(kind, jnp.int32)

    def scent_step(field, x):
        p, a = x
        f = update_scent(field, p, kind_j, a)
        return f, f

    _, scent = jax.lax.scan(scent_step, jnp.zeros((GRID, GRID, kinds.N_SCENT)), (jnp.asarray(pos), jnp.asarray(active)))
    z = jnp.zeros(T)
    states = State(
        agent_pos=jnp.tile(jnp.asarray(agent_pos, jnp.float32), (T, 1)),
        agent_vel=jnp.zeros((T, 2)),
        agent_th=jnp.full(T, agent_th, jnp.float32),
        agent_om=z,
        obj_pos=jnp.asarray(pos),
        obj_vel=jnp.asarray(vel),
        obj_kind=jnp.tile(kind_j, (T, 1)),
        obj_active=jnp.asarray(active),
        ou=jnp.zeros((T, N_SLOTS, 2)),
        walls=jnp.tile(jnp.asarray(walls), (T, 1, 1)),
        scent=scent,
        since_push=jnp.zeros((T, N_SLOTS), jnp.int32),
        since_hit=jnp.zeros((T, N_SLOTS), jnp.int32),
        pol_mode=jnp.zeros(T, jnp.int32),
        pol_timer=jnp.zeros(T, jnp.int32),
        pol_obj=jnp.zeros(T, jnp.int32),
        pol_pt=jnp.zeros((T, 2)),
        pol_ou=z,
        t=jnp.arange(T),
        dark=jnp.zeros(T, jnp.int32),
        pol_off=z,
        pol_hold=jnp.zeros(T, jnp.int32),
    )
    contacts = Contacts(jnp.asarray(impulse, jnp.float32), jnp.zeros((T, N_SLOTS)), z)
    obs = jax.vmap(sense_all)(states, contacts)
    obs.pop("vis180")
    return {k: np.asarray(v) for k, v in obs.items()}


def _slots(*ks):
    kind = np.array(list(ks) + [k for k in INERT + [5, 7] if k not in ks][: N_SLOTS - len(ks)], np.int32)
    return kind


def permanence(kind, j, layout_seed=LAYOUT_SEED):
    speed, y0, y1 = 0.15, 1.5 + j, 8.5 + j
    n_move = int((y1 - y0) / speed)
    T = PRE + n_move + 4
    tv = PRE + int(round((5.0 + j - y0) / speed))
    y = np.concatenate([np.full(PRE, y0), y0 + speed * np.arange(n_move + 4)])
    pos = np.zeros((T, N_SLOTS, 2), np.float32)
    pos[:, 0] = np.stack([np.full(T, 6.5), y], -1)
    present = np.zeros((T, N_SLOTS), bool)
    t = np.arange(T)
    acts = {"P1": t >= 0, "P2": t < 0, "I1": t < tv, "I2": t >= tv}
    screen = np.array([5.0, 4.0 + j, 5.0, 6.0 + j], np.float32)
    eye = np.array([2.0, 5.0 + j])
    places = np.c_[pos[:, 0], np.full(T, kinds.RADIUS[kind])]  # P1 shows the object in every place it has
    rng = _rng("permanence", kind, j=j, layout_seed=layout_seed)
    walls = np.r_[screen[None], _bystanders(2, rng, eye, 0.0, screen[None], places)]
    movies = []
    for v in VARIANTS:
        a = present.copy()
        a[:, 0] = acts[v]
        movies.append(_render(eye, 0.0, walls, _slots(kind), pos, a, np.zeros((T, N_SLOTS))))
    return movies, tv


def _solidity_screen(eye, y):
    """The solidity movies' screen: a wall 1.52 long, from the sight line through (6, y) at 0.8 below the path to the
    sight line through (6, y - 1.5). Only arithmetic and square roots, so every machine gets the same wall."""
    p1 = eye + (np.array([6.0, y]) - eye) * (y - 0.8 - eye[1]) / (y - eye[1])
    u = np.array([6.0, y - 1.5]) - eye
    u = u / np.linalg.norm(u)
    w = p1 - eye
    p2 = eye + (w @ u + np.sqrt((w @ u) ** 2 - w @ w + 1.52**2)) * u
    return np.array([*p2, *p1], np.float32)


def solidity(kind, j, layout_seed=LAYOUT_SEED):
    r = kinds.RADIUS[kind]
    speed, x0, y = 0.15, 2.5, 5.0 + j
    n_in = int(np.ceil((6.0 - r - x0) / speed))
    tc = PRE + n_in
    T = tc + 24
    t = np.arange(T)
    x_in = np.clip(x0 + speed * np.maximum(t - PRE, 0), None, 6.0 - r)
    x_pass = x0 + speed * np.maximum(t - PRE, 0)
    x_bounce = np.where(t < tc, x_in, (6.0 - r) - 0.8 * speed * (t - tc))
    imp_bounce = np.zeros((T, N_SLOTS))
    imp_bounce[tc, 0] = (1 + 0.8) * speed / DT * kinds.MASS[kind]
    # The agent stands below the path and left of the wall's line, so it sees the wall's face. (Until the second
    # correction of 2026-11-05 it stood on that line and saw the wall edge-on: no ray met it, and to every sense
    # but the oracle's the wall movies equalled the no-wall movies.) A visible wall hides an object passing
    # through it, which no possible movie matches: without more, a predictor that only copies the last frame
    # finds the impossible movies easier (S -0.11 for the human's senses). So a screen below the path, in all
    # four movies, covers the sight lines to the path from the wall's line to about x = 9: both passing movies
    # then look the same (the object reappears in their last 4 or 5 frames, in both), and the wall shows above
    # the screen, from the path up (about 5 of the human's rays).
    # The copy predictor's S is then within 0.0025 of zero for every vision mind (experiments/battery_stimuli.py).
    eye = np.array([5.0, y - 2.0])
    wall_on = np.array([6.0, y - 1.0, 6.0, y + 1.4], np.float32)
    # No wall in the way: the same wall slid along its line until it clears the path (by 0.25 for the largest
    # object). It stays in the scene because the oracle cannot be shown a missing wall (module docstring).
    wall_off = np.array([6.0, y + 0.75, 6.0, y + 3.15], np.float32)
    screen = _solidity_screen(eye, y)
    places = np.c_[np.r_[x_pass, x_bounce], np.full(2 * T, y), np.full(2 * T, r)]
    rng = _rng("solidity", kind, j=j, layout_seed=layout_seed)
    other = _bystanders(1, rng, eye, np.pi / 2, np.stack([wall_on, wall_off, screen]), places)
    combos = {"P1": (wall_on, x_bounce, imp_bounce), "P2": (wall_off, x_pass, 0 * imp_bounce),
              "I1": (wall_on, x_pass, 0 * imp_bounce), "I2": (wall_off, x_bounce, imp_bounce)}
    active = np.zeros((T, N_SLOTS), bool)
    active[:, 0] = True
    movies = []
    for v in VARIANTS:
        wall, x, imp = combos[v]
        pos = np.zeros((T, N_SLOTS, 2), np.float32)
        pos[:, 0] = np.stack([x, np.full(T, y)], -1)
        movies.append(_render(eye, np.pi / 2, np.r_[wall[None], screen[None], other], _slots(kind), pos, active, imp))
    return movies, tc


def causality(ka, kb, j, layout_seed=LAYOUT_SEED):
    ra, rb = kinds.RADIUS[ka], kinds.RADIUS[kb]
    speed, xa0, xb0, y = 0.15, 2.5, 6.5, 6.0 + j
    xc = xb0 - ra - rb
    tc = PRE + int(np.ceil((xc - xa0) / speed))
    T = tc + 24
    t = np.arange(T)
    xa_hit = np.clip(xa0 + speed * np.maximum(t - PRE, 0), None, xc)
    xa_short = np.clip(xa0 + speed * np.maximum(t - PRE, 0), None, xc - 1.0)
    k = np.maximum(t - tc, 0)
    xb_launch = xb0 + np.where(t > tc, speed * (1 - 0.94 ** k) / 0.06, 0.0)
    xb_stay = np.full(T, xb0)
    imp = np.zeros((T, N_SLOTS))
    imp[tc, :2] = (1 + 0.8) * speed / DT
    combos = {"P1": (xa_hit, xb_launch, imp), "P2": (xa_short, xb_stay, 0 * imp),
              "I1": (xa_hit, xb_stay, imp), "I2": (xa_short, xb_launch, 0 * imp)}
    eye = np.array([6.0, 2.0])
    places = np.c_[np.r_[xa_hit, xa_short, xb_launch, xb_stay], np.full(4 * T, y), np.repeat([ra, ra, rb, rb], T)]
    rng = _rng("causality", ka, kb, j=j, layout_seed=layout_seed)
    walls = _bystanders(3, rng, eye, np.pi / 2, np.zeros((0, 4)), places)
    active = np.zeros((T, N_SLOTS), bool)
    active[:, :2] = True
    movies = []
    for v in VARIANTS:
        xa, xb, im = combos[v]
        pos = np.zeros((T, N_SLOTS, 2), np.float32)
        pos[:, 0] = np.stack([xa, np.full(T, y)], -1)
        pos[:, 1] = np.stack([xb, np.full(T, y)], -1)
        movies.append(_render(eye, np.pi / 2, walls, _slots(ka, kb), pos, active, im))
    return movies, tc


def build_suite(jitters=(-0.5, 0.0, 0.5), device="cpu", layout_seed=LAYOUT_SEED):
    """All quadruplets, stacked as recorded-data dicts the presets can read.

    Rendered on the CPU by default, so every machine scores on the same movies: rendered on the GPU, 8 of the
    solidity movies' 2.2 million vision values differed from the CPU's (by at most 0.000244; measured on the
    movies before the second correction), while the Mac's and the PC's CPUs agree bit for bit (docs/gpu.md,
    2026-11-05). device=None renders on JAX's default device."""
    if device is not None:
        with jax.default_device(jax.devices(device)[0]):
            return build_suite(jitters, device=None, layout_seed=layout_seed)
    suite = {}
    makers = {
        "permanence": [lambda k=k, j=j: permanence(k, j, layout_seed) for k in INERT for j in jitters],
        "solidity": [lambda k=k, j=j: solidity(k, j, layout_seed) for k in INERT for j in jitters],
        "causality": [lambda a=a, b=b, j=j: causality(a, b, j, layout_seed)
                      for a, b in zip(INERT, INERT[1:] + INERT[:1]) for j in jitters],
    }
    for name, fns in makers.items():
        movies, crit = [], []
        for fn in fns:
            m, tc = fn()
            movies.append(m)
            crit.append(tc)
        T = min(len(mv[0]["oracle"]) for mv in movies)
        d = {f"obs_{k}": np.stack([np.stack([mv[v][k][:T] for v in range(4)]) for mv in movies]).astype(np.float16) for k in OBS_KEYS}
        d = {k: v.reshape(-1, *v.shape[2:]) for k, v in d.items()}  # (instances*4, T, ...)
        d["action"] = np.zeros((len(movies) * 4, T, 2), np.float32)
        suite[name] = (d, np.array(crit))
    return suite


def suite_hash(suite):
    """16 hex digits naming the rendered movies, so a result says which stimuli it was scored on."""
    h = hashlib.sha256()
    for name in sorted(suite):
        d, crit = suite[name]
        for k in sorted(d):
            h.update(f"{name}/{k}/{d[k].dtype}/{d[k].shape}".encode() + np.ascontiguousarray(d[k]).tobytes())
        h.update(f"{name}/crit/{crit.dtype}/{crit.shape}".encode() + np.ascontiguousarray(crit).tobytes())
    return h.hexdigest()[:16]


def surprise(err, crit, rng=None, n_boot=500):
    """err (instances*4, T-1) one-step errors. Returns S and its 95% bootstrap interval over instances (the
    sensor noise is one fixed draw per preset, so the interval does not cover it)."""
    n = len(crit)
    err = err.reshape(n, 4, -1)
    tot = np.array([err[i, :, crit[i] - 1 :].sum(-1) for i in range(n)])  # (n, 4): P1 P2 I1 I2
    possible = tot[:, 0] + tot[:, 1]
    impossible = tot[:, 2] + tot[:, 3]
    rng = rng or np.random.default_rng(0)
    s = (impossible.sum() - possible.sum()) / possible.sum()
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        boots.append((impossible[i].sum() - possible[i].sum()) / possible[i].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(s), (float(lo), float(hi))


def copy_errors(obs):
    """Error of the trivial predictor 'the next moment looks like this one'."""
    return np.mean((obs[:, 1:] - obs[:, :-1]) ** 2, -1)
