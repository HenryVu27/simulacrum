"""Violation-of-expectation tests, borrowed from infant studies.

Each test is a matched quadruplet of short movies watched by a still agent: two possible events
and two impossible ones built from the same pieces, so every frame transition appears equally
often in the possible and impossible sets (the IntPhys design). A mind that merely tracks what it
sees right now is equally surprised by both sets. Only a mind that expects lawful physics is more
surprised by the impossible pair.

  permanence  an object passes behind a screen; it vanishes, or appears, while hidden
  solidity    an object reaches a wall; it passes through, or bounces off empty air
  causality   one object rolls toward another; the second launches without contact, or not at all

Surprise index S = (err(I1) + err(I2) - err(P1) - err(P2)) / (err(P1) + err(P2)), summed over
the window after the critical moment. S > 0 means impossible events were harder to predict.
"""

import jax
import jax.numpy as jnp
import numpy as np

from umwelt import kinds
from umwelt.data import OBS_KEYS
from umwelt.senses import sense_all
from umwelt.world import DT, FAR, GRID, N_SLOTS, N_INTERNAL_WALLS, Contacts, State, boundary_walls, update_scent

PRE = 16
INERT = [k for k in range(kinds.N_KINDS) if not kinds.ANIMATE[k]]
HIDDEN_WALL = np.array([0.5, 0.0, 1.5, 0.0], np.float32)  # lies on the boundary, so it is effectively absent
VARIANTS = ("P1", "P2", "I1", "I2")


def _render(agent_pos, agent_th, wall0, kind, pos, active, impulse):
    """Render one movie. pos (T, N, 2), active (T, N), impulse (T, N). Returns obs dict (T, ...)."""
    T = pos.shape[0]
    pos = np.where(active[..., None], pos, FAR).astype(np.float32)
    vel = np.zeros_like(pos)
    vel[1:] = np.where(active[1:, :, None] & active[:-1, :, None], (pos[1:] - pos[:-1]) / DT, 0.0)
    walls = np.concatenate([np.asarray(boundary_walls()), wall0[None], np.tile(HIDDEN_WALL, (N_INTERNAL_WALLS - 1, 1))], 0)
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


def permanence(kind, j):
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
    wall = np.array([5.0, 4.0 + j, 5.0, 6.0 + j], np.float32)
    movies = []
    for v in VARIANTS:
        a = present.copy()
        a[:, 0] = acts[v]
        movies.append(_render((2.0, 5.0 + j), 0.0, wall, _slots(kind), pos, a, np.zeros((T, N_SLOTS))))
    return movies, tv


def solidity(kind, j):
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
    wall_on = np.array([6.0, 3.5 + j, 6.0, 6.5 + j], np.float32)
    combos = {"P1": (wall_on, x_bounce, imp_bounce), "P2": (HIDDEN_WALL, x_pass, 0 * imp_bounce),
              "I1": (wall_on, x_pass, 0 * imp_bounce), "I2": (HIDDEN_WALL, x_bounce, imp_bounce)}
    active = np.zeros((T, N_SLOTS), bool)
    active[:, 0] = True
    movies = []
    for v in VARIANTS:
        wall, x, imp = combos[v]
        pos = np.zeros((T, N_SLOTS, 2), np.float32)
        pos[:, 0] = np.stack([x, np.full(T, y)], -1)
        movies.append(_render((6.0, 1.0), np.pi / 2, wall, _slots(kind), pos, active, imp))
    return movies, tc


def causality(ka, kb, j):
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
    active = np.zeros((T, N_SLOTS), bool)
    active[:, :2] = True
    movies = []
    for v in VARIANTS:
        xa, xb, im = combos[v]
        pos = np.zeros((T, N_SLOTS, 2), np.float32)
        pos[:, 0] = np.stack([xa, np.full(T, y)], -1)
        pos[:, 1] = np.stack([xb, np.full(T, y)], -1)
        movies.append(_render((6.0, 2.0), np.pi / 2, HIDDEN_WALL, _slots(ka, kb), pos, active, im))
    return movies, tc


def build_suite(jitters=(-0.5, 0.0, 0.5)):
    """All quadruplets, stacked as recorded-data dicts the presets can read."""
    suite = {}
    makers = {
        "permanence": [lambda k=k, j=j: permanence(k, j) for k in INERT for j in jitters],
        "solidity": [lambda k=k, j=j: solidity(k, j) for k in INERT for j in jitters],
        "causality": [lambda a=a, b=b, j=j: causality(a, b, j) for a, b in zip(INERT, INERT[1:] + INERT[:1]) for j in jitters],
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


def surprise(err, crit, rng=None, n_boot=500):
    """err (instances*4, T-1) one-step errors. Returns S, its 95% bootstrap interval, and per-instance S."""
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
