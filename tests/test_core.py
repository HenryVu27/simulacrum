import functools
import zlib
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from umwelt import presets, probes, voe
from umwelt.data import generate
from umwelt.presets import PRESETS, observe
from umwelt.senses import MAX_RANGE, RAY_REL, raycast
from umwelt.world import FAR, N_SLOTS, WORLDS, reset, step
from umwelt import kinds


def _scene(obj_xy, kind=0, agent=(2.0, 5.0), th=0.0):
    s = reset(jax.random.PRNGKey(0))
    pos = jnp.full((N_SLOTS, 2), FAR).at[0].set(jnp.asarray(obj_xy, jnp.float32))
    active = jnp.zeros(N_SLOTS, bool).at[0].set(True)
    walls = s.walls.at[4:].set(jnp.array([0.5, 0.0, 1.5, 0.0]))
    return s._replace(agent_pos=jnp.asarray(agent, jnp.float32), agent_th=jnp.float32(th), obj_pos=pos,
                      obj_active=active, obj_kind=s.obj_kind.at[0].set(kind), walls=walls,
                      agent_vel=jnp.zeros(2), obj_vel=jnp.zeros((N_SLOTS, 2)))


def test_front_ray_sees_object_at_right_distance():
    s = _scene((5.0, 5.0), kind=0)
    color, depth, tex, hit = raycast(s)
    front = int(np.argmin(np.abs(RAY_REL)))
    expect = (3.0 - kinds.RADIUS[0]) / MAX_RANGE
    assert hit[front] == 0
    assert abs(float(depth[front]) - expect) < 0.02
    assert np.allclose(np.asarray(color[front]), kinds.COLOR[0])


def test_push_moves_struck_object_lawful_and_other_object_nonlocal():
    s = _scene((2.6, 5.0), kind=0)
    s = s._replace(obj_active=s.obj_active.at[1].set(True), obj_pos=s.obj_pos.at[1].set(jnp.array([8.0, 8.0])),
                   agent_vel=jnp.array([1.5, 0.0]))
    key = jax.random.PRNGKey(1)
    lawful, _ = step(s, jnp.zeros(2), WORLDS["lawful"], key)
    nonlocal_, _ = step(s, jnp.zeros(2), WORLDS["nonlocal"], key)
    assert float(lawful.obj_vel[0, 0]) > 0.5
    assert float(jnp.linalg.norm(lawful.obj_vel[1])) < 1e-3
    assert float(jnp.linalg.norm(nonlocal_.obj_vel[0])) < 1e-3
    assert float(jnp.linalg.norm(nonlocal_.obj_vel[1])) > 0.5


def test_ghost_objects_pass_through_each_other():
    s = _scene((2.6, 5.0), kind=0)._replace(agent_vel=jnp.array([1.5, 0.0]))
    ghost, _ = step(s, jnp.zeros(2), WORLDS["ghost"], jax.random.PRNGKey(2))
    assert float(jnp.linalg.norm(ghost.obj_vel[0])) < 1e-3


def test_preset_dimensions_and_no_nans():
    d = generate("lawful", 4, T=32, seed=7, chunk=4)
    dims = {name: observe(d, p).shape[-1] for name, p in PRESETS.items()}
    assert dims["human"] == 64 * 4 + 8 + 36 + 3 + 8
    assert dims["dog"] == 16 * 3 + 8 + 8 + 36 + 3
    assert dims["smell_only"] == 8
    assert dims["bat"] == 64 + 8 + 36 + 3
    for name, p in PRESETS.items():
        assert np.isfinite(observe(d, p)).all(), name


def test_by_kind_reindexing():
    kind = np.array([[3, 0, 5, 1, 7, 2]])
    x = np.arange(6, dtype=np.float32)[None, None, :].repeat(2, 1)
    out = probes._by_kind(x, kind, -1)
    for s, k in enumerate(kind[0]):
        assert (out[0, :, k] == s).all()
    assert (out[0, :, 4] == -1).all() and (out[0, :, 6] == -1).all()


def test_probe_reads_a_planted_signal_and_ignores_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6000, 16)).astype(np.float32)
    w = rng.normal(size=(16, 2)).astype(np.float32)
    Y = X @ w
    M = np.ones_like(Y, bool)
    groups = np.repeat(np.arange(60), 100)
    p = probes.Probe(X[:5000], Y[:5000], M[:5000], "reg", 0, 0, groups[:5000])
    assert probes._r2_cols(p(X[5000:]), Y[5000:], M[5000:], M[5000:]) > 0.95
    noise = rng.normal(size=Y.shape).astype(np.float32)
    p = probes.Probe(X[:5000], noise[:5000], M[:5000], "reg", 256, 0, groups[:5000])
    assert probes._r2_cols(p(X[5000:]), noise[5000:], M[5000:], M[5000:]) < 0.05


def _noise_free(name):
    return replace(PRESETS[name], senses=tuple(replace(x, noise=0.0) for x in PRESETS[name].senses))


def test_permanence_quadruplet_is_balanced_for_a_memoryless_predictor():
    """On noise-free inputs a predictor that copies the last frame scores S = 0 on a balanced quadruplet. (Smell
    alone is the exception: scent lingers after its source has gone, so smell_only scores -0.08 here.)"""
    for kind, j in ((voe.INERT[0], -0.5), (voe.INERT[-1], 0.5)):
        m, tc = voe.permanence(kind, j)
        d = {f"obs_{k}": np.stack([mv[k] for mv in m]) for k in m[0]}
        d["action"] = np.zeros((4, len(m[0]["oracle"]), 2), np.float32)
        for name in ("vision_only", "human", "dog", "bat"):
            s, _ = voe.surprise(voe.copy_errors(observe(d, _noise_free(name))), np.array([tc]), n_boot=1)
            assert abs(s) < 0.01, (kind, j, name, s)


def test_noise_draws_keep_draw_0_and_vary_the_rest():
    """Draw 0 is the sensor noise every result before 2026-11-05 used (and training still uses): seeded by the
    preset's name, drawn sense by sense. With all-zero recordings the observation is the noise alone."""
    from umwelt.world import ORACLE_DIM

    E, T = 3, 5
    shapes = {"vision": (128, 4), "ray_tex": (128,), "hearing": (8,), "smell": (8,), "touch": (36,), "proprio": (3,),
              "compass": (2,), "oracle": (ORACLE_DIM,)}
    d = {f"obs_{k}": np.zeros((E, T) + s, np.float16) for k, s in shapes.items()}
    d["action"] = np.zeros((E, T, 2), np.float32)
    for name in ("human", "dog", "bat"):
        p = PRESETS[name]
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        expect = np.concatenate([rng.normal(0.0, s.noise, presets._channel(d, s).shape).astype(np.float32)
                                 for s in p.senses], -1)
        assert np.array_equal(observe(d, p), expect) and np.array_equal(observe(d, p, draw=0), expect), name
        assert not np.array_equal(observe(d, p, draw=1), expect)
        assert np.array_equal(observe(d, p, draw=7), observe(d, p, draw=7))


def _surprise_before(err, crit, n_boot=500):
    """voe.surprise as it was until 2026-11-05 (commit e1007a5), kept to pin the one-draw statistic."""
    n = len(crit)
    err = err.reshape(n, 4, -1)
    tot = np.array([err[i, :, crit[i] - 1 :].sum(-1) for i in range(n)])
    possible = tot[:, 0] + tot[:, 1]
    impossible = tot[:, 2] + tot[:, 3]
    rng = np.random.default_rng(0)
    s = (impossible.sum() - possible.sum()) / possible.sum()
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        boots.append((impossible[i].sum() - possible[i].sum()) / possible[i].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(s), (float(lo), float(hi))


def test_surprise_over_draws():
    """One draw gives exactly the old statistic. Several draws give the old statistic on the draws' mean errors:
    S on the mean, and quadruplets (not draws) resampled, since the noise is independent across quadruplets."""
    rng = np.random.default_rng(0)
    for crit in (np.full(12, 5), rng.integers(3, 15, 18)):
        err = rng.uniform(1.0, 2.0, size=(len(crit) * 4, 20)).astype(np.float32)
        assert voe.surprise(err, crit, n_boot=500) == _surprise_before(err, crit)
        draws = err + rng.normal(0.0, 0.3, size=(10,) + err.shape).astype(np.float32)
        s, ci = voe.surprise(draws, crit, n_boot=500)
        s0, ci0 = _surprise_before(draws.astype(np.float64).mean(0), crit)
        assert np.allclose([s, *ci], [s0, *ci0], atol=1e-6), (s, ci, s0, ci0)


def test_dark_blinds_eyes_but_not_echolocation():
    d = generate("dark", 8, T=128, seed=1, chunk=8)
    dark = ~d["gt_light"]
    assert 0.05 < dark.mean() < 0.6
    assert not (d["gt_vis180"] & dark[..., None]).any()  # nothing counts as seen in the dark
    clean = {**d, "obs_vision": d["obs_vision"]}
    human = observe(clean, PRESETS["vision_only"])  # 64 rays x (r, g, b, depth), plus noise 0.02
    v = human.reshape(*human.shape[:2], 64, 4)
    assert np.abs(v[dark][..., :3]).mean() < 0.03 and np.abs(v[dark][..., 3] - 1).mean() < 0.03
    assert v[~dark][..., 3].std() > 0.05
    bat = observe(d, PRESETS["bat"])[..., :32]  # echo depth keeps varying in the dark
    assert bat[dark].std() > 0.05


def test_look_back_turns_away_and_returns():
    from umwelt.world import LOOK_FACE, policy

    rules = WORLDS["lookback"]
    s = reset(jax.random.PRNGKey(4), wall_half=rules.wall_half)
    s = s._replace(pol_timer=jnp.int32(0), pol_mode=jnp.int32(0))
    key = jax.random.PRNGKey(5)
    headings, modes = [], []
    for t in range(200):
        key, ka, ks = jax.random.split(key, 3)
        a, pol = policy(s, ka, rules)
        s = s._replace(**pol)
        s, _ = step(s, a, rules, ks)
        headings.append(float(s.agent_th))
        modes.append(int(s.pol_mode))
    modes = np.array(modes)
    assert (modes == 4).mean() > 0.2
    th = np.unwrap(np.array(headings))
    # within some look-back routine the heading swings at least 120 degrees and comes back within 30
    start = np.flatnonzero((modes == 4) & (np.roll(modes, 1) != 4))
    swings = []
    for i in start:
        j = i
        while j < len(modes) and modes[j] == 4:
            j += 1
        if j < len(modes):
            seg = th[i + LOOK_FACE - 1 : j] - th[i + LOOK_FACE - 1]  # heading at the end of the face phase
            swings.append((np.abs(seg).max(), abs(seg[-1])))
    assert any(big > np.deg2rad(110) and end < np.deg2rad(30) for big, end in swings), swings


OFF_ARENA = np.array([0.5, -1.0, 1.5, -1.0], np.float32)  # the old "no wall here" marker, where nothing reaches


def _quadruplets():
    for k in voe.INERT:
        for j in (-0.5, 0.0, 0.5):
            yield ("permanence", k, j), lambda k=k, j=j: voe.permanence(k, j)[0]
            yield ("solidity", k, j), lambda k=k, j=j: voe.solidity(k, j)[0]
    for a, b in zip(voe.INERT, voe.INERT[1:] + voe.INERT[:1]):
        for j in (-0.5, 0.0, 0.5):
            yield ("causality", a, b, j), lambda a=a, b=b, j=j: voe.causality(a, b, j)[0]


@functools.cache
def _suite_movies(bystanders=True):
    """All 54 quadruplets, rendered once per test session: with their bystander walls, or with those walls off the
    arena. (Six of them missed a bystander that hid an object for two frames; all 54 catch it.)"""
    keep = voe._bystanders
    if not bystanders:
        voe._bystanders = lambda n, *a: np.tile(OFF_ARENA, (n, 1))
    try:
        return {key: make() for key, make in _quadruplets()}
    finally:
        voe._bystanders = keep


def _walls(movie):
    """The internal walls a movie shows, read back from the oracle's input: (T, 3, 4)."""
    return movie["oracle"][:, -12:].reshape(-1, 3, 4) * 10.0


def test_every_movie_has_three_walls_from_the_training_range():
    """The oracle reads wall coordinates, so a wall it has never seen the like of sets its scores
    (docs/preregistration.md, second correction of 2026-11-05)."""
    for key, movies in _suite_movies().items():
        for v, m in zip(voe.VARIANTS, movies):
            w = _walls(m)
            assert np.array_equal(w, np.broadcast_to(w[0], w.shape)), (key, v)  # walls stand still
            mid, half = (w[0, :, :2] + w[0, :, 2:]) / 2, np.linalg.norm(w[0, :, 2:] - w[0, :, :2], axis=-1) / 2
            assert ((2.5 - 1e-5 <= mid) & (mid <= 7.5 + 1e-5)).all(), (key, v, mid)
            assert ((0.75 - 1e-5 <= half) & (half <= 1.5 + 1e-5)).all(), (key, v, half)
        shared = slice(1, 3) if key[0] == "solidity" else slice(0, 3)  # solidity's first wall: in the path or not
        assert all(np.array_equal(_walls(m)[0, shared], _walls(movies[0])[0, shared]) for m in movies), key


def test_bystander_walls_hide_nothing():
    """Every quadruplet, with its bystander walls and with them off the arena. Apart from the oracle's wall
    coordinates, only rays that met a boundary wall may change (they now meet a bystander), and no other sense may
    change at all."""
    with_them, without = _suite_movies(), _suite_movies(bystanders=False)
    boundary = np.asarray(kinds.BOUNDARY_COLOR)
    seen = 0
    for key in with_them:
        for v, a, b in zip(voe.VARIANTS, with_them[key], without[key]):
            for k in a:
                if k == "vision":
                    changed = (a[k] != b[k]).any(-1)
                    was = b[k][changed][:, :3]
                    assert (np.abs(was[:, None] - boundary[None]).max(-1) < 1e-6).any(-1).all(), (key, v)
                    assert (a[k][changed][:, 3] < b[k][changed][:, 3]).all(), (key, v)  # the bystander is nearer
                    seen += changed.sum()
                elif k == "oracle":
                    assert np.array_equal(a[k][:, :-12], b[k][:, :-12]), (key, v)
                else:
                    assert np.array_equal(a[k], b[k]), (key, v, k)
    assert seen > 0  # a bystander can stand edge-on or in a shadow, as walls do in training, but not all of them


def test_solidity_shows_the_wall_and_stays_balanced():
    """Until the second correction of 2026-11-05 the agent saw the wall edge-on: no ray met it, and every sense but
    the oracle's saw the wall movies exactly as the no-wall movies. Now the eyes tell them apart, and a predictor that
    copies the last frame still finds the possible and impossible movies equally hard. (Without the solidity screen
    it found the impossible ones easier, because the wall hid the object passing through it.)"""
    human = np.abs(np.rad2deg(RAY_REL)) <= 90
    for kind in (voe.INERT[0], voe.INERT[-1]):
        for j in (-0.5, 0.5):
            (p1, p2, i1, i2), tc = voe.solidity(kind, j)
            differ = ((p1["vision"][voe.PRE:tc] != i2["vision"][voe.PRE:tc]).any(-1) & human).sum(-1)
            assert differ.min() >= 2, (kind, j)  # same motion, wall in the path or slid away
            d = {f"obs_{k}": np.stack([m[k] for m in (p1, p2, i1, i2)]) for k in p1}
            d["action"] = np.zeros((4, len(p1["oracle"]), 2), np.float32)
            for name in ("human", "bat", "chimera", "dog"):
                s, _ = voe.surprise(voe.copy_errors(observe(d, _noise_free(name))), np.array([tc]), n_boot=1)
                assert abs(s) < 0.01, (kind, j, name, s)
