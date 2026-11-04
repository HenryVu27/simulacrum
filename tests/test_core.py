import jax
import jax.numpy as jnp
import numpy as np

from umwelt import probes, voe
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


def test_permanence_quadruplet_is_balanced_for_a_memoryless_predictor():
    m, tc = voe.permanence(0, 0.0)
    d = {f"obs_{k}": np.stack([mv[k] for mv in m])[None].reshape(4, *m[0][k].shape) for k in m[0]}
    d["action"] = np.zeros((4, len(m[0]["oracle"]), 2), np.float32)
    o = observe(d, PRESETS["vision_only"])
    s, _ = voe.surprise(voe.copy_errors(o), np.array([tc]), n_boot=1)
    assert abs(s) < 0.1


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
