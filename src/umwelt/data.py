"""Roll out the shared world once and record every sense, so every mind sees the same history."""

import os
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from umwelt.senses import sense_all
from umwelt.world import WORLDS, WorldRules, cause_labels, policy, reset, step

DATA_DIR = Path(os.environ.get("UMWELT_RESULTS", Path(__file__).resolve().parents[2] / "results")) / "data"
OBS_KEYS = ("vision", "ray_tex", "hearing", "smell", "touch", "proprio", "compass", "oracle")


def _gt(s):
    return dict(
        agent_pos=s.agent_pos,
        agent_th=s.agent_th,
        agent_vel=s.agent_vel,
        obj_pos=s.obj_pos,
        obj_vel=s.obj_vel,
        obj_active=s.obj_active,
        obj_kind=s.obj_kind.astype(jnp.int8),
        cause=cause_labels(s),
    )


def rollout(key, rules: WorldRules, T: int):
    k0, key = jax.random.split(key)
    s = reset(k0)

    def body(carry, _):
        s, key = carry
        key, ka, kp = jax.random.split(key, 3)
        action, pol = policy(s, ka)
        s = s._replace(**pol)
        s, c = step(s, action, rules, kp)
        obs = sense_all(s, c)
        vis = obs.pop("vis180")
        return (s, key), (action, obs, dict(_gt(s), vis180=vis))

    _, (actions, obs, gt) = jax.lax.scan(body, (s, key), None, length=T)
    return actions, obs, gt


@partial(jax.jit, static_argnums=(1, 2))
def rollout_batch(keys, rules: WorldRules, T: int):
    return jax.vmap(rollout, in_axes=(0, None, None))(keys, rules, T)


def generate(world: str, n_episodes: int, T: int = 256, seed: int = 0, chunk: int = 128):
    rules = WORLDS[world]
    keys = jax.random.split(jax.random.PRNGKey(seed), n_episodes)
    parts = []
    for i in range(0, n_episodes, chunk):
        a, o, g = rollout_batch(keys[i : i + chunk], rules, T)
        rec = {"action": np.asarray(a, np.float32)}
        rec.update({f"obs_{k}": np.asarray(o[k], np.float16) for k in OBS_KEYS})
        rec.update({f"gt_{k}": np.asarray(v) for k, v in g.items()})
        parts.append(rec)
    return {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}


def save(world: str, n_episodes: int, T: int = 256, seed: int = 0):
    """World names ending in _mj are simulated by the MuJoCo audit engine instead of Flatland."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if world.endswith("_mj"):
        from umwelt import world_mj

        d = world_mj.generate(world[:-3], n_episodes, T, seed)
    else:
        d = generate(world, n_episodes, T, seed)
    path = DATA_DIR / f"{world}_s{seed}_e{n_episodes}_t{T}.npz"
    np.savez(path, **d)
    print(f"{world}: {n_episodes} episodes x {T} steps in {time.time() - t0:.0f}s -> {path.name}")
    return path


def load(world: str, n_episodes: int, T: int = 256, seed: int = 0):
    path = DATA_DIR / f"{world}_s{seed}_e{n_episodes}_t{T}.npz"
    if not path.exists():
        save(world, n_episodes, T, seed)
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
