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
        light=s.dark == 0,  # eyes see nothing in the dark; presets.py applies this at the eye
    )


def rollout(key, rules: WorldRules, T: int):
    k0, key = jax.random.split(key)
    s = reset(k0, wall_half=rules.wall_half)

    def body(carry, _):
        s, key = carry
        key, ka, kp = jax.random.split(key, 3)
        action, pol = policy(s, ka, rules)
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


def chunks(world: str, n_episodes: int, T: int = 256, seed: int = 0, chunk: int = 128):
    """Yield (first episode, record) for consecutive chunks of episodes."""
    rules = WORLDS[world]
    keys = jax.random.split(jax.random.PRNGKey(seed), n_episodes)
    for i in range(0, n_episodes, chunk):
        a, o, g = rollout_batch(keys[i : i + chunk], rules, T)
        rec = {"action": np.asarray(a, np.float32)}
        rec.update({f"obs_{k}": np.asarray(o[k], np.float16) for k in OBS_KEYS})
        rec.update({f"gt_{k}": np.asarray(v) for k, v in g.items()})
        yield i, rec


def generate(world: str, n_episodes: int, T: int = 256, seed: int = 0, chunk: int = 128):
    parts = [rec for _, rec in chunks(world, n_episodes, T, seed, chunk)]
    return {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}


def _dir(world, n_episodes, T, seed):
    return DATA_DIR / f"{world}_s{seed}_e{n_episodes}_t{T}"


def save(world: str, n_episodes: int, T: int = 256, seed: int = 0):
    """World names ending in _mj are simulated by the MuJoCo audit engine instead of Flatland.

    Each array is stored as its own .npy file so runs can memory-map it instead of holding it in RAM.
    """
    t0 = time.time()
    out = _dir(world, n_episodes, T, seed)
    tmp = out.with_name(out.name + ".partial")  # renamed when complete, so a crash never leaves a half recording
    tmp.mkdir(parents=True, exist_ok=True)
    if world.endswith("_mj"):
        from umwelt import world_mj

        for k, v in world_mj.generate(world[:-3], n_episodes, T, seed).items():
            np.save(tmp / f"{k}.npy", v)
    else:  # stream chunks straight to disk, so a large recording never sits in memory
        files = {}
        for i, rec in chunks(world, n_episodes, T, seed):
            for k, v in rec.items():
                if k not in files:
                    files[k] = np.lib.format.open_memmap(tmp / f"{k}.npy", "w+", v.dtype, (n_episodes,) + v.shape[1:])
                files[k][i : i + len(v)] = v
        for f in files.values():
            f.flush()
        del files
    tmp.rename(out)
    print(f"{world}: {n_episodes} episodes x {T} steps in {time.time() - t0:.0f}s -> {out.name}")
    return out


def load(world: str, n_episodes: int, T: int = 256, seed: int = 0):
    out = _dir(world, n_episodes, T, seed)
    legacy = out.with_suffix(".npz")
    if not out.exists() and legacy.exists():  # convert an older single-file recording once
        out.mkdir(parents=True)
        with np.load(legacy) as f:
            for k in f.files:
                np.save(out / f"{k}.npy", f[k])
        legacy.unlink()
    if not out.exists():
        save(world, n_episodes, T, seed)
    return {f.stem: np.load(f, mmap_mode="r") for f in sorted(out.glob("*.npy"))}
