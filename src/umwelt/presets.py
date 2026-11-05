"""Minds as settings on dials: which senses, how sharp, whether actions are known, whether memory exists.

Each preset turns the recorded superset of senses into one mind's observation stream.
"""

import zlib
from dataclasses import dataclass

import numpy as np

from umwelt.senses import N_RAYS, RAY_REL


@dataclass(frozen=True)
class Sense:
    kind: str  # vision, echo, hearing, smell, touch, proprio, compass, oracle
    fov_deg: float = 180.0
    n_rays: int = 64
    color: str = "rgb"  # rgb, dichromat
    gain: float = 1.0
    noise: float = 0.02


@dataclass(frozen=True)
class Preset:
    name: str
    senses: tuple
    uses_action: bool = True  # efference copy: does the mind know what it just did?
    recurrent: bool = True  # memory
    note: str = ""


HUMAN_EYES = Sense("vision", fov_deg=180, n_rays=64, color="rgb")
DOG_EYES = Sense("vision", fov_deg=240, n_rays=16, color="dichromat")
WEAK_NOSE = Sense("smell", gain=2.0, noise=0.05)
STRONG_NOSE = Sense("smell", gain=50.0, noise=0.01)
EARS = Sense("hearing")
SHARP_EARS = Sense("hearing", gain=1.5)
SKIN = Sense("touch")
BODY = Sense("proprio")
ECHO = Sense("echo", n_rays=32)
COMPASS = Sense("compass", noise=0.0)
HUMAN = (HUMAN_EYES, EARS, SKIN, BODY, WEAK_NOSE)

PRESETS = {
    p.name: p
    for p in [
        Preset("oracle", (Sense("oracle", noise=0.0),), note="sees the true world state; ceiling control"),
        Preset("human", HUMAN, note="color vision 180 deg, hearing, touch, body sense, weak smell"),
        Preset("dog", (DOG_EYES, STRONG_NOSE, SHARP_EARS, SKIN, BODY), note="blurry two-color vision 240 deg, strong smell"),
        Preset("blind", (EARS, SKIN, BODY, WEAK_NOSE), note="human without vision"),
        Preset("deafblind", (SKIN, BODY, WEAK_NOSE), note="touch, body sense and weak smell only"),
        Preset("vision_only", (HUMAN_EYES,), note="human eyes, nothing else"),
        Preset("smell_only", (STRONG_NOSE,), note="Condillac's statue: smell and nothing else"),
        Preset("hearing_only", (EARS,), note="Strawson's sound-only being"),
        Preset("touch_only", (SKIN, BODY), note="touch and body sense"),
        Preset("bat", (ECHO, SHARP_EARS, SKIN, BODY), note="echolocation instead of eyes"),
        Preset("chimera", (STRONG_NOSE, ECHO, COMPASS), note="smell, echolocation and a magnetic compass"),
        Preset("passive", HUMAN, uses_action=False, note="yoked kitten: human senses, never knows its own actions"),
        Preset("passive_novest", HUMAN[:3] + (WEAK_NOSE,), uses_action=False, note="yoked, and no body sense either"),
        Preset("memoryless", HUMAN, recurrent=False, note="human senses, no memory beyond the present moment"),
    ]
}


def _pool(x, n):
    """Average neighbouring rays into n wider receptive fields."""
    groups = np.array_split(np.arange(x.shape[-2]), n)
    return np.stack([x[..., g, :].mean(-2) for g in groups], -2)


def _channel(d, s: Sense):
    if s.kind == "vision":
        v = d["obs_vision"].astype(np.float32)
        if "gt_light" in d:  # in the dark, eyes see black and no depth; echolocation is unaffected
            v = np.where(d["gt_light"][..., None, None], v, np.array([0.0, 0.0, 0.0, 1.0], np.float32))
        keep = np.abs(np.rad2deg(RAY_REL)) <= s.fov_deg / 2
        v = _pool(v[..., keep, :], s.n_rays)
        if s.color == "dichromat":
            v = np.stack([v[..., 2], (v[..., 0] + v[..., 1]) / 2, v[..., 3]], -1)
        return v.reshape(*v.shape[:-2], -1)
    if s.kind == "echo":
        step = N_RAYS // s.n_rays
        depth = d["obs_vision"][..., ::step, 3].astype(np.float32)
        tex = d["obs_ray_tex"][..., ::step].astype(np.float32)
        return np.concatenate([depth, tex], -1)
    if s.kind == "smell":
        return np.log1p(s.gain * np.maximum(d["obs_smell"].astype(np.float32), 0.0))
    if s.kind == "hearing":
        return s.gain * d["obs_hearing"].astype(np.float32)
    return d[f"obs_{s.kind}"].astype(np.float32)


def observe(d, preset: Preset, dtype=np.float32, chunk=128, path=None, draw=0):
    """The observation stream (episodes, time, dim) this mind receives from recorded data d.

    Built in chunks of episodes so a large recording never has to sit in memory as float32. With path,
    the stream is written to a memory-mapped .npy file there instead of held in memory. draw picks the sensor
    noise: draw 0 is the one every result so far used, seeded by the preset's name; others are fresh draws.
    """
    seed = zlib.crc32(preset.name.encode())
    rng = np.random.default_rng(seed if draw == 0 else [seed, draw])
    E = len(d["action"])
    out = None
    for i in range(0, E, chunk):
        part = {k: v[i : i + chunk] for k, v in d.items() if k.startswith("obs_") or k in ("action", "gt_light")}
        xs = []
        for s in preset.senses:
            x = _channel(part, s)
            if s.noise > 0:
                x = x + rng.normal(0.0, s.noise, x.shape).astype(np.float32)
            xs.append(x)
        x = np.concatenate(xs, -1)
        if out is None:
            shape = (E,) + x.shape[1:]
            out = np.empty(shape, dtype) if path is None else np.lib.format.open_memmap(path, "w+", dtype, shape)
        out[i : i + chunk] = x
    return out


def actions(d, preset: Preset):
    a = d["action"].astype(np.float32)
    return a if preset.uses_action else np.zeros_like(a)
