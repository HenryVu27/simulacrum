"""Claim 5 (independent part): my own float64 NumPy forward pass of the stored networks (GRU, memoryless, JEPA) on
the committed new movies, my own noise draw and standardisation, my own S and bootstrap; compared with the JSON."""
import json
import pickle
import zlib
from pathlib import Path

import jax
import numpy as np
from common import S, voe
from umwelt import learner, learner_jepa, presets

LIVE = Path("/Users/henryvu/personal/umwelt/results/runs")
COMMITTED = S / "repo_c35" / "results" / "runs"
SAMPLE = [("main/lawful_s0", "oracle"), ("main/lawful_s0", "human"), ("main/lawful_s0", "dog"), ("main/lawful_s0", "memoryless"),
          ("main/lawful_s0", "passive"), ("main/ghost_s0", "bat"), ("main/lawful_s1", "chimera"), ("v2/dark_s0", "human"),
          ("main/lawful_mj_s0", "vision_only"), ("jepa/lawful_s0", "human"), ("long/lawful_s0", "human"), ("main/impermanent_s0", "smell_only")]


def gelu(x):  # jax.nn.gelu default (approximate=True)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def lin(p, x):
    return x @ np.asarray(p["w"], np.float64) + np.asarray(p["b"], np.float64)


def core(p, e):  # e (B, T, E)
    if "core" in p:
        return np.tanh(lin(p["core"], e))
    H = np.asarray(p["gru_h"]["w"]).shape[0]
    xs = lin(p["gru_x"], e)
    h = np.zeros((e.shape[0], H))
    hs = []
    for t in range(e.shape[1]):
        x, hh = xs[:, t], lin(p["gru_h"], h)
        z = 1 / (1 + np.exp(-(x[:, :H] + hh[:, :H])))
        r = 1 / (1 + np.exp(-(x[:, H:2 * H] + hh[:, H:2 * H])))
        n = np.tanh(x[:, 2 * H:] + r * hh[:, 2 * H:])
        h = (1 - z) * n + z * h
        hs.append(h)
    return np.stack(hs, 1)


def errors_pixel(p, o, a):
    e = gelu(lin(p["enc2"], gelu(lin(p["enc1"], np.concatenate([o, a], -1)))))
    h = core(p, e)
    a_next = np.concatenate([a[:, 1:], np.zeros_like(a[:, :1])], 1)
    near = lin(p["dec2"], gelu(lin(p["dec1"], np.concatenate([h, a_next], -1))))
    return np.mean((near[:, :-1] - o[:, 1:]) ** 2, -1)


def errors_jepa(p, o, a):
    z = lin(p["jenc2"], gelu(lin(p["jenc1"], o)))
    e = gelu(lin(p["jin"], np.concatenate([z, a], -1)))
    h = core(p, e)
    a1 = np.concatenate([a[:, 1:], np.zeros_like(a[:, :1])], 1)  # next action, k = 1
    pred = lin(p["jp1_2"], gelu(lin(p["jp1_1"], np.concatenate([h, a1], -1))))
    t = lin(p["tenc2"], gelu(lin(p["tenc1"], o)))
    t = (t - t.mean(-1, keepdims=True)) / (t.std(-1, keepdims=True) + 1e-5)
    return np.mean((pred[:, :-1] - t[:, 1:]) ** 2, -1)


def my_observe(d, preset):
    """presets.observe re-implemented: channels, then N(0, noise) from default_rng(crc32(name)) in sense order."""
    rng = np.random.default_rng(zlib.crc32(preset.name.encode()))
    xs = []
    for s in preset.senses:
        if s.kind == "vision":
            v = d["obs_vision"].astype(np.float32)
            v = v[..., np.abs(np.rad2deg(presets.RAY_REL)) <= s.fov_deg / 2, :]
            groups = np.array_split(np.arange(v.shape[-2]), s.n_rays)
            v = np.stack([v[..., g, :].mean(-2) for g in groups], -2)
            if s.color == "dichromat":
                v = np.stack([v[..., 2], (v[..., 0] + v[..., 1]) / 2, v[..., 3]], -1)
            x = v.reshape(*v.shape[:-2], -1)
        elif s.kind == "echo":
            st = 128 // s.n_rays
            x = np.concatenate([d["obs_vision"][..., ::st, 3].astype(np.float32), d["obs_ray_tex"][..., ::st].astype(np.float32)], -1)
        elif s.kind == "smell":
            x = np.log1p(s.gain * np.maximum(d["obs_smell"].astype(np.float32), 0))
        elif s.kind == "hearing":
            x = s.gain * d["obs_hearing"].astype(np.float32)
        else:
            x = d[f"obs_{s.kind}"].astype(np.float32)
        if s.noise > 0:
            x = x + rng.normal(0.0, s.noise, x.shape).astype(np.float32)
        xs.append(x)
    return np.concatenate(xs, -1)


def my_surprise(err, crit, n_boot=500):
    n = len(crit)
    err = err.reshape(n, 4, -1)
    tot = np.array([err[i, :, crit[i] - 1:].sum(-1) for i in range(n)])
    P, I = tot[:, 0] + tot[:, 1], tot[:, 2] + tot[:, 3]
    rng = np.random.default_rng(0)
    b = [(I[i].sum() - P[i].sum()) / P[i].sum() for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return (I.sum() - P.sum()) / P.sum(), np.percentile(b, [2.5, 97.5])


suite = voe.build_suite()
print("suite hash", voe.suite_hash(suite))
worst = 0.0
for run, name in SAMPLE:
    jf = COMMITTED / run / f"{name}.json"
    rec = json.loads(jf.read_text())
    saved = pickle.load(open(LIVE / run / f"{name}_params.pkl", "rb"))
    cfg = learner.LearnerConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in rec["config"].items()})
    jepa = cfg.objective == "jepa"
    p = presets.PRESETS[rec["preset"]]
    nets = [("voe", jax.tree.map(np.asarray, saved["params"]))]
    if "voe_untrained" in rec and run == "main/lawful_s0":
        L = learner_jepa if jepa else learner
        nets.append(("voe_untrained", jax.tree.map(np.asarray, L.init(jax.random.PRNGKey(10_000 + rec["seed"]), rec["obs_dim"], cfg, p.recurrent))))
    for field, params in nets:
        line = []
        for test, (d, crit) in suite.items():
            o = ((my_observe(d, p) - saved["mu"]) / saved["sd"]).astype(np.float32).astype(np.float64)
            a = d["action"].astype(np.float64) if p.uses_action else np.zeros_like(d["action"], np.float64)
            err = (errors_jepa if jepa else errors_pixel)(params, o, a)
            s, ci = my_surprise(err, crit)
            want = rec[field][test]
            worst = max(worst, abs(s - want["S"]))
            line.append(f"{test} mine {s:+.5f} stored {want['S']:+.5f} (ci mine [{ci[0]:+.3f},{ci[1]:+.3f}] stored [{want['ci'][0]:+.3f},{want['ci'][1]:+.3f}])")
        print(f"{run:20} {name:12} {field:13} " + " | ".join(line), flush=True)
print("largest |mine - stored| over the sample:", worst)
