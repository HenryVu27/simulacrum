"""Independent recompute of dog and oracle surprise scores: own observation builder, own float64 numpy forward
pass of the stored GRU networks, own surprise statistic. Compared with the stored new values and the kept old ones."""
import json, pickle, sys, zlib
import numpy as np
import jax
from umwelt import learner, presets
from umwelt.senses import RAY_REL

suites = pickle.load(open(sys.argv[1], "rb"))
ROOT = "/Users/henryvu/personal/umwelt/results/runs/"
RUNS = sys.argv[2].split(",")
MINDS = sys.argv[3].split(",")

def my_observe(d, mind):
    if mind == "oracle":
        return d["obs_oracle"].astype(np.float32)
    rng = np.random.default_rng(zlib.crc32(mind.encode()))
    assert mind == "dog" and len(d["action"]) <= 128
    v = d["obs_vision"].astype(np.float32)
    keep = np.abs(np.rad2deg(RAY_REL)) <= 120
    v = v[..., keep, :]
    groups = np.array_split(np.arange(v.shape[-2]), 16)
    v = np.stack([v[..., g, :].mean(-2) for g in groups], -2)
    v = np.stack([v[..., 2], (v[..., 0] + v[..., 1]) / 2, v[..., 3]], -1).reshape(*v.shape[:-2], -1)
    parts = [(v, 0.02), (np.log1p(50.0 * np.maximum(d["obs_smell"].astype(np.float32), 0.0)), 0.01),
             (1.5 * d["obs_hearing"].astype(np.float32), 0.02), (d["obs_touch"].astype(np.float32), 0.02),
             (d["obs_proprio"].astype(np.float32), 0.02)]
    return np.concatenate([x + rng.normal(0.0, s, x.shape).astype(np.float32) for x, s in parts], -1)

def gelu(x):  # jax.nn.gelu default: tanh approximation
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def sig(x):
    return 1 / (1 + np.exp(-x))

def my_errors(p, obs, act):
    P = jax.tree_util.tree_map(lambda a: np.asarray(a, np.float64), p)
    lin = lambda q, x: x @ q["w"] + q["b"]
    e = gelu(lin(P["enc2"], gelu(lin(P["enc1"], np.concatenate([obs, act], -1)))))
    if "core" in P:
        h = np.tanh(lin(P["core"], e))
    else:
        H = P["gru_h"]["w"].shape[0]
        xs = lin(P["gru_x"], e)
        hh_, out = np.zeros((obs.shape[0], H)), []
        for t in range(obs.shape[1]):
            x, hh = xs[:, t], lin(P["gru_h"], hh_)
            z, r = sig(x[:, :H] + hh[:, :H]), sig(x[:, H:2 * H] + hh[:, H:2 * H])
            n = np.tanh(x[:, 2 * H:] + r * hh[:, 2 * H:])
            hh_ = (1 - z) * n + z * hh_
            out.append(hh_)
        h = np.stack(out, 1)
    a_next = np.concatenate([act[:, 1:], np.zeros_like(act[:, :1])], 1)
    near = lin(P["dec2"], gelu(lin(P["dec1"], np.concatenate([h, a_next], -1))))
    return np.mean((near[:, :-1] - obs[:, 1:]) ** 2, -1)

def my_S(err, crit, n_boot=500):
    n = len(crit)
    err = err.reshape(n, 4, -1)
    tot = np.stack([err[i, :, crit[i] - 1:].sum(-1) for i in range(n)])
    P, I = tot[:, 0] + tot[:, 1], tot[:, 2] + tot[:, 3]
    s = (I.sum() - P.sum()) / P.sum()
    rng = np.random.default_rng(0)
    b = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        b.append((I[i].sum() - P[i].sum()) / P[i].sum())
    return s, np.percentile(b, [2.5, 97.5])

def scores(params, suite, mind, mu, sd):
    out = {}
    for sc, (d, crit) in suite.items():
        raw = my_observe(d, mind)
        assert np.array_equal(raw, presets.observe(d, presets.PRESETS[mind])), "own observe != presets.observe"
        o = (raw - mu) / sd
        a = np.zeros(o.shape[:2] + (2,))
        s, ci = my_S(my_errors(params, o.astype(np.float64), a), crit)
        sc_copy, _ = my_S(np.mean((o[:, 1:] - o[:, :-1]) ** 2, -1).astype(np.float64), crit, n_boot=1)
        out[sc] = (s, ci, sc_copy)
    return out

for run in RUNS:
    for mind in MINDS:
        rec = json.load(open(ROOT + run + f"/{mind}.json"))
        saved = pickle.load(open(ROOT + run + f"/{mind}_params.pkl", "rb"))
        cfg = learner.LearnerConfig()  # the preregistered defaults; compare with the stored config
        stored_cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in rec["config"].items()}
        assert all(getattr(cfg, k) == v for k, v in stored_cfg.items()), (stored_cfg, cfg)
        nets = {"voe": saved["params"]}
        if "voe_untrained" in rec:
            nets["voe_untrained"] = learner.init(jax.random.PRNGKey(10_000 + rec["seed"]), rec["obs_dim"], cfg, presets.PRESETS[mind].recurrent)
        for field, net in nets.items():
            for label, suite, stored in (("old", suites["old"], rec[f"{field}_before_wall_fix"]), ("new", suites["new"], rec[field])):
                mine = scores(net, suite, mind, saved["mu"], saved["sd"])
                for sc, (s, ci, scp) in mine.items():
                    st = stored[sc]
                    print(f"{run:19} {mind:6} {field:13} {label} {sc:10} mine S {s:+.6f} ci [{ci[0]:+.4f},{ci[1]:+.4f}] copy {scp:.4f} | stored S {st['S']:+.6f} "
                          f"ci [{st['ci'][0]:+.4f},{st['ci'][1]:+.4f}] copy {st['S_copy']:.4f} | dS {s - st['S']:+.1e}", flush=True)
