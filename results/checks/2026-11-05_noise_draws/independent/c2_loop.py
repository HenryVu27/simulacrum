"""My own loop: per stored mind, per net, per test, per noise draw k in [k0, k1), the summed post-critical errors
(float32 the way voe.totals sums, and float64 by a mask) for the network and for the copy predictor.
Noise is added by my own code (clean channel + default_rng(seed if k == 0 else [seed, k]).normal per sense, in
sense order) and also checked bitwise against presets.observe(d, p, draw=k) for k < 3.
Usage: c2_loop.py SNAPSHOT OUTDIR k0 k1 run[/mind] ..."""
import sys, json, pickle, zlib, time
from dataclasses import replace
from pathlib import Path
import numpy as np, jax
from umwelt import learner, learner_jepa, presets, voe

snap, outdir, k0, k1 = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
targets = sys.argv[5:]
outdir.mkdir(parents=True, exist_ok=True)
suite = voe.build_suite()
assert voe.suite_hash(suite) == "016fc3d72807e39d"


def my_observe(d, p, k):
    seed = zlib.crc32(p.name.encode())
    rng = np.random.default_rng(seed if k == 0 else [seed, k])
    xs = []
    for s in p.senses:
        x = presets._channel(d, s)
        if s.noise > 0:
            x = x + rng.normal(0.0, s.noise, x.shape).astype(np.float32)
        xs.append(x)
    return np.concatenate(xs, -1).astype(np.float32)


def totals32(err, crit):  # the summation voe.totals does, per instance slice, in err's dtype
    n = len(crit)
    e = err.reshape(n, 4, -1)
    return np.array([e[i, :, crit[i] - 1:].sum(-1) for i in range(n)])


def totals64(err, crit):  # independent: float64 mask over frames t >= crit-1
    n = len(crit)
    e = err.reshape(n, 4, -1).astype(np.float64)
    t = np.arange(e.shape[-1])
    m = t[None, None, :] >= (np.asarray(crit) - 1)[:, None, None]
    return (e * m).sum(-1)


for tg in targets:
    run, _, only = tg.partition(":")
    folder = snap / "results" / "runs" / run
    names = [only] if only else sorted(f.stem for f in folder.glob("*.json") if f.stem != "translation")
    for name in names:
        t0 = time.time()
        rec = json.loads((folder / f"{name}.json").read_text())
        saved = pickle.load(open(folder / f"{name}_params.pkl", "rb"))
        p = presets.PRESETS[rec["preset"]]
        cfg = learner.LearnerConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in rec["config"].items()})
        L = learner_jepa if cfg.objective == "jepa" else learner
        nets = {"trained": saved["params"]}
        if "voe_untrained" in rec:
            nets["untrained"] = L.init(jax.random.PRNGKey(10_000 + rec["seed"]), rec["obs_dim"], cfg, p.recurrent)
        noisy = any(s.noise > 0 for s in p.senses)
        ks = list(range(k0, k1)) if noisy else [0]
        out = {}
        for test, (d, crit) in suite.items():
            a = presets.actions(d, p)
            T32 = {net: [] for net in nets}; T64 = {net: [] for net in nets}; C32, C64 = [], []
            for k in ks:
                raw = my_observe(d, p, k)
                if k < 3:
                    assert np.array_equal(raw, presets.observe(d, p, draw=k)), (run, name, test, k)
                o = (raw - saved["mu"]) / saved["sd"]
                for net, params in nets.items():
                    err = learner.batched(L._errors_jit, params, o, a)
                    T32[net].append(totals32(err, crit)); T64[net].append(totals64(err, crit))
                ce = np.mean((o[:, 1:] - o[:, :-1]) ** 2, -1)
                C32.append(totals32(ce, crit)); C64.append(totals64(ce, crit))
            for net in nets:
                out[f"{net}/{test}/t32"] = np.stack(T32[net]); out[f"{net}/{test}/t64"] = np.stack(T64[net])
            out[f"copy/{test}/t32"] = np.stack(C32); out[f"copy/{test}/t64"] = np.stack(C64)
            out[f"{test}/crit"] = crit
        out["draws"] = np.array(ks)
        fn = outdir / f"{run.replace('/', '__')}__{name}__{k0}_{k1}.npz"
        np.savez(fn, **out)
        print(f"{run} {name} draws {len(ks)} nets {list(nets)} {time.time() - t0:.1f}s", flush=True)
