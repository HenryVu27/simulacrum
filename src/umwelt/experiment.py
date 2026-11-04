"""Train every mind in one world, then run the probe battery, the surprise tests and translation."""

import json
import os
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from umwelt import learner, presets, probes, translate, voe
from umwelt.data import load

RESULTS = Path(os.environ.get("UMWELT_RESULTS", Path(__file__).resolve().parents[2] / "results"))


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _voe_scores(params, suite, preset, mu, sd):
    out = {}
    for name, (d, crit) in suite.items():
        o = (presets.observe(d, preset) - mu) / sd
        a = presets.actions(d, preset)
        s, ci = voe.surprise(learner.batched(learner._errors_jit, params, o, a), crit)
        s_copy, _ = voe.surprise(voe.copy_errors(o), crit, n_boot=1)
        out[name] = {"S": s, "ci": ci, "S_copy": s_copy}
    return out


def run_world(world, seed, preset_names, cfg: learner.LearnerConfig, n_train=1024, n_test=256, n_probe=1024,
              n_keep=128, controls=True, tag="main"):
    out_dir = RESULTS / "runs" / tag / f"{world}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    d = load(world, n_train + n_test, 256, seed=0)
    tr = np.arange(n_train)
    te = np.arange(n_train, n_train + n_test)
    pr = tr[:n_probe]
    gt_keys = [k for k in d if k.startswith("gt_")]
    T_tr = probes.targets({k: d[k][pr] for k in gt_keys})
    T_te = probes.targets({k: d[k][te] for k in gt_keys})
    suite = voe.build_suite()
    keep_tr, keep_te = {}, {}

    for name in preset_names:
        path = out_dir / f"{name}.json"
        if path.exists():
            print(f"[{world} s{seed}] {name}: done already, loading latents", flush=True)
            with open(out_dir / f"{name}_latents.pkl", "rb") as f:
                keep_tr[name], keep_te[name] = pickle.load(f)
            continue
        p = presets.PRESETS[name]
        t0 = time.time()
        o = presets.observe(d, p)
        a = presets.actions(d, p)
        mu, sd = learner.standardizer(o[tr])
        o -= mu
        o /= sd
        print(f"[{world} s{seed}] {name}: obs dim {o.shape[-1]}, training", flush=True)
        timing, last = {}, [time.time()]

        def tick(label):
            now = time.time()
            timing[label] = round(now - last[0], 1)
            last[0] = now

        params, log = learner.train(o[tr], a[tr], cfg, seed=seed, recurrent=p.recurrent, verbose=False)
        tick("train")
        H_pr = learner.batched(learner._latents_jit, params, o[pr], a[pr])
        H_te = learner.batched(learner._latents_jit, params, o[te], a[te])
        test_err = float(learner.batched(learner._errors_jit, params, o[te], a[te])[:, 16:].mean())
        tick("latents")
        rec = {"world": world, "seed": seed, "preset": name, "obs_dim": int(o.shape[-1]),
               "config": asdict(cfg), "train_log": log, "test_error": test_err}
        rec["probes"] = probes.run_battery(H_pr, T_tr, H_te, T_te, seed)
        tick("probes_mlp")
        rec["probes_linear"] = probes.run_battery(H_pr, T_tr, H_te, T_te, seed, hidden=0)
        tick("probes_linear")
        rec["voe"] = _voe_scores(params, suite, p, mu, sd)
        tick("voe")
        if controls:
            p0 = learner.init(jax.random.PRNGKey(10_000 + seed), o.shape[-1], cfg, p.recurrent)
            H0_pr = learner.batched(learner._latents_jit, p0, o[pr], a[pr])
            H0_te = learner.batched(learner._latents_jit, p0, o[te], a[te])
            rec["probes_untrained"] = probes.run_battery(H0_pr, T_tr, H0_te, T_te, seed)
            rec["voe_untrained"] = _voe_scores(p0, suite, p, mu, sd)
            tick("untrained")
            rec["probes_raw"] = probes.run_battery(o[pr], T_tr, o[te], T_te, seed)
            tick("raw")
        rec["timing"] = timing
        rec["seconds"] = time.time() - t0
        with open(out_dir / f"{name}_params.pkl", "wb") as f:
            pickle.dump({"params": jax.device_get(params), "mu": mu, "sd": sd}, f)
        keep_tr[name] = H_pr[:n_keep].astype(np.float16)
        keep_te[name] = H_te.astype(np.float16)
        with open(out_dir / f"{name}_latents.pkl", "wb") as f:
            pickle.dump((keep_tr[name], keep_te[name]), f)
        with open(path, "w") as f:
            json.dump(_jsonable(rec), f, indent=1)
        pb = rec["probes"]
        print(f"[{world} s{seed}] {name}: {rec['seconds']:.0f}s {timing} test_err {test_err:.3f} | "
              + " ".join(f"{k}={pb[k]:.2f}" for k in probes.CONCEPTS)
              + " | voe " + " ".join(f"{k}={v['S']:.2f}" for k, v in rec["voe"].items()), flush=True)

    names, cka, trans = translate.matrices(
        {k: v.astype(np.float32) for k, v in keep_tr.items()}, {k: v.astype(np.float32) for k, v in keep_te.items()}
    )
    with open(out_dir / "translation.json", "w") as f:
        json.dump({"names": names, "cka": cka.tolist(), "translation_r2": trans.tolist()}, f)
    print(f"[{world} s{seed}] all minds done in {(time.time() - t_start) / 60:.1f} min", flush=True)
