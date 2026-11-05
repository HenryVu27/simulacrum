"""Train every mind in one world, then run the probe battery, the surprise tests and translation."""

import json
import os
import pickle
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from umwelt import learner, learner_jepa, presets, probes, translate, voe
from umwelt.data import load, recording_hash

RESULTS = Path(os.environ.get("UMWELT_RESULTS", Path(__file__).resolve().parents[2] / "results"))


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _voe_scores(params, suite, preset, mu, sd, L=learner):
    out = {}
    for name, (d, crit) in suite.items():
        o = (presets.observe(d, preset) - mu) / sd
        a = presets.actions(d, preset)
        s, ci = voe.surprise(learner.batched(L._errors_jit, params, o, a), crit)
        s_copy, _ = voe.surprise(voe.copy_errors(o), crit, n_boot=1)
        out[name] = {"S": s, "ci": ci, "S_copy": s_copy}
    return out


# Settings that change the numbers: every mind under one tag must share them (docs/gpu.md, E7 and E7b).
SETTINGS = ("backend", "matmul_precision", "xla_flags", "recording_hash")


def _code_version():
    if os.environ.get("UMWELT_CODE"):  # set by the GPU job runner from the pushed manifest
        return os.environ["UMWELT_CODE"]
    try:
        repo = Path(__file__).resolve().parents[2]
        head = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--", "src"], capture_output=True, text=True, check=False).stdout
        return head + (" with uncommitted changes in src" if dirty.strip() else "")
    except OSError:
        return ""


def provenance(world, n_episodes, data_seed, T=256):
    """Where and how a result was computed, so a result file says whether it can be pooled with another."""
    return {"machine": platform.node(), "platform": platform.platform(), "backend": jax.default_backend(),
            "device": jax.devices()[0].device_kind, "jax": jax.__version__,
            "matmul_precision": str(jax.config.jax_default_matmul_precision), "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "code": _code_version(), "recording": f"{world}_s{data_seed}_e{n_episodes}_t{T}",
            "recording_hash": recording_hash(world, n_episodes, T, data_seed)}


def check_same_settings(path, prov):
    """A mind already done under this tag is reused only if it was made with the same settings."""
    old = json.loads(Path(path).read_text()).get("provenance")
    if old is None:
        print(f"warning: {path} predates provenance records; reusing it unchecked", flush=True)
        return
    changed = {k: (old.get(k), prov[k]) for k in SETTINGS if old.get(k) != prov[k]}
    if changed:
        raise RuntimeError(f"{path} was made with other settings {changed}; use a new tag, or delete that run folder")


def run_world(world, seed, preset_names, cfg: learner.LearnerConfig, n_train=1024, n_test=256, n_probe=1024,
              n_keep=128, controls=True, tag="main", data_seed=0):
    if (RESULTS / "skip" / f"{world}_s{seed}").exists():
        print(f"[{world} s{seed}] skipped: results/skip/{world}_s{seed} exists", flush=True)
        return
    out_dir = RESULTS / "runs" / tag / f"{world}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    d = load(world, n_train + n_test, 256, seed=data_seed)
    prov = provenance(world, n_train + n_test, data_seed)
    tr = slice(0, n_train)  # slices, not index arrays, so memory-mapped data is viewed rather than copied
    te = slice(n_train, n_train + n_test)
    pr = slice(0, n_probe)
    gt_keys = [k for k in d if k.startswith("gt_")]
    T_tr = probes.targets({k: np.asarray(d[k][pr]) for k in gt_keys})
    T_te = probes.targets({k: np.asarray(d[k][te]) for k in gt_keys})
    suite = voe.build_suite()
    keep_tr, keep_te = {}, {}

    for name in preset_names:
        path = out_dir / f"{name}.json"
        if path.exists():
            check_same_settings(path, prov)
            print(f"[{world} s{seed}] {name}: done already, loading latents", flush=True)
            with open(out_dir / f"{name}_latents.pkl", "rb") as f:
                keep_tr[name], keep_te[name] = pickle.load(f)
            continue
        p = presets.PRESETS[name]
        t0 = time.time()
        (RESULTS / "cache").mkdir(exist_ok=True)
        cache = RESULTS / "cache" / f"{tag}_{world}_s{seed}_d{data_seed}_{name}.npy"
        o = presets.observe(d, p, dtype=np.float16, path=cache)  # memory-mapped: large recordings fit
        a = presets.actions(d, p)
        mu, sd = learner.standardizer(o[tr])
        for i in range(0, len(o), 128):
            o[i : i + 128] = (o[i : i + 128].astype(np.float32) - mu) / sd
        print(f"[{world} s{seed}] {name}: obs dim {o.shape[-1]}, training", flush=True)
        timing, last = {}, [time.time()]

        def tick(label):
            now = time.time()
            timing[label] = round(now - last[0], 1)
            last[0] = now

        L = learner_jepa if cfg.objective == "jepa" else learner
        params, log = L.train(o[tr], a[tr], cfg, seed=seed, recurrent=p.recurrent, verbose=False)
        tick("train")
        H_pr = learner.batched(L._latents_jit, params, o[pr], a[pr])
        H_te = learner.batched(L._latents_jit, params, o[te], a[te])
        step_err = learner.batched(L._errors_jit, params, o[te], a[te])  # (episodes, T-1), for event-level checks
        np.save(out_dir / f"{name}_test_errors.npy", step_err.astype(np.float16))
        test_err = float(step_err[:, 16:].mean())
        tick("latents")
        rec = {"world": world, "seed": seed, "data_seed": data_seed, "n_train": n_train, "preset": name, "obs_dim": int(o.shape[-1]),
               "config": asdict(cfg), "provenance": prov, "train_log": log, "test_error": test_err}
        rec["probes"] = probes.run_battery(H_pr, T_tr, H_te, T_te, seed)
        tick("probes_mlp")
        rec["probes_linear"] = probes.run_battery(H_pr, T_tr, H_te, T_te, seed, hidden=0)
        tick("probes_linear")
        rec["voe"] = _voe_scores(params, suite, p, mu, sd, L)
        tick("voe")
        if controls:
            p0 = L.init(jax.random.PRNGKey(10_000 + seed), o.shape[-1], cfg, p.recurrent)
            H0_pr = learner.batched(L._latents_jit, p0, o[pr], a[pr])
            H0_te = learner.batched(L._latents_jit, p0, o[te], a[te])
            rec["probes_untrained"] = probes.run_battery(H0_pr, T_tr, H0_te, T_te, seed)
            rec["voe_untrained"] = _voe_scores(p0, suite, p, mu, sd, L)
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
        del o
        cache.unlink()
        pb = rec["probes"]
        print(f"[{world} s{seed}] {name}: {rec['seconds']:.0f}s {timing} test_err {test_err:.3f} | "
              + " ".join(f"{k}={pb[k]:.2f}" for k in probes.CONCEPTS)
              + " | voe " + " ".join(f"{k}={v['S']:.2f}" for k, v in rec["voe"].items()), flush=True)

    names, cka, trans = translate.matrices(
        {k: v.astype(np.float32) for k, v in keep_tr.items()}, {k: v.astype(np.float32) for k, v in keep_te.items()}
    )
    with open(out_dir / "translation.json", "w") as f:
        json.dump({"names": names, "cka": cka.tolist(), "translation_r2": trans.tolist(), "provenance": prov}, f)
    print(f"[{world} s{seed}] all minds done in {(time.time() - t_start) / 60:.1f} min", flush=True)
