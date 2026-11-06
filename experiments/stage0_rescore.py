"""v3 Stage 0, last part (docs/preregistration.md, Amendments 5 and 6): every stored v1 and v2 mind rescored with the
probe Stage 0 fixed (ridge, 1,024 fitting episodes), without retraining. These are corrections to past results, not
new tests.

For each run and mind: the fixed probe on its hidden states, fit on the run's first 1,024 training episodes and scored
on the run's own 256 held-out episodes: object_now and object_hidden (the pooled readout), object_seen (a readout fit
on objects in view, as in Amendment 6), and the permanence curve (skill above a mind that knows nothing, by steps
since the object was last in view; split dark and lit where the world has dark spells). The same for each mind's raw
senses, once per recording. Paired bootstrap intervals (1,000 resamples of the held-out episodes) for each mind minus
its raw senses, and for the human minus the memoryless mind. Then H1 (v1, lawful world, seeds 0 and 1) and P1 (v2,
dark world, seeds 0 and 1) by their preregistered rules, on the rescored numbers.

    uv run python experiments/stage0_rescore.py      # on the GPU PC (docs/gpu.md): heavy for a laptop
"""

import importlib.util
import json
import pickle
import time
from pathlib import Path

import numpy as np

from umwelt import learner, learner_jepa, presets, probes
from umwelt.data import load
from umwelt.experiment import RESULTS

spec = importlib.util.spec_from_file_location("stage0_rerun", Path(__file__).with_name("stage0_rerun.py"))
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

OUT = RESULTS / "v3" / "stage0_rescore"
RUNS = ["main/lawful_s0", "main/lawful_s1", "main/ghost_s0", "main/impermanent_s0", "main/nonlocal_s0", "main/lawful_mj_s0",
        "v2/dark_s0", "v2/dark_s1", "long/lawful_s0", "jepa/lawful_s0", "ab_drop/lawful_s0", "ab_h5/lawful_s0",
        "ab_h15/lawful_s0", "ab_h30/lawful_s0"]  # the pilot's files are not in git and are left out
N_FIT, N_TEST, TR, TE, N_BOOT = 1024, 256, 8, 4, 1000
BINS = [("1-5 steps", (1, 5)), ("6-20 steps", (6, 20))]


def masks(Rte):
    """Named (mask, readout) pairs for every score: rows x kinds."""
    pres, vis, since = Rte["present"], Rte["vis"], Rte["since"]
    out = {"object_now": (pres, "pool"), "object_hidden": (pres & ~vis, "pool"), "object_seen": (pres & vis, "seen")}
    for name, (a, b) in BINS:
        m = pres & ~vis & (since >= a) & (since <= b)
        out[name] = (m, "pool")
        if "dark" in Rte:
            out[name + ", dark"], out[name + ", lit"] = (m & Rte["dark"], "pool"), (m & ~Rte["dark"], "pool")
    return out


def score_sums(X, Xte, Rtr, Rte, ep):
    """Per-episode sums for every score, and for the no-information baseline the permanence skill subtracts."""
    m2 = lambda m: np.repeat(m, 2, axis=1)  # noqa: E731
    groups = np.repeat(np.arange(N_FIT), len(range(probes.PAST, 256, TR)))
    ytr = Rtr["ego"].reshape(len(X), -1)
    pred = {"pool": probes.RidgeProbe(X, ytr, m2(Rtr["present"]), "reg", 0, 0, groups)(Xte),
            "seen": probes.RidgeProbe(X, ytr, m2(Rtr["present"] & Rtr["vis"]), "reg", 0, 8, groups)(Xte)}
    Y, ref = Rte["ego"].reshape(len(Xte), -1), m2(Rte["present"])
    ptr = Rtr["present"]
    mean = np.array([[Rtr["ego"][ptr[:, k], k, j].mean() if ptr[:, k].any() else 0.0 for j in range(2)] for k in range(ptr.shape[1])])
    const = np.broadcast_to(mean.reshape(1, -1), Y.shape)
    return {name: (rr.sums(pred[ro], Y, m2(m), ref, ep, N_TEST), rr.sums(const, Y, m2(m), ref, ep, N_TEST))
            for name, (m, ro) in masks(Rte).items()}


def value(name, s, w):
    """A score from per-episode sums under episode weights w: R^2 for the readouts, skill for the curve bins."""
    r = rr.r2_from(s[0], w)
    if name.startswith(("object_",)):
        return r
    b = rr.r2_from(s[1], w)
    return (r - b) / (1 - b)


def diff(a, b, W):
    """Point and 95% paired bootstrap interval of a - b for every score both have."""
    out = {}
    for name in a:
        d = [value(name, a[name], w) - value(name, b[name], w) for w in W]
        point = value(name, a[name], np.ones(N_TEST)) - value(name, b[name], np.ones(N_TEST))
        out[name] = [float(point), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    W = np.stack([np.bincount(np.random.default_rng(b).integers(0, N_TEST, N_TEST), minlength=N_TEST) for b in range(N_BOOT)])
    raw_cache, all_runs = {}, {}
    for run in RUNS:
        folder = RESULTS / "runs" / run
        files = sorted(f for f in folder.glob("*.json") if f.stem != "translation")
        rec0 = json.loads(files[0].read_text())
        world, n_train, ds = rec0["world"], rec0.get("n_train", 1024), rec0.get("data_seed", 0)
        d = load(world, n_train + N_TEST, seed=ds)
        fit, te = slice(0, N_FIT), slice(n_train, n_train + N_TEST)
        sub = {name: {k: np.asarray(v[s]) for k, v in d.items()} for name, s in (("fit", fit), ("te", te))}
        R = {}
        for name, stride in (("fit", TR), ("te", TE)):
            g = {k: v for k, v in sub[name].items() if k.startswith("gt_")}
            T = probes.targets(g)
            if "gt_light" in g:
                T["dark"] = np.repeat(~g["gt_light"][..., None], probes.K, -1)
            R[name] = {k: T[k][:, probes.PAST :: stride].reshape(-1, *T[k].shape[2:]) for k in ("ego", "present", "vis", "since", "dark") if k in T}
        ep = np.repeat(np.arange(N_TEST), len(range(probes.PAST, 256, TE)))

        def rows(H, stride):
            return H[:, probes.PAST :: stride].reshape(-1, H.shape[-1]).astype(np.float32)

        t0 = time.time()
        res = {"world": world, "data_seed": ds, "n_train": n_train, "minds": {}}
        sums = {}
        for f in files:
            rec = json.loads(f.read_text())
            name, p = rec["preset"], presets.PRESETS[rec["preset"]]
            with open(folder / f"{f.stem}_params.pkl", "rb") as fh:
                saved = pickle.load(fh)
            L = learner_jepa if rec["config"].get("objective") == "jepa" else learner
            H = {}
            for part in ("fit", "te"):
                o = (presets.observe(sub[part], p) - saved["mu"]) / saved["sd"]
                H[part] = learner.batched(L._latents_jit, saved["params"], o, presets.actions(sub[part], p))
            sums[name] = score_sums(rows(H["fit"], TR), rows(H["te"], TE), R["fit"], R["te"], ep)
            key = (world, ds, name)
            if key not in raw_cache:  # raw senses depend on the recording and the senses, not the run
                o = {part: presets.observe(sub[part], p) for part in ("fit", "te")}
                raw_cache[key] = score_sums(rows(o["fit"], TR), rows(o["te"], TE), R["fit"], R["te"], ep)
            raw = raw_cache[key]
            res["minds"][name] = {"trained": {k: value(k, s, np.ones(N_TEST)) for k, s in sums[name].items()},
                                  "raw senses": {k: value(k, s, np.ones(N_TEST)) for k, s in raw.items()},
                                  "trained minus raw": diff(sums[name], raw, W)}
        if {"human", "memoryless"} <= set(sums):
            res["human minus memoryless"] = diff(sums["human"], sums["memoryless"], W)
            res["memoryless minus human raw senses"] = diff(sums["memoryless"], raw_cache[(world, ds, "human")], W)
        all_runs[run] = res
        tag = run.replace("/", "_")
        (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1))
        h = res["minds"].get("human")
        print(f"{run}: {len(files)} minds in {time.time() - t0:.0f} s" + (
            f"; human object_hidden {h['trained']['object_hidden']:+.3f} (raw {h['raw senses']['object_hidden']:+.3f}), "
            f"1-5 steps {h['trained']['1-5 steps']:+.3f} (raw {h['raw senses']['1-5 steps']:+.3f})" if h else ""), flush=True)
    verdicts(all_runs)


def verdicts(runs):
    """H1 and P1 by their preregistered rules (docs/preregistration.md), on the rescored numbers."""
    def mean(seeds, mind, part, score):
        return float(np.mean([runs[s]["minds"][mind][part][score] for s in seeds]))

    lawful, dark = ["main/lawful_s0", "main/lawful_s1"], ["v2/dark_s0", "v2/dark_s1"]
    h, m, raw = (mean(lawful, "human", "trained", "object_hidden"), mean(lawful, "memoryless", "trained", "object_hidden"),
                 mean(lawful, "human", "raw senses", "object_hidden"))
    out = {"H1": {"human": h, "memoryless": m, "human raw senses": raw,
                  "passes": bool(h - m > 0.05 and abs(m - raw) <= 0.05),
                  "rule": "object_hidden(human) - object_hidden(memoryless) > 0.05 and |memoryless - raw| <= 0.05, mean of seeds 0 and 1"}}
    p1 = {}
    for b in ("1-5 steps", "6-20 steps", "1-5 steps, dark", "1-5 steps, lit", "6-20 steps, dark", "6-20 steps, lit"):
        hh, mm, rr_ = mean(dark, "human", "trained", b), mean(dark, "memoryless", "trained", b), mean(dark, "human", "raw senses", b)
        p1[b] = {"human": hh, "memoryless": mm, "human raw senses": rr_, "beats raw by 0.05": hh - rr_ > 0.05,
                 "beats memoryless by 0.05": hh - mm > 0.05}
    out["P1"] = {**p1, "passes": all(p1[b]["beats raw by 0.05"] and p1[b]["beats memoryless by 0.05"] for b in ("1-5 steps", "6-20 steps")),
                 "rule": "the human's permanence skill at 1 to 5 and 6 to 20 steps exceeds its raw senses and the memoryless mind by more than 0.05 each, mean of seeds 0 and 1 (dark and lit split reported, not part of the rule)"}
    (OUT / "verdicts.json").write_text(json.dumps(out, indent=1))
    print(f"H1: human {h:+.3f}, memoryless {m:+.3f}, raw {raw:+.3f}: {'PASS' if out['H1']['passes'] else 'FAIL'}")
    for b, v in p1.items():
        print(f"P1 {b:18} human {v['human']:+.3f} raw {v['human raw senses']:+.3f} memoryless {v['memoryless']:+.3f}")
    print(f"P1: {'PASS' if out['P1']['passes'] else 'FAIL'}")


if __name__ == "__main__":
    main()
