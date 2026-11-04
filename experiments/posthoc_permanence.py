"""Post-hoc, not preregistered: how fast does each mind's knowledge of an object decay once it leaves view?

The preregistered object_hidden score pools objects that left view a moment ago with objects never seen
at all, which no mind can know about. This script reloads each trained mind, refits the egocentric
object probe on all training episodes, and scores it separately by time since the object was last in view.
Scores are skill above a mind that knows nothing (0) up to perfect (1). A mind with object permanence
should stay above its raw-senses curve for some steps after an object disappears; a reactive mind
falls to that curve at once.

    uv run python experiments/posthoc_permanence.py lawful 0            # world, learner seed
    uv run python experiments/posthoc_permanence.py occlusion 0 v2 1   # plus run tag and data seed
"""

import json
import pickle
import sys

import numpy as np

from umwelt import learner, learner_jepa, presets, probes
from umwelt.data import load
from umwelt.experiment import RESULTS

BINS = [("in view", None), ("1-5 steps", (1, 5)), ("6-20 steps", (6, 20)), ("21-60 steps", (21, 60)),
        ("over 60 steps", (61, 999)), ("never seen", (1000, 1000))]


def curve(Htr, Ttr, Hte, Tte, seed=0, stride_tr=8, stride_te=4):
    Xtr, Xte = probes._prep(Htr, stride_tr), probes._prep(Hte, stride_te)
    groups = np.repeat(np.arange(Htr.shape[0]), len(range(probes.PAST, Htr.shape[1], stride_tr)))
    R = {k: (probes._rows(Ttr, k, stride_tr), probes._rows(Tte, k, stride_te)) for k in ("ego", "present", "vis", "since")}
    m2 = lambda m: np.repeat(m, 2, axis=1)
    p = probes.Probe(Xtr, R["ego"][0].reshape(len(Xtr), -1), m2(R["present"][0]), "reg", 256, seed, groups)
    pred, Y = p(Xte), R["ego"][1].reshape(len(Xte), -1)
    pres, vis, since = R["present"][1], R["vis"][1], R["since"][1]
    ptr = R["present"][0]
    ytr = R["ego"][0]
    mean = np.array([[ytr[ptr[:, k], k, j].mean() if ptr[:, k].any() else 0.0 for j in range(2)] for k in range(ptr.shape[1])])
    const = np.broadcast_to(mean.reshape(1, -1), Y.shape)
    out = {}
    for name, rng in BINS:
        m = pres & vis if rng is None else pres & ~vis & (since >= rng[0]) & (since <= rng[1])
        score = probes._r2_cols(pred, Y, m2(m), m2(pres))
        base = probes._r2_cols(const, Y, m2(m), m2(pres))
        out[name] = (score - base) / (1 - base)  # skill above a mind that knows nothing
        out[name + " (n)"] = int(m.sum())
    if "dark" in Tte:  # exploratory (Amendment 2): hidden because the lights are out, or out of view while lit
        dark = probes._rows(Tte, "dark", stride_te)
        for name, rng in BINS[1:3]:
            for label, dm in (("dark", dark), ("lit", ~dark)):
                m = pres & ~vis & (since >= rng[0]) & (since <= rng[1]) & dm
                score = probes._r2_cols(pred, Y, m2(m), m2(pres))
                base = probes._r2_cols(const, Y, m2(m), m2(pres))
                out[f"{name}, {label}"] = (score - base) / (1 - base)
                out[f"{name}, {label} (n)"] = int(m.sum())
    return out


def main(world="lawful", seed=0, tag="main", data_seed=0):
    run = RESULTS / "runs" / tag / f"{world}_s{seed}"
    n_train = json.loads((run / "human.json").read_text()).get("n_train", 1024)
    full = load(world, n_train + 256, seed=data_seed)
    # Probes are fit on the first 1,024 training episodes (as in the runner) and scored on the run's held-out ones.
    d = {k: np.concatenate([v[:1024], v[n_train : n_train + 256]]) for k, v in full.items()}
    tr, te = slice(0, 1024), slice(1024, 1280)
    gt = {k: v for k, v in d.items() if k.startswith("gt_")}
    Ttr = probes.targets({k: np.asarray(v[tr]) for k, v in gt.items()})
    Tte = probes.targets({k: np.asarray(v[te]) for k, v in gt.items()})
    if "gt_light" in gt:
        Tte["dark"] = np.repeat(~np.asarray(gt["gt_light"][te])[..., None], probes.K, -1)
    results = {}
    for f in sorted(run.glob("*_params.pkl")):
        mind = f.name[: -len("_params.pkl")]
        with open(f, "rb") as fh:
            saved = pickle.load(fh)
        params, mu, sd = saved["params"], saved["mu"], saved["sd"]
        L = learner_jepa if "jenc1" in params else learner
        pr = presets.PRESETS[mind]
        o = presets.observe(d, pr, dtype=np.float16)
        for i in range(0, len(o), 128):
            o[i : i + 128] = (o[i : i + 128].astype(np.float32) - mu) / sd
        a = presets.actions(d, pr)
        Htr = learner.batched(L._latents_jit, params, o[tr], a[tr])
        Hte = learner.batched(L._latents_jit, params, o[te], a[te])
        results[mind] = {"trained": curve(Htr, Ttr, Hte, Tte)}
        if mind in ("human", "vision_only", "dog", "smell_only", "memoryless"):
            results[mind]["raw_senses"] = curve(o[tr], Ttr, o[te], Tte)
        row = results[mind]["trained"]
        print(f"{mind:15s} " + " ".join(f"{n}={row[n]:.2f}" for n, _ in BINS), flush=True)
        if "raw_senses" in results[mind]:
            row = results[mind]["raw_senses"]
            print(f"{'  raw senses':15s} " + " ".join(f"{n}={row[n]:.2f}" for n, _ in BINS), flush=True)
    out = RESULTS / f"posthoc_permanence_{tag}_{world}_s{seed}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "lawful", int(a[1]) if len(a) > 1 else 0, a[2] if len(a) > 2 else "main",
         int(a[3]) if len(a) > 3 else 0)
