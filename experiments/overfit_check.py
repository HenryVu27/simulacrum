"""Diagnosis (prediction error only): does a mind with memory memorize its training episodes?

Trains human and memoryless exactly as calibration does, then scores each on 256 training episodes and
on the 256 held-out episodes. A mind that generalizes scores about the same on both. A mind that has
memorized episodes (for example, when the lights come back in each one) scores far better on episodes
it trained on. Error is also traced through dark spells on both sets.

    uv run python experiments/overfit_check.py dark human 4000
"""

import json
import sys

import numpy as np

from umwelt import learner, presets
from umwelt.data import load
from umwelt.experiment import RESULTS


def dark_course(err, lit):
    lit = lit[:, 1:]
    into = np.zeros_like(lit, int)
    for t in range(1, lit.shape[1]):
        into[:, t] = np.where(~lit[:, t], np.where(lit[:, t - 1], 1, into[:, t - 1] + 1), 0)
    burn = np.arange(lit.shape[1]) >= 16
    return [float(err[(into == k) & burn[None]].mean()) for k in range(1, 31, 3)]


def main(world, mind, steps, n_train=1024):
    d = load(world, 1280, 256, seed=0)
    p = presets.PRESETS[mind]
    o = presets.observe(d, p, dtype=np.float16)
    a = presets.actions(d, p)
    tr = slice(0, n_train)
    mu, sd = learner.standardizer(o[tr])
    for i in range(0, len(o), 128):
        o[i : i + 128] = (o[i : i + 128].astype(np.float32) - mu) / sd
    params, _ = learner.train(o[tr], a[tr], learner.LearnerConfig(steps=steps), seed=0, recurrent=p.recurrent, verbose=False)
    lit = np.asarray(d["gt_light"]) if "gt_light" in d else np.ones(o.shape[:2], bool)
    out = {"world": world, "mind": mind, "steps": steps, "n_train": n_train}
    for name, sl in (("train_episodes", slice(0, 256)), ("test_episodes", slice(1024, 1280))):
        err = learner.batched(learner._errors_jit, params, o[sl], a[sl])
        L = lit[sl][:, 1:]
        burn = np.arange(L.shape[1]) >= 16
        out[name] = {"all": float(err[:, 16:].mean()), "lit": float(err[L & burn].mean()),
                     "dark": float(err[~L & burn].mean()) if (~L).any() else None,
                     "dark_course": dark_course(err, lit[sl]) if (~L).any() else None}
        print(f"[{world} {mind} u{steps} n{n_train}] {name}: all {out[name]['all']:.3f} lit {out[name]['lit']:.3f} "
              f"dark {out[name]['dark']} course {[round(x, 2) for x in out[name]['dark_course'] or []]}", flush=True)
    path = RESULTS / "calibration" / "overfit" / f"{world}_{mind}_u{steps}_n{n_train}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1024)
