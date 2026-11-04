"""Amendment 2 calibration: how much does memory cut prediction error in a world?

Trains human and memoryless (data seed 0, learner seed 0, the v1 learner) and reports memory value
M = 1 - err(human) / err(memoryless) on held-out episodes. By the preregistered rule only prediction
error is looked at here: no probes, no surprise tests, and the trained models are not kept.

The breakdown says where the gap comes from: steps in the dark, the first lit step after a dark spell,
and steps where an object comes back into view.

    uv run python experiments/calibrate.py occlusion
"""

import json
import sys
import time

import numpy as np

from umwelt import learner, presets
from umwelt.data import load
from umwelt.experiment import RESULTS
from umwelt.probes import steps_since_seen


def errors(world, mind, cfg):
    d = load(world, 1280, 256, seed=0)
    tr, te = slice(0, 1024), slice(1024, 1280)
    p = presets.PRESETS[mind]
    o = presets.observe(d, p, dtype=np.float16)
    a = presets.actions(d, p)
    mu, sd = learner.standardizer(o[tr])
    for i in range(0, len(o), 128):
        o[i : i + 128] = (o[i : i + 128].astype(np.float32) - mu) / sd
    t0 = time.time()
    params, _ = learner.train(o[tr], a[tr], cfg, seed=0, recurrent=p.recurrent, verbose=False)
    err = learner.batched(learner._errors_jit, params, o[te], a[te])  # (E, T-1): error predicting step t+1
    return err, time.time() - t0, d


def main(world):
    cfg = learner.LearnerConfig()
    out = {"world": world}
    errs = {}
    for mind in ("human", "memoryless"):
        errs[mind], secs, d = errors(world, mind, cfg)
        out[f"seconds_{mind}"] = round(secs)
        print(f"[{world}] {mind} trained in {secs:.0f}s, test error {errs[mind][:, 16:].mean():.4f}", flush=True)
    te = slice(1024, 1280)
    vis = np.asarray(d["gt_vis180"][te]) & np.asarray(d["gt_obj_active"][te])
    lit = np.asarray(d["gt_light"][te]) if "gt_light" in d else np.ones(vis.shape[:2], bool)  # v1 data: always lit
    light, lit_prev = lit[:, 1:], lit[:, :-1]
    since = steps_since_seen(vis)
    returns = (vis[:, 1:] & ~vis[:, :-1] & (since[:, :-1] < 1000) & (since[:, :-1] >= 3)).any(-1)
    masks = {"all": np.ones_like(light, bool), "dark": ~light, "lights_on": light & ~lit_prev,
             "object_returns": returns, "other": light & lit_prev & ~returns}
    burn = np.arange(light.shape[1]) >= 16  # same burn-in as v1 test_error
    for name, m in masks.items():
        m = m & burn[None]
        h, z = errs["human"][m].mean() if m.any() else np.nan, errs["memoryless"][m].mean() if m.any() else np.nan
        out[name] = {"share_of_steps": float(m.mean()), "err_human": float(h), "err_memoryless": float(z),
                     "M": float(1 - h / z) if m.any() else None}
    out["M"] = out["all"]["M"]
    path = RESULTS / "calibration" / f"{world}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"[{world}] M = {out['M']:.3f} | " + " ".join(
        f"{k}: M={v['M']:.3f} ({v['share_of_steps']:.0%} of steps)" for k, v in out.items()
        if isinstance(v, dict) and v["M"] is not None), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
