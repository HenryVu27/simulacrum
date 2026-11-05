"""Surprise scores over 20 sensor-noise draws (docs/preregistration.md, 2026-11-05, after the second correction).

Every surprise score so far saw one draw of sensor noise, the same in every run of a mind (presets.observe seeds
it with the preset's name). The draw moved permanence scores by as much as the effects H6 and P4 test (bat:
-0.0052 on that draw, +0.0001 over 20), and the interval did not cover it. Now each score averages the errors of
voe.NOISE_DRAWS draws (draw 0 is the old one), and the interval resamples draws as well as quadruplets
(voe.surprise). Minds without sensor noise (the oracle) are unchanged. Nothing is retrained.

For each stored mind it first rescores with one draw and requires the stored scores exactly, then scores with
voe.NOISE_DRAWS draws and writes those, keeping the previous ones under voe_before_noise_draws and
voe_untrained_before_noise_draws.

    uv run python experiments/recompute_noise.py            # check only
    uv run python experiments/recompute_noise.py --write    # check, then write
"""

import argparse
import importlib.util
import json
import pickle
import platform
import time
from pathlib import Path

import jax

from umwelt import learner, learner_jepa, presets, voe
from umwelt.experiment import RESULTS, _code_version, _jsonable, _voe_scores

spec = importlib.util.spec_from_file_location("recompute_battery", Path(__file__).with_name("recompute_battery.py"))
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)
WHAT = f"surprise scores over {voe.NOISE_DRAWS} sensor-noise draws (docs/preregistration.md, 2026-11-05)"


def scores(rec, saved, suite, draws):
    cfg = learner.LearnerConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in rec["config"].items()})
    L = learner_jepa if cfg.objective == "jepa" else learner
    p = presets.PRESETS[rec["preset"]]
    out = {"voe": _voe_scores(saved["params"], suite, p, saved["mu"], saved["sd"], L, draws=draws)}
    if "voe_untrained" in rec:
        p0 = L.init(jax.random.PRNGKey(10_000 + rec["seed"]), rec["obs_dim"], cfg, p.recurrent)
        out["voe_untrained"] = _voe_scores(p0, suite, p, saved["mu"], saved["sd"], L, draws=draws)
    return json.loads(json.dumps(_jsonable(out)))


def main(write):
    t0 = time.time()
    suite = voe.build_suite()
    stimuli = voe.suite_hash(suite)
    report = []
    for run in rb.RUNS:
        d = RESULTS / "runs" / run
        if not d.exists():
            print(f"{run}: not here", flush=True)
            continue
        for name in rb.minds(d):
            f = d / f"{name}.json"
            rec = json.loads(f.read_text())
            assert rec.get("provenance", {}).get("stimuli") == stimuli, f"{run}/{name}: scored on other movies"
            with open(d / f"{name}_params.pkl", "rb") as fh:
                saved = pickle.load(fh)
            stored = {k: rec[f"{k}_before_noise_draws"] if f"{k}_before_noise_draws" in rec else rec[k]
                      for k in ("voe", "voe_untrained") if k in rec}
            assert scores(rec, saved, suite, 1) == stored, f"{run}/{name}: one draw does not reproduce the stored scores"
            new = scores(rec, saved, suite, voe.NOISE_DRAWS)
            for k in new:
                for test in new[k]:
                    a, b = stored[k][test], new[k][test]
                    report.append({"run": run, "mind": name, "net": "trained" if k == "voe" else "untrained", "test": test,
                                   "S_before": a["S"], "S_after": b["S"], "ci_before": a["ci"], "ci_after": b["ci"],
                                   "S_copy_before": a["S_copy"], "S_copy_after": b["S_copy"]})
            if write:
                for k in new:
                    rec[f"{k}_before_noise_draws"] = stored[k]
                    rec[k] = new[k]
                rec["provenance"]["voe_draws"] = voe.NOISE_DRAWS
                kept = [c for c in rec.get("corrections", []) if not (isinstance(c, dict) and c.get("what") == WHAT)]
                rec["corrections"] = kept + [{"date": "2026-11-05", "what": WHAT, "script": "experiments/recompute_noise.py",
                                              "code": _code_version(), "machine": platform.node()}]
                f.write_text(json.dumps(rec, indent=1))
        print(f"{run}: one draw reproduces all {len(rb.minds(d))} minds' stored scores exactly"
              + ("; written" if write else "") + f" ({time.time() - t0:.0f} s)", flush=True)
    out = RESULTS / "noise_draws_recompute.json"
    if write:
        out.write_text(json.dumps({"draws": voe.NOISE_DRAWS, "stimuli": stimuli, "scores": report}, indent=1))
    print("wrote" if write else "checked; not written:", out, f"({len(report)} scores, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
