"""Second correction to the surprise-test stimuli (docs/preregistration.md, 2026-11-05): recompute every mind's
surprise scores on the redesigned movies.

The independent check of the first correction found two more defects in the movies:
- The oracle reads wall coordinates, and an unused wall slot held an off-arena marker, unlike any wall it saw in
  training. Where the marker lay set the oracle's scores (permanence 0 to 0.57 across ten places).
- In the solidity movies the agent stood on the wall's line, so no ray met the wall: to every sense but the
  oracle's, the wall movies were exactly the no-wall movies.
Now every movie has three real walls in the range training draws from, and the solidity agent sees the wall's face
(src/umwelt/voe.py). That changes every mind's inputs, so every mind's scores are recomputed from its stored
network. Nothing is retrained.

For each stored run it first rebuilds the previous movies (voe.py as of BEFORE, read from git) and requires them
to reproduce every stored score exactly, so the recompute path is the one that made them. Then it scores the new
movies and writes them into each result file. The previous scores are kept under voe_before_battery_fix and
voe_untrained_before_battery_fix; the scores from before the first correction stay under *_before_wall_fix.

    uv run python experiments/recompute_battery.py            # check only
    uv run python experiments/recompute_battery.py --write    # check, then write

Kept as the record: its scores used one sensor-noise draw, as every score did then, and were rescored over 20
draws the same day (recompute_noise.py). It still scores with one draw; do not run it with --write again.
"""

import argparse
import json
import pickle
import platform
import subprocess
import sys
import time
import types
from pathlib import Path

import jax

from umwelt import learner, learner_jepa, presets, voe
from umwelt.experiment import RESULTS, _code_version, _jsonable, _voe_scores

BEFORE = "d6c6763"  # the last commit with the previous movies
# Every stored run whose surprise scores are used. The pilot's files are not in git. The bench_* runs time the
# code and are not in git either; bench_mac-cpu keeps a network, but its scores are used nowhere, so it is left
# as it was (it still holds the previous movies' scores).
RUNS = ["main/lawful_s0", "main/lawful_s1", "main/ghost_s0", "main/impermanent_s0", "main/nonlocal_s0",
        "main/lawful_mj_s0", "v2/dark_s0", "v2/dark_s1", "pilot/lawful_s0",
        "long/lawful_s0", "jepa/lawful_s0", "ab_drop/lawful_s0", "ab_h5/lawful_s0", "ab_h15/lawful_s0", "ab_h30/lawful_s0"]
WHAT = ("surprise-test stimuli redesigned: three real walls in every movie, solidity wall seen face-on "
        "(docs/preregistration.md, second correction, 2026-11-05)")


def previous_voe():
    """voe.py as of BEFORE, as a module."""
    repo = Path(__file__).resolve().parents[1]
    src = subprocess.run(["git", "-C", repo, "show", f"{BEFORE}:src/umwelt/voe.py"], capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("voe_before")
    exec(compile(src, f"{BEFORE}:src/umwelt/voe.py", "exec"), mod.__dict__)
    return mod


def scores(rec, saved, suite):
    """A stored mind's surprise scores on a suite: its trained network, and its untrained control if it has one."""
    cfg = learner.LearnerConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in rec["config"].items()})
    L = learner_jepa if cfg.objective == "jepa" else learner
    p = presets.PRESETS[rec["preset"]]
    out = {"voe": _voe_scores(saved["params"], suite, p, saved["mu"], saved["sd"], L, draws=1)}
    if "voe_untrained" in rec:  # the run's untrained-network control, rebuilt from its seed
        p0 = L.init(jax.random.PRNGKey(10_000 + rec["seed"]), rec["obs_dim"], cfg, p.recurrent)
        out["voe_untrained"] = _voe_scores(p0, suite, p, saved["mu"], saved["sd"], L, draws=1)
    return json.loads(json.dumps(_jsonable(out)))  # the same rounding to JSON as the stored files


def minds(d):
    return sorted(f.stem for f in d.glob("*.json") if f.stem != "translation")


def main(write):
    t0 = time.time()
    old, new = previous_voe().build_suite(), voe.build_suite()
    stimuli = voe.suite_hash(new)
    print(f"suites built in {time.time() - t0:.0f} s; new stimuli {stimuli}, previous {voe.suite_hash(old)}", flush=True)
    report = []
    for run in RUNS:
        d = RESULTS / "runs" / run
        if not d.exists():
            print(f"{run}: not here", flush=True)
            continue
        for name in minds(d):
            f = d / f"{name}.json"
            rec = json.loads(f.read_text())
            with open(d / f"{name}_params.pkl", "rb") as fh:
                saved = pickle.load(fh)
            stored = {k: rec[f"{k}_before_battery_fix"] if f"{k}_before_battery_fix" in rec else rec[k]
                      for k in ("voe", "voe_untrained") if k in rec}  # rerun after writing: start from the kept values
            again = scores(rec, saved, old)
            assert again == stored, f"{run}/{name}: the previous movies do not reproduce the stored scores"
            fixed = scores(rec, saved, new)
            for k in fixed:
                for test in fixed[k]:
                    a, b = stored[k][test], fixed[k][test]
                    report.append({"run": run, "mind": name, "net": "trained" if k == "voe" else "untrained", "test": test,
                                   "S_before": a["S"], "S_after": b["S"], "ci_before": a["ci"], "ci_after": b["ci"],
                                   "S_copy_before": a["S_copy"], "S_copy_after": b["S_copy"]})
            if write:
                for k in fixed:
                    rec[f"{k}_before_battery_fix"] = stored[k]
                    rec[k] = fixed[k]
                rec.setdefault("provenance", {})
                was = rec["provenance"].get("stimuli")
                rec["provenance"]["stimuli"] = stimuli  # what the surprise scores in this file were scored on
                kept = [c for c in rec.get("corrections", []) if not (isinstance(c, dict) and c.get("what") == WHAT)]
                rec["corrections"] = kept + [{"date": "2026-11-05", "what": WHAT, "script": "experiments/recompute_battery.py",
                                              "code": _code_version(), "machine": platform.node(), "stimuli": stimuli,
                                              "stimuli_before": was or "not recorded"}]
                f.write_text(json.dumps(rec, indent=1))
        print(f"{run}: previous movies reproduce all {len(minds(d))} minds' stored scores exactly"
              + ("; written" if write else ""), flush=True)
    out = RESULTS / "battery_fix_recompute.json"
    if write:  # check mode leaves the record alone
        out.write_text(json.dumps({"stimuli": stimuli, "before": BEFORE, "scores": report}, indent=1))
    print("wrote" if write else "checked; not written:", out, f"({len(report)} scores, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    sys.exit(main(ap.parse_args().write))
