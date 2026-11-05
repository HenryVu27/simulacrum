"""Correction to the surprise-test stimuli (docs/preregistration.md, 2026-11-05): recompute surprise scores.

Kept as the record of the first correction. It runs only on voe.py as of cf27e28 to d6c6763 (it moves
voe.HIDDEN_WALL); the second correction removed that marker and is recomputed by recompute_battery.py.

voe.HIDDEN_WALL ("no wall here") lay on the boundary wall, so some of the dog's rays met two walls at the same
distance and took their colour from last-bit rounding, which differs between devices. It now lies outside the
arena. Only two minds' surprise-test inputs change (every preset's observations of the suite were compared):
the dog's permanence movies (vision) and all three of the oracle's (it reads wall coordinates). Their scores are
recomputed here from the stored trained networks; nothing is retrained.

For each stored run it first recomputes the scores with the old movies and requires them to equal the stored
values exactly (so the recompute path is the one that made them), and recomputes one unaffected mind (human)
with the new movies, which must equal its stored values. Then it writes the new scores into the dog's and the
oracle's result files, keeping the old ones under voe_before_wall_fix and voe_untrained_before_wall_fix.

    uv run python experiments/recompute_voe.py            # check only
    uv run python experiments/recompute_voe.py --write    # check, then write
"""

import argparse
import json
import pickle
import platform

import jax
import numpy as np

from umwelt import learner, learner_jepa, presets, voe
from umwelt.experiment import RESULTS, _code_version, _jsonable, _voe_scores

OLD_HIDDEN_WALL = np.array([0.5, 0.0, 1.5, 0.0], np.float32)
AFFECTED = ("dog", "oracle")
CONTROL = "human"
RUNS = ["main/lawful_s0", "main/lawful_s1", "main/ghost_s0", "main/impermanent_s0", "main/nonlocal_s0",
        "main/lawful_mj_s0", "v2/dark_s0", "v2/dark_s1", "pilot/lawful_s0"]  # the pilot's files are not in git
WHAT = "surprise-test stimuli: hidden wall moved off the boundary (docs/preregistration.md, 2026-11-05)"


def suite_with(wall):
    keep = voe.HIDDEN_WALL
    voe.HIDDEN_WALL = wall  # the movie makers read it when they run
    try:
        return voe.build_suite()
    finally:
        voe.HIDDEN_WALL = keep


def scores(rec, saved, suite):
    cfg = learner.LearnerConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in rec["config"].items()})
    L = learner_jepa if cfg.objective == "jepa" else learner
    p = presets.PRESETS[rec["preset"]]
    out = {"voe": _voe_scores(saved["params"], suite, p, saved["mu"], saved["sd"], L)}
    if "voe_untrained" in rec:  # the run's untrained-network control, rebuilt from its seed
        p0 = L.init(jax.random.PRNGKey(10_000 + rec["seed"]), rec["obs_dim"], cfg, p.recurrent)
        out["voe_untrained"] = _voe_scores(p0, suite, p, saved["mu"], saved["sd"], L)
    return json.loads(json.dumps(_jsonable(out)))  # the same rounding to JSON as the stored files


def main(write):
    old, new = suite_with(OLD_HIDDEN_WALL), suite_with(voe.HIDDEN_WALL)
    report = []
    for run in RUNS:
        d = RESULTS / "runs" / run
        for name in AFFECTED + (CONTROL,):
            f = d / f"{name}.json"
            if not f.exists():
                print(f"{run}: no {name}", flush=True)
                continue
            rec = json.loads(f.read_text())
            with open(d / f"{name}_params.pkl", "rb") as fh:
                saved = pickle.load(fh)
            stored = {k: rec[k] for k in ("voe", "voe_untrained") if k in rec}
            if "voe_before_wall_fix" in rec:  # already corrected: the old values are kept beside the new
                stored = {k: rec[f"{k}_before_wall_fix"] for k in stored}
            again = scores(rec, saved, old)
            assert again == stored, f"{run}/{name}: the old movies do not reproduce the stored scores"
            fixed = scores(rec, saved, new)
            if name == CONTROL:
                assert fixed == stored, f"{run}/{name}: an unaffected mind changed"
                continue
            for k in fixed:
                for test in fixed[k]:
                    a, b = stored[k][test]["S"], fixed[k][test]["S"]
                    report.append({"run": run, "mind": name, "score": f"{k}/{test}", "S_before": a, "S_after": b,
                                   "change": b - a, "S_copy_before": stored[k][test]["S_copy"],
                                   "S_copy_after": fixed[k][test]["S_copy"]})
            if write:
                for k in fixed:
                    rec.setdefault(f"{k}_before_wall_fix", stored[k])
                    rec[k] = fixed[k]
                kept = [c for c in rec.get("corrections", []) if not (isinstance(c, str) and "hidden wall" in c)
                        and not (isinstance(c, dict) and c.get("what") == WHAT)]
                rec["corrections"] = kept + [{"date": "2026-11-05", "what": WHAT, "script": "experiments/recompute_voe.py",
                                              "code": _code_version(), "machine": platform.node()}]
                f.write_text(json.dumps(rec, indent=1))
        print(f"{run}: old movies reproduce the stored scores exactly; {CONTROL} unchanged by the new movies"
              + ("; written" if write else ""), flush=True)
    out = RESULTS / "wall_fix_recompute.json"
    if write:  # check mode leaves the record alone
        out.write_text(json.dumps(report, indent=1))
    for r in report:
        if r["change"] != 0:
            print(f"  {r['run']:20} {r['mind']:7} {r['score']:28} {r['S_before']:+.5f} -> {r['S_after']:+.5f} ({r['change']:+.5f})")
    print("wrote" if write else "checked; not written:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
