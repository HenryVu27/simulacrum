"""Where does a mind's solidity surprise come from? (Post hoc, after the second correction of 2026-11-05.)

The independent check of that correction found trained minds whose solidity interval lies above 0 (the bat in
most runs, the jepa human). S adds two contrasts, each over P1 + P2:
  pass   = I1 - P2: the object passes through the wall, against passing where no wall is;
  bounce = I2 - P1: the object bounces where no wall is, against bouncing off the wall.
The hope was that they would show which expectation a mind holds. They do not: each also carries the main effect
of where the wall stands, which the design cancels only in their sum. For every vision mind the networks err
about 0.1 less (of the possible movies' error) with the wall in the path than with it slid away, whatever the
object does, so pass is about -0.1 and bounce about +0.1 in every world.

The test is the world contrast on S itself. In the ghost world objects pass through internal walls, so a mind
that learned in the lawful world that walls stop objects should score higher than its ghost-trained twin.
A difference in the stimuli that only a learned predictor picks up would show in both worlds alike.

Errors are scored with the stored networks on the movies the battery uses, over voe.NOISE_DRAWS draws of each
preset's sensor noise (the stored scores; asserted) and again without noise. Intervals: bootstrap over the 18
quadruplets of the draws' mean, as in voe.surprise. (Its first version used one noise draw, as every score did
then, and a later one 20 draws; the conclusion did not change.)

    uv run python experiments/solidity_split.py > results/checks/2026-11-05_battery/solidity_split.txt
"""

import importlib.util
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np

from umwelt import learner, learner_jepa, presets, voe
from umwelt.experiment import RESULTS

spec = importlib.util.spec_from_file_location("recompute_battery", Path(__file__).with_name("recompute_battery.py"))
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)

RUNS = ["main/lawful_s0", "main/lawful_s1", "main/ghost_s0", "main/impermanent_s0", "main/nonlocal_s0",
        "main/lawful_mj_s0", "v2/dark_s0", "v2/dark_s1", "jepa/lawful_s0", "long/lawful_s0"]


def contrasts(err, crit, rng, n_boot=voe.N_BOOT):
    """err (draws, n*4, T-1). Returns {name: (value, (lo, hi))} for S, pass and bounce on the draws' mean errors,
    resampling quadruplets as voe.surprise does."""
    mean = voe.totals(err, crit).mean(0)  # (n, 4): P1 P2 I1 I2
    n = len(mean)

    def stats(t):
        base = t[:, 0].sum() + t[:, 1].sum()
        return {"S": (t[:, 2].sum() + t[:, 3].sum() - base) / base, "pass": (t[:, 2].sum() - t[:, 1].sum()) / base,
                "bounce": (t[:, 3].sum() - t[:, 0].sum()) / base}

    point = stats(mean)
    boots = [stats(mean[rng.integers(0, n, n)]) for _ in range(n_boot)]
    return {k: (v, tuple(np.percentile([b[k] for b in boots], [2.5, 97.5]))) for k, v in point.items()}


def main():
    d, crit = voe.build_suite()["solidity"]
    rows = []
    for run in RUNS:
        folder = RESULTS / "runs" / run
        for name in rb.minds(folder):
            rec = json.loads((folder / f"{name}.json").read_text())
            with open(folder / f"{name}_params.pkl", "rb") as fh:
                saved = pickle.load(fh)
            cfg = learner.LearnerConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in rec["config"].items()})
            L = learner_jepa if cfg.objective == "jepa" else learner
            p = presets.PRESETS[name]
            for noise in ("usual", "none"):
                q = p if noise == "usual" else replace(p, senses=tuple(replace(s, noise=0.0) for s in p.senses))
                draws = voe.NOISE_DRAWS if noise == "usual" and any(s.noise > 0 for s in p.senses) else 1
                err = np.stack([learner.batched(L._errors_jit, saved["params"], (presets.observe(d, q, draw=k) - saved["mu"]) / saved["sd"],
                                                presets.actions(d, q)) for k in range(draws)])
                c = contrasts(err, crit, np.random.default_rng(0))
                rows.append({"run": run, "mind": name, "noise": noise, **{k: [float(v), [float(x) for x in ci]] for k, (v, ci) in c.items()}})
                if noise == "usual":
                    assert abs(c["S"][0] - rec["voe"]["solidity"]["S"]) < 1e-6, (run, name)  # the stored score
    out = RESULTS / "checks" / "2026-11-05_battery" / "solidity_split.json"
    out.write_text(json.dumps(rows, indent=1))

    def fmt(x):
        v, (lo, hi) = x
        star = "*" if lo > 0 or hi < 0 else " "
        return f"{v:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}"

    for noise in ("usual", "none"):
        print(f"\nsensor noise: {noise}. S = pass + bounce; * = 95% interval excludes 0")
        for r in rows:
            if r["noise"] == noise:
                print(f"  {r['run']:20} {r['mind']:15} S {fmt(r['S'])}  pass {fmt(r['pass'])}  bounce {fmt(r['bounce'])}")
    print("\nworld contrast, usual noise: lawful (mean of seeds 0 and 1) minus ghost, per mind")
    get = {(r["run"], r["mind"]): r for r in rows if r["noise"] == "usual"}
    for name in rb.minds(RESULTS / "runs" / "main/ghost_s0"):
        lw = [get[(f"main/lawful_s{s}", name)] for s in (0, 1)]
        gh = get[("main/ghost_s0", name)]
        print(f"  {name:15} " + "  ".join(f"{k} {np.mean([x[k][0] for x in lw]) - gh[k][0]:+.4f}" for k in ("S", "pass", "bounce")))


if __name__ == "__main__":
    main()
