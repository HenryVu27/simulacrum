"""Do surprise conclusions depend on which sensor-noise draws a score saw? (docs/preregistration.md, surprise
scores over sensor-noise draws.)

Every score is made from the errors of voe.NOISE_DRAWS draws (draws 0 to 199), S on their mean and a bootstrap
over quadruplets (voe.surprise). This scores the same networks on a disjoint set of draws (200 to 399) and
compares, and shows what one draw did before:

1. one draw, as every score was made until 2026-11-05 (draws 0 to 19 one at a time): how far S moves between
   draws, how often "interval excludes 0" changes, how often the one-draw interval contains the noise-averaged S;
2. the statistic, on draws 0 to 199 (set A) and 200 to 399 (set B): how far S moves, whether "interval excludes
   0" agrees, whether A's interval contains B's S;
3. the parts of H6 (v1) and P4 (v2), which compare permanence scores, on set A and on set B.

Scores: every mind with sensor noise in lawful seed 0 and dark seed 0, trained and untrained, three tests; and
the permanence scores of trained minds in lawful seed 1, impermanent seed 0 and dark seed 1, which H6 and P4 use.

    uv run python experiments/noise_draws_check.py > results/checks/2026-11-05_noise_draws/check.txt
"""

import json
import pickle

import jax
import numpy as np

from umwelt import learner, learner_jepa, presets, voe
from umwelt.experiment import RESULTS

FULL = ["main/lawful_s0", "v2/dark_s0"]
PERMANENCE_ONLY = ["main/lawful_s1", "main/impermanent_s0", "v2/dark_s1"]
K = voe.NOISE_DRAWS


def excludes(ci):
    return ci[0] > 0 or ci[1] < 0


def score_all():
    suite = voe.build_suite()
    rows = []
    for run in FULL + PERMANENCE_ONLY:
        folder = RESULTS / "runs" / run
        for f in sorted(folder.glob("*.json")):
            if f.stem == "translation":
                continue
            rec = json.loads(f.read_text())
            p = presets.PRESETS[rec["preset"]]
            if all(s.noise == 0 for s in p.senses):
                continue
            with open(folder / f"{f.stem}_params.pkl", "rb") as fh:
                saved = pickle.load(fh)
            cfg = learner.LearnerConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in rec["config"].items()})
            L = learner_jepa if cfg.objective == "jepa" else learner
            nets = {"trained": saved["params"]}
            if run in FULL and "voe_untrained" in rec:
                nets["untrained"] = L.init(jax.random.PRNGKey(10_000 + rec["seed"]), rec["obs_dim"], cfg, p.recurrent)
            tests = voe.TESTS if run in FULL else ("permanence",)
            for net, params in nets.items():
                for test in tests:
                    d, crit = suite[test]
                    a = presets.actions(d, p)
                    err = np.stack([learner.batched(L._errors_jit, params, (presets.observe(d, p, draw=k) - saved["mu"]) / saved["sd"], a)
                                    for k in range(2 * K)])
                    rows.append({"run": run, "mind": f.stem, "net": net, "test": test,
                                 "single": [voe.surprise(e, crit) for e in err[:20]],
                                 "A": voe.surprise(err[:K], crit), "B": voe.surprise(err[K:], crit)})
                    stored = rec["voe" if net == "trained" else "voe_untrained"][test]
                    assert (rows[-1]["A"][0], list(rows[-1]["A"][1])) == (stored["S"], stored["ci"]), (run, f.stem, net, test)  # set A is the stored score
            print(f"{run} {f.stem}: done", flush=True)
    return rows


def verdict_parts(rows, which):
    """The parts of H6 (v1) and P4 (v2) that use permanence scores, on draw set `which` (report.py, report_v2.py)."""
    get = {(r["run"], r["mind"]): r[which] for r in rows if r["net"] == "trained" and r["test"] == "permanence"}
    lawful = [f"main/lawful_s{s}" for s in (0, 1)]
    dark = [f"v2/dark_s{s}" for s in (0, 1)]
    minds = sorted({m for (run, m) in get if run == "main/impermanent_s0"})
    h6_above = [m for m in ("human", "vision_only") if all(get[(r, m)][1][0] > 0 for r in lawful)]
    h6_null = all(not excludes(get[(r, "memoryless")][1]) for r in lawful)
    lower = np.mean([get[("main/impermanent_s0", m)][0] < np.mean([get[(r, m)][0] for r in lawful]) for m in minds])
    p4_above = all(get[(r, "human")][1][0] > 0 for r in dark)
    p4_null = all(not excludes(get[(r, "memoryless")][1]) for r in dark)
    s2, s1 = np.mean([get[(r, "human")][0] for r in dark]), np.mean([get[(r, "human")][0] for r in lawful])
    return (f"H6: intervals above 0 for {h6_above or 'none'}; memoryless null {h6_null}; share of {len(minds)} minds less "
            f"surprised after impermanent training {lower:.2f} (the oracle, which has no sensor noise, is left out here "
            f"and in the report). P4: human intervals above 0 {p4_above}; memoryless null {p4_null}; human mean S "
            f"{s2:+.5f} here against {s1:+.5f} in v1, above {s2 > s1}")


def main():
    rows = score_all()
    out = RESULTS / "checks" / "2026-11-05_noise_draws" / "check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    full = [r for r in rows if r["run"] in FULL]

    def contains(ci, x):
        return ci[0] <= x <= ci[1]

    print(f"\n1. one draw, {len(full)} scores of lawful seed 0 and dark seed 0 (minds with sensor noise, trained and "
          "untrained, three tests), draws 0 to 19")
    sd = np.array([np.std([s for s, _ in r["single"]], ddof=1) for r in full])
    half = np.array([np.mean([(ci[1] - ci[0]) / 2 for _, ci in r["single"]]) for r in full])
    print(f"   S varies between draws with sd {np.median(sd):.4f} (median), {sd.max():.4f} (largest): {np.median(sd / half):.2f} "
          f"of the interval's half-width (median), {np.max(sd / half):.2f} (largest)")
    flips = np.mean([np.mean([excludes(ci) != excludes(r["single"][0][1]) for _, ci in r["single"][1:]]) for r in full])
    print(f"   'interval excludes 0' differs from draw 0's in {100 * flips:.1f}% of (score, draw) pairs")
    cover = np.mean([np.mean([contains(ci, r["B"][0]) for _, ci in r["single"]]) for r in full])
    print(f"   a one-draw interval contains the {K}-draw S of set B {100 * cover:.1f}% of the time: within a run the old "
          "interval did cover the noise; the point estimate carried it, and every run of a mind shared the draw")

    print(f"\n2. the statistic over {K} draws, set A (draws 0 to {K - 1}) against set B (draws {K} to {2 * K - 1}), same scores")
    dS = np.array([abs(r["A"][0] - r["B"][0]) for r in full])
    halfA = np.array([(r["A"][1][1] - r["A"][1][0]) / 2 for r in full])
    print(f"   S differs by {np.median(dS):.5f} (median), {dS.max():.5f} (largest): {np.median(dS / halfA):.3f} of the "
          f"half-width (median), {np.max(dS / halfA):.2f} (largest)")
    agree = [excludes(r["A"][1]) == excludes(r["B"][1]) for r in full]
    print(f"   'interval excludes 0' agrees in {100 * np.mean(agree):.1f}% of scores ({len(agree) - sum(agree)} disagree)")
    print(f"   A's interval contains B's S in {100 * np.mean([contains(r['A'][1], r['B'][0]) for r in full]):.1f}% of scores")
    for r in full:
        if excludes(r["A"][1]) != excludes(r["B"][1]):
            print(f"   disagree: {r['run']} {r['mind']} {r['net']} {r['test']}: A {r['A'][0]:+.5f} {np.round(r['A'][1], 5).tolist()}, "
                  f"B {r['B'][0]:+.5f} {np.round(r['B'][1], 5).tolist()}")

    print("\n3. verdict parts that use permanence scores")
    for which in ("A", "B"):
        print(f"   set {which}: {verdict_parts(rows, which)}")


if __name__ == "__main__":
    main()
