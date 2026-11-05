"""Does scoring over sensor-noise draws make surprise conclusions independent of the draw? (docs/preregistration.md,
2026-11-05, surprise scores over noise draws.)

For every mind with sensor noise in lawful seed 0 and dark seed 0, trained and untrained, each test, using draws
0 to 39:
- one draw, as every score was made until now: S and its interval for each of draws 0 to 19;
- the new statistic (voe.surprise over 20 draws): once on draws 0 to 19 (set A), once on draws 20 to 39 (set B).
Reported: how far S moves between draws, how often "the interval excludes 0" disagrees, and how often an interval
contains an independent estimate of the noise-averaged S (set B's), against which a single draw is judged too.

    uv run python experiments/noise_draws_check.py > results/checks/2026-11-05_noise_draws/check.txt
"""

import json
import pickle

import jax
import numpy as np

from umwelt import learner, learner_jepa, presets, voe
from umwelt.experiment import RESULTS

RUNS = ["main/lawful_s0", "v2/dark_s0"]
K = voe.NOISE_DRAWS


def excludes(ci):
    return ci[0] > 0 or ci[1] < 0


def main():
    suite = voe.build_suite()
    rows = []
    for run in RUNS:
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
            if "voe_untrained" in rec:
                nets["untrained"] = L.init(jax.random.PRNGKey(10_000 + rec["seed"]), rec["obs_dim"], cfg, p.recurrent)
            for net, params in nets.items():
                for test, (d, crit) in suite.items():
                    a = presets.actions(d, p)
                    err = np.stack([learner.batched(L._errors_jit, params, (presets.observe(d, p, draw=k) - saved["mu"]) / saved["sd"], a)
                                    for k in range(2 * K)])
                    single = [voe.surprise(e, crit) for e in err[:K]]
                    A, B = voe.surprise(err[:K], crit), voe.surprise(err[K:], crit)
                    rows.append({"run": run, "mind": f.stem, "net": net, "test": test, "single": single, "A": A, "B": B})
            print(f"{run} {f.stem}: done", flush=True)
    out = RESULTS / "checks" / "2026-11-05_noise_draws" / "check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))

    def contains(ci, x):
        return ci[0] <= x <= ci[1]

    print(f"\n{len(rows)} scores (minds with sensor noise, trained and untrained, three tests, two runs)")
    sd = np.array([np.std([s for s, _ in r["single"]], ddof=1) for r in rows])
    half = np.array([np.mean([(ci[1] - ci[0]) / 2 for _, ci in r["single"]]) for r in rows])
    print(f"one draw: S moves between draws by sd {np.median(sd):.4f} (median), {sd.max():.4f} (largest); "
          f"that is {np.median(sd / half):.2f} of the interval's half-width (median), {np.max(sd / half):.2f} (largest)")
    flips = np.mean([np.mean([excludes(ci) != excludes(r["single"][0][1]) for _, ci in r["single"][1:]]) for r in rows])
    print(f"one draw: 'interval excludes 0' differs from draw 0's in {100 * flips:.1f}% of (score, draw) pairs")
    cover1 = np.mean([np.mean([contains(ci, r["B"][0]) for _, ci in r["single"]]) for r in rows])
    print(f"one draw: an interval contains the noise-averaged S of 20 other draws {100 * cover1:.1f}% of the time")
    dS = np.array([abs(r["A"][0] - r["B"][0]) for r in rows])
    halfA = np.array([(r["A"][1][1] - r["A"][1][0]) / 2 for r in rows])
    print(f"{K} draws: S of set A and set B differ by {np.median(dS):.4f} (median), {dS.max():.4f} (largest); "
          f"{np.median(dS / halfA):.2f} of the half-width (median), {np.max(dS / halfA):.2f} (largest)")
    agree = np.mean([excludes(r["A"][1]) == excludes(r["B"][1]) for r in rows])
    print(f"{K} draws: 'interval excludes 0' agrees between set A and set B in {100 * agree:.1f}% of scores "
          f"({sum(excludes(r['A'][1]) != excludes(r['B'][1]) for r in rows)} disagree)")
    coverA = np.mean([contains(r["A"][1], r["B"][0]) for r in rows])
    print(f"{K} draws: set A's interval contains set B's S in {100 * coverA:.1f}% of scores")
    for r in rows:
        if excludes(r["A"][1]) != excludes(r["B"][1]):
            print(f"  disagree: {r['run']} {r['mind']} {r['net']} {r['test']}: A {r['A'][0]:+.4f} {np.round(r['A'][1], 4).tolist()}, "
                  f"B {r['B'][0]:+.4f} {np.round(r['B'][1], 4).tolist()}")


if __name__ == "__main__":
    main()
