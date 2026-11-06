"""Instance-level check of the new interval with a variable the bootstrap is meant to cover: the bystander walls are a
per-quadruplet random draw (voe._rng per test, kind, jitter, layout seed). Score stored minds on layout seeds 0..L-1,
each over noise draws 0..19 with the new statistic (voe.surprise), and compare the SD of S across layouts with the
interval's implied SD (half-width/1.96). Usage: c3_layouts.py SNAPSHOT OUT L run ..."""
import sys, json, pickle, time
from pathlib import Path
import numpy as np, jax
from umwelt import learner, learner_jepa, presets, voe

snap, out, L = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
runs = sys.argv[4:]
t0 = time.time()
suites = {s: voe.build_suite(layout_seed=s) for s in range(L)}
print(f"{L} suites in {time.time() - t0:.0f}s", flush=True)
rows = []
for run in runs:
    folder = snap / "results/runs" / run
    for f in sorted(folder.glob("*.json")):
        if f.stem == "translation": continue
        rec = json.loads(f.read_text()); p = presets.PRESETS[rec["preset"]]
        if all(s.noise == 0 for s in p.senses): continue
        saved = pickle.load(open(folder / f"{f.stem}_params.pkl", "rb"))
        cfg = learner.LearnerConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in rec["config"].items()})
        Lm = learner_jepa if cfg.objective == "jepa" else learner
        nets = {"trained": saved["params"]}
        if "voe_untrained" in rec:
            nets["untrained"] = Lm.init(jax.random.PRNGKey(10_000 + rec["seed"]), rec["obs_dim"], cfg, p.recurrent)
        for net, params in nets.items():
            for test in voe.TESTS:
                res = []
                for s, suite in suites.items():
                    d, crit = suite[test]
                    a = presets.actions(d, p)
                    err = np.stack([learner.batched(Lm._errors_jit, params, (presets.observe(d, p, draw=k) - saved["mu"]) / saved["sd"], a) for k in range(20)])
                    S, ci = voe.surprise(err, crit)
                    S1, ci1 = voe.surprise(err[0], crit)
                    res.append((S, ci, S1, ci1))
                rows.append(dict(run=run, mind=f.stem, net=net, test=test, S=[r[0] for r in res], ci=[r[1] for r in res],
                                 S1=[r[2] for r in res], ci1=[r[3] for r in res]))
        print(f"{run} {f.stem} done {time.time() - t0:.0f}s", flush=True)
json.dump(rows, open(out, "w"))
