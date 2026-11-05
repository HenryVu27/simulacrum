"""Claim 5 (pipeline part): every stored mind, scored by the project's own _voe_scores on the previous movies and on
the committed new movies, compared for exact equality with the committed JSON (voe, voe_untrained,
*_before_battery_fix). My own loop; the committed files come from git (c35cc6e), the networks from the working tree."""
import json
import pickle
import time
from pathlib import Path

import jax
import numpy as np
from common import S, old_voe, voe
from umwelt import learner, learner_jepa, presets
from umwelt.experiment import _jsonable, _voe_scores

LIVE = Path("/Users/henryvu/personal/umwelt/results/runs")
COMMITTED = S / "repo_c35" / "results" / "runs"
RUNS = sorted({p.parent.relative_to(LIVE).as_posix() for p in LIVE.glob("*/*/*_params.pkl")})
print("runs with networks:", RUNS)
t0 = time.time()
old, new = old_voe().build_suite(), voe.build_suite()
print(f"suites in {time.time()-t0:.0f}s; hashes: new {voe.suite_hash(new)} (recorded 016fc3d72807e39d), previous {voe.suite_hash(old)} (recorded e8b112a5e2a11879)", flush=True)
pickle.dump({"new": new}, open(S / "suite_new.pkl", "wb"))
n_minds = n_scores = 0
bad = []
for run in RUNS:
    for pkl in sorted((LIVE / run).glob("*_params.pkl")):
        name = pkl.name[: -len("_params.pkl")]
        jf = COMMITTED / run / f"{name}.json"
        src = "git"
        if not jf.exists():
            jf, src = LIVE / run / f"{name}.json", "disk (not in git)"
        rec = json.loads(jf.read_text())
        saved = pickle.load(open(pkl, "rb"))
        cfg = learner.LearnerConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in rec["config"].items()})
        L = learner_jepa if cfg.objective == "jepa" else learner
        p = presets.PRESETS[rec["preset"]]
        nets = {"voe": saved["params"]}
        if "voe_untrained" in rec:
            nets["voe_untrained"] = L.init(jax.random.PRNGKey(10_000 + rec["seed"]), rec["obs_dim"], cfg, p.recurrent)
        for k, params in nets.items():
            for label, suite, field in (("previous", old, f"{k}_before_battery_fix"), ("new", new, k)):
                got = json.loads(json.dumps(_jsonable(_voe_scores(params, suite, p, saved["mu"], saved["sd"], L))))
                want = rec.get(field)
                n_scores += 3
                if got != want:
                    bad.append((run, name, k, label, src))
                    print(f"  MISMATCH {run} {name} {k} {label}: got {got} want {want}", flush=True)
        n_minds += 1
    print(f"{run}: done ({time.time()-t0:.0f}s)", flush=True)
print(f"\n{n_minds} minds, {n_scores} test scores compared; mismatches: {len(bad)} {bad}")
