"""Is a solidity interval above 0 a property of the mind or of the one fixed noise draw? presets.observe seeds its
noise with crc32(preset name), so every run of a preset sees the same noise on the surprise movies, and the
bootstrap (over quadruplets) does not vary it. Rescore with the stored draw, with no noise, and with 20 other draws."""
import json
import pickle
from pathlib import Path

import numpy as np
from common import S, voe
from umwelt import learner, learner_jepa, presets

LIVE = Path("/Users/henryvu/personal/umwelt/results/runs")
suite = pickle.load(open(S / "suite_new.pkl", "rb"))["new"] if (S / "suite_new.pkl").exists() else voe.build_suite()
d, crit = suite["solidity"]
SAMPLE = [("main/lawful_s0", "bat"), ("main/lawful_s1", "bat"), ("main/lawful_mj_s0", "bat"), ("jepa/lawful_s0", "human"),
          ("main/lawful_s0", "memoryless"), ("main/lawful_s0", "human"), ("main/lawful_s1", "passive")]


def observe_seed(d, p, seed):
    rng = np.random.default_rng(seed)
    xs = []
    for s in p.senses:
        x = presets._channel(d, s)
        if s.noise > 0 and seed is not None:
            x = x + rng.normal(0.0, s.noise, x.shape).astype(np.float32)
        xs.append(x)
    return np.concatenate(xs, -1)


for run, name in SAMPLE:
    rec = json.loads((S / "repo_c35/results/runs" / run / f"{name}.json").read_text())
    saved = pickle.load(open(LIVE / run / f"{name}_params.pkl", "rb"))
    cfg = learner.LearnerConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in rec["config"].items()})
    L = learner_jepa if cfg.objective == "jepa" else learner
    p = presets.PRESETS[name]
    a = presets.actions(d, p)

    def S_for(o):
        o = (o - saved["mu"]) / saved["sd"]
        s, ci = voe.surprise(learner.batched(L._errors_jit, saved["params"], o, a), crit)
        return s, ci
    s0, ci0 = S_for(presets.observe(d, p))
    sn, cin = S_for(observe_seed(d, p, None))
    draws = [S_for(observe_seed(d, p, 1000 + i))[0] for i in range(20)]
    print(f"{run:18} {name:11} stored draw S {s0:+.4f} [{ci0[0]:+.4f},{ci0[1]:+.4f}] (json {rec['voe']['solidity']['S']:+.4f}) | "
          f"no noise {sn:+.4f} [{cin[0]:+.4f},{cin[1]:+.4f}] | 20 other draws: mean {np.mean(draws):+.4f} sd {np.std(draws):.4f} "
          f"min {min(draws):+.4f} max {max(draws):+.4f}", flush=True)
