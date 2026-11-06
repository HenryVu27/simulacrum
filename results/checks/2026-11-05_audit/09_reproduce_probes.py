"""Audit: recompute stored probe batteries from stored networks (lawful seed 0, data seed 0), as the runner does."""
import json, pickle, sys
import numpy as np
sys.path.insert(0, "/Users/henryvu/personal/umwelt/src")
from umwelt import learner, presets, probes
from umwelt.data import load
d = load("lawful", 1280)
gt = [k for k in d if k.startswith("gt_")]
Ttr = probes.targets({k: np.asarray(d[k][:1024]) for k in gt}); Tte = probes.targets({k: np.asarray(d[k][1024:]) for k in gt})
R = "/Users/henryvu/personal/umwelt/results/runs/main/lawful_s0/"
for mind in ("human", "memoryless"):
    sv = pickle.load(open(R + f"{mind}_params.pkl", "rb")); p = presets.PRESETS[mind]
    o = presets.observe(d, p, dtype=np.float16)
    for i in range(0, len(o), 128):
        o[i:i + 128] = (o[i:i + 128].astype(np.float32) - sv["mu"]) / sv["sd"]
    a = presets.actions(d, p)
    Hpr = learner.batched(learner._latents_jit, sv["params"], o[:1024], a[:1024]); Hte = learner.batched(learner._latents_jit, sv["params"], o[1024:], a[1024:])
    err = learner.batched(learner._errors_jit, sv["params"], o[1024:], a[1024:])
    st = json.load(open(R + f"{mind}.json"))
    print(f"{mind}: test_error recomputed {err[:, 16:].mean():.6f} stored {st['test_error']:.6f}", flush=True)
    got = probes.run_battery(Hpr, Ttr, Hte, Tte, 0)
    for c in probes.CONCEPTS:
        print(f"   {c:20s} recomputed {got[c]:+.5f} stored {st['probes'][c]:+.5f} diff {got[c]-st['probes'][c]:+.1e}", flush=True)
