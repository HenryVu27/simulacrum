"""How much would memory help? Probe raw senses at one moment versus a stack of the last 32 steps.

If the stacked history scores far higher on hidden objects and number, the information is there in
the sensory past, and a learner that stays reactive is leaving it unused.
"""
import json
import sys

import numpy as np

from umwelt import presets, probes
from umwelt.data import load

mind = sys.argv[1] if len(sys.argv) > 1 else "human"
d = load("lawful", 1280)
o = presets.observe(d, presets.PRESETS[mind], dtype=np.float16)
mu, sd = o[:1024].reshape(-1, o.shape[-1]).astype(np.float32).mean(0), o[:1024].reshape(-1, o.shape[-1]).astype(np.float32).std(0) + 1e-3
ts = np.arange(0, 256, 4)
lags = [0, 8, 16, 24]
now = ((o[:, ts].astype(np.float32) - mu) / sd).astype(np.float16)
stack = np.concatenate([((o[:, np.maximum(ts - L, 0)].astype(np.float32) - mu) / sd).astype(np.float16) for L in lags], -1)
gt = {k: np.asarray(v) for k, v in d.items() if k.startswith("gt_")}
T_all = probes.targets(gt)
T_sub = {k: v[:, ts] for k, v in T_all.items()}
tr, te = slice(0, 1024), slice(1024, 1280)
Ttr = {k: v[tr] for k, v in T_sub.items()}
Tte = {k: v[te] for k, v in T_sub.items()}
out = {}
for name, H in (("now", now), ("last_24_steps_4_snapshots", stack)):
    out[name] = probes.run_battery(H[tr], Ttr, H[te], Tte, seed=0, stride_tr=1, stride_te=1)
    print(name, " ".join(f"{c}={v:.2f}" for c, v in out[name].items()), flush=True)
json.dump(out, open(f"results/memory_headroom_{mind}.json", "w"), indent=1)
