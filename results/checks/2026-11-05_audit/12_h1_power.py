"""Audit: could H1 (object_hidden of human exceeds memoryless by > 0.05, pooled over every hidden object) pass at all
with this probe? Positive control: give the probe a perfect explicit memory (copies of the raw human senses from
earlier steps) and see how far pooled object_hidden rises over raw senses now. Lawful world, data seed 0, the
runner's probe exactly (run_battery: 1,024 fitting episodes, stride 8 / 4, MLP 256)."""
import sys
import numpy as np
sys.path.insert(0, "/Users/henryvu/personal/umwelt/src")
from umwelt import learner, presets, probes
from umwelt.data import load
d = load("lawful", 1280)
p = presets.PRESETS["human"]
o = presets.observe(d, p)
mu, sd = learner.standardizer(o[:1024])
o = (o - mu) / sd
gt = [k for k in d if k.startswith("gt_")]
Ttr = probes.targets({k: np.asarray(d[k][:1024]) for k in gt}); Tte = probes.targets({k: np.asarray(d[k][1024:]) for k in gt})
def stack(lags):
    return np.concatenate([np.roll(o, L, axis=1) for L in lags], -1)  # rows used start at t >= 10, so no wraparound for lags <= 10
for lags in ((0,), (0, 2, 4), (0, 3, 6, 9)):
    H = stack(lags)
    r = probes.run_battery(H[:1024], Ttr, H[1024:], Tte, 0)
    print(f"raw human senses, lags {lags}: object_hidden {r['object_hidden']:+.3f} object_seen {r['object_seen']:+.3f} self_location {r['self_location']:+.3f} elapsed {r['elapsed']:+.3f} number {r['number']:+.3f}", flush=True)
    del H
