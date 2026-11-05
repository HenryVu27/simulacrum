"""The balance claim is for raw noise-free units. The networks see (obs - mu) / sd with each mind's training mu, sd.
Copy-predictor S in that space, noise-free, per test, for the five minds (main/lawful_s0 and v2/dark_s0)."""
import pickle
from pathlib import Path
import numpy as np
from common import S, voe
from umwelt import presets
from balance import noise_free

LIVE = Path("/Users/henryvu/personal/umwelt/results/runs")
suite = pickle.load(open(S / "suite_new.pkl", "rb"))["new"]
for run in ("main/lawful_s0", "v2/dark_s0"):
    for name in ("human", "vision_only", "dog", "bat", "chimera"):
        saved = pickle.load(open(LIVE / run / f"{name}_params.pkl", "rb"))
        row = []
        for test, (d, crit) in suite.items():
            raw = presets.observe(d, noise_free(presets.PRESETS[name]))
            st = (raw - saved["mu"]) / saved["sd"]
            noisy = (presets.observe(d, presets.PRESETS[name]) - saved["mu"]) / saved["sd"]
            vals = [voe.surprise(voe.copy_errors(o), crit, n_boot=1)[0] for o in (raw, st, noisy)]
            row.append(f"{test} raw {vals[0]:+.4f} std {vals[1]:+.4f} std+noise {vals[2]:+.4f}")
        print(f"{run:15} {name:12} " + " | ".join(row), flush=True)
