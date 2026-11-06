"""Dependence between runs of one mind: draw k is the same noise in every run (seed = crc32(preset name), same movies).
Per mind and test, correlation across draws 0..19 of single-draw S between runs, and the shared part of the 20-draw
residual (SD of the 20-draw mean's noise that is common to all runs)."""
import sys
from pathlib import Path
import numpy as np
T = Path(sys.argv[1])
def S(t):
    P = t[..., 0].sum(-1) + t[..., 1].sum(-1)
    return (t[..., 2].sum(-1) + t[..., 3].sum(-1) - P) / P
RUNS = ["main/lawful_s0", "main/lawful_s1", "main/ghost_s0", "main/impermanent_s0", "main/nonlocal_s0", "main/lawful_mj_s0", "v2/dark_s0", "v2/dark_s1"]
print("trained nets, single-draw S per draw; corr = median pairwise correlation over the 8 runs; shared SD = SD over draws of the across-run mean minus its expected independent part")
for mind in ("smell_only", "vision_only", "human", "bat", "chimera", "dog", "touch_only", "hearing_only", "memoryless"):
    for test in ("permanence", "solidity", "causality"):
        X = np.array([S(np.load(T / f"{r.replace('/', '__')}__{mind}__0_20.npz")[f"trained/{test}/t64"]) for r in RUNS])  # (runs, 20)
        C = np.corrcoef(X)
        iu = np.triu_indices(len(RUNS), 1)
        sd1 = X.std(1, ddof=1).mean()
        print(f"  {mind:13} {test:10} single-draw SD {sd1:.4f}; corr across runs median {np.median(C[iu]):+.2f} (min {C[iu].min():+.2f}); "
              f"SD of 20-draw mean {sd1 / np.sqrt(20):.4f}, of which shared by all runs ~{np.sqrt(max(np.median(C[iu]), 0)) * sd1 / np.sqrt(20):.4f}")
# untrained: lawful_s0, dark_s0 (same untrained network: PRNGKey(10000 + seed 0))
import json
for mind in ("chimera", "bat", "human"):
    a = np.load(T / f"main__lawful_s0__{mind}__0_20.npz"); b = np.load(T / f"v2__dark_s0__{mind}__0_20.npz")
    if "untrained/solidity/t64" in a and "untrained/solidity/t64" in b:
        x, y = S(a["untrained/solidity/t64"]), S(b["untrained/solidity/t64"])
        print(f"  untrained {mind} solidity, lawful_s0 vs dark_s0 (same untrained weights, same draws): corr {np.corrcoef(x, y)[0, 1]:+.3f}, 20-draw S {S(a['untrained/solidity/t64'].mean(0)):+.4f} vs {S(b['untrained/solidity/t64'].mean(0)):+.4f}")
