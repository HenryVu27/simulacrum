"""Which stored minds' solidity (and other) intervals exclude 0 on the new movies; and the S_copy the networks' own
(standardised, noisy) inputs give."""
import json
from pathlib import Path
rec = json.loads(Path("repo_c35/results/battery_fix_recompute.json").read_text())
rows = rec["scores"]
print("scores:", len(rows), "stimuli", rec["stimuli"])
for test in ("solidity", "permanence", "causality"):
    for net in ("trained", "untrained"):
        sel = [r for r in rows if r["test"] == test and r["net"] == net]
        pos = [r for r in sel if r["ci_after"][0] > 0]
        neg = [r for r in sel if r["ci_after"][1] < 0]
        pos_b = [r for r in sel if r["ci_before"][0] > 0]
        print(f"{test:10} {net:9}: {len(sel)} scores; interval above 0 after: {len(pos)} (before: {len(pos_b)}); below 0 after: {len(neg)}")
        if test == "solidity":
            for r in pos:
                print(f"     above 0: {r['run']:20} {r['mind']:15} S {r['S_after']:+.4f} [{r['ci_after'][0]:+.4f}, {r['ci_after'][1]:+.4f}]  S_copy {r['S_copy_after']:+.4f}")
print("\nS_copy (standardised, noisy inputs, as stored) on solidity after the redesign, by mind (min..max over runs):")
by = {}
for r in rows:
    if r["test"] == "solidity" and r["net"] == "trained":
        by.setdefault(r["mind"], []).append(r["S_copy_after"])
for m, v in sorted(by.items()):
    print(f"  {m:15} {min(v):+.4f} .. {max(v):+.4f}")
