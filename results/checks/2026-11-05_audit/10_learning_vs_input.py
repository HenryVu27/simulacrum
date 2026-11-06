"""Audit: how much of what each mind 'knows' (trained probe score) is already in its raw current senses, or in an
untrained random network fed those senses? Lawful seed 0, oracle excluded, MLP probes."""
import json
from pathlib import Path
import numpy as np
R = Path("/Users/henryvu/personal/umwelt/results/runs/main/lawful_s0")
C = ("object_seen", "object_hidden", "object_now", "object_past", "object_allo_seen", "object_allo_hidden", "number", "self_motion", "self_location", "cause", "elapsed", "relations")
rec = {f.stem: json.loads(f.read_text()) for f in R.glob("*.json") if f.stem not in ("translation",)}
tr, raw, un = [], [], []
for m, r in rec.items():
    if m == "oracle": continue
    for c in C:
        tr.append(r["probes"][c]); raw.append(r["probes_raw"][c]); un.append(r["probes_untrained"][c])
tr, raw, un = map(np.array, (tr, raw, un))
print(f"across 13 minds x 12 concepts: r(trained, raw senses) = {np.corrcoef(tr, raw)[0,1]:.3f}; r(trained, untrained) = {np.corrcoef(tr, un)[0,1]:.3f}")
print(f"R^2 of trained on max(raw, untrained): {np.corrcoef(tr, np.maximum(raw, un))[0,1]**2:.3f}")
d = tr - np.maximum(raw, un)
print(f"trained minus the better of (raw senses, untrained network): median {np.median(d):+.3f}, share above +0.05: {np.mean(d > 0.05):.2f}")
print("per concept, trained minus max(raw, untrained), by mind:")
minds = [m for m in rec if m != "oracle"]
for i, c in enumerate(C):
    vals = {m: rec[m]["probes"][c] - max(rec[m]["probes_raw"][c], rec[m]["probes_untrained"][c]) for m in minds}
    top = sorted(vals.items(), key=lambda kv: -kv[1])[:3]
    print(f"  {c:20s} median {np.median(list(vals.values())):+.3f} largest: " + ", ".join(f"{m} {v:+.3f}" for m, v in top))
