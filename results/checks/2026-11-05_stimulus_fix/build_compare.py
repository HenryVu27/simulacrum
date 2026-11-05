"""Independent check: build the surprise suite with the pre-fix voe module (from git) and the current one,
then compare raw sense arrays and every preset's observation stream."""
import sys, time, pickle
import numpy as np
sys.path.insert(0, sys.argv[1])  # scratch/mods for voe_old
import voe_old
from umwelt import voe, presets

t = time.time()
old = voe_old.build_suite()
new = voe.build_suite()
print("built both suites in %.1fs" % (time.time() - t))
print("old HIDDEN_WALL", voe_old.HIDDEN_WALL, "new", voe.HIDDEN_WALL)
with open(sys.argv[2], "wb") as f:
    pickle.dump({"old": old, "new": new}, f)

for sc in old:
    do, co = old[sc]
    dn, cn = new[sc]
    assert np.array_equal(co, cn)
    print(f"== {sc}: shapes", {k: v.shape for k, v in do.items() if k.startswith('obs_')}, "crit", co.tolist()[:6])
    for k in do:
        a, b = do[k], dn[k]
        same = np.array_equal(a.view(np.uint8) if a.dtype != np.float32 else a, b.view(np.uint8) if b.dtype != np.float32 else b)
        if not same:
            diff = a != b
            print(f"   raw {k}: {diff.sum()} of {diff.size} entries differ ({100*diff.mean():.3f}%), movies with a difference: {np.unique(np.nonzero(diff)[0]).size}")
for name, p in presets.PRESETS.items():
    row = []
    for sc in old:
        a = presets.observe(old[sc][0], p)
        b = presets.observe(new[sc][0], p)
        if a.tobytes() == b.tobytes():
            row.append(f"{sc}: identical")
        else:
            d = a != b
            row.append(f"{sc}: {100*d.mean():.3f}% entries differ (max abs {np.abs(a-b).max():.4f})")
    print(f"{name:15}", " | ".join(row))
