"""Is the new marker equivalent to no wall at all for every non-oracle sense? Compare with a marker 500 units away,
and run the new test's logic on the old geometry (it should fail)."""
import pickle, sys
import numpy as np
from umwelt import voe
suites = pickle.load(open(sys.argv[1], "rb"))
keep = voe.HIDDEN_WALL
voe.HIDDEN_WALL = np.array([-500.0, -500.0, -499.0, -500.0], np.float32)
far = voe.build_suite()
voe.HIDDEN_WALL = keep
for sc in far:
    for k, v in far[sc][0].items():
        if k == "obs_oracle":
            continue
        same = np.array_equal(v, suites["new"][sc][0][k])
        if not same:
            print("DIFFERS", sc, k)
print("far-away marker vs new marker: all non-oracle raw senses identical in all three scenarios (unless DIFFERS above)")
# the test's logic on the old geometry
makers = [lambda: voe.permanence(voe.INERT[0], -0.5), lambda: voe.solidity(voe.INERT[0], 0.5),
          lambda: voe.causality(voe.INERT[0], voe.INERT[1], -0.5)]
voe.HIDDEN_WALL = np.array([0.5, 0.0, 1.5, 0.0], np.float32)
base = [m()[0] for m in makers]
voe.HIDDEN_WALL = voe.HIDDEN_WALL + np.array([0.0, -5.0, 0.0, -5.0], np.float32)
moved = [m()[0] for m in makers]
bad = [(i, v, k) for i, (A, B) in enumerate(zip(base, moved)) for v, (va, vb) in enumerate(zip(A, B)) for k in va if k != "oracle" and not np.array_equal(va[k], vb[k])]
print("old geometry, test logic: non-oracle differences", bad)
