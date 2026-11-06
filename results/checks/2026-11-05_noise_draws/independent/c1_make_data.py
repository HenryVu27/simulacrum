"""Make test inputs once: the surprise suite and a 300-episode lawful + dark recording (crosses the 128-episode chunk)."""
import sys, pickle, numpy as np
from umwelt import voe
from umwelt.data import generate
out = sys.argv[1]
suite = voe.build_suite()
rec = {"lawful": generate("lawful", 300, T=24, seed=3, chunk=128), "dark": generate("dark", 140, T=24, seed=4, chunk=128)}
with open(out, "wb") as f:
    pickle.dump({"suite": suite, "rec": rec, "hash": voe.suite_hash(suite)}, f)
print("suite hash", voe.suite_hash(suite), {k: v[0]["action"].shape for k, v in suite.items()}, {k: v[1].tolist() for k, v in suite.items()})
