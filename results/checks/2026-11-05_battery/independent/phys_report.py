import pickle
import numpy as np
from common import BYSTANDER_SLOTS
rows = pickle.load(open("phys.pkl", "rb"))
print("agent: smallest distance to any internal wall per test")
for test in ("permanence", "solidity", "causality"):
    a = [r[4] for r in rows if r[0][0] == test and r[2] == "agent"]
    print(f"  {test:10} {min(a):.3f}")
print("objects touching a boundary wall:", [r for r in rows if r[3] == "boundary"][:5])
print("object-to-wall clearance (min distance - r), per test, variant and wall slot (min over quads):")
for test in ("permanence", "solidity", "causality"):
    for v in ("P1", "P2", "I1", "I2"):
        line = []
        for i in range(3):
            g = [r[4] - r[5] for r in rows if r[0][0] == test and r[1] == v and r[3] == i]
            role = "bys" if i in BYSTANDER_SLOTS[test] else ("event" if not (test == "solidity" and i == 1) else "screen")
            line.append(f"slot{i}({role}) {min(g):+.3f}" if g else f"slot{i} (no object)")
        print(f"  {test:10} {v}: " + ", ".join(line))
