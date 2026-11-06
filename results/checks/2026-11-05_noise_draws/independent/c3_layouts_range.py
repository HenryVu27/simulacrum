"""Committed preregistration: 'Across layouts a score moves by at most 0.68 of its own 95 percent interval width'
(one-draw intervals). The same ratio with 20-draw scores and the new interval, lawful s0, layouts 0..4."""
import sys, json
import numpy as np
rows = json.load(open(sys.argv[1]))
for label, Sk, ck in (("20 draws, new interval", "S", "ci"), ("one draw (draw 0), old interval", "S1", "ci1")):
    r = []
    for x in rows:
        s = np.array(x[Sk][:5]); w = x[ck][0][1] - x[ck][0][0]
        r.append(((s.max() - s.min()) / w, x["mind"], x["net"], x["test"]))
    r.sort(reverse=True); a = np.array([v[0] for v in r])
    print(f"{label}: spread over layouts 0-4 / interval width on layout 0: max {a.max():.2f} ({r[0][1]} {r[0][2]} {r[0][3]}), median {np.median(a):.2f}")
