"""Claim 6 probe: the trained oracle's permanence S (main/lawful_s0) when the three internal walls are put in other
slot orders (training draws the three walls i.i.d., so slot order carries no meaning there)."""
import itertools
import numpy as np
from common import ORIG_RENDER, Capture, voe
exec(compile((open("oracle_layouts.py").read().split("for run in (")[0]), "ol", "exec"))

calls = []
for k in voe.INERT:
    for j in (-0.5, 0.0, 0.5):
        with Capture(voe) as cap:
            _, tc = voe.permanence(k, j)
        calls.append((cap.calls, tc))
for perm in itertools.permutations(range(3)):
    movies, crit = [], []
    for cs, tc in calls:
        movies.append([ORIG_RENDER(c["agent"], c["th"], c["internal"][list(perm)], c["kind"], c["pos"].astype(np.float32), c["active"], c["impulse"]) for c in cs])
        crit.append(tc)
    T = min(len(mv[0]["oracle"]) for mv in movies)
    d = {f"obs_{k}": np.stack([np.stack([mv[v][k][:T] for v in range(4)]) for mv in movies]).astype(np.float16) for k in OBS_KEYS}
    d = {k: v.reshape(-1, *v.shape[2:]) for k, v in d.items()}
    d["action"] = np.zeros((len(movies) * 4, T, 2), np.float32)
    s, ci = S_of("main/lawful_s0", "oracle", d, np.array(crit))
    print(f"slot order {perm} (screen in slot {perm.index(0)}): S {s:+.4f} [{ci[0]:+.3f}, {ci[1]:+.3f}]", flush=True)
