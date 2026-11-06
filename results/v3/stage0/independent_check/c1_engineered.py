"""Check 1: recompute the engineered fixtures (pre-embedding) from raw ground-truth arrays with plain loops,
compare with stage0.engineered on a sample of episodes. Also: kind uniqueness per episode, and how far hidden
objects move while hidden in the dark world (relevant to 'last seen, updated' and to the movies)."""
import importlib.util, sys
import numpy as np
from pathlib import Path
REPO = Path("/Users/henryvu/personal/umwelt")
spec = importlib.util.spec_from_file_location("stage0", REPO / "experiments/stage0.py")
s0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s0)
from umwelt.data import load
from umwelt import probes

d = load("dark", 8448, seed=0)
rng = np.random.default_rng(123)
eps = np.sort(rng.choice(np.r_[np.arange(8192), np.arange(8192, 8448)], 60, replace=False))
gt = {k: np.asarray(v[eps]) for k, v in d.items() if k.startswith("gt_")}
times = np.arange(probes.PAST, 256, 8)
pos, apos, th = gt["gt_obj_pos"], gt["gt_agent_pos"], gt["gt_agent_th"]
act, vis, kind = gt["gt_obj_active"], gt["gt_vis180"], gt["gt_obj_kind"]
print("kind constant over time per slot:", bool((kind == kind[:, :1]).all()))
E, T, N = act.shape
dup = 0
for e in range(E):
    ks = kind[e, 0][act[e].any(0)]
    dup += len(ks) != len(set(ks.tolist()))
print("episodes in sample with a repeated kind among ever-active slots:", dup, "of", E)

def ego(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])

mine = {w: np.zeros((E, len(times), 8, 3), np.float32) for w in ("full", "last seen", "last seen, updated")}
for e in range(E):
    for ti, t in enumerate(times):
        for n in range(N):
            k = int(kind[e, 0, n])
            if act[e, t, n]:
                mine["full"][e, ti, k] = [*(ego(pos[e, t, n] - apos[e, t], th[e, t]) / 10 * 4), 1.0]
            seen_ts = np.flatnonzero(vis[e, : t + 1, n])
            if len(seen_ts) and act[e, t, n]:
                tl = seen_ts[-1]
                mine["last seen"][e, ti, k] = [*(ego(pos[e, tl, n] - apos[e, tl], th[e, tl]) / 10 * 4), 1.0]
                mine["last seen, updated"][e, ti, k] = [*(ego(pos[e, tl, n] - apos[e, t], th[e, t]) / 10 * 4), 1.0]
for w in mine:
    theirs = s0.engineered(gt, w, times)
    diff = np.abs(theirs - mine[w].reshape(E, len(times), -1))
    print(f"{w:20s} max abs diff {diff.max():.2e}  nonzero entries {np.count_nonzero(mine[w])}")

# How much do hidden objects move while out of view? (all 8448 episodes, every 4th step)
P = np.asarray(d["gt_obj_pos"][:2000]); V = np.asarray(d["gt_vis180"][:2000]); A = np.asarray(d["gt_obj_active"][:2000])
ls = s0.last_seen({"gt_vis180": V})
e_, t_, n_ = np.nonzero(A & ~V & (ls >= 0))
moved = np.linalg.norm(P[e_, t_, n_] - P[e_, ls[e_, t_, n_], n_], axis=-1)
since = t_ - ls[e_, t_, n_]
for lo, hi in ((1, 5), (6, 20), (21, 60)):
    m = (since >= lo) & (since <= hi)
    print(f"hidden {lo}-{hi} steps: room displacement since last seen, median {np.median(moved[m]):.3f}, mean {moved[m].mean():.3f}, "
          f"90th pct {np.percentile(moved[m], 90):.3f}, share > 0.5: {np.mean(moved[m] > 0.5):.3f} (n={m.sum()})")
light = np.asarray(d["gt_light"][:2000])
print("share of hidden (ever seen) moments that are dark:", float(np.mean(~light[e_, t_])))
