"""Blind-spot check for the fixed probe (ridge, 1,024 episodes) on the Stage 0 rerun recording (dark, data seed 3): a
fixture holding exactly the information of 'last seen, updated' (each object's room position when last seen, plus the
agent's current pose), but stored allocentrically, so egocentric positions are a nonlinear function of it. Embedded the
same way (s0.embed). Then the same features with products pose x position added (which span the rotation exactly)."""
import importlib.util
from pathlib import Path
import numpy as np
REPO = Path("/Users/henryvu/personal/simulacrum")
spec = importlib.util.spec_from_file_location("s0", REPO / "experiments/stage0.py"); s0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s0)
from umwelt import probes
from umwelt.data import load
from umwelt.world import ARENA
d = load("dark", 8448, seed=3)
ttr, tte = np.arange(10, 256, 8), np.arange(10, 256, 4)
gf = {k: np.asarray(v[:1024]) for k, v in d.items() if k.startswith("gt_")}
gt_ = {k: np.asarray(v[8192:8448]) for k, v in d.items() if k.startswith("gt_")}
def allo(g, times, products=False):
    pos, ap, th = g["gt_obj_pos"], g["gt_agent_pos"], g["gt_agent_th"]
    act, kind = g["gt_obj_active"], g["gt_obj_kind"][:, 0].astype(int)
    E = pos.shape[0]; e = np.arange(E)[:, None, None]; n = np.arange(pos.shape[2])[None, None, :]
    ls = s0.last_seen(g)[:, times]; seen = ls >= 0
    then = pos[e, np.maximum(ls, 0), n] / ARENA * 4  # room position when last seen
    flag = seen & act[:, times]
    then = np.where(flag[..., None], then, 0.0)
    obj = probes._by_kind(np.concatenate([then, flag[..., None].astype(np.float32)], -1).astype(np.float32), kind).reshape(E, len(times), -1)
    a = ap[:, times] / ARENA * 4; c, s = np.cos(th[:, times]), np.sin(th[:, times])
    pose = np.stack([a[..., 0], a[..., 1], c, s], -1)
    f = [obj, pose]
    if products:  # pose x object terms, enough for a linear map to compute the rotation into the agent's frame
        ob = obj.reshape(E, len(times), 8, 3)[..., :2]
        f += [(ob * c[..., None, None]).reshape(E, len(times), -1), (ob * s[..., None, None]).reshape(E, len(times), -1),
              (a[..., None, :] * c[..., None, None]).reshape(E, len(times), -1), (a[..., None, :] * s[..., None, None]).reshape(E, len(times), -1)]
    return np.concatenate(f, -1).astype(np.float32)
Ttr, Tte = probes.targets(gf), probes.targets(gt_)
def R(T, times): return {k: T[k][:, times].reshape(-1, *T[k].shape[2:]) for k in ("ego", "present", "vis", "since")}
Rtr, Rte = R(Ttr, ttr), R(Tte, tte)
m2 = lambda m: np.repeat(m, 2, 1)
ytr, yte = Rtr["ego"].reshape(-1, 16), Rte["ego"].reshape(-1, 16)
groups = np.repeat(np.arange(1024), len(ttr))
base = np.array([Rtr["ego"][Rtr["present"][:, k], k, j].mean() for k in range(8) for j in range(2)])
def r2(pred, m, ref):
    sc, n = [], []
    for c in range(16):
        if m[:, c].sum() < 30 or ref[:, c].sum() < 30: continue
        v = yte[ref[:, c], c].astype(np.float64).var(); sc.append(1 - np.mean((pred[m[:, c], c] - yte[m[:, c], c]) ** 2) / v); n.append(m[:, c].sum())
    return np.average(sc, weights=n)
def curve(pred):
    pres, vis, since = Rte["present"], Rte["vis"], Rte["since"]; const = np.broadcast_to(base, pred.shape); out = {}
    for name, lo, hi in (("in view", None, None), ("1-5", 1, 5), ("6-20", 6, 20), ("21-60", 21, 60)):
        m = pres & vis if lo is None else pres & ~vis & (since >= lo) & (since <= hi)
        a, b = r2(pred, m2(m), m2(pres)), r2(const, m2(m), m2(pres)); out[name] = (a - b) / (1 - b)
    return out
for label, prod in (("last seen, allocentric (same information as 'last seen, updated')", False), ("same, with pose x position products", True)):
    A, B = s0.embed(allo(gf, ttr, prod), allo(gt_, tte, prod))
    pred = probes.RidgeProbe(A.reshape(-1, 128), ytr, m2(Rtr["present"]), "reg", 0, 0, groups)(B.reshape(-1, 128))
    print(f"{label:72} ridge skill: " + "  ".join(f"{k} {v:+.3f}" for k, v in curve(pred).items()))
A, B = s0.embed(s0.engineered(gf, "last seen, updated", ttr), s0.engineered(gt_, "last seen, updated", tte))
pred = probes.RidgeProbe(A.reshape(-1, 128), ytr, m2(Rtr["present"]), "reg", 0, 0, groups)(B.reshape(-1, 128))
print(f"{'last seen, updated (egocentric, as in Stage 0)':72} ridge skill: " + "  ".join(f"{k} {v:+.3f}" for k, v in curve(pred).items()))
