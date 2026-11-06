"""The lawful mind's gate, recomputed from the stored networks (results/v3/stage0_rerun/{lawful,twin}_params.pkl) on the
256 held-out episodes of dark data seed 3, with the return mask written here; the channel checked against the truth on
many episodes; and two diagnostics (not part of the gate): returns split into first-lit steps and lit re-entries, and the
lawful mind given a wrong channel (another episode's), to see whether it uses the channel at all."""
import importlib.util, pickle, json
from pathlib import Path
import numpy as np
REPO = Path("/Users/henryvu/personal/simulacrum")
spec = importlib.util.spec_from_file_location("rr", REPO / "experiments/stage0_rerun.py"); rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
from umwelt import learner, presets, probes
from umwelt.data import load
from umwelt.world import ARENA
d = load("dark", 8448, seed=3)
raw = np.load(REPO / "results/v3/stage0_rerun/raw_obs.npy", mmap_mode="r")
TE0 = 8192; te = slice(TE0, TE0 + 256)
# 1. channel against the truth: own computation of egocentric positions by kind, on 512 episodes (256 fit, 256 held out)
def own_truth(sl):
    pos, ap, th = np.asarray(d["gt_obj_pos"][sl], np.float64), np.asarray(d["gt_agent_pos"][sl], np.float64), np.asarray(d["gt_agent_th"][sl], np.float64)
    act, kind = np.asarray(d["gt_obj_active"][sl]), np.asarray(d["gt_obj_kind"][sl])[:, 0].astype(int)
    rel = pos - ap[:, :, None]
    c, s = np.cos(th)[:, :, None], np.sin(th)[:, :, None]
    ego = np.stack([c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]], -1) / ARENA * 4
    out = np.zeros(pos.shape[:2] + (8, 3))
    for e in range(len(kind)):
        for slot, k in enumerate(kind[e]):
            out[e, :, k, :2] = np.where(act[e, :, slot, None], ego[e, :, slot], 0.0)
            out[e, :, k, 2] = act[e, :, slot]
    return out.reshape(*out.shape[:2], 24)
for sl in (slice(0, 256), te):
    ch = rr.channel_from_oracle(d["obs_oracle"][sl])
    dev = np.abs(ch - own_truth(sl))
    print(f"channel vs own truth, episodes {sl.start}-{sl.stop-1}: max |dev| {dev.max():.4f}, 99.9th pct {np.percentile(dev, 99.9):.4f}; flags exact: {bool((ch[..., 2::3] == own_truth(sl)[..., 2::3]).all())}")
chan_te = rr.channel_from_oracle(d["obs_oracle"][te])
# 2. returns mask, own implementation
vis = np.asarray(d["gt_vis180"][te]) & np.asarray(d["gt_obj_active"][te])
E, T, N = vis.shape
since = np.full((E, T, N), 1000)
for e in range(E):
    for n in range(N):
        last = None
        for t in range(T):
            if vis[e, t, n]: last = t
            since[e, t, n] = 1000 if last is None else min(t - last, 1000)
ret = np.zeros((E, T - 1), bool)
for t in range(T - 1):
    ret[:, t] = (vis[:, t + 1] & ~vis[:, t] & (since[:, t] >= 3) & (since[:, t] < 1000)).any(-1)
ret &= (np.arange(T - 1) >= 16)[None]
lit = np.asarray(d["gt_light"][te])
first_lit = lit[:, 1:] & ~lit[:, :-1]
print(f"return steps: {ret.sum()} ({ret.mean():.2%} of steps); of which first lit step {(ret & first_lit).sum()}, lit re-entry {(ret & ~first_lit).sum()}")
acts = presets.actions(d, presets.PRESETS["human"])[te]
def errs(name, extra):
    with open(REPO / f"results/v3/stage0_rerun/{name}_params.pkl", "rb") as f: sv = pickle.load(f)
    o = raw[te] if extra is None else np.concatenate([raw[te], extra.astype(np.float16)], -1)
    o = ((o.astype(np.float32) - sv["mu"]) / sv["sd"]).astype(np.float16)  # as the script: stored back into a float16 memmap
    return learner.batched(rr.sense_errors, sv["params"], o, acts)
res = {}
res["twin"] = errs("twin", None)
res["lawful"] = errs("lawful", chan_te)
rng = np.random.default_rng(0)
res["lawful, other episode's channel"] = errs("lawful", chan_te[np.roll(np.arange(256), 1)])
res["lawful, channel zeroed"] = errs("lawful", np.zeros_like(chan_te))
for k, e in res.items():
    print(f"{k:34} returns {e[ret].mean():.5f}  first-lit {e[ret & first_lit].mean():.5f}  lit re-entry {e[ret & ~first_lit].mean():.5f}  all after burn-in {e[:, 16:].mean():.5f}")
r = res["lawful"][ret].mean() / res["twin"][ret].mean()
print(f"gate ratio lawful/twin at returns: {r:.4f} (stored 0.99155), needs <= 0.90: passes {r <= 0.9}")
print(f"lit re-entry only ratio: {res['lawful'][ret & ~first_lit].mean() / res['twin'][ret & ~first_lit].mean():.4f}")
