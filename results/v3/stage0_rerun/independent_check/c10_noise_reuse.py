"""stage0_rescore.py calls presets.observe separately on the fitting slice and the held-out slice, and observe seeds its
noise by the preset's name and restarts it per call: so held-out episode j gets the same sensor noise as fitting episode j
(j < 256), and not the noise the run itself gave it. Check that, and how much it moves the human's scores (lawful s0)."""
import json, pickle
from pathlib import Path
import numpy as np
REPO = Path("/Users/henryvu/personal/simulacrum")
from umwelt import learner, presets, probes
from umwelt.data import load
d = load("lawful", 1280, seed=0)
p = presets.PRESETS["human"]
fit = {k: np.asarray(v[:1024]) for k, v in d.items()}; te = {k: np.asarray(v[1024:1280]) for k, v in d.items()}
o_fit, o_te = presets.observe(fit, p), presets.observe(te, p)
o_full = presets.observe(d, p)  # as the run itself built it
clean = lambda dd: np.concatenate([presets._channel({k: v for k, v in dd.items()}, s) for s in p.senses], -1)
n_fit = o_fit[:256] - clean({k: v[:256] for k, v in fit.items() if k.startswith("obs_") or k in ("action", "gt_light")})
n_te = o_te - clean({k: v for k, v in te.items() if k.startswith("obs_") or k in ("action", "gt_light")})
print("held-out noise == fitting episodes 0-255 noise:", bool(np.array_equal(n_fit, n_te)), "| max |diff|", float(np.abs(n_fit - n_te).max()))
print("fitting slice noise == the run's own noise:", bool(np.array_equal(o_fit, o_full[:1024])), "| held-out == the run's own:", bool(np.array_equal(o_te, o_full[1024:])))
# effect on the human's rescored numbers: held-out observations with the run's own noise instead
with open(REPO / "results/runs/main/lawful_s0/human_params.pkl", "rb") as f: sv = pickle.load(f)
Ttr, Tte = probes.targets({k: v for k, v in fit.items() if k.startswith("gt_")}), probes.targets({k: v for k, v in te.items() if k.startswith("gt_")})
def R(T, s): return {k: T[k][:, 10::s].reshape(-1, *T[k].shape[2:]) for k in ("ego", "present", "vis", "since")}
Rtr, Rte = R(Ttr, 8), R(Tte, 4); m2 = lambda m: np.repeat(m, 2, 1)
ytr, yte = Rtr["ego"].reshape(-1, 16), Rte["ego"].reshape(-1, 16); groups = np.repeat(np.arange(1024), 31)
base = np.array([Rtr["ego"][Rtr["present"][:, k], k, j].mean() for k in range(8) for j in range(2)])
def sc(Xf, Xt):
    pr = probes.RidgeProbe(Xf, ytr, m2(Rtr["present"]), "reg", 0, 0, groups)(Xt); pres, vis, since = Rte["present"], Rte["vis"], Rte["since"]
    out = {"object_hidden": probes._r2_cols(pr, yte, m2(pres & ~vis), m2(pres))}
    for b, lo, hi in (("1-5", 1, 5), ("6-20", 6, 20)):
        m = pres & ~vis & (since >= lo) & (since <= hi)
        a, c = probes._r2_cols(pr, yte, m2(m), m2(pres)), probes._r2_cols(np.broadcast_to(base, pr.shape), yte, m2(m), m2(pres)); out[b] = (a - c) / (1 - c)
    return out
a = presets.actions(fit, p); at = presets.actions(te, p)
H = lambda o, act: learner.batched(learner._latents_jit, sv["params"], (o - sv["mu"]) / sv["sd"], act)
Hf = H(o_fit, a)[:, 10::8].reshape(-1, 128)
for label, ot in (("rescore's held-out noise (= fitting eps 0-255)", o_te), ("the run's own held-out noise", o_full[1024:])):
    tr = sc(Hf, H(ot, at)[:, 10::4].reshape(-1, 128)); rw = sc(o_fit[:, 10::8].reshape(-1, 311), ot[:, 10::4].reshape(-1, 311))
    print(f"{label:48} trained " + " ".join(f"{k} {v:+.4f}" for k, v in tr.items()) + " | raw " + " ".join(f"{k} {v:+.4f}" for k, v in rw.items()))
