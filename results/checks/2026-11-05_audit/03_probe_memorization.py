"""Audit: probes are fit on hidden states of episodes the learner trained on (v1: about 47 passes over each step).
If a recurrent mind memorized those episodes, a probe fit there can learn mappings that do not transfer.
Compare, with the same probe code and the same number of fitting episodes (128):
  (a) fit on 128 learner-training episodes (the stored *_latents.pkl keep_tr), score on held-out episodes 128..255
  (b) fit on held-out episodes 0..127 (never trained on), score on the same held-out 128..255
Concepts: object_hidden (raw scale), plus skill in the 1-5 and 6-20 steps-since-seen bins."""
import pickle, sys, json
import numpy as np
sys.path.insert(0, "/Users/henryvu/personal/umwelt/src")
from umwelt import probes
from umwelt.data import load

def bins_scores(pred, Y, pres, vis, since, ytr_mean):
    m2 = lambda m: np.repeat(m, 2, axis=1)
    const = np.broadcast_to(ytr_mean.reshape(1, -1), Y.shape)
    out = {"object_hidden": probes._r2_cols(pred, Y, m2(pres & ~vis), m2(pres)), "object_seen": probes._r2_cols(pred, Y, m2(pres & vis), m2(pres))}
    for name, (lo, hi) in (("1-5", (1, 5)), ("6-20", (6, 20))):
        m = pres & ~vis & (since >= lo) & (since <= hi)
        s, b = probes._r2_cols(pred, Y, m2(m), m2(pres)), probes._r2_cols(const, Y, m2(m), m2(pres))
        out[f"skill {name}"] = (s - b) / (1 - b)
    return out

def run(tag, world, seed, data_seed, n_train, minds):
    d = load(world, n_train + 256, seed=data_seed)
    gt = {k: v for k, v in d.items() if k.startswith("gt_")}
    Ttr = probes.targets({k: np.asarray(v[:128]) for k, v in gt.items()})
    Tte = probes.targets({k: np.asarray(v[n_train:n_train + 256]) for k, v in gt.items()})
    Tfit_b = {k: v[:128] for k, v in Tte.items()}; Tev = {k: v[128:] for k, v in Tte.items()}
    for mind in minds:
        Htr, Hte = pickle.load(open(f"/Users/henryvu/personal/umwelt/results/runs/{tag}/{world}_s{seed}/{mind}_latents.pkl", "rb"))
        Htr, Hte = Htr.astype(np.float32), Hte.astype(np.float32)
        res = {}
        for label, Hfit, Tfit in (("(a) fit on training episodes", Htr, Ttr), ("(b) fit on held-out episodes", Hte[:128], Tfit_b)):
            X = probes._prep(Hfit, 8 // 2)  # stride 4 for both so row counts match
            groups = np.repeat(np.arange(Hfit.shape[0]), len(range(probes.PAST, Hfit.shape[1], 4)))
            R = {k: probes._rows(Tfit, k, 4) for k in ("ego", "present")}
            Rv = {k: probes._rows(Tev, k, 4) for k in ("ego", "present", "vis", "since")}
            ytr_mean = np.array([[R["ego"][R["present"][:, k], k, j].mean() if R["present"][:, k].any() else 0.0 for j in range(2)] for k in range(8)])
            p = probes.Probe(X, R["ego"].reshape(len(X), -1), np.repeat(R["present"], 2, 1), "reg", 256, 0, groups)
            Xe = probes._prep(Hte[128:], 4)
            res[label] = bins_scores(p(Xe), Rv["ego"].reshape(len(Xe), -1), Rv["present"], Rv["vis"], Rv["since"], ytr_mean)
        for label, r in res.items():
            print(f"{tag}/{world}_s{seed} {mind:12s} {label}: " + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)

run("main", "lawful", 0, 0, 1024, ["human", "memoryless", "vision_only", "passive_novest"])
run("v2", "dark", 0, 1, 8192, ["human", "memoryless"])
