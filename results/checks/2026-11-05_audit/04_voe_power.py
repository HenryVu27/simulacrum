"""Audit of the permanence surprise test.
1. Reproduce stored surprise scores from stored networks (experiment._voe_scores, 200 draws, 2000 resamples).
2. Power: what S would this same network show if it had object permanence? For each permanence quadruplet,
   D = summed (over the scoring window) per-step squared difference, averaged over the mind's standardized
   observation dims, between the impossible movie's frames and the lawful continuation an object-permanent
   predictor would expect (I1 vs P1: the object should come out; I2 vs P2: nothing should come out).
   'first frame' counts only the first frame where they differ (a mind that updates at once);
   'whole window' counts every frame (a mind that keeps expecting).
   S_ideal ~= sum(D) / sum(err(P1)+err(P2)) of the trained network itself (its own denominator)."""
import json, pickle, sys, time
import numpy as np
sys.path.insert(0, "/Users/henryvu/personal/umwelt/src")
from umwelt import experiment, learner, presets, voe
t0 = time.time()
suite = voe.build_suite()
print("suite hash", voe.suite_hash(suite), "(stored provenance stimuli 016fc3d72807e39d)", f"{time.time()-t0:.0f}s", flush=True)
R = "/Users/henryvu/personal/umwelt/results/runs/"
def stored(run, mind):
    return json.load(open(R + run + f"/{mind}.json"))
for run, mind in (("main/lawful_s0", "human"), ("main/lawful_s0", "smell_only"), ("v2/dark_s0", "human")):
    sv = pickle.load(open(R + run + f"/{mind}_params.pkl", "rb"))
    t1 = time.time()
    got = experiment._voe_scores(sv["params"], suite, presets.PRESETS[mind], sv["mu"], sv["sd"])
    st = stored(run, mind)["voe"]
    for test in voe.TESTS:
        print(f"[reproduce] {run} {mind} {test}: recomputed S {got[test]['S']:+.7f} ci [{got[test]['ci'][0]:+.6f}, {got[test]['ci'][1]:+.6f}] | stored S {st[test]['S']:+.7f} ci [{st[test]['ci'][0]:+.6f}, {st[test]['ci'][1]:+.6f}]", flush=True)
    print(f"   ({time.time()-t1:.0f}s)")

# Power analysis on permanence
d, crit = suite["permanence"]
n = len(crit)
for run, mind in (("main/lawful_s0", "human"), ("main/lawful_s0", "vision_only"), ("main/lawful_s0", "dog"), ("v2/dark_s0", "human"), ("main/lawful_s0", "bat")):
    sv = pickle.load(open(R + run + f"/{mind}_params.pkl", "rb"))
    p = presets.PRESETS[mind]
    clean = presets.Preset(p.name, tuple(presets.Sense(s.kind, s.fov_deg, s.n_rays, s.color, s.gain, 0.0) for s in p.senses), p.uses_action, p.recurrent)
    oc = (presets.observe(d, clean) - sv["mu"]) / sv["sd"]           # noise-free standardized frames
    oc = oc.reshape(n, 4, *oc.shape[1:])                               # (instance, variant P1 P2 I1 I2, T, D)
    a = presets.actions(d, p)
    errs = []
    for k in range(20):
        o = (presets.observe(d, p, draw=k) - sv["mu"]) / sv["sd"]
        errs.append(learner.batched(learner._errors_jit, sv["params"], o, a))
    tot = voe.totals(np.stack(errs), crit).mean(0)                     # (instance, 4)
    possible = tot[:, 0] + tot[:, 1]
    Dw, Df = [], []
    for i in range(n):
        c = crit[i]
        for imp, ref in ((2, 0), (3, 1)):
            diff = ((oc[i, imp, c:] - oc[i, ref, c:]) ** 2).mean(-1)   # frames c..T-1 = targets of error steps c-1..T-2
            Dw.append(diff.sum())
            nz = np.flatnonzero(diff > 1e-9)
            Df.append(diff[nz[0]] if len(nz) else 0.0)
    Dw, Df = np.array(Dw), np.array(Df)
    S_obs = (tot[:, 2] + tot[:, 3] - possible).sum() / possible.sum()
    print(f"[power] {run} {mind}: obs dims {oc.shape[-1]}; possible-movie error {possible.sum():.2f} (per instance {possible.mean():.3f});"
          f" S over 20 draws {S_obs:+.5f}; S_ideal first-frame {Df.sum()/possible.sum():.4f}; S_ideal whole-window {Dw.sum()/possible.sum():.4f};"
          f" instances with any visible difference {np.mean(Dw.reshape(n,2).sum(1) > 0):.2f}", flush=True)
