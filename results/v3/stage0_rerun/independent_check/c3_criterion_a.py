"""Criterion (a) for ridge at 1,024 episodes, recomputed with an independent per-episode bootstrap (own sums code),
from c2's saved Mac predictions; compared with scores.json. Also with other bootstrap RNG seeds."""
import json
from pathlib import Path
import numpy as np
REPO = Path("/Users/henryvu/personal/simulacrum"); OUTD = Path(__file__).parent / "out"
t = np.load(OUTD / "c2_targets.npz"); yte, pres, vis, loc, ep = t["yte"].astype(np.float64), t["pres"], t["vis"], t["loc"].astype(np.float64), t["ep"]
m2 = lambda m: np.repeat(m, 2, 1)
def ep_stats(pred, Y, m, ref):
    # per episode and column: sse over m, n over m, sum Y, sum Y^2, n over ref
    out = []
    for a in (np.where(m, (pred - Y) ** 2, 0.0), m.astype(float), np.where(ref, Y, 0.0), np.where(ref, Y * Y, 0.0), ref.astype(float)):
        out.append(np.stack([a[ep == e].sum(0) for e in range(256)]))
    return out
def r2(st, w):
    sse, nm, s1, s2, nr = (np.einsum("e,ec->c", w, x) for x in st)
    ok = (nm >= 30) & (nr >= 30)
    var = s2[ok] / nr[ok] - (s1[ok] / nr[ok]) ** 2
    return np.sum(nm[ok] * (1 - sse[ok] / nm[ok] / var)) / nm[ok].sum()
def metrics(f, s):
    p = np.load(OUTD / f"c2_{f}_s{s}.npz")
    ones = np.ones_like(loc, bool)
    return {"object_now": ep_stats(p["pp"], yte, m2(pres), m2(pres)), "object_hidden": ep_stats(p["pp"], yte, m2(pres & ~vis), m2(pres)),
            "object_seen": ep_stats(p["ps"], yte, m2(pres & vis), m2(pres)), "self_location": ep_stats(p["pl"], loc, ones, ones)}
S = {(f, s): metrics(f, s) for f in ("raw", "b24", "b369") for s in (0, 1, 2)}
stored = json.loads((REPO / "results/v3/stage0_rerun/scores.json").read_text())["ridge"]["buffer minus raw"]
NAME = {"b24": "buffer 2,4", "b369": "buffer 3,6,9"}
ok_all = True
for bseed_label, W in (("as scripted (default_rng(b), b=0..999)", [np.bincount(np.random.default_rng(b).integers(0, 256, 256), minlength=256) for b in range(1000)]),
                       ("fresh RNG 777, 4000 resamples", [np.bincount(r, minlength=256) for r in np.random.default_rng(777).integers(0, 256, (4000, 256))])):
    print("bootstrap:", bseed_label)
    W = np.array(W, float)
    for f in ("b24", "b369"):
        for k in ("object_now", "object_hidden", "object_seen", "self_location"):
            dif = lambda w: np.mean([r2(S[(f, s)][k], w) - r2(S[("raw", s)][k], w) for s in (0, 1, 2)])
            point = dif(np.ones(256)); boots = np.array([dif(w) for w in W]); lo, hi = np.percentile(boots, [2.5, 97.5])
            sp, (slo, shi) = stored[f"{NAME[f]}, {k}"]
            ok_all &= hi >= 0
            print(f"  {NAME[f]:13} {k:14} mine {point:+.5f} [{lo:+.5f}, {hi:+.5f}]  stored {sp:+.5f} [{slo:+.5f}, {shi:+.5f}]  upper>=0 {hi >= 0}")
print("criterion a holds in every check:", ok_all)
