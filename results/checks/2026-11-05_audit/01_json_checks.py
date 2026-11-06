"""Audit: recompute v1 aggregate numbers straight from results/runs/main/*/*.json (no repo code)."""
import json, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np

R = Path("/Users/henryvu/personal/umwelt/results/runs")
CONCEPTS = ("object_seen", "object_hidden", "object_now", "object_past", "object_allo_seen", "object_allo_hidden",
            "number", "self_motion", "self_location", "cause", "elapsed", "relations")
def load(tag):
    runs = {}
    for d in sorted((R / tag).glob("*_s*")):
        w, s = d.name.rsplit("_s", 1)
        runs[(w, int(s))] = {f.stem: json.loads(f.read_text()) for f in d.glob("*.json") if f.name != "translation.json"}
    return runs
main = load("main")
print("runs:", sorted(main))

# 1. seed reproducibility, lawful s0 vs s1
a, b, lab = [], [], []
for m in main[("lawful", 0)]:
    for c in CONCEPTS:
        x, y = main[("lawful", 0)][m]["probes"][c], main[("lawful", 1)][m]["probes"][c]
        if np.isfinite(x) and np.isfinite(y):
            a.append(x); b.append(y); lab.append((m, c))
a, b = np.array(a), np.array(b); d = np.abs(a - b)
print(f"\n[seeds] n={len(a)} mean|diff|={d.mean():.4f} max={d.max():.4f} at {lab[d.argmax()]}")
print("  largest 5:", [(lab[i], round(float(a[i]),3), round(float(b[i]),3)) for i in np.argsort(-d)[:5]])
# per-concept seed SD relative to between-mind SD (oracle excluded)
print("  per concept: sd of seed diff / 2**.5 vs sd across minds (oracle excluded)")
for c in CONCEPTS:
    ms = [m for m in main[("lawful",0)] if m != "oracle"]
    x = np.array([main[("lawful",0)][m]["probes"][c] for m in ms]); y = np.array([main[("lawful",1)][m]["probes"][c] for m in ms])
    print(f"   {c:20s} seed sd {np.std(x-y)/2**.5:.4f}  mind sd {np.std((x+y)/2):.4f}")

# 2. MuJoCo agreement: STATUS says r 0.997 over 168, mean abs 0.017, max 0.12; within-concept lowest cause 0.91
def tab(w, field="probes"):
    acc = defaultdict(lambda: defaultdict(list))
    for (ww, s), minds in main.items():
        if ww == w:
            for m, r in minds.items():
                for c, v in r.get(field, {}).items(): acc[m][c].append(v)
    return {m: {c: float(np.nanmean(v)) for c, v in cs.items()} for m, cs in acc.items()}
for variant, A in (("lawful seed-mean", tab("lawful")), ("lawful seed 0", {m: r["probes"] for m, r in main[("lawful",0)].items()})):
    B = tab("lawful_mj")
    pairs = [(A[m][c], B[m][c], m, c) for m in A for c in CONCEPTS if np.isfinite(A[m][c]) and np.isfinite(B[m][c])]
    x, y = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    print(f"\n[mujoco, {variant}] n={len(x)} r={np.corrcoef(x,y)[0,1]:.4f} mean|d|={np.abs(x-y).mean():.4f} max={np.abs(x-y).max():.4f}")
    rs = {}
    for c in CONCEPTS:
        sel = [i for i, p in enumerate(pairs) if p[3] == c and p[2] != "oracle"]
        rs[c] = np.corrcoef(x[sel], y[sel])[0, 1]
    print("  within-concept r (oracle excluded):", {c: round(float(v), 3) for c, v in rs.items()})
    # how does engine difference compare with seed difference? (mind-concept level, oracle excluded)
    dm = np.array([abs(p[0]-p[1]) for p in pairs if p[2] != "oracle"])
    print(f"  engine |diff| (oracle excl) mean {dm.mean():.4f}, 95th pct {np.percentile(dm,95):.4f}")

# 3. ANOVA shares, recomputed independently, and the Lewontin point: shares depend on which minds/worlds are sampled
def shares(concept, minds_keep=None, worlds_keep=None, field="probes"):
    cells = defaultdict(list)
    for (w, s), ms in main.items():
        if w.endswith("_mj"): continue
        if worlds_keep and w not in worlds_keep: continue
        for m, r in ms.items():
            if m == "oracle": continue
            if minds_keep and m not in minds_keep: continue
            v = r[field].get(concept, np.nan)
            if np.isfinite(v): cells[(m, w)].append(v)
    minds = sorted({m for m, _ in cells}); worlds = sorted({w for _, w in cells})
    minds = [m for m in minds if all((m, w) in cells for w in worlds)]
    Y = [[np.array(cells[(m, w)]) for w in worlds] for m in minds]
    allv = np.concatenate([y for row in Y for y in row]); gm = allv.mean()
    cm = np.array([[y.mean() for y in row] for row in Y]); n = np.array([[len(y) for y in row] for row in Y])
    mm = (cm*n).sum(1)/n.sum(1); wm = (cm*n).sum(0)/n.sum(0)
    ssm = np.sum(n.sum(1)*(mm-gm)**2); ssw = np.sum(n.sum(0)*(wm-gm)**2); ssc = np.sum(n*(cm-gm)**2)
    ssr = sum(np.sum((y-y.mean())**2) for row in Y for y in row); tot = ssc + ssr
    # balanced alternative: seed-mean cells, one per cell (Type III-like for this design)
    gm2 = cm.mean(); ssm2 = len(worlds)*np.sum((cm.mean(1)-gm2)**2); ssw2 = len(minds)*np.sum((cm.mean(0)-gm2)**2); tot2 = np.sum((cm-gm2)**2)
    return dict(mind=ssm/tot, world=ssw/tot, inter=(ssc-ssm-ssw)/tot, noise=ssr/tot, mind_bal=ssm2/tot2, world_bal=ssw2/tot2,
                n_minds=len(minds), worlds=worlds)
print("\n[ANOVA] recomputed shares (all 13 non-oracle minds), weighted and balanced-on-cell-means")
for c in CONCEPTS:
    s = shares(c)
    print(f"  {c:20s} mind {s['mind']:.2f} world {s['world']:.2f} inter {s['inter']:.2f} noise {s['noise']:.3f} | balanced mind {s['mind_bal']:.2f} world {s['world_bal']:.2f} | worlds {s['worlds']}")
vis_minds = ["human", "vision_only", "passive", "passive_novest", "memoryless"]
print("\n[ANOVA, Lewontin check] only the five human-eyed minds (human, vision_only, passive, passive_novest, memoryless)")
for c in ("object_seen", "object_hidden", "number", "self_motion", "cause", "elapsed"):
    s = shares(c, minds_keep=vis_minds)
    print(f"  {c:15s} mind {s['mind']:.2f} world {s['world']:.2f} inter {s['inter']:.2f} noise {s['noise']:.2f}")
print("  H7 verdict on this mind set: world>mind for object_hidden and cause, mind>world for seen/number/self_motion ->",
      all(shares(c, vis_minds)['world'] > shares(c, vis_minds)['mind'] for c in ("object_hidden","cause")) and
      all(shares(c, vis_minds)['mind'] > shares(c, vis_minds)['world'] for c in ("object_seen","number","self_motion")))
print("\n[ANOVA, cause without nonlocal] (lawful, impermanent only; ghost has no cause)")
s = shares("cause", worlds_keep=["lawful", "impermanent"]); print(f"  cause mind {s['mind']:.2f} world {s['world']:.2f} inter {s['inter']:.2f} noise {s['noise']:.2f}")
s = shares("elapsed", worlds_keep=["lawful", "ghost", "nonlocal"]); print(f"  elapsed without impermanent: mind {s['mind']:.2f} world {s['world']:.2f}")

# 4. V2 with the preregistered nine/ten concepts versus twelve, and margins
print("\n[V2] trained-minus-untrained wins per mind, lawful seed-mean; and wins with margin > 0.02 (about 2 seed sd)")
L, L0 = tab("lawful"), tab("lawful", "probes_untrained")
orig10 = ("object_seen", "object_hidden", "object_now", "object_past", "number", "self_motion", "self_location", "cause", "elapsed", "relations")
for m in L:
    if m not in L0 or not L0[m]: continue
    w12 = sum(L[m][c] > L0[m][c] for c in CONCEPTS); w10 = sum(L[m][c] > L0[m][c] for c in orig10)
    wm = sum(L[m][c] > L0[m][c] + 0.02 for c in CONCEPTS)
    print(f"  {m:15s} wins 12: {w12:2d}  wins original 10: {w10:2d}  wins by >0.02: {wm:2d}")

# 5. H6 share and H6/P4 surprise values
def perm(w, m, s=None):
    return [r[m]["voe"]["permanence"] for (ww, ss), r in main.items() if ww == w and m in r and (s is None or ss == s)]
common = [m for m in main[("lawful",0)] if m != "oracle"]
diffs = {m: np.mean([v["S"] for v in perm("impermanent", m)]) - np.mean([v["S"] for v in perm("lawful", m)]) for m in common}
print("\n[H6] impermanent minus lawful permanence S per mind:", {m: f"{v:+.5f}" for m, v in diffs.items()})
print("  share < 0:", np.mean([v < 0 for v in diffs.values()]), " (n minds", len(diffs), ")")
for m in ("human", "vision_only", "memoryless"):
    print(f"  {m} lawful:", [(round(v['S'],5), [round(x,5) for x in v['ci']]) for v in perm("lawful", m)])
