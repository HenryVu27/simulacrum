"""Audit: is the 'cause' probe target (agent vs another object set this inert object moving) decodable from
proximity alone? If so, the probe measures 'is a moving object near me', and the nonlocal world's drop is the
cue breaking, not the world forcing a concept. Scored the way probes.py scores cause: per-kind balanced
accuracy, chance-normalized, count-weighted; train rows = episodes 0..1023 stride 8, test = 1024..1279 stride 4."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
D = "/Users/henryvu/personal/umwelt/results/data/{}_s0_e1280_t256/"
ANIMATE = np.array([0, 0, 0, 0, 0, 1, 0, 1], bool)
def rows(world, sl, stride):
    g = {k: np.load(D.format(world) + f"gt_{k}.npy", mmap_mode="r")[sl] for k in ("agent_pos", "agent_th", "obj_pos", "obj_vel", "obj_active", "obj_kind", "cause")}
    g = {k: np.asarray(v)[:, 10::stride] for k, v in g.items()}
    rel = g["obj_pos"] - g["agent_pos"][:, :, None]
    dist = np.linalg.norm(rel, axis=-1)
    ang = np.arctan2(rel[..., 1], rel[..., 0]) - g["agent_th"][:, :, None]
    speed = np.linalg.norm(g["obj_vel"], axis=-1)
    kind = g["obj_kind"].astype(int)
    m = g["obj_active"] & ~ANIMATE[kind] & np.isin(g["cause"], (1, 2))
    X = np.stack([dist, np.cos(ang), np.sin(ang), speed], -1)[m]
    return X, (g["cause"][m] == 1), kind[m]
def score(y, p, kind):
    s, c = [], []
    for k in range(8):
        sel = kind == k
        yy, pp = y[sel], p[sel]
        if sel.sum() < 30 or yy.all() or (~yy).all(): continue
        b = 0.5 * (pp[yy].mean() + (~pp[~yy]).mean()); s.append((b - .5) / .5); c.append(sel.sum())
    return np.average(s, weights=c)
for world in ("lawful", "impermanent", "nonlocal", "lawful_mj"):
    Xtr, ytr, ktr = rows(world, slice(0, 1024), 8)
    Xte, yte, kte = rows(world, slice(1024, 1280), 4)
    out = [f"{world:12s} n_train {len(ytr)} n_test {len(yte)} share agent-caused {yte.mean():.2f}"]
    for name, cols in (("distance only", [0]), ("distance+direction+speed", [0, 1, 2, 3])):
        for mname, model in (("logistic", LogisticRegression(class_weight="balanced", max_iter=1000)),
                             ("boosted trees", HistGradientBoostingClassifier(class_weight="balanced", max_iter=200))):
            model.fit(Xtr[:, cols], ytr)
            out.append(f"   {name:26s} {mname:13s} cause score {score(yte, model.predict(Xte[:, cols]).astype(bool), kte):.3f}")
    print("\n".join(out))
print("\nStored MLP-probe cause scores (seed-mean), for comparison: lawful human 0.52, smell_only 0.74, blind 0.65, oracle 0.43;"
      " nonlocal human 0.24, smell_only 0.32, oracle 0.30")
