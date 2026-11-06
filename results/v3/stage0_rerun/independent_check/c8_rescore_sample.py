"""Rescore a sample of stored minds with the fixed probe (ridge, 1,024 fitting episodes), with targets, masks, skill and
the per-episode bootstrap written here (independent of probes.targets / stage0_rescore.score_sums), and compare with
results/v3/stage0_rescore/<run>.json. Optional: the same trained network with its memory cut (hidden state reset to 0
before every step), as a control that keeps the learned nonlinear encoding of the present and removes the past.
usage: c8_rescore_sample.py TAG/WORLD_sSEED mind1,mind2 [lesion]"""
import json, pickle, sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
REPO = Path("/Users/henryvu/personal/simulacrum"); OUTD = Path(__file__).parent / "out"; OUTD.mkdir(exist_ok=True)
from umwelt import learner, learner_jepa, presets, probes
from umwelt.data import load
from umwelt.world import ARENA
run, minds = sys.argv[1], sys.argv[2].split(","); do_lesion = len(sys.argv) > 3
folder = REPO / "results/runs" / run
rec0 = json.loads((folder / f"{minds[0]}.json").read_text())
world, n_train, ds = rec0["world"], rec0.get("n_train", 1024), rec0.get("data_seed", 0)
print(f"{run}: world {world}, data seed {ds}, n_train {n_train}; fit episodes 0-1023, held-out {n_train}-{n_train+255}")
d = load(world, n_train + 256, seed=ds)
sub = {"fit": {k: np.asarray(v[:1024]) for k, v in d.items()}, "te": {k: np.asarray(v[n_train:n_train + 256]) for k, v in d.items()}}
STRIDE = {"fit": 8, "te": 4}
def own_targets(g, stride):
    pos, ap, th = g["gt_obj_pos"].astype(np.float64), g["gt_agent_pos"].astype(np.float64), g["gt_agent_th"].astype(np.float64)
    act, kind, vis180 = g["gt_obj_active"], g["gt_obj_kind"][:, 0].astype(int), g["gt_vis180"]
    E, T, N = act.shape
    rel = pos - ap[:, :, None]; c, s = np.cos(th)[:, :, None], np.sin(th)[:, :, None]
    ego = np.stack([c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]], -1) / ARENA * 4
    since = np.full((E, T, N), 1000); last = np.full((E, N), -1)
    for t in range(T):
        last = np.where(vis180[:, t], t, last); since[:, t] = np.where(last >= 0, np.minimum(t - last, 1000), 1000)
    times = np.arange(10, T, stride)
    out = {"ego": np.zeros((E, len(times), 8, 2)), "present": np.zeros((E, len(times), 8), bool), "vis": np.zeros((E, len(times), 8), bool),
           "since": np.full((E, len(times), 8), 1000)}
    for e in range(E):
        for n, k in enumerate(kind[e]):
            out["ego"][e, :, k] = ego[e, times, n]; out["present"][e, :, k] = act[e, times, n]
            out["vis"][e, :, k] = vis180[e, times, n]; out["since"][e, :, k] = since[e, times, n]
    out = {k: v.reshape(E * len(times), *v.shape[2:]) for k, v in out.items()}
    if "gt_light" in g: out["dark"] = np.repeat(~g["gt_light"][:, times].reshape(-1)[:, None], 8, 1)
    return out
R = {p: own_targets({k: v for k, v in sub[p].items() if k.startswith("gt_")}, STRIDE[p]) for p in ("fit", "te")}
# cross-check own targets against the repo's on the held-out rows
T_repo = probes.targets({k: v for k, v in sub["te"].items() if k.startswith("gt_")})
for k in ("ego", "present", "vis", "since"):
    a = T_repo[k][:, 10::4].reshape(R["te"][k].shape)
    print(f"  own target {k} vs probes.targets: max |diff| {np.abs(a.astype(float) - R['te'][k].astype(float)).max():.2e}")
n_te_rows = len(range(10, 256, 4)); ep = np.repeat(np.arange(256), n_te_rows)
m2 = lambda m: np.repeat(m, 2, 1)
ytr, yte = R["fit"]["ego"].reshape(-1, 16), R["te"]["ego"].reshape(-1, 16)
groups = np.repeat(np.arange(1024), len(range(10, 256, 8)))
base = np.array([R["fit"]["ego"][R["fit"]["present"][:, k], k, j].mean() if R["fit"]["present"][:, k].any() else 0.0 for k in range(8) for j in range(2)])
pres, vis, since = R["te"]["present"], R["te"]["vis"], R["te"]["since"]
MASKS = {"object_now": (pres, "pool"), "object_hidden": (pres & ~vis, "pool"), "object_seen": (pres & vis, "seen")}
for b, (lo, hi) in (("1-5 steps", (1, 5)), ("6-20 steps", (6, 20))):
    m = pres & ~vis & (since >= lo) & (since <= hi); MASKS[b] = (m, "pool")
    if "dark" in R["te"]: MASKS[b + ", dark"], MASKS[b + ", lit"] = (m & R["te"]["dark"], "pool"), (m & ~R["te"]["dark"], "pool")
def epsums(pred, m):
    M, ref = m2(m), m2(pres)
    parts = [np.where(M, (pred - yte) ** 2, 0), M.astype(float), np.where(ref, yte, 0), np.where(ref, yte ** 2, 0), ref.astype(float)]
    return [np.stack([x[ep == e].sum(0) for e in range(256)]) for x in parts]
def r2w(st, w):
    sse, nm, s1, s2, nr = (w @ x for x in st); ok = (nm >= 30) & (nr >= 30)
    return np.sum(nm[ok] * (1 - (sse[ok] / nm[ok]) / (s2[ok] / nr[ok] - (s1[ok] / nr[ok]) ** 2))) / nm[ok].sum()
const = np.broadcast_to(base, yte.shape)
CB = {name: epsums(const, m) for name, (m, _) in MASKS.items()}
def scores(Xf, Xt):
    pred = {"pool": probes.RidgeProbe(Xf, ytr, m2(R["fit"]["present"]), "reg", 0, 0, groups)(Xt),
            "seen": probes.RidgeProbe(Xf, ytr, m2(R["fit"]["present"] & R["fit"]["vis"]), "reg", 0, 8, groups)(Xt)}
    return {name: epsums(pred[ro], m) for name, (m, ro) in MASKS.items()}
def val(name, st, w):
    r = r2w(st, w)
    if name.startswith("object_"): return r
    b = r2w(CB[name], w); return (r - b) / (1 - b)
W = np.stack([np.bincount(np.random.default_rng(b).integers(0, 256, 256), minlength=256) for b in range(1000)]).astype(float)
one = np.ones(256)
def lesion_fn(p, obs, act):
    e = learner.encode(p, obs, act); B, T, E = e.shape
    return learner.core_scan(p, e.reshape(1, B * T, E)).reshape(B, T, -1)
lesion_jit = jax.jit(lesion_fn)
stored = json.loads((REPO / "results/v3/stage0_rescore" / f"{run.replace('/', '_')}.json").read_text())
ST = {}
for mind in minds:
    t0 = time.time()
    rec = json.loads((folder / f"{mind}.json").read_text()); p = presets.PRESETS[mind]
    with open(folder / f"{mind}_params.pkl", "rb") as f: sv = pickle.load(f)
    L = learner_jepa if rec["config"].get("objective") == "jepa" else learner
    feats = {"trained": {}, "raw senses": {}}
    if do_lesion: feats["memory cut"] = {}
    for part in ("fit", "te"):
        o_raw = presets.observe(sub[part], p); o = (o_raw - sv["mu"]) / sv["sd"]; a = presets.actions(sub[part], p)
        sl = slice(10, None, STRIDE[part])
        feats["trained"][part] = learner.batched(L._latents_jit, sv["params"], o, a)[:, sl].reshape(-1, 128 if "gru_h" in sv["params"] or "core" in sv["params"] else -1)
        feats["raw senses"][part] = o_raw[:, sl].reshape(-1, o_raw.shape[-1]).astype(np.float32)
        if do_lesion: feats["memory cut"][part] = learner.batched(lesion_jit, sv["params"], o, a)[:, sl].reshape(-1, 128)
    for kind, F in feats.items():
        ST[(mind, kind)] = scores(F["fit"], F["te"])
        mine = {k: val(k, s, one) for k, s in ST[(mind, kind)].items()}
        sk = stored["minds"][mind].get(kind)
        line = "  ".join(f"{k} {v:+.4f}" + (f" ({sk[k]:+.4f})" if sk else "") for k, v in mine.items())
        print(f"  {mind:12} {kind:10} {line}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
def diff(a, b, keys):
    out = {}
    for k in keys:
        pt = val(k, ST[a][k], one) - val(k, ST[b][k], one)
        bs = np.array([val(k, ST[a][k], w) - val(k, ST[b][k], w) for w in W]); lo, hi = np.percentile(bs, [2.5, 97.5])
        out[k] = (pt, lo, hi)
    return out
keys = [k for k in MASKS if k not in ("object_now", "object_seen")]
pairs = [(("human", "trained"), ("human", "raw senses"), "trained minus raw", ("minds", "human", "trained minus raw"))]
if "memoryless" in minds: pairs.append((("human", "trained"), ("memoryless", "trained"), "human minus memoryless", ("human minus memoryless",)))
if do_lesion:
    for mnd in minds: pairs.append(((mnd, "trained"), (mnd, "memory cut"), f"{mnd} trained minus memory cut", None))
    for mnd in minds: pairs.append(((mnd, "memory cut"), (mnd, "raw senses"), f"{mnd} memory cut minus raw", None))
for a, b, label, path in pairs:
    if a[0] not in minds or b[0] not in minds: continue
    ref = stored
    if path:
        for k in path: ref = ref[k]
    dd = diff(a, b, keys)
    print(f"{label}: " + "  ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" + (f" (stored {ref[k][0]:+.3f} [{ref[k][1][0]:+.3f}, {ref[k][1][1]:+.3f}])" if path else "") for k, v in dd.items()), flush=True)
