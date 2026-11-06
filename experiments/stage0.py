"""v3 Stage 0 (docs/preregistration.md, Amendment 5): an instrument that registers memory.

Fixtures with known memory, on the dark world (data seed 0, the v2 recording: 8,192 training episodes and 256
held-out ones), scored by probe candidates in the preregistered order; the first candidate that meets every
criterion is fixed for the rest of v3.

Fixtures:
- raw: the human mind's current observation (the raw-senses control);
- buffer 2,4 and buffer 3,6,9: raw plus exact copies of the observation that many steps back (an explicit memory);
- engineered states, 24 numbers (per kind of object: x, y and a flag) embedded into 128 dimensions by a fixed
  random linear map, plus noise of 1 percent of each unit's variance (the amendment's "noise of the hidden
  states' scale" made concrete here):
  full (every object's true egocentric position, in view or not; the flag says it exists),
  last seen (its egocentric position when last in view, frozen since; the flag says it was ever seen),
  last seen, updated (its room position when last in view, re-expressed in the agent's current pose);
- track: the v2 learner for the human mind trained with an auxiliary loss that predicts every object's egocentric
  position (learner.LearnerConfig.track), a mind whose memory is known to be trained in.

Probe candidates, in order: the battery as it is (1,024 fitting episodes, at most 60,000 rows); the same with
4,096 and with 8,192 fitting episodes (the row cap scaled with them, 60,000 per 1,024 episodes); ridge linear
probes, scored only if those fail. The amendment's "MLP with early stopping on a held-out split of the fitting
episodes" is the battery as it is (its probes already stop early on held-out episodes), so it adds no candidate.

Criteria (all must hold): every buffer scores at least as high as raw on object_seen, object_hidden and
self_location; last seen, updated exceeds raw by more than 0.30 in permanence skill at 1 to 5 and at 6 to 20
steps; full exceeds 0.80 at every delay; track exceeds raw by more than 0.10 at 1 to 5 steps.

    uv run python experiments/stage0.py track      # train the tracking mind (about 10 minutes on the Mac)
    uv run python experiments/stage0.py score      # score every fixture with every candidate, then choose
    uv run python experiments/stage0.py ridge      # if no MLP candidate passes: ridge probes, in the same order
    uv run python experiments/stage0.py surprise   # the surprise test on the tracking mind
"""

import json
import pickle
import sys
import time

import numpy as np

from umwelt import learner, presets, probes
from umwelt.data import load
from umwelt.experiment import RESULTS, _voe_scores
from umwelt.world import ARENA

OUT = RESULTS / "v3" / "stage0"
N_TRAIN, N_TEST = 8192, 256
TR, TE = 8, 4  # row strides, as in the battery
CANDIDATES = [("battery", 1024, 60_000), ("battery, 4,096 episodes", 4096, 240_000), ("battery, 8,192 episodes", 8192, 480_000)]
# Scored only if every MLP candidate fails, in the same order of data (probes.RidgeProbe).
RIDGE = [("ridge", 1024, 60_000), ("ridge, 4,096 episodes", 4096, 240_000), ("ridge, 8,192 episodes", 8192, 480_000)]
BINS = [("in view", None), ("1-5 steps", (1, 5)), ("6-20 steps", (6, 20)), ("21-60 steps", (21, 60)),
        ("over 60 steps", (61, 999)), ("never seen", (1000, 1000))]


def data():
    d = load("dark", N_TRAIN + N_TEST, seed=0)
    gt = {k: v for k, v in d.items() if k.startswith("gt_")}
    T = probes.targets({k: np.asarray(v) for k, v in gt.items()})
    T["dark"] = np.repeat(~np.asarray(gt["gt_light"])[..., None], probes.K, -1)
    return d, gt, T


def rows(a, stride):
    return a[:, probes.PAST::stride]


def last_seen(gt):
    """For each (episode, time, slot), the latest time at or before it when the object was in view, or -1."""
    vis = np.asarray(gt["gt_vis180"])
    E, T, N = vis.shape
    out = np.full((E, T, N), -1, np.int32)
    last = np.full((E, N), -1, np.int32)
    for t in range(T):
        last = np.where(vis[:, t], t, last)
        out[:, t] = last
    return out


def engineered(gt, which, times):
    """(E, len(times), 24): per kind, (x, y, flag) for the fixture `which`, at the given times."""
    pos, apos, th = np.asarray(gt["gt_obj_pos"]), np.asarray(gt["gt_agent_pos"]), np.asarray(gt["gt_agent_th"])
    active, kind = np.asarray(gt["gt_obj_active"]), np.asarray(gt["gt_obj_kind"])[:, 0].astype(int)
    E = pos.shape[0]
    e = np.arange(E)[:, None, None]
    if which == "full":
        rel, flag = pos[:, times] - apos[:, times, None], active[:, times]
        ego = probes._ego(rel, th[:, times, None])
    else:
        ls = last_seen(gt)[:, times]  # (E, nt, N)
        seen = ls >= 0
        tl = np.maximum(ls, 0)
        n = np.arange(pos.shape[2])[None, None, :]
        then = pos[e, tl, n]  # room position when last seen
        if which == "last seen":
            ego = probes._ego(then - apos[e, tl], th[e, tl])
        else:  # last seen, updated: re-expressed in the agent's current pose
            ego = probes._ego(then - apos[:, times, None], th[:, times, None])
        flag = seen & active[:, times]
    ego = np.where(flag[..., None], ego / ARENA * 4, 0.0).astype(np.float32)
    feats = np.concatenate([ego, flag[..., None].astype(np.float32)], -1)  # (E, nt, N, 3)
    return probes._by_kind(feats, kind).reshape(E, len(times), -1)


def embed(train, test, seed=0):
    """Fixed random linear map of standardized features into 128 dimensions, plus noise at 1% of each unit's variance."""
    mu, sd = train.reshape(-1, train.shape[-1]).mean(0), train.reshape(-1, train.shape[-1]).std(0) + 1e-6
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1.0 / np.sqrt(train.shape[-1]), (train.shape[-1], 128)).astype(np.float32)
    out = []
    for x in (train, test):
        z = ((x - mu) / sd) @ W
        out.append((z + rng.normal(0.0, 0.1, z.shape) * z.reshape(-1, 128).std(0)).astype(np.float32))
    return out


def score(Xtr, Xte, Rtr, Rte, groups, max_rows, seed=0, cls=probes.Probe):
    """object_seen, object_hidden, self_location, and the permanence curve (skill above a no-information mind)."""
    m2 = lambda m: np.repeat(m, 2, axis=1)  # noqa: E731
    p = cls(Xtr, Rtr["ego"].reshape(len(Xtr), -1), m2(Rtr["present"]), "reg", 256, seed, groups, max_rows)
    pred, Y = p(Xte), Rte["ego"].reshape(len(Xte), -1)
    pres, vis, since, dark = Rte["present"], Rte["vis"], Rte["since"], Rte["dark"]
    out = {"object_seen": probes._r2_cols(pred, Y, m2(pres & vis), m2(pres)),
           "object_hidden": probes._r2_cols(pred, Y, m2(pres & ~vis), m2(pres))}
    ptr, ytr = Rtr["present"], Rtr["ego"]
    mean = np.array([[ytr[ptr[:, k], k, j].mean() if ptr[:, k].any() else 0.0 for j in range(2)] for k in range(ptr.shape[1])])
    const = np.broadcast_to(mean.reshape(1, -1), Y.shape)

    def skill(m):
        s, b = probes._r2_cols(pred, Y, m2(m), m2(pres)), probes._r2_cols(const, Y, m2(m), m2(pres))
        return float((s - b) / (1 - b))

    for name, rng in BINS:
        m = pres & vis if rng is None else pres & ~vis & (since >= rng[0]) & (since <= rng[1])
        out[name] = skill(m)
        if rng is not None and rng[1] <= 20:
            out[name + ", dark"] = skill(m & dark)
            out[name + ", lit"] = skill(m & ~dark)
    q = cls(Xtr, Rtr["loc"], np.ones_like(Rtr["loc"], bool), "reg", 256, seed + 4, groups, max_rows)
    out["self_location"] = probes._r2_cols(q(Xte), Rte["loc"], np.ones_like(Rte["loc"], bool), np.ones_like(Rte["loc"], bool))
    return out


def track_targets(T):
    y = T["ego"].reshape(*T["ego"].shape[:2], -1).astype(np.float32)
    m = np.repeat(T["present"], 2, axis=-1)
    return y, m


def train_track():
    OUT.mkdir(parents=True, exist_ok=True)
    d, gt, T = data()
    p = presets.PRESETS["human"]
    cache = OUT / "track_obs.npy"
    o = presets.observe(d, p, dtype=np.float16, path=cache)
    tr = slice(0, N_TRAIN)
    mu, sd = learner.standardizer(o[tr])
    for i in range(0, len(o), 128):
        o[i : i + 128] = (o[i : i + 128].astype(np.float32) - mu) / sd
    cfg = learner.LearnerConfig(track=1.0)
    t0 = time.time()
    params, log = learner.train(o[tr], presets.actions(d, p)[tr], cfg, seed=0, track=tuple(x[tr] for x in track_targets(T)))
    print(f"tracking mind trained in {time.time() - t0:.0f} s; last loss {log[-1]}", flush=True)
    with open(OUT / "track_params.pkl", "wb") as f:
        pickle.dump({"params": params, "mu": mu, "sd": sd, "log": log}, f)
    cache.unlink()


def main_score(ridge=False):
    OUT.mkdir(parents=True, exist_ok=True)
    d, gt, T = data()
    p = presets.PRESETS["human"]
    raw = presets.observe(d, p, dtype=np.float16, path=OUT / "raw_obs.npy")
    times_tr = np.arange(probes.PAST, raw.shape[1], TR)
    times_te = np.arange(probes.PAST, raw.shape[1], TE)
    fit, te = slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_TEST)
    with open(OUT / "track_params.pkl", "rb") as f:
        saved = pickle.load(f)
    acts = presets.actions(d, p)

    def track_rows(eps, times):
        H = []
        for i in range(eps.start, eps.stop, 256):
            j = min(i + 256, eps.stop)
            x = (raw[i:j].astype(np.float32) - saved["mu"]) / saved["sd"]
            H.append(learner.batched(learner._latents_jit, saved["params"], x, acts[i:j])[:, times])
        return np.concatenate(H)

    def raw_rows(eps, times, lags=(0,)):
        out = []
        for i in range(eps.start, eps.stop, 512):
            chunk = raw[i : min(i + 512, eps.stop)]
            out.append(np.concatenate([chunk[:, times - L].astype(np.float32) for L in lags], -1))
        return np.concatenate(out)

    # Every fixture's rows, built once for all 8,192 fitting episodes (E, rows, D); candidates use the first n_fit.
    t0 = time.time()
    fixtures = {"raw": (raw_rows(fit, times_tr), raw_rows(te, times_te)),
                "buffer 2,4": (raw_rows(fit, times_tr, (0, 2, 4)), raw_rows(te, times_te, (0, 2, 4))),
                "buffer 3,6,9": (raw_rows(fit, times_tr, (0, 3, 6, 9)), raw_rows(te, times_te, (0, 3, 6, 9)))}
    gtf = {k: np.asarray(v[fit]) for k, v in gt.items()}
    gte = {k: np.asarray(v[te]) for k, v in gt.items()}
    for which in ("full", "last seen", "last seen, updated"):
        fixtures[which] = tuple(embed(engineered(gtf, which, times_tr), engineered(gte, which, times_te)))
    fixtures["track"] = (track_rows(fit, times_tr), track_rows(te, times_te))
    print(f"fixtures built in {time.time() - t0:.0f} s: " + ", ".join(f"{k} {v[0].shape}" for k, v in fixtures.items()), flush=True)
    Rte = {k: rows(T[k][te], TE).reshape(-1, *T[k].shape[2:]) for k in ("ego", "present", "vis", "since", "dark", "loc")}

    results = json.loads((OUT / "scores.json").read_text()) if ridge else {}
    for cname, n_fit, max_rows in (RIDGE if ridge else CANDIDATES):
        Rtr = {k: rows(T[k][:n_fit], TR).reshape(-1, *T[k].shape[2:]) for k in ("ego", "present", "vis", "since", "dark", "loc")}
        groups = np.repeat(np.arange(n_fit), len(times_tr))
        results[cname] = {}
        for fname, (a, b) in fixtures.items():
            t0 = time.time()
            r = results[cname][fname] = score(a[:n_fit].reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), Rtr, Rte, groups,
                                              max_rows, cls=probes.RidgeProbe if ridge else probes.Probe)
            print(f"{cname:26} {fname:20} seen {r['object_seen']:+.3f} hidden {r['object_hidden']:+.3f} "
                  f"self {r['self_location']:+.3f} | 1-5 {r['1-5 steps']:+.3f} (dark {r['1-5 steps, dark']:+.3f}, "
                  f"lit {r['1-5 steps, lit']:+.3f}) 6-20 {r['6-20 steps']:+.3f} | lowest bin "
                  f"{min(r[n] for n, _ in BINS):+.3f} ({time.time() - t0:.0f} s)", flush=True)
        checks = criteria(results[cname])
        results[cname]["criteria"] = checks
        print(f"{cname}: " + "; ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in checks.items()), flush=True)
        (OUT / "scores.json").write_text(json.dumps(results, indent=1))
        if all(checks.values()):
            print(f"chosen: {cname} (the first candidate that meets every criterion)")
            (OUT / "chosen.json").write_text(json.dumps({"candidate": cname, "fitting_episodes": n_fit, "max_rows": max_rows}))
            break
    else:
        print("no ridge candidate passes either: this battery cannot be shown to measure memory (Amendment 5)" if ridge
              else "no MLP candidate passes; ridge linear probes are next (Amendment 5)")
    (OUT / "raw_obs.npy").unlink()


def criteria(r):
    raw = r["raw"]
    return {
        "buffers never lower a score": all(r[b][c] >= raw[c] for b in ("buffer 2,4", "buffer 3,6,9")
                                           for c in ("object_seen", "object_hidden", "self_location")),
        "last seen, updated beats raw by 0.30": all(r["last seen, updated"][b] - raw[b] > 0.30 for b in ("1-5 steps", "6-20 steps")),
        "full above 0.80 at every delay": all(r["full"][n] > 0.80 for n, _ in BINS),
        "track beats raw by 0.10 at 1-5 steps": r["track"]["1-5 steps"] - raw["1-5 steps"] > 0.10,
    }


def surprise():
    """The surprise test on the tracking mind (Amendment 5: its permanence S must have an interval above 0), beside
    the v2 learner's human mind trained on the same recording's world (dark, data seed 1, learner seed 0)."""
    from umwelt import voe

    with open(OUT / "track_params.pkl", "rb") as f:
        saved = pickle.load(f)
    p = presets.PRESETS["human"]
    out = {"track": _voe_scores(saved["params"], voe.build_suite(), p, saved["mu"], saved["sd"])}
    v2 = json.loads((RESULTS / "runs" / "v2" / "dark_s1" / "human.json").read_text())["voe"]
    out["v2 human (dark seed 1, stored)"] = v2
    for name, r in out.items():
        print(f"{name:32} " + "  ".join(f"{t} {r[t]['S']:+.4f} [{r[t]['ci'][0]:+.4f}, {r[t]['ci'][1]:+.4f}]" for t in voe.TESTS))
    ok = out["track"]["permanence"]["ci"][0] > 0
    print(f"tracking mind's permanence interval above 0: {ok}")
    (OUT / "surprise.json").write_text(json.dumps({**out, "permanence interval above 0": ok}, indent=1))


if __name__ == "__main__":
    {"track": train_track, "score": main_score, "ridge": lambda: main_score(ridge=True), "surprise": surprise}[sys.argv[1]]()
