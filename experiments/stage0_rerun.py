"""v3 Stage 0 rerun (docs/preregistration.md, Amendment 6), on a fresh recording: the dark world, data seed 3.

Probe candidates, in order: the MLP probe with real early stopping (probes.Probe(early_stop=True)) at 1,024, 4,096
and 8,192 fitting episodes, then ridge (probes.RidgeProbe) at the same sizes. The first that meets every criterion
is fixed for the rest of v3:
  a. memory loses nothing: for each buffer (raw senses plus exact copies 2 and 4, or 3, 6 and 9 steps back) and each
     of object_now, object_hidden, self_location (the pooled probe) and object_seen (a probe fit on objects in view
     only), the buffer's score minus raw's, averaged over probe seeds 0 to 2, has a paired bootstrap interval (1,000
     resamples of the 256 held-out episodes) whose upper end is at or above 0;
  b. last seen, updated beats raw senses by more than 0.30 at 1 to 5 and 6 to 20 steps (probe seed 0);
  c. full scores above 0.80 at every delay;
  d. the tracking mind (retrained on this recording, as in Stage 0) beats raw senses by more than 0.10 at 1 to 5 steps.

The lawful mind checks the surprise test: the human mind with one more input, every kind of object's egocentric
position and a flag as lawful physics has it (in recordings the truth; in the surprise movies the possible movie of
the same pair: I1 gets P1's, I2 gets P2's). Its twin is the same mind without that input. Gate: its one-step error at
object-return steps on held-out episodes is at least 10 percent below its twin's. Check: its permanence S (200 noise
draws) has an interval above 0. Errors are over the human's senses only (the 311 dimensions both minds predict), so
the two are compared on the same thing.

    uv run python experiments/stage0_rerun.py track     # retrain the tracking mind on data seed 3
    uv run python experiments/stage0_rerun.py probes    # score the candidates in order; fix the first that passes
    uv run python experiments/stage0_rerun.py lawful    # train the lawful mind and its twin, gate, surprise test
"""

import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from umwelt import learner, presets, probes, voe
from umwelt.data import load
from umwelt.experiment import RESULTS
from umwelt.world import ARENA

spec = importlib.util.spec_from_file_location("stage0", Path(__file__).with_name("stage0.py"))
s0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s0)

OUT = RESULTS / "v3" / "stage0_rerun"
SEED = 3
N_TRAIN, N_TEST = 8192, 256
TR, TE = 8, 4
SENSES = 311  # the human mind's observation dimensions
CANDIDATES = [("MLP, early stopping", 1024, 60_000, "mlp"), ("MLP, early stopping, 4,096 episodes", 4096, 240_000, "mlp"),
              ("MLP, early stopping, 8,192 episodes", 8192, 480_000, "mlp"), ("ridge", 1024, 60_000, "ridge"),
              ("ridge, 4,096 episodes", 4096, 240_000, "ridge"), ("ridge, 8,192 episodes", 8192, 480_000, "ridge")]
PROBE_SEEDS = (0, 1, 2)
N_BOOT = 1000
BUFFERS = {"buffer 2,4": (0, 2, 4), "buffer 3,6,9": (0, 3, 6, 9)}
METRICS = ("object_now", "object_hidden", "object_seen", "self_location")


def data():
    d = load("dark", N_TRAIN + N_TEST, seed=SEED)
    gt = {k: v for k, v in d.items() if k.startswith("gt_")}
    T = probes.targets({k: np.asarray(v) for k, v in gt.items()})
    T["dark"] = np.repeat(~np.asarray(gt["gt_light"])[..., None], probes.K, -1)
    return d, gt, T


def make(kind, X, Y, M, seed, groups, max_rows):
    if kind == "mlp":
        return probes.Probe(X, Y, M, "reg", 256, seed, groups, max_rows, early_stop=True)
    return probes.RidgeProbe(X, Y, M, "reg", 256, seed, groups, max_rows)


# ---------------------------------------------------------------- scores from per-episode sums (exact, resamplable)

def sums(pred, Y, m, ref, ep, n_ep):
    """Per-episode sums from which _r2_cols can be recomputed for any resample of episodes: (n_ep, columns) each."""
    out = []
    for a in (m * (pred - Y) ** 2, m.astype(float), ref * Y, ref * Y**2, ref.astype(float)):
        s = np.zeros((n_ep, a.shape[1]))
        np.add.at(s, ep, a)
        out.append(s)
    return out


def r2_from(sm, w):
    """_r2_cols from per-episode sums, episodes weighted by w (their counts in a resample)."""
    sse, nm, s1, s2, nr = (w @ s for s in sm)
    ok = (nm >= 30) & (nr >= 30)
    var = s2[ok] / nr[ok] - (s1[ok] / nr[ok]) ** 2 + 1e-9
    scores = 1 - (sse[ok] / nm[ok]) / var
    return float(np.average(scores, weights=nm[ok])) if ok.any() else float("nan")


def metric_sums(pp, ps, pl, Rte, ep):
    """The four criterion-(a) metrics' per-episode sums from the pooled, seen-only and self-location predictions."""
    m2 = lambda m: np.repeat(m, 2, axis=1)  # noqa: E731
    Y, pres, vis, loc = Rte["ego"].reshape(len(pp), -1), Rte["present"], Rte["vis"], Rte["loc"]
    ones = np.ones_like(loc, bool)
    return {"object_now": sums(pp, Y, m2(pres), m2(pres), ep, N_TEST),
            "object_hidden": sums(pp, Y, m2(pres & ~vis), m2(pres), ep, N_TEST),
            "object_seen": sums(ps, Y, m2(pres & vis), m2(pres), ep, N_TEST),
            "self_location": sums(pl, loc, ones, ones, ep, N_TEST)}


def curve(pred, Rtr, Rte):
    """Permanence skill above a no-information mind, by steps since last in view (Stage 0's bins)."""
    m2 = lambda m: np.repeat(m, 2, axis=1)  # noqa: E731
    Y = Rte["ego"].reshape(len(pred), -1)
    pres, vis, since, dark = Rte["present"], Rte["vis"], Rte["since"], Rte["dark"]
    ptr, ytr = Rtr["present"], Rtr["ego"]
    mean = np.array([[ytr[ptr[:, k], k, j].mean() if ptr[:, k].any() else 0.0 for j in range(2)] for k in range(ptr.shape[1])])
    const = np.broadcast_to(mean.reshape(1, -1), Y.shape)

    def skill(m):
        s, b = probes._r2_cols(pred, Y, m2(m), m2(pres)), probes._r2_cols(const, Y, m2(m), m2(pres))
        return float((s - b) / (1 - b))

    out = {}
    for name, rng in s0.BINS:
        m = pres & vis if rng is None else pres & ~vis & (since >= rng[0]) & (since <= rng[1])
        out[name] = skill(m)
        if rng is not None and rng[1] <= 20:
            out[name + ", dark"], out[name + ", lit"] = skill(m & dark), skill(m & ~dark)
    return out


# ---------------------------------------------------------------- the tracking mind, retrained on data seed 3

def train_track():
    OUT.mkdir(parents=True, exist_ok=True)
    d, gt, T = data()
    p = presets.PRESETS["human"]
    o = presets.observe(d, p, dtype=np.float16, path=OUT / "track_obs.npy")
    tr = slice(0, N_TRAIN)
    mu, sd = learner.standardizer(o[tr])
    for i in range(0, len(o), 128):
        o[i : i + 128] = (o[i : i + 128].astype(np.float32) - mu) / sd
    t0 = time.time()
    params, log = learner.train(o[tr], presets.actions(d, p)[tr], learner.LearnerConfig(track=1.0), seed=0,
                                track=tuple(x[tr] for x in s0.track_targets(T)), verbose=False)
    print(f"tracking mind trained in {time.time() - t0:.0f} s; last loss {log[-1]}", flush=True)
    with open(OUT / "track_params.pkl", "wb") as f:
        pickle.dump({"params": params, "mu": mu, "sd": sd, "log": log}, f)
    (OUT / "track_obs.npy").unlink()


# ---------------------------------------------------------------- the probe candidates

def main_probes():
    OUT.mkdir(parents=True, exist_ok=True)
    d, gt, T = data()
    p = presets.PRESETS["human"]
    raw = presets.observe(d, p, dtype=np.float16, path=OUT / "raw_obs.npy")
    times_tr, times_te = np.arange(probes.PAST, 256, TR), np.arange(probes.PAST, 256, TE)
    fit, te = slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_TEST)
    with open(OUT / "track_params.pkl", "rb") as f:
        saved = pickle.load(f)
    acts = presets.actions(d, p)

    def raw_rows(eps, times, lags=(0,)):
        return np.concatenate([np.concatenate([raw[i : min(i + 512, eps.stop)][:, times - L].astype(np.float32) for L in lags], -1)
                               for i in range(eps.start, eps.stop, 512)])

    def track_rows(eps, times):
        return np.concatenate([learner.batched(learner._latents_jit, saved["params"],
                                               (raw[i : min(i + 256, eps.stop)].astype(np.float32) - saved["mu"]) / saved["sd"],
                                               acts[i : min(i + 256, eps.stop)])[:, times] for i in range(eps.start, eps.stop, 256)])

    t0 = time.time()
    fixtures = {"raw": (raw_rows(fit, times_tr), raw_rows(te, times_te))}
    for name, lags in BUFFERS.items():
        fixtures[name] = (raw_rows(fit, times_tr, lags), raw_rows(te, times_te, lags))
    gtf = {k: np.asarray(v[fit]) for k, v in gt.items()}
    gte = {k: np.asarray(v[te]) for k, v in gt.items()}
    for which in ("full", "last seen", "last seen, updated"):
        fixtures[which] = tuple(s0.embed(s0.engineered(gtf, which, times_tr), s0.engineered(gte, which, times_te)))
    fixtures["track"] = (track_rows(fit, times_tr), track_rows(te, times_te))
    print(f"fixtures built in {time.time() - t0:.0f} s", flush=True)
    keys = ("ego", "present", "vis", "since", "dark", "loc")
    Rte = {k: s0.rows(T[k][te], TE).reshape(-1, *T[k].shape[2:]) for k in keys}
    ep = np.repeat(np.arange(N_TEST), len(times_te))
    W = np.stack([np.bincount(np.random.default_rng(b).integers(0, N_TEST, N_TEST), minlength=N_TEST) for b in range(N_BOOT)])
    m2 = lambda m: np.repeat(m, 2, axis=1)  # noqa: E731

    results = {}
    for cname, n_fit, max_rows, kind in CANDIDATES:
        Rtr = {k: s0.rows(T[k][:n_fit], TR).reshape(-1, *T[k].shape[2:]) for k in keys}
        groups = np.repeat(np.arange(n_fit), len(times_tr))
        ytr = Rtr["ego"].reshape(len(Rtr["ego"]), -1)
        res = results[cname] = {"seeds": {}, "curves": {}}
        stats = {}
        for fname in ("raw",) + tuple(BUFFERS):
            a, b = fixtures[fname]
            a, b = a[:n_fit].reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
            for seed in PROBE_SEEDS:
                t0 = time.time()
                pool = make(kind, a, ytr, m2(Rtr["present"]), seed, groups, max_rows)
                seen = make(kind, a, ytr, m2(Rtr["present"] & Rtr["vis"]), seed + 8, groups, max_rows)
                loc = make(kind, a, Rtr["loc"], np.ones_like(Rtr["loc"], bool), seed + 4, groups, max_rows)
                pp, ps, pl = pool(b), seen(b), loc(b)
                sm = stats[(fname, seed)] = metric_sums(pp, ps, pl, Rte, ep)
                point = {k: r2_from(v, np.ones(N_TEST)) for k, v in sm.items()}
                Y = Rte["ego"].reshape(len(b), -1)
                dev = abs(point["object_now"] - probes._r2_cols(pp, Y, m2(Rte["present"]), m2(Rte["present"])))
                assert dev < 1e-4, dev  # per-episode sums in float64 against the battery's float32 arithmetic
                steps = {n: getattr(q, "best_step", None) for n, q in (("pooled", pool), ("seen", seen), ("self", loc))}
                res["seeds"][f"{fname}, seed {seed}"] = {**point, "best_steps": steps}
                if seed == 0:
                    res["curves"][fname] = curve(pp, Rtr, Rte)
                print(f"{cname:38} {fname:13} seed {seed}: " + " ".join(f"{k} {v:+.3f}" for k, v in point.items())
                      + f" | best steps {steps} ({time.time() - t0:.0f} s)", flush=True)
        for fname in ("full", "last seen", "last seen, updated", "track"):
            a, b = fixtures[fname]
            pool = make(kind, a[:n_fit].reshape(-1, a.shape[-1]), ytr, m2(Rtr["present"]), 0, groups, max_rows)
            res["curves"][fname] = curve(pool(b.reshape(-1, b.shape[-1])), Rtr, Rte)
            c = res["curves"][fname]
            print(f"{cname:38} {fname:20} 1-5 {c['1-5 steps']:+.3f} 6-20 {c['6-20 steps']:+.3f} lowest bin "
                  f"{min(c[n] for n, _ in s0.BINS):+.3f}", flush=True)
        # criterion a: paired bootstrap of buffer minus raw, each averaged over probe seeds
        res["buffer minus raw"] = {}
        ok_a = True
        for bname in BUFFERS:
            for metric in METRICS:
                def diff(w):
                    return np.mean([r2_from(stats[(bname, s)][metric], w) - r2_from(stats[("raw", s)][metric], w) for s in PROBE_SEEDS])
                point = diff(np.ones(N_TEST))
                boots = np.array([diff(w) for w in W])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                res["buffer minus raw"][f"{bname}, {metric}"] = [point, [float(lo), float(hi)]]
                ok_a &= hi >= 0
                print(f"   {bname:13} {metric:14} buffer minus raw {point:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
        cv = res["curves"]
        checks = {"a. memory loses nothing": bool(ok_a),
                  "b. last seen, updated beats raw by 0.30": all(cv["last seen, updated"][k] - cv["raw"][k] > 0.30 for k in ("1-5 steps", "6-20 steps")),
                  "c. full above 0.80 at every delay": all(cv["full"][n] > 0.80 for n, _ in s0.BINS),
                  "d. track beats raw by 0.10 at 1-5 steps": cv["track"]["1-5 steps"] - cv["raw"]["1-5 steps"] > 0.10}
        res["criteria"] = checks
        print(f"{cname}: " + "; ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in checks.items()), flush=True)
        (OUT / "scores.json").write_text(json.dumps(results, indent=1))
        if all(checks.values()):
            print(f"chosen: {cname} (the first candidate that meets every criterion)")
            (OUT / "chosen.json").write_text(json.dumps({"candidate": cname, "fitting_episodes": n_fit, "max_rows": max_rows, "kind": kind}))
            break
    else:
        print("no candidate passes: v3 stops (Amendment 6)")
    (OUT / "raw_obs.npy").unlink()


# ---------------------------------------------------------------- the lawful mind

def channel_from_oracle(orc):
    """Every kind's egocentric position (scaled as the probe targets) and a flag, from oracle vectors (..., T, 58)."""
    orc = np.asarray(orc, np.float32)
    apos, th = orc[..., :2] * ARENA, np.arctan2(orc[..., 2], orc[..., 3])
    bk = orc[..., 6 : 6 + probes.K * 5].reshape(*orc.shape[:-1], probes.K, 5)
    act = bk[..., 0] > 0.5
    ego = probes._ego(bk[..., 1:3] * ARENA - apos[..., None, :], th[..., None]) / ARENA * 4
    ego = np.where(act[..., None], ego, 0.0)
    return np.concatenate([ego, act[..., None]], -1).reshape(*orc.shape[:-1], -1).astype(np.float32)


def lawful_channel_for_movies(d):
    """The possible movie's channel for every movie of the quadruplet: I1 gets P1's, I2 gets P2's."""
    ch = channel_from_oracle(d["obs_oracle"])
    n = len(ch) // 4
    ch = ch.reshape(n, 4, *ch.shape[1:]).copy()
    ch[:, 2], ch[:, 3] = ch[:, 0], ch[:, 1]
    return ch.reshape(n * 4, *ch.shape[2:])


@jax.jit
def sense_errors(p, obs, act):
    """One-step squared error over the human's senses only (the first 311 dimensions), (B, T-1)."""
    h = learner.latents(p, obs, act)
    near, _ = learner.predict(p, h, act)
    return jnp.mean((near[:, :-1, :SENSES] - obs[:, 1:, :SENSES]) ** 2, -1)


def lawful():
    OUT.mkdir(parents=True, exist_ok=True)
    d, gt, T = data()
    p = presets.PRESETS["human"]
    raw = presets.observe(d, p, dtype=np.float16, path=OUT / "raw_obs.npy")
    E = len(raw)
    chan = channel_from_oracle(d["obs_oracle"])  # recordings are lawful: the truth, in view or not
    full = s0.engineered({k: np.asarray(v[:4]) for k, v in gt.items()}, "full", np.arange(256))
    assert np.abs(chan[:4] - full).max() < 1e-2, "the oracle-based channel must equal the ground truth"
    acts = presets.actions(d, p)
    te = slice(N_TRAIN, N_TRAIN + N_TEST)
    vis = np.asarray(d["gt_vis180"][te]) & np.asarray(d["gt_obj_active"][te])
    since = probes.steps_since_seen(vis)
    returns = (vis[:, 1:] & ~vis[:, :-1] & (since[:, :-1] < 1000) & (since[:, :-1] >= 3)).any(-1) & (np.arange(255) >= 16)[None]
    out = {}
    for name in ("twin", "lawful"):
        D = SENSES + (chan.shape[-1] if name == "lawful" else 0)
        o = np.lib.format.open_memmap(OUT / f"{name}_obs.npy", "w+", np.float16, (E, 256, D))
        for i in range(0, E, 256):
            j = min(i + 256, E)
            o[i:j] = raw[i:j] if name == "twin" else np.concatenate([raw[i:j], chan[i:j].astype(np.float16)], -1)
        mu, sd = learner.standardizer(o[:N_TRAIN])
        for i in range(0, E, 128):
            o[i : i + 128] = (o[i : i + 128].astype(np.float32) - mu) / sd
        t0 = time.time()
        params, log = learner.train(o[:N_TRAIN], acts[:N_TRAIN], learner.LearnerConfig(), seed=0, verbose=False)
        err = learner.batched(sense_errors, params, o[te], acts[te])
        out[name] = {"error at object returns": float(err[returns].mean()), "error, all steps after burn-in": float(err[:, 16:].mean()),
                     "train seconds": round(time.time() - t0)}
        with open(OUT / f"{name}_params.pkl", "wb") as f:
            pickle.dump({"params": params, "mu": mu, "sd": sd, "log": log}, f)
        print(f"{name}: {out[name]}", flush=True)
        # surprise test: senses with their 200 noise draws; the lawful channel has no noise
        suite = voe.build_suite()
        out[name]["voe"] = {}
        for test, (dm, crit) in suite.items():
            extra = lawful_channel_for_movies(dm) if name == "lawful" else None
            am = presets.actions(dm, p)
            errs = []
            for k in range(voe.NOISE_DRAWS):
                x = presets.observe(dm, p, draw=k)
                if extra is not None:
                    x = np.concatenate([x, extra], -1)
                errs.append(learner.batched(sense_errors, params, (x - mu) / sd, am))
            s, ci = voe.surprise(np.stack(errs), crit)
            out[name]["voe"][test] = {"S": s, "ci": ci}
            print(f"   {name} {test:10} S {s:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]", flush=True)
        (OUT / f"{name}_obs.npy").unlink()
    ratio = out["lawful"]["error at object returns"] / out["twin"]["error at object returns"]
    out["gate: error at returns at least 10 percent below the twin's"] = {"ratio": ratio, "passes": bool(ratio <= 0.9)}
    perm = out["lawful"]["voe"]["permanence"]["ci"]
    out["check: permanence interval above 0"] = bool(perm[0] > 0)
    print(f"gate: lawful error at returns / twin's = {ratio:.3f} ({'PASS' if ratio <= 0.9 else 'FAIL'}); "
          f"permanence interval above 0: {perm[0] > 0}")
    (OUT / "lawful.json").write_text(json.dumps(out, indent=1))
    (OUT / "raw_obs.npy").unlink()


if __name__ == "__main__":
    {"track": train_track, "probes": main_probes, "lawful": lawful}[sys.argv[1]]()
