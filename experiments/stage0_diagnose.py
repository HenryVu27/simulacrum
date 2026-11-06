"""v3 Stage 0 diagnostics, after its criteria failed (docs/preregistration.md, Amendment 5; not preregistered).

1. Does adding memory lose information, or does one shared readout trade accuracy between objects in view and
   hidden ones? For raw senses and the 3, 6, 9 buffer: object_now (all present objects, the probe's own target)
   beside object_seen and object_hidden, and object_seen from a probe fit on objects in view only.
2. Does the tracking mind track the hidden object inside the surprise-test movies? Its own tracking head (no probe)
   read on the permanence movies while the object is behind the screen, against the object's true egocentric
   position, and against keeping the last position seen.

    uv run python experiments/stage0_diagnose.py > results/v3/stage0/diagnose.txt
"""

import importlib.util
import pickle
from pathlib import Path

import jax
import numpy as np

from umwelt import learner, presets, probes, voe

spec = importlib.util.spec_from_file_location("stage0", Path(__file__).with_name("stage0.py"))
s0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s0)


def shared_readout():
    d, gt, T = s0.data()
    p = presets.PRESETS["human"]
    raw = presets.observe(d, p, dtype=np.float16, path=s0.OUT / "raw_obs.npy")
    times_tr, times_te = np.arange(probes.PAST, 256, s0.TR), np.arange(probes.PAST, 256, s0.TE)
    n_fit = 1024
    fit, te = slice(0, n_fit), slice(s0.N_TRAIN, s0.N_TRAIN + s0.N_TEST)
    Rtr = {k: s0.rows(T[k][fit], s0.TR).reshape(-1, *T[k].shape[2:]) for k in ("ego", "present", "vis")}
    Rte = {k: s0.rows(T[k][te], s0.TE).reshape(-1, *T[k].shape[2:]) for k in ("ego", "present", "vis")}
    groups = np.repeat(np.arange(n_fit), len(times_tr))
    m2 = lambda m: np.repeat(m, 2, axis=1)  # noqa: E731

    def X(eps, times, lags):
        return np.concatenate([raw[eps][:, times - L].astype(np.float32) for L in lags], -1).reshape(-1, 311 * len(lags))

    print("1. one readout for objects in view and hidden (1,024 fitting episodes)")
    for cls in (probes.Probe, probes.RidgeProbe):
        for name, lags in (("raw", (0,)), ("buffer 3,6,9", (0, 3, 6, 9))):
            a, b = X(fit, times_tr, lags), X(te, times_te, lags)
            pr = cls(a, Rtr["ego"].reshape(len(a), -1), m2(Rtr["present"]), "reg", 256, 0, groups)
            pred, Y = pr(b), Rte["ego"].reshape(len(b), -1)
            pres, vis = Rte["present"], Rte["vis"]
            seen_only = cls(a, Rtr["ego"].reshape(len(a), -1), m2(Rtr["present"] & Rtr["vis"]), "reg", 256, 0, groups)
            print(f"   {cls.__name__:10} {name:13} object_now {probes._r2_cols(pred, Y, m2(pres), m2(pres)):+.3f}  "
                  f"seen {probes._r2_cols(pred, Y, m2(pres & vis), m2(pres)):+.3f}  hidden "
                  f"{probes._r2_cols(pred, Y, m2(pres & ~vis), m2(pres)):+.3f}  | seen, probe fit on seen only "
                  f"{probes._r2_cols(seen_only(b), Y, m2(pres & vis), m2(pres)):+.3f}", flush=True)
    (s0.OUT / "raw_obs.npy").unlink()


def tracking_in_movies():
    with open(s0.OUT / "track_params.pkl", "rb") as f:
        saved = pickle.load(f)
    P = saved["params"]
    p = presets.PRESETS["human"]
    print("\n2. the tracking mind's own head on the permanence movies (P1: object always there), frames behind the screen")
    errs, frozen = [], []
    for k in voe.INERT:
        for j in (-0.5, 0.0, 0.5):
            movies, tv = voe.permanence(k, j)
            m = movies[0]  # P1
            dd = {f"obs_{key}": m[key][None] for key in m}
            dd["action"] = np.zeros((1, len(m["oracle"]), 2), np.float32)
            o = (presets.observe(dd, p) - saved["mu"]) / saved["sd"]
            h = learner.latents(P, jax.numpy.asarray(o, jax.numpy.float32), jax.numpy.asarray(dd["action"]))
            out = np.asarray(learner._lin(P["trk2"], jax.nn.gelu(learner._lin(P["trk1"], h))))[0].reshape(-1, 8, 2)
            # true egocentric position of the object (agent at (2, 5 + j) facing +x), scaled as the targets are
            y = np.concatenate([np.full(voe.PRE, 1.5 + j), 1.5 + j + 0.15 * np.arange(len(o[0]) - voe.PRE)])
            true = np.stack([np.full(len(y), (6.5 - 2.0) / 10 * 4), (y - (5.0 + j)) / 10 * 4], -1)
            vis = m["vision"][:, :, :3]
            seen = (np.abs(vis - voe.kinds.COLOR[k]).max(-1) < 1e-6).any(-1)
            hidden = ~seen & (np.arange(len(seen)) >= voe.PRE)
            last = np.maximum.accumulate(np.where(seen, np.arange(len(seen)), 0))
            errs.append(np.linalg.norm(out[hidden, k] - true[hidden], axis=-1))
            frozen.append(np.linalg.norm(true[last][hidden] - true[hidden], axis=-1))
    errs, frozen = np.concatenate(errs), np.concatenate(frozen)
    print(f"   {len(errs)} hidden frames: mean distance from the truth, tracking head {errs.mean():.3f}; keeping the last "
          f"position seen {frozen.mean():.3f}; the object moves {0.15 / 10 * 4:.3f} per frame in these units")


if __name__ == "__main__":
    shared_readout()
    tracking_in_movies()
