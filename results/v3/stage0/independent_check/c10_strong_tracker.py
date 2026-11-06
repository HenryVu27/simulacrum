"""Exploratory (not Stage 0, not preregistered): would a tracking mind that tracks show permanence surprise?
Same recipe as stage0.train_track but a larger tracking-loss weight; then its head on held-out episodes and in the
permanence movies, and its permanence S over 200 noise draws. Usage: c10_strong_tracker.py WEIGHT [STEPS]"""
import pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from common import *  # noqa
from umwelt import learner, voe, kinds

w = float(sys.argv[1]); steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
d, gt, T = get_data()
raw = get_raw(d)
hp = presets.PRESETS["human"]
std_path = SCR / "std_obs_human.npy"
mu, sd = learner.standardizer(raw[:N_TRAIN])
if not std_path.exists():
    o = np.lib.format.open_memmap(std_path, "w+", np.float16, raw.shape)
    for i in range(0, len(raw), 256):
        o[i:i + 256] = (np.asarray(raw[i:i + 256]).astype(np.float32) - mu) / sd
    o.flush(); del o
o = np.load(std_path, mmap_mode="r")
y = T["ego"].reshape(*T["ego"].shape[:2], -1).astype(np.float32); m = np.repeat(T["present"], 2, -1).astype(np.float32)
cfg = learner.LearnerConfig(track=w, steps=steps)
t0 = time.time()
params, log = learner.train(o[:N_TRAIN], presets.actions(d, hp)[:N_TRAIN], cfg, seed=0, track=(y[:N_TRAIN], m[:N_TRAIN]), verbose=False)
print(f"weight {w}, {steps} updates: trained in {time.time() - t0:.0f} s; log first {log[0]} last {log[-1]}", flush=True)
pickle.dump({"params": jax.device_get(params), "mu": mu, "sd": sd, "log": log}, open(SCR / f"track_w{w:g}_s{steps}.pkl", "wb"))
head = lambda h: np.asarray(learner._lin(params["trk2"], jax.nn.gelu(learner._lin(params["trk1"], jnp.asarray(h)))))
te = slice(N_TRAIN, N_TRAIN + N_TEST)
H = learner.batched(learner._latents_jit, params, np.asarray(o[te]).astype(np.float32), presets.actions(d, hp)[te])
out = head(H).reshape(N_TEST, 256, 8, 2)
Y, pres, vis, since = T["ego"][te], T["present"][te], T["vis"][te], T["since"][te]
gte = {k: np.asarray(v[te]) for k, v in gt.items()}
upd = s0.engineered(gte, "last seen, updated", np.arange(256)).reshape(N_TEST, 256, 8, 3)[..., :2]
yy, mm = y[te][:, 16:], m[te][:, 16:]
print(f"  held-out tracking loss {(mm * (out.reshape(N_TEST, 256, 16)[:, 16:] - yy) ** 2).sum() / mm.sum():.4f} (target variance ~1.99; weight-1 mind 1.2025)")
for label, msk in (("in view", pres & vis), ("hidden 1-5", pres & ~vis & (since >= 1) & (since <= 5)), ("hidden 6-20", pres & ~vis & (since >= 6) & (since <= 20))):
    print(f"  {label:12s} head mean distance {np.linalg.norm(out - Y, axis=-1)[msk].mean():.3f}; last seen, updated {np.linalg.norm(upd - Y, axis=-1)[msk].mean():.3f}")
errs, frz, seen_err = [], [], []
for k in voe.INERT:
    for j in (-0.5, 0.0, 0.5):
        mv, tv = voe.permanence(k, j)
        mo = mv[0]; Tm = len(mo["oracle"])
        orc = mo["oracle"]; apos = orc[:, :2] * 10; opos = orc[:, 6 + 5 * k + 1: 6 + 5 * k + 3] * 10
        true = (opos - apos) / 10 * 4  # agent faces +x (th = 0)
        x = (presets.observe({f"obs_{kk}": mo[kk][None] for kk in mo} | {"action": np.zeros((1, Tm, 2), np.float32)}, hp) - mu) / sd
        h = learner.latents(params, jnp.asarray(x, jnp.float32), jnp.zeros((1, Tm, 2), jnp.float32))
        pr = head(h)[0].reshape(Tm, 8, 2)[:, k]
        seen = (np.abs(mo["vision"][:, :, :3] - kinds.COLOR[k]).max(-1) < 1e-6).any(-1)
        t = np.arange(Tm); hid = ~seen & (t >= voe.PRE)
        last = np.maximum.accumulate(np.where(seen, t, -1))
        errs.append(np.linalg.norm(pr - true, axis=-1)[hid]); frz.append(np.linalg.norm(true[last] - true, axis=-1)[hid])
        seen_err.append(np.linalg.norm(pr - true, axis=-1)[seen & (t >= voe.PRE)])
print(f"  movies: head error hidden {np.concatenate(errs).mean():.3f} (frozen {np.concatenate(frz).mean():.3f}); head error seen {np.concatenate(seen_err).mean():.3f}", flush=True)
suite = voe.build_suite()
dd, crit = suite["permanence"]
a = presets.actions(dd, hp)
E = np.mean([learner.batched(learner._errors_jit, params, (presets.observe(dd, hp, draw=k) - mu) / sd, a) for k in range(voe.NOISE_DRAWS)], 0)
s, ci = voe.surprise(E[None].repeat(2, 0), crit)  # the draw-mean, as voe.surprise takes it
print(f"  permanence S {s:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}]", flush=True)
