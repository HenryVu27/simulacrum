"""Check 6: the tracking head on held-out recorded episodes (in distribution), against simple baselines, and split
by whether the agent is still (the movies' condition). Also: how often the training agent is still, how many
objects a training scene has, and how the head does on single-object moments."""
import pickle
import jax, jax.numpy as jnp, numpy as np
from common import *  # noqa
from umwelt import learner
saved = pickle.load(open(REPO / "results/v3/stage0/track_params.pkl", "rb"))
P, mu, sd = saved["params"], saved["mu"], saved["sd"]
d, gt, T = get_data()
raw = get_raw(d)
te = slice(N_TRAIN, N_TRAIN + N_TEST)
x = (np.asarray(raw[te]).astype(np.float32) - mu) / sd
a = presets.actions(d, presets.PRESETS["human"])[te]
H = learner.batched(learner._latents_jit, P, x, a)
out = np.asarray(learner._lin(P["trk2"], jax.nn.gelu(learner._lin(P["trk1"], jnp.asarray(H))))).reshape(*H.shape[:2], 8, 2)
Y, pres, vis, since = T["ego"][te], T["present"][te], T["vis"][te], T["since"][te]
# frozen-updated baseline: last seen room position re-expressed now (stage0.engineered verified in check 1)
gte = {k: np.asarray(v[te]) for k, v in gt.items()}
upd = s0.engineered(gte, "last seen, updated", np.arange(256)).reshape(N_TEST, 256, 8, 3)[..., :2]
frz = s0.engineered(gte, "last seen", np.arange(256)).reshape(N_TEST, 256, 8, 3)[..., :2]
Ttr = {k: T[k][:1024] for k in ("ego", "present")}
mean = np.array([[Ttr["ego"][..., k, j][Ttr["present"][..., k]].mean() for j in range(2)] for k in range(8)])
speed = np.linalg.norm(np.asarray(d["gt_agent_vel"][te]), axis=-1)
om = np.abs(np.diff(np.asarray(d["gt_agent_th"][te]), axis=1, prepend=np.asarray(d["gt_agent_th"][te])[:, :1]))
still = (speed < 0.05) & (om < 0.01)
nobj = np.asarray(d["gt_obj_active"][te]).sum(-1)
print(f"held-out moments with the agent still (speed < 0.05, turn < 0.01 rad/step): {still[:, 10:].mean():.3f}; "
      f"objects per scene: {np.bincount(nobj[:, 10:].ravel()) / nobj[:, 10:].size}")
print(f"training-agent speed: median {np.median(speed):.3f}, 10th pct {np.percentile(speed, 10):.3f}")
def rep(label, m):
    e = np.linalg.norm(out - Y, axis=-1)[m]
    c = np.linalg.norm(mean[None, None] - Y, axis=-1)[m]
    u = np.linalg.norm(upd - Y, axis=-1)[m]
    f = np.linalg.norm(frz - Y, axis=-1)[m]
    sse = ((out - Y) ** 2).sum(-1)[m].sum(); sst = ((mean[None, None] - Y) ** 2).sum(-1)[m].sum()
    print(f"  {label:34s} n={m.sum():7d}  mean distance: head {e.mean():.3f} | kind mean {c.mean():.3f} | last seen updated {u.mean():.3f} | last seen frozen {f.mean():.3f} | head skill vs kind mean {1 - sse / sst:+.3f}")
base = pres.copy()
print("tracking head on held-out recorded episodes (ego units = metres * 0.4):")
rep("in view", base & vis)
rep("hidden 1-5 steps", base & ~vis & (since >= 1) & (since <= 5))
rep("hidden 6-20 steps", base & ~vis & (since >= 6) & (since <= 20))
rep("hidden, never seen", base & ~vis & (since >= 1000))
st = np.repeat(still[..., None], 8, -1)
rep("in view, agent still", base & vis & st)
rep("hidden 1-20, agent still", base & ~vis & (since >= 1) & (since <= 20) & st)
one = np.repeat((nobj == 1)[..., None], 8, -1)
rep("in view, one object in scene", base & vis & one)
# distance and bearing like the movies: object 4.5 m ahead (ego x 1.8), lateral within 1.4
mov = (np.abs(Y[..., 0] - 1.8) < 0.2) & (np.abs(Y[..., 1]) < 1.4)
rep("in view, at the movies' distance", base & vis & mov)
# objects moving while hidden in view of a still agent? speed of objects while in view
ov = np.linalg.norm(np.asarray(d["gt_obj_vel"][te]), axis=-1)
print(f"object speed while present: share above 1.0 m/s (movie objects move 1.5 m/s): {(ov[np.asarray(d['gt_obj_active'][te])] > 1.0).mean():.4f}")
