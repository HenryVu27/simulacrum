"""Check 7: recompute the tracking mind's permanence S (200 noise draws, own totals, own bootstrap over the 18
quadruplets), and check the tracking head's loss on held-out episodes against its training log."""
import pickle
import jax, jax.numpy as jnp, numpy as np
from common import *  # noqa
from umwelt import learner, voe
saved = pickle.load(open(REPO / "results/v3/stage0/track_params.pkl", "rb"))
P, mu, sd = saved["params"], saved["mu"], saved["sd"]
hp = presets.PRESETS["human"]
print("training log, last entries (step, loss, 1-step, far):", saved["log"][-2:])
l = saved["log"]
print("implied tracking-loss term: first {:.4f}, last {:.4f}".format(l[0][1] - l[0][2] - 0.5 * l[0][3], l[-1][1] - l[-1][2] - 0.5 * l[-1][3]))
d, gt, T = get_data()
raw = get_raw(d)
te = slice(N_TRAIN, N_TRAIN + N_TEST)
x = (np.asarray(raw[te]).astype(np.float32) - mu) / sd
H = learner.batched(learner._latents_jit, P, x, presets.actions(d, hp)[te])
out = np.asarray(learner._lin(P["trk2"], jax.nn.gelu(learner._lin(P["trk1"], jnp.asarray(H)))))
y = T["ego"][te].reshape(N_TEST, 256, -1); m = np.repeat(T["present"][te], 2, -1)
print(f"tracking loss on held-out episodes (t >= 16, present objects): {(m[:, 16:] * (out[:, 16:] - y[:, 16:]) ** 2).sum() / m[:, 16:].sum():.4f}; "
      f"loss of predicting 0: {(m[:, 16:] * y[:, 16:] ** 2).sum() / m[:, 16:].sum():.4f}; target variance per coordinate: "
      f"{np.mean([y[:, 16:, c][m[:, 16:, c]].var() for c in range(16)]):.4f}")

suite = voe.build_suite()
dd, crit = suite["permanence"]
a = presets.actions(dd, hp)
errs = []
for k in range(voe.NOISE_DRAWS):
    o = (presets.observe(dd, hp, draw=k) - mu) / sd
    errs.append(learner.batched(learner._errors_jit, P, o, a))
E = np.stack(errs).mean(0)  # (72, T-1)
n = len(crit)
E = E.reshape(n, 4, -1)
tot = np.array([[E[i, v, crit[i] - 1:].sum() for v in range(4)] for i in range(n)])
poss, imp = tot[:, 0] + tot[:, 1], tot[:, 2] + tot[:, 3]
S = (imp.sum() - poss.sum()) / poss.sum()
rng = np.random.default_rng(12345)
b = []
for _ in range(5000):
    i = rng.integers(0, n, n)
    b.append((imp[i].sum() - poss[i].sum()) / poss[i].sum())
print(f"permanence S {S:+.5f}, 95% bootstrap [{np.percentile(b, 2.5):+.5f}, {np.percentile(b, 97.5):+.5f}] (own code, 5,000 resamples); stored -0.00034 [-0.00096, +0.00021]")
print("per-quadruplet (I1 - P1, I2 - P2) summed error after the critical frame:")
print(np.round(np.c_[tot[:, 2] - tot[:, 0], tot[:, 3] - tot[:, 1]], 3).T)
