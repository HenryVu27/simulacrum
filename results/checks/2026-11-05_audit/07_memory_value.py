"""Audit: recompute the v2 manipulation check (M overall and at object returns) from stored per-step errors,
with an episode bootstrap, and the calibration stage-3 values that chose the world."""
import numpy as np
U = "/Users/henryvu/personal/umwelt/results/"
def steps_since_seen(vis):
    E, T, N = vis.shape; out = np.full((E, T, N), 1000, np.int32); last = np.full((E, N), -10_000)
    for t in range(T):
        last = np.where(vis[:, t], t, last); out[:, t] = np.minimum(t - last, 1000)
    return out
def masks(world, data_seed, n_train):
    D = U + f"data/{world}_s{data_seed}_e{n_train+256}_t256/"
    vis = np.load(D + "gt_vis180.npy", mmap_mode="r")[n_train:] & np.load(D + "gt_obj_active.npy", mmap_mode="r")[n_train:]
    vis = np.asarray(vis); since = steps_since_seen(vis)
    ret = (vis[:, 1:] & ~vis[:, :-1] & (since[:, :-1] < 1000) & (since[:, :-1] >= 3)).any(-1)
    burn = np.arange(ret.shape[1])[None] >= 16
    return ret & burn, np.broadcast_to(burn, ret.shape)
def mv(eh, ez, ret, burn, rng, B=2000):
    M = 1 - eh[burn].mean() / ez[burn].mean(); Mr = 1 - eh[ret].mean() / ez[ret].mean()
    E = len(eh); bs = []
    for _ in range(B):
        i = rng.integers(0, E, E)
        bs.append(1 - eh[i][ret[i]].mean() / ez[i][ret[i]].mean())
    return M, Mr, np.percentile(bs, [2.5, 97.5]), ret.mean()
rng = np.random.default_rng(0)
ret, burn = masks("dark", 1, 8192)
for s in (0, 1):
    eh = np.load(U + f"runs/v2/dark_s{s}/human_test_errors.npy").astype(np.float32)
    ez = np.load(U + f"runs/v2/dark_s{s}/memoryless_test_errors.npy").astype(np.float32)
    M, Mr, ci, share = mv(eh, ez, ret, burn, rng)
    print(f"[confirmatory dark data seed 1, learner seed {s}] M {M:.4f}  M at returns {Mr:.4f}  episode-bootstrap 95% [{ci[0]:.3f}, {ci[1]:.3f}]  return-step share {share:.3f}")
ret0, burn0 = masks("dark", 0, 8192)
for w in ("dark", "occlusion"):
    if w == "occlusion": ret0, burn0 = masks("occlusion", 0, 8192)
    eh = np.load(U + f"calibration/errors/{w}_n8192_human.npy"); ez = np.load(U + f"calibration/errors/{w}_n8192_memoryless.npy")
    M, Mr, ci, share = mv(eh, ez, ret0, burn0, rng)
    print(f"[calibration stage 3, {w}, data seed 0, learner seed 0] M {M:.4f}  M at returns {Mr:.4f}  episode-bootstrap 95% [{ci[0]:.3f}, {ci[1]:.3f}]")
