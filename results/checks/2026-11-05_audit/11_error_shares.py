"""Audit of Amendment 4's erratum: share of the memoryless mind's error (after the 16-step burn-in) at object-return
steps, and with first-lit steps added, in the dark world (stage 2 and stage 3 error arrays)."""
import numpy as np
U = "/Users/henryvu/personal/umwelt/results/"
def since_seen(vis):
    E, T, N = vis.shape; out = np.full((E, T, N), 1000, np.int32); last = np.full((E, N), -10_000)
    for t in range(T):
        last = np.where(vis[:, t], t, last); out[:, t] = np.minimum(t - last, 1000)
    return out
for name, n_train in (("dark_u16000", 1024), ("dark_n8192", 8192)):
    D = U + f"data/dark_s0_e{n_train+256}_t256/"
    vis = np.asarray(np.load(D + "gt_vis180.npy", mmap_mode="r")[n_train:]) & np.asarray(np.load(D + "gt_obj_active.npy", mmap_mode="r")[n_train:])
    lit = np.asarray(np.load(D + "gt_light.npy", mmap_mode="r")[n_train:])
    s = since_seen(vis)
    ret = (vis[:, 1:] & ~vis[:, :-1] & (s[:, :-1] < 1000) & (s[:, :-1] >= 3)).any(-1)
    on = lit[:, 1:] & ~lit[:, :-1]
    burn = np.arange(ret.shape[1])[None] >= 16
    ez = np.load(U + f"calibration/errors/{name}_memoryless.npy")
    tot = ez[np.broadcast_to(burn, ez.shape)].sum()
    print(f"{name}: return steps {np.mean(ret[:, 16:]):.3f} of steps carry {ez[ret & burn].sum()/tot:.3f} of memoryless error; "
          f"returns or first lit step: {np.mean((ret | on)[:, 16:]):.3f} of steps, {ez[(ret | on) & burn].sum()/tot:.3f} of error")
