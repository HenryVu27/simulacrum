"""How much of the error at return steps could object knowledge remove? Twin's error at lit re-entries versus ordinary lit
steps (lit now and before, no return), on the same held-out episodes; also by sense block (vision vs the rest)."""
import importlib.util, pickle
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
REPO = Path("/Users/henryvu/personal/simulacrum")
spec = importlib.util.spec_from_file_location("rr", REPO / "experiments/stage0_rerun.py"); rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
from umwelt import learner, presets, probes
from umwelt.data import load
d = load("dark", 8448, seed=3)
raw = np.load(REPO / "results/v3/stage0_rerun/raw_obs.npy", mmap_mode="r")
te = slice(8192, 8448)
vis = np.asarray(d["gt_vis180"][te]) & np.asarray(d["gt_obj_active"][te])
since = probes.steps_since_seen(vis)
ret = (vis[:, 1:] & ~vis[:, :-1] & (since[:, :-1] < 1000) & (since[:, :-1] >= 3)).any(-1) & (np.arange(255) >= 16)[None]
lit = np.asarray(d["gt_light"][te]); first_lit = lit[:, 1:] & ~lit[:, :-1]
ordinary = lit[:, 1:] & lit[:, :-1] & ~ret & (np.arange(255) >= 16)[None]
reentry = ret & ~first_lit
acts = presets.actions(d, presets.PRESETS["human"])[te]
@jax.jit
def dim_err(p, obs, act):
    h = learner.latents(p, obs, act); near, _ = learner.predict(p, h, act)
    return (near[:, :-1, :311] - obs[:, 1:, :311]) ** 2
def per_dim(name, extra):
    with open(REPO / f"results/v3/stage0_rerun/{name}_params.pkl", "rb") as f: sv = pickle.load(f)
    o = raw[te] if extra is None else np.concatenate([raw[te], extra.astype(np.float16)], -1)
    o = ((o.astype(np.float32) - sv["mu"]) / sv["sd"]).astype(np.float16)
    return learner.batched(dim_err, sv["params"], o, acts)
chan = rr.channel_from_oracle(d["obs_oracle"][te])
for name, extra in (("twin", None), ("lawful", chan)):
    e = per_dim(name, extra)
    v = e[..., :256].mean(-1); r = e[..., 256:].mean(-1); a = e.mean(-1)
    print(f"{name:7} all dims: ordinary lit {a[ordinary].mean():.4f}  lit re-entry {a[reentry].mean():.4f}  first-lit {a[first_lit & ret].mean():.4f} | "
          f"vision dims (256): ordinary {v[ordinary].mean():.4f} re-entry {v[reentry].mean():.4f} | other 55 dims: ordinary {r[ordinary].mean():.4f} re-entry {r[reentry].mean():.4f}")
    tot = a[ret].sum(); print(f"         share of summed return-step error from first-lit steps: {a[ret & first_lit].sum() / tot:.3f}; "
          f"re-entry excess over ordinary lit, as share of gate's summed error: {(a[reentry].mean() - a[ordinary].mean()) * reentry.sum() / tot:.4f}")
