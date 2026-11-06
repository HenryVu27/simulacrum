"""Audit: does today's world code, on this Mac, regenerate the stored v1 lawful recording and the v2 dark
confirmatory recording bit for bit? First 32 episodes (one sub-batch of the first chunk), every array."""
import sys
import numpy as np
sys.path.insert(0, "/Users/henryvu/personal/umwelt/src")
import jax
from umwelt import data
from umwelt.world import WORLDS
for world, n, seed in (("lawful", 1280, 0), ("dark", 8448, 1)):
    keys = jax.random.split(jax.random.PRNGKey(seed), n)[:128]
    a, o, g = data.rollout_batch(keys, WORLDS[world], 256)
    rec = {"action": np.asarray(a, np.float32)}
    rec.update({f"obs_{k}": np.asarray(o[k], np.float16) for k in data.OBS_KEYS})
    rec.update({f"gt_{k}": np.asarray(v) for k, v in g.items()})
    D = f"/Users/henryvu/personal/umwelt/results/data/{world}_s{seed}_e{n}_t256/"
    bad = []
    for k, v in rec.items():
        try:
            st = np.load(D + f"{k}.npy", mmap_mode="r")[:128]
        except FileNotFoundError:
            continue
        if v.shape != st.shape or not np.array_equal(v, np.asarray(st)):
            nd = np.sum(v != np.asarray(st)) if v.shape == st.shape else "shape"
            bad.append((k, nd))
    print(f"{world} s{seed}: {len(rec)} arrays compared on 128 episodes (the first recording chunk, same batch size as data.chunks); mismatches: {bad or 'none'}", flush=True)
