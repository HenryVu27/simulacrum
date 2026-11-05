"""Does pausing a GPU job (C:\\gpu-pause, see job.sh) change its result? Also run-to-run determinism.

Trains the current learner on fixed random data and writes the loss log and a hash of the trained
weights. Run it as remote jobs with --xla_gpu_deterministic_ops=true: once straight through, once paused
and resumed partway. The hashes must match; if two straight runs already differ, the GPU is not
deterministic and the pause cannot be judged by hash.

    XLA_FLAGS=--xla_gpu_deterministic_ops=true uv run python experiments/gpu/pause_check.py results/remote/pause_a.json
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import jax
import numpy as np

from umwelt import learner


def main(out, steps):
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((256, 256, 311), dtype=np.float32).astype(np.float16)
    act = rng.standard_normal((256, 256, 2), dtype=np.float32)
    t0 = time.time()
    params, log = learner.train(obs, act, learner.LearnerConfig(steps=steps), verbose=False, log_every=100)
    secs = time.time() - t0
    leaves = jax.tree_util.tree_leaves(params)
    sha = hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in leaves)).hexdigest()[:16]
    rec = {"steps": steps, "seconds": secs, "params_sha": sha, "backend": jax.default_backend(),
           "xla_flags": os.environ.get("XLA_FLAGS", ""), "log": log}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "log"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args()
    main(a.out, a.steps)
