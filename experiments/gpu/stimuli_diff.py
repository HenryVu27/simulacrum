"""Where do the surprise-test movies rendered on the GPU differ from those rendered on the CPU?

stimuli_check.sh found different voe.suite_hash values: e8b112a5e2a11879 on the Mac's CPU and the PC's CPU,
ceb701ed0ddd1c3a on the GPU at full float32 with deterministic operations, 3776a5c3ffed017d at the GPU's
defaults (docs/gpu.md, setup log 2026-11-05). This renders the movies on both devices in one process and
compares every array: how many values differ, by how much, and in which frames. (The answer, 8 values in
the solidity movies' vision, is why voe.build_suite now renders on the CPU.)

    experiments/gpu/remote.sh run stimuli-diff bash -c 'JAX_DEFAULT_MATMUL_PRECISION=highest \
        XLA_FLAGS=--xla_gpu_deterministic_ops=true uv run python experiments/gpu/stimuli_diff.py; \
        uv run python experiments/gpu/stimuli_diff.py'
"""

import os

import jax
import numpy as np

from umwelt import voe

print(f"settings: precision {jax.config.jax_default_matmul_precision}, XLA_FLAGS '{os.environ.get('XLA_FLAGS', '')}'")
gpu = voe.build_suite(device=None)  # JAX's default device; experiments render on the CPU since then
cpu = voe.build_suite(device="cpu")
print(f"suite_hash: {jax.default_backend()} {voe.suite_hash(gpu)}, cpu {voe.suite_hash(cpu)}")
for test in sorted(gpu):
    (g, gc), (c, cc) = gpu[test], cpu[test]
    assert np.array_equal(gc, cc), f"{test}: critical frames differ"
    for k in sorted(g):
        a, b = g[k].astype(np.float64), c[k].astype(np.float64)
        diff = a != b
        if not diff.any():
            print(f"  {test:10} {k:12} identical ({a.size} values)")
            continue
        movies, frames = np.nonzero(diff.reshape(diff.shape[0], diff.shape[1], -1).any(-1))
        print(f"  {test:10} {k:12} {diff.sum()} of {a.size} values differ, largest {np.abs(a - b).max():.3g};"
              f" {len(set(movies))} of {diff.shape[0]} movies, frames {frames.min()} to {frames.max()}")
