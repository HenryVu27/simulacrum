#!/usr/bin/env bash
# Does the persistent compile cache change results? (docs/gpu.md, E6 follow-up.) At the GPU's defaults, a run
# without the cache differed from runs with it in 36 of 74 numbers (probes, up to 0.003), while the
# cache-empty and cache-full runs were identical. Guess: without a cache each process autotunes kernels
# afresh and may pick different ones. Two tests, one full mind each time (human, lawful world):
# 1. default settings, no cache, again: if it differs from the first no-cache run, the variation is the
#    GPU's run to run, not the cache;
# 2. full float32 with deterministic operations, the settings for reported work: no cache, cache empty,
#    cache full. All three must be identical for the cache to be safe there.
#
#   experiments/gpu/remote.sh run cache-check bash experiments/gpu/cache_check.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
py() { uv run --no-sync python "$@"; }
nocache() { env -u JAX_COMPILATION_CACHE_DIR "$@"; }

echo "== 1. default settings, no cache, a second time"
nocache uv run --no-sync python experiments/gpu/bench.py pc-gpu-nocache2 --parts e2e

echo "== 2. full float32, deterministic operations: no cache, cache empty, cache full"
export JAX_DEFAULT_MATMUL_PRECISION=highest XLA_FLAGS=--xla_gpu_deterministic_ops=true
nocache uv run --no-sync python experiments/gpu/bench.py pc-gpu-f32det-nocache --parts e2e
export JAX_COMPILATION_CACHE_DIR=$PWD/results/jax_cache_check
rm -rf "${JAX_COMPILATION_CACHE_DIR:?}"
py experiments/gpu/bench.py pc-gpu-f32det-cache-cold --parts e2e
py experiments/gpu/bench.py pc-gpu-f32det-cache-warm --parts e2e
echo "== cache check done"
