#!/usr/bin/env bash
# Are the surprise-test movies the same on the GPU as on the CPUs, and does a job use the compile cache?
# (docs/gpu.md, E7b after the stimulus fix.) Hashes the rendered movies (voe.suite_hash, which every result's
# provenance now records) in three separate processes: at full float32 with deterministic operations twice
# (the second should read the first's compiles: faster, no new cache entries), then at the GPU's defaults.
# The Mac's CPU and the PC's CPU both give e8b112a5e2a11879. job.sh logs the cache's entries before and after.
#
#   experiments/gpu/remote.sh run stimuli-check bash experiments/gpu/stimuli_check.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
h() {
  local s=$SECONDS
  uv run --no-sync python -c "import jax; from umwelt import voe; print('suite_hash', voe.suite_hash(voe.build_suite()), jax.default_backend())"
  echo "  $((SECONDS - s)) s, cache entries $(find "$JAX_COMPILATION_CACHE_DIR" -name '*-cache' | wc -l)"
}
echo "== full float32, deterministic operations: once, then again from the cache"
(
  export JAX_DEFAULT_MATMUL_PRECISION=highest XLA_FLAGS=--xla_gpu_deterministic_ops=true
  h
  h
)
echo "== the GPU's defaults (TF32 products where a call does not name its precision, nondeterministic operations)"
h
