#!/usr/bin/env bash
# Can GPU jobs use JAX's cuda_async allocator? Inside WSL the default (BFC) allocator cannot get one block
# of 4 GiB or more, so any compiled step that needs that much working memory fails (128 minds at once
# needed 8.12 GiB); cuda_async can. It is adopted only if training gives the same weights (deterministic
# operations, against pause_check's hash 69122ab38190b194) and is no slower. docs/gpu.md, setup log.
#
#   experiments/gpu/remote.sh run alloc-check bash experiments/gpu/alloc_check.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
py() { uv run --no-sync python "$@"; }
out=results/hardware/pause_check
echo "== same weights under cuda_async (deterministic operations)"
XLA_FLAGS=--xla_gpu_deterministic_ops=true XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
  py experiments/gpu/pause_check.py "$out/alloc_async.json" --steps 3000
echo "== speed, default allocator, then cuda_async"
py experiments/gpu/bench.py pc-gpu-bfc --parts update train
XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async py experiments/gpu/bench.py pc-gpu-async --parts update train
echo "== alloc check done"
