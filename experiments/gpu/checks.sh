#!/usr/bin/env bash
# Every hardware check on the GPU in one job (docs/gpu.md, E1 to E6). Rerun after a driver, CUDA or
# JAX upgrade. Start it from the Mac, then pull and compare there:
#
#   experiments/gpu/remote.sh push results/hardware/mac-cpu/data_dark.npz
#   experiments/gpu/remote.sh run gpu-checks bash experiments/gpu/checks.sh
#   experiments/gpu/remote.sh pull results/hardware
#   for t in pc-gpu pc-gpu-rerun pc-gpu-highest pc-gpu-det pc-gpu-det-rerun; do
#     uv run python experiments/gpu/repro.py compare mac-cpu $t; done
set -euo pipefail
cd "$(dirname "$0")/../.."
ref=results/hardware/mac-cpu/data_dark.npz
repro() { uv run --frozen python experiments/gpu/repro.py make "$@" --ref "$ref"; }
det=--xla_gpu_deterministic_ops=true

echo "== E1 to E3, and E5: default precision, rerun, full float32, deterministic ops"
repro pc-gpu
repro pc-gpu-rerun
repro pc-gpu-highest --precision highest
XLA_FLAGS=$det repro pc-gpu-det
XLA_FLAGS=$det repro pc-gpu-det-rerun

echo "== E6: speed"
uv run --frozen python experiments/gpu/bench.py pc-gpu
uv run --frozen python experiments/gpu/bench.py pc-gpu-highest --precision highest --parts update train
JAX_PLATFORMS=cpu uv run --frozen python experiments/gpu/bench.py pc-wsl-cpu --parts update train  # the PC's CPU under Linux

echo "== E4: calibration stage 3 end to end, default precision, then full float32"
for prec in default highest; do
  tag=pc-gpu; [ "$prec" = highest ] && tag=pc-gpu-highest
  if [ "$prec" = highest ]; then export JAX_DEFAULT_MATMUL_PRECISION=highest; fi
  uv run --frozen python experiments/calibrate.py dark --episodes 8192
  mkdir -p "results/hardware/$tag/stage3"
  mv results/calibration/errors/dark_n8192_* results/calibration/dark_n8192.json "results/hardware/$tag/stage3/"
done
echo "== all checks done"
