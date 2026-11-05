#!/usr/bin/env bash
# Every hardware check on the GPU in one job (docs/gpu.md, E1 to E6). Rerun after a driver, CUDA or
# JAX upgrade, with new tags if the old results should be kept. From the Mac:
#
#   experiments/gpu/remote.sh push results/hardware/mac-cpu/data_dark.npz
#   experiments/gpu/remote.sh run gpu-checks bash experiments/gpu/checks.sh
#   experiments/gpu/remote.sh status gpu-checks
#   experiments/gpu/remote.sh pull results/hardware
#
# then compare on the Mac:
#
#   R="uv run python experiments/gpu/repro.py compare"
#   for t in pc-gpu pc-gpu-highest pc-gpu-det; do $R mac-cpu $t; done   # E1 to E3, E5 precision
#   $R pc-gpu pc-gpu-rerun; $R pc-gpu-det pc-gpu-det-rerun              # E5 run-to-run determinism
#   for t in pc-gpu pc-gpu-highest; do                                   # E4
#     uv run python experiments/gpu/compare_stage3.py results/hardware/mac-cpu/stage3_ref results/hardware/$t/stage3; done
#   uv run python experiments/gpu/bench_table.py                         # E6
set -euo pipefail
cd "$(dirname "$0")/../.."
py() { uv run --no-sync python "$@"; }  # never let uv change the environment (a plain sync drops the CUDA plugin)
ref=results/hardware/mac-cpu/data_dark.npz
det=--xla_gpu_deterministic_ops=true
repro() { py experiments/gpu/repro.py make "$@" --ref "$ref"; }

py -c "import jax; b = jax.default_backend(); print('backend', b, jax.devices()); assert b == 'gpu', 'JAX does not see the GPU'"
[ -f "$ref" ] || { echo "missing $ref: push it with the code (see the header)"; exit 1; }

echo "== E1 to E3, and E5: default precision, rerun, full float32, deterministic ops"
repro pc-gpu
repro pc-gpu-rerun
repro pc-gpu-highest --precision highest
XLA_FLAGS=$det repro pc-gpu-det
XLA_FLAGS=$det repro pc-gpu-det-rerun

echo "== E6: speed"
py experiments/gpu/bench.py pc-gpu
py experiments/gpu/bench.py pc-gpu-highest --precision highest --parts update train
JAX_PLATFORMS=cpu py experiments/gpu/bench.py pc-wsl-cpu --parts update train  # the PC's CPU under Linux

echo "== E4: calibration stage 3 end to end, default precision, then full float32"
# Each pass gets its own results folder, so each records its own episodes under its own precision.
for tag in pc-gpu pc-gpu-highest; do
  root=results/stage3_runs/$tag  # outside results/hardware, so pulling that folder skips the recordings
  if [ "$tag" = pc-gpu-highest ]; then prec=highest; else prec=default; fi
  UMWELT_RESULTS=$root JAX_DEFAULT_MATMUL_PRECISION=$prec py experiments/calibrate.py dark --episodes 8192
  mkdir -p "results/hardware/$tag/stage3"
  cp "$root"/calibration/errors/dark_n8192_* "$root"/calibration/dark_n8192.json "results/hardware/$tag/stage3/"
done
echo "== all checks done"
