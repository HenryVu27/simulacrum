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
#   S="uv run python experiments/gpu/compare_stage3.py"; H=results/hardware
#   for t in pc-gpu pc-gpu-highest pc-gpu-pcdata; do $S $H/mac-cpu/stage3_ref $H/$t/stage3; done  # E4
#   $S $H/pc-cpu/stage3 $H/pc-gpu-pcdata/stage3                         # E4, same episodes: GPU against CPU only
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
# Other GPU load at the start (Henry's desktop: an animated wallpaper and the compositor used about 20%)
for _ in 1 2 3 4 5; do /usr/lib/wsl/lib/nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; sleep 1; done
py experiments/gpu/bench.py pc-gpu
py experiments/gpu/bench.py pc-gpu-highest --precision highest --parts update train
JAX_PLATFORMS=cpu py experiments/gpu/bench.py pc-wsl-cpu --parts update train  # the PC's CPU under Linux

echo "== E4: calibration stage 3 end to end, default precision, then full float32"
# Each pass gets its own results folder, so each records its own episodes under its own precision.
stage3() {  # tag, precision
  root=results/stage3_runs/$1  # outside results/hardware, so pulling that folder skips the recordings
  UMWELT_RESULTS=$root JAX_DEFAULT_MATMUL_PRECISION=$2 py experiments/calibrate.py dark --episodes 8192
  mkdir -p "results/hardware/$1/stage3"
  cp "$root"/calibration/errors/dark_n8192_* "$root"/calibration/dark_n8192.json "results/hardware/$1/stage3/"
}
stage3 pc-gpu default
stage3 pc-gpu-highest highest

# E4 once more on the very episodes the PC's CPU run used (recorded under Windows, results/hardware/pc-cpu/stage3),
# so against that run only the arithmetic differs: GPU against CPU, same bytes in.
pcdata=${PC_CPU_DATA:-/mnt/c/umwelt/results/data/dark_s0_e8448_t256}
if [ -d "$pcdata" ]; then
  echo "== E4 on the PC CPU run's episodes ($pcdata)"
  mkdir -p results/stage3_runs/pc-gpu-pcdata/data
  rm -rf results/stage3_runs/pc-gpu-pcdata/data/dark_s0_e8448_t256
  cp -r "$pcdata" results/stage3_runs/pc-gpu-pcdata/data/
  stage3 pc-gpu-pcdata default
else
  echo "== skipped E4 on the PC CPU run's episodes: $pcdata not found"
fi
echo "== all checks done"
