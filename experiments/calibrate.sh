#!/bin/bash
# Amendment 2 calibration: human and memoryless in each v2 world, plus lawful as a reproduction check.
cd "$(dirname "$0")/.."
mkdir -p results/logs
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for w in lookback dark occluders occlusion lawful; do
  uv run python experiments/calibrate.py "$w" > "results/logs/calib_$w.log" 2>&1 &
done
wait
grep -h "M = " results/logs/calib_*.log
