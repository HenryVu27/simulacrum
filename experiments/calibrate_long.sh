#!/bin/bash
# Amendment 3 calibration stage 2: human and memoryless at 16,000 updates, one process per mind.
cd "$(dirname "$0")/.."
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for w in dark occlusion lawful; do
  for m in human memoryless; do
    uv run python experiments/calibrate.py "$w" --steps 16000 --mind "$m" > "results/logs/calib16k_${w}_$m.log" 2>&1 &
  done
done
wait
grep -h "M = " results/logs/calib16k_*.log
