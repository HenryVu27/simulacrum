#!/bin/bash
# Amendment 4 calibration stage 3: 8,192 training episodes, 4,000 updates, one process per mind.
cd "$(dirname "$0")/.."
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for w in dark occlusion; do  # record first, one at a time, so no two processes write the same recording
  uv run python -c "from umwelt.data import load; load('$w', 8448, 256, 0)" > "results/logs/calib_n8192_data_$w.log" 2>&1
done
for w in dark occlusion; do
  for m in human memoryless; do
    uv run python experiments/calibrate.py "$w" --episodes 8192 --mind "$m" > "results/logs/calib_n8192_${w}_$m.log" 2>&1 &
  done
done
wait
grep -h "M = \|Traceback\|Error" results/logs/calib_n8192_*.log
