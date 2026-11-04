#!/bin/bash
# Resume the lawful seed-1 run once a slot frees up, then write the final report when everything is done.
cd "$(dirname "$0")/.."
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
until [ "$(pgrep -f 'python.*umwelt run' | wc -l)" -le 3 ]; do sleep 30; done
uv run umwelt run --world lawful --seed 1 --steps 4000 --no-controls >> results/logs/lawful_s1.log 2>&1
until [ "$(pgrep -f 'python.*umwelt run' | wc -l)" -eq 0 ]; do sleep 30; done
uv run umwelt report > results/logs/report_final.log 2>&1
echo "followup finished $(date)" >> results/logs/report_final.log
