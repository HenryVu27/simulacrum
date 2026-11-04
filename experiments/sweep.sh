#!/bin/bash
# Main sweep, four single-threaded queues (several single-threaded processes beat one multi-threaded
# one on this laptop). Each run resumes if interrupted: finished minds are skipped.
cd "$(dirname "$0")/.."
mkdir -p results/logs
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
run() { uv run umwelt run --world "$1" --seed "$2" --steps 4000 $3 >> "results/logs/$1_s$2.log" 2>&1; }
( run lawful 0; run ghost 0 --no-controls ) &
( run lawful_mj 0; run ghost_mj 0 --no-controls ) &
( run lawful 1 --no-controls; run nonlocal 0 --no-controls ) &
( run impermanent 0 --no-controls; run impermanent_mj 0 --no-controls ) &
wait
uv run umwelt report > results/logs/report.log 2>&1
echo "sweep finished $(date)" >> results/logs/report.log
