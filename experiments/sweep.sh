#!/bin/bash
# Main sweep: 4 Flatland worlds x 2 seeds, plus the MuJoCo audit (3 worlds x 1 seed).
# Three queues run in parallel; each run resumes if interrupted (finished minds are skipped).
cd "$(dirname "$0")/.."
mkdir -p results/logs
run() { uv run umwelt run --world "$1" --seed "$2" --steps 4000 $3 >> "results/logs/$1_s$2.log" 2>&1; }
( run lawful 0; run impermanent 0; run lawful_mj 0 ) &
( run ghost 0; run nonlocal 0; run impermanent_mj 0 ) &
( run lawful 1 --no-controls; run impermanent 1 --no-controls; run ghost 1 --no-controls; run nonlocal 1 --no-controls; run ghost_mj 0 ) &
wait
uv run umwelt report > results/logs/report.log 2>&1
echo "sweep finished $(date)" >> results/logs/report.log
