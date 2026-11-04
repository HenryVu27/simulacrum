#!/bin/bash
# Amendment 2 confirmatory run: the chosen memory world, fresh data (data seed 1), learner seeds 0 and 1,
# all 14 minds. Four single-threaded queues; minds the hypotheses need come first in each queue.
#   experiments/sweep_v2.sh occlusion
set -u
W="$1"
cd "$(dirname "$0")/.."
mkdir -p results/logs
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
LOG="results/logs/v2_$W"
run() { uv run umwelt run --world "$W" --tag v2 --data-seed 1 --steps 4000 "$@"; }
# Record the world once before the queues start, so no two processes write it at the same time.
uv run python -c "from umwelt.data import save; save('$W', 1280, 256, 1)" > "${LOG}_data.log" 2>&1
queue() {
  local q="$1"; shift
  run --seed 0 --presets "$@" >> "${LOG}_q$q.log" 2>&1
  run --seed 1 --no-controls --presets "$@" >> "${LOG}_q$q.log" 2>&1
}
queue 1 human memoryless smell_only &
queue 2 passive_novest dog bat &
queue 3 vision_only oracle blind chimera &
queue 4 passive touch_only deafblind hearing_only &
wait
# Every mind is done, so these passes only load latents and write translation.json for each seed.
run --seed 0 >> "${LOG}_translate.log" 2>&1 &
run --seed 1 --no-controls >> "${LOG}_translate.log" 2>&1 &
uv run python experiments/posthoc_permanence.py "$W" 0 v2 1 > "${LOG}_posthoc_s0.log" 2>&1 &
uv run python experiments/posthoc_permanence.py "$W" 1 v2 1 > "${LOG}_posthoc_s1.log" 2>&1 &
wait
echo "v2 sweep finished $(date)" >> "${LOG}_translate.log"
