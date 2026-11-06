#!/usr/bin/env bash
# Round 1 close (docs/round1.md): the memory-cut control for both dark-world learner seeds, saved. The Stage 0 rerun's
# independent check ran it once and kept only a summary; this runs that check's own script, unchanged except for the
# repository path it names, on the stored v2 dark-world human, memoryless, passive and passive_novest minds (the last two
# have no efference copy), and writes its full output to
# results/v3/round1_memory_cut. Each line compares the fixed probe's scores at 1 to 5 and 6 to 20 steps, dark and lit,
# for the trained network, the same network with its hidden state reset every step, and raw senses.
#
#   experiments/gpu/remote.sh run NAME env JAX_DEFAULT_MATMUL_PRECISION=highest \
#       XLA_FLAGS=--xla_gpu_deterministic_ops=true bash experiments/round1_memory_cut.sh
set -euo pipefail
cd "$(dirname "$0")/.."
src=results/v3/stage0_rerun/independent_check/c8_rescore_sample.py
out=results/v3/round1_memory_cut
mkdir -p "$out"
tmp=$(mktemp -d)
trap 'rm -rf "${tmp:?}"' EXIT
sed "s#/Users/henryvu/personal/simulacrum#$PWD#" "$src" > "$tmp/c8.py"
for run in v2/dark_s0 v2/dark_s1; do
  uv run python "$tmp/c8.py" "$run" human,memoryless,passive,passive_novest lesion 2>&1 | grep -v Warn | tee "$out/${run//\//_}.txt"
done
