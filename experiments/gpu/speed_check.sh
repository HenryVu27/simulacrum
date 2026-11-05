#!/usr/bin/env bash
# Two questions left by E6 (docs/gpu.md):
# 1. The training loop as experiments run it, timed properly (two run lengths, so compiling is separated
#    from the cost of an update), at the default precision and at full float32.
# 2. Why one full mind gains only 1.6 times on the GPU against 2.4 for its training: the guess is compile
#    time on many small functions. One full mind three times: no compile cache, then a persistent compile
#    cache that starts empty, then the same cache full. If the guess holds, only the last run is faster.
#
#   experiments/gpu/remote.sh run speed-check bash experiments/gpu/speed_check.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
py() { uv run --no-sync python "$@"; }

echo "== training loop, default precision, then full float32"
py experiments/gpu/bench.py pc-gpu --parts train
py experiments/gpu/bench.py pc-gpu-highest --precision highest --parts train

echo "== one full mind: no cache"
py experiments/gpu/bench.py pc-gpu-nocache --parts e2e
cache=$PWD/results/jax_cache
rm -rf "$cache"
export JAX_COMPILATION_CACHE_DIR=$cache JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
py -c "import jax; print('cache dir', jax.config.jax_compilation_cache_dir, 'min seconds', jax.config.jax_persistent_cache_min_compile_time_secs)"
echo "== one full mind: cache empty"
py experiments/gpu/bench.py pc-gpu-cache-cold --parts e2e
echo "== one full mind: cache full"
py experiments/gpu/bench.py pc-gpu-cache-warm --parts e2e
echo "cache: $(du -sh "$cache" | cut -f1), $(find "$cache" -type f | wc -l) entries"
echo "== speed check done"
