#!/usr/bin/env bash
# E7 (docs/gpu.md): a full run on the GPU, every mind and every test, to compare score by score with the
# Mac's stored run (results/runs/main/lawful_s0) using experiments/gpu/compare_e7.py. Lawful world, data
# seed 0, learner seed 0, default sizes. The run records its own world, as a real GPU run would. One
# configuration per job, so no job holds the shared GPU for long:
#
#   experiments/gpu/remote.sh run e7-f32det bash experiments/gpu/e7.sh f32det
#
#   f32det        full float32 and deterministic GPU operations (the candidate standard for real work)
#   f32det-rerun  the same again: does a whole run repeat bit for bit?
#   default       the GPU's defaults: TF32 matrix products, nondeterministic operations (the fast path)
#   f32det-macdata  as f32det, on the Mac's own recording (E7b): push results/data/lawful_s0_e1280_t256
#                 first; it is copied in, so only the arithmetic differs from the Mac's run
set -euo pipefail
cd "$(dirname "$0")/../.."
cfg=${1:?configuration: f32det, f32det-rerun or default}
case "$cfg" in
  f32det | f32det-rerun | f32det-macdata) export JAX_DEFAULT_MATMUL_PRECISION=highest XLA_FLAGS=--xla_gpu_deterministic_ops=true ;;
  default) ;;
  *) echo "unknown configuration: $cfg"; exit 2 ;;
esac
root=$PWD/results/e7_runs/$cfg  # its own world and run folder, outside results/hardware so a pull skips them
rm -rf "${root:?}"
rec=lawful_s0_e1280_t256
if [ "$cfg" = f32det-macdata ]; then
  [ -d "results/data/$rec" ] || { echo "push the Mac's recording first: remote.sh push results/data/$rec"; exit 2; }
  mkdir -p "$root/data"
  cp -r "results/data/$rec" "$root/data/"
fi
UMWELT_RESULTS=$root uv run --no-sync umwelt run --world lawful --seed 0 --tag e7
out=results/hardware/e7/$cfg/lawful_s0
mkdir -p "$out"
cp "$root"/runs/e7/lawful_s0/*.json "$out/"
{ echo "configuration $cfg"; echo "JAX_DEFAULT_MATMUL_PRECISION=${JAX_DEFAULT_MATMUL_PRECISION:-}"; echo "XLA_FLAGS=${XLA_FLAGS:-}"
  uv run --no-sync python -c "import jax; print('jax', jax.__version__, jax.default_backend(), jax.devices()[0].device_kind)"
  echo "recording $rec hash $(cd "$root/data/$rec" && sha256sum ./*.npy | sha256sum | cut -c1-16)"; } > "$out/../env.txt"
echo "== E7 $cfg done: $out"
