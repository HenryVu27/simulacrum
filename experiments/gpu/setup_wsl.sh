#!/usr/bin/env bash
# One-time setup of umwelt inside WSL2 Ubuntu on the GPU PC. remote.sh copies the code to ~/umwelt and
# runs this there; it is safe to run again. See docs/gpu.md.
#
# Needs: WSL2 with Ubuntu, and the Windows NVIDIA driver (WSL passes it through as
# /usr/lib/wsl/lib/libcuda.so). Do not install a Linux NVIDIA driver inside WSL.
#
# It runs outside the GPU lock (job.sh needs this setup first), so it stays light on a shared card: the GPU
# check is a small product run only when the card is nearly idle, without JAX's up-front reservation of 75
# percent of its memory, and the test suite runs on the CPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "== GPU seen from WSL"
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv

echo "== uv"
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "== packages (the locked set, plus JAX's CUDA 13 build)"
uv sync --frozen --extra gpu

echo "== JAX on the GPU"
used=$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' \r')
if ! [ "$used" -lt 6000 ] 2>/dev/null; then
  echo "skipped: the GPU is busy (${used} MiB in use, perhaps the other project); rerun setup later"
  exit 3
fi
JAX_PLATFORMS=cuda uv run --no-sync python - <<'EOF'  # --no-sync: nothing may change the environment after the sync above
import jax, jax.numpy as jnp
print("jax", jax.__version__, "backend", jax.default_backend(), "devices", jax.devices())
assert jax.default_backend() == "gpu", "JAX did not find the GPU"
x = jax.random.normal(jax.random.PRNGKey(0), (1024, 1024))
y = (x @ x).block_until_ready()
print("1024 x 1024 matmul ok, mean", float(y.mean()))
EOF

echo "== tests (on the CPU: they check the code, and leave the shared GPU alone)"
JAX_PLATFORMS=cpu uv run --no-sync pytest -q
