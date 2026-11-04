#!/usr/bin/env bash
# One-time setup of umwelt inside WSL2 Ubuntu on the GPU PC. remote.sh copies the code to ~/umwelt and
# runs this there; it is safe to run again. See docs/gpu.md.
#
# Needs: WSL2 with Ubuntu, and the Windows NVIDIA driver (WSL passes it through as
# /usr/lib/wsl/lib/libcuda.so). Do not install a Linux NVIDIA driver inside WSL.
set -euo pipefail
cd "$(dirname "$0")/../.."

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
uv run --frozen python - <<'EOF'
import jax, jax.numpy as jnp
print("jax", jax.__version__, "backend", jax.default_backend(), "devices", jax.devices())
assert jax.default_backend() == "gpu", "JAX did not find the GPU"
x = jax.random.normal(jax.random.PRNGKey(0), (4096, 4096))
y = (x @ x).block_until_ready()
print("4096 x 4096 matmul ok, mean", float(y.mean()))
EOF

echo "== tests"
uv run --frozen pytest -q
