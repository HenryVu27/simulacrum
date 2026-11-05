"""How much GPU memory can one JAX process get inside WSL? (docs/gpu.md, setup log.)

JAX's default reservation (75 percent of the card) failed at setup with only 1.5 GB in use, and
settled near 4 GB. This finds the real limit two ways: one array of each size in turn (around 4 GiB
finely), then 1 GiB arrays held together until an allocation fails. Run it under each allocator; the
output is the record (speed_check.sh saves it to results/hardware/memprobe_<allocator>.txt).

    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python experiments/gpu/memprobe.py
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async uv run python experiments/gpu/memprobe.py
"""

import json
import os

import jax
import jax.numpy as jnp

GIB = 2**30


def alloc(gib):
    return jnp.ones((int(gib * GIB) // 4,), jnp.float32).block_until_ready()


def main():
    dev = jax.devices()[0]
    print("device", dev, "preallocate", os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
          "allocator", os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default"))
    print("memory_stats", json.dumps(dev.memory_stats()))
    single = {}
    for g in (2, 3.5, 3.99, 4.0, 4.01, 6, 8.2, 10, 12):
        try:
            x = alloc(g)
            single[g] = True
            del x
        except Exception as e:  # noqa: BLE001
            single[g] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
    print("one array of N GiB:", json.dumps(single))
    held = []
    try:
        for _ in range(16):
            held.append(alloc(1))
    except Exception as e:  # noqa: BLE001
        print("held arrays stopped:", type(e).__name__, str(e).splitlines()[0][:120])
    print(f"held {len(held)} x 1 GiB at once")
    print("memory_stats", json.dumps(dev.memory_stats()))


if __name__ == "__main__":
    main()
