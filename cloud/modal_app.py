"""Run the umwelt sweep on Modal GPUs. One container per (world, seed); results land in a Modal volume.

    modal setup                                  # once: log in
    modal run cloud/modal_app.py                 # full sweep: 4 worlds x 2 seeds
    modal run cloud/modal_app.py --seeds 0,1,2,3 --steps 8000
    modal volume get umwelt-results runs results/   # pull results back, then: uv run umwelt report
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("umwelt")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("jax[cuda12]", "optax", "numpy", "scipy", "scikit-learn", "matplotlib")
    .add_local_dir(ROOT / "src" / "umwelt", "/root/umwelt")
)
volume = modal.Volume.from_name("umwelt-results", create_if_missing=True)


@app.function(image=image, gpu="L4", timeout=8 * 3600, volumes={"/results": volume})
def run(world: str, seed: int, steps: int, tag: str):
    import os

    os.environ["UMWELT_RESULTS"] = "/results"
    from umwelt import learner
    from umwelt.experiment import run_world
    from umwelt.presets import PRESETS

    run_world(world, seed, list(PRESETS), learner.LearnerConfig(steps=steps), tag=tag)
    volume.commit()


@app.local_entrypoint()
def main(worlds: str = "lawful,impermanent,ghost,nonlocal", seeds: str = "0,1", steps: int = 4000, tag: str = "cloud"):
    calls = [run.spawn(w, int(s), steps, tag) for w in worlds.split(",") for s in seeds.split(",")]
    for c in calls:
        c.get()
