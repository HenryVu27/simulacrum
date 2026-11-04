# Running umwelt in the cloud

The v1 sweep fits on a laptop (Apple M2 Pro, about five to six hours with three parallel queues). The cloud
path is for the next sizes: more seeds, longer training, the full sweep over every combination of sense
settings, and frontier-model probing.

No cloud account was set up for this project. Creating accounts and entering credentials is something
only Henry can do. Once an account exists, either path below works unchanged.

## Option 1: Modal (simplest)

```bash
pip install modal
modal setup                                   # opens a browser to log in
modal run cloud/modal_app.py                  # 4 worlds x 2 seeds, one L4 GPU container each
modal run cloud/modal_app.py --seeds 0,1,2,3,4 --steps 8000 --tag big
modal volume get umwelt-results runs results/ # copy results back
uv run umwelt report --tag big
```

Check the JAX CUDA wheel name against the JAX install guide before the first run; the image installs
`jax[cuda12]`, and newer JAX releases may name a CUDA 13 extra instead.

## Option 2: any machine with an NVIDIA GPU

```bash
git clone <this repo> && cd umwelt
uv sync && uv pip install "jax[cuda12]"
uv run umwelt data
bash experiments/sweep.sh
```

## What the GPU buys

The learner is small (a 128-unit recurrent network), so one run is only a few times faster on a GPU.
The gain is parallelism: dozens of (world, seed, mind) cells at once. That makes the factorial
sweep (hundreds of sense configurations instead of 14 presets) and five or more seeds per cell affordable.
