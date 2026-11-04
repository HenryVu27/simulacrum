"""Object kinds: the fixed table of what each kind of thing is like in every sense."""

import numpy as np

N_KINDS = 8
N_SCENT = 4
N_BANDS = 4

# Red and green are close on the blue-yellow axis on purpose, so a dichromat confuses them.
COLOR = np.array(
    [
        [0.90, 0.10, 0.10],  # red
        [0.10, 0.80, 0.20],  # green
        [0.10, 0.20, 0.90],  # blue
        [0.90, 0.90, 0.10],  # yellow
        [0.80, 0.10, 0.80],  # magenta
        [0.10, 0.80, 0.80],  # cyan
        [1.00, 0.50, 0.00],  # orange
        [0.95, 0.95, 0.95],  # white
    ],
    dtype=np.float32,
)

_rng = np.random.default_rng(0)
# Scent signature per kind; strength varies so some kinds are faint.
SCENT = (_rng.dirichlet(np.ones(N_SCENT), size=N_KINDS) * np.array([1.0, 0.6, 1.2, 0.3, 0.9, 1.1, 0.5, 0.8])[:, None]).astype(np.float32)
# Sound spectrum per kind, heard when it collides (and, for animate kinds, as a faint hum).
SPECTRUM = _rng.dirichlet(np.ones(N_BANDS) * 0.7, size=N_KINDS).astype(np.float32)
TEXTURE = _rng.uniform(0.0, 1.0, size=N_KINDS).astype(np.float32)
TEMPERATURE = _rng.uniform(0.0, 1.0, size=N_KINDS).astype(np.float32)
RADIUS = np.array([0.35, 0.40, 0.45, 0.30, 0.35, 0.40, 0.50, 0.30], dtype=np.float32)
MASS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 0.7], dtype=np.float32)
ANIMATE = np.array([0, 0, 0, 0, 0, 1, 0, 1], dtype=bool)

# Walls: four boundary walls with faint distinct tints (landmarks), then internal occluders.
BOUNDARY_COLOR = np.array(
    [[0.60, 0.60, 0.72], [0.72, 0.60, 0.60], [0.60, 0.72, 0.60], [0.66, 0.66, 0.66]], dtype=np.float32
)
INTERNAL_COLOR = np.array([0.30, 0.30, 0.30], dtype=np.float32)
WALL_TEXTURE = 0.8
WALL_TEMPERATURE = 0.4
WALL_SPECTRUM = np.array([0.6, 0.3, 0.1, 0.0], dtype=np.float32)
