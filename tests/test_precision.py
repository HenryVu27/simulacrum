"""Every matrix product in the world simulation and the senses runs in full float32 (docs/gpu.md).

On a CPU the default precision is already full float32, so a product left at the default changes nothing
here and shows up only on a GPU (TF32 there flipped which object a ray hits). So this checks the traced
programs instead: every dot_general in a world rollout, and in the senses the surprise-test movies are
rendered with, must say HIGHEST.
"""

import re

import jax
import jax.numpy as jnp

from umwelt.data import rollout
from umwelt.senses import sense_all
from umwelt.world import N_SLOTS, WORLDS, Contacts, reset

DOT = re.compile(r"dot_general\[(.*?)\n\s*\]", re.DOTALL)
FULL = "precision=(Precision.HIGHEST, Precision.HIGHEST)"


def _products(jaxpr):
    blocks = DOT.findall(str(jaxpr))
    assert blocks, "no matrix products found: the check would pass vacuously"
    return blocks


def test_world_rollouts_use_full_precision():
    for name, rules in WORLDS.items():
        jaxpr = jax.make_jaxpr(rollout, static_argnums=(1, 2))(jax.random.PRNGKey(0), rules, 3)
        loose = [b for b in _products(jaxpr) if FULL not in b]
        assert not loose, f"{name}: {len(loose)} matrix products not at full precision: {loose[:1]}"


def test_senses_use_full_precision():
    s = reset(jax.random.PRNGKey(0))
    c = Contacts(jnp.zeros(N_SLOTS), jnp.zeros(N_SLOTS), jnp.float32(0.0))
    loose = [b for b in _products(jax.make_jaxpr(sense_all)(s, c)) if FULL not in b]
    assert not loose, f"{len(loose)} matrix products in the senses not at full precision: {loose[:1]}"
