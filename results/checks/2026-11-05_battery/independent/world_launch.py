"""Causality movies vs the training world: A (moving 1.5/s) hits B at rest. Movie: A stops dead, B leaves at 1.5/s,
both get impulse 2.7. World: what speeds and impulses?"""
import jax
import jax.numpy as jnp
from umwelt import kinds
from umwelt.world import FAR, N_SLOTS, WORLDS, reset, step
for ka, kb in ((0, 1), (4, 6), (6, 0)):
    s = reset(jax.random.PRNGKey(0))
    ra, rb = float(kinds.RADIUS[ka]), float(kinds.RADIUS[kb])
    pos = jnp.full((N_SLOTS, 2), FAR).at[0].set(jnp.array([5.0 - ra - rb - 0.2, 6.0])).at[1].set(jnp.array([5.0, 6.0]))
    vel = jnp.zeros((N_SLOTS, 2)).at[0].set(jnp.array([1.5, 0.0]))
    walls = s.walls.at[4:].set(jnp.array([0.5, -1.0, 1.5, -1.0]))
    s = s._replace(agent_pos=jnp.array([2.0, 1.0]), agent_vel=jnp.zeros(2), obj_pos=pos, obj_vel=vel, walls=walls,
                   obj_active=jnp.zeros(N_SLOTS, bool).at[:2].set(True), obj_kind=s.obj_kind.at[0].set(ka).at[1].set(kb))
    for t in range(6):
        v_before = s.obj_vel[:2, 0]
        s, c = step(s, jnp.zeros(2), WORLDS["lawful"], jax.random.PRNGKey(t))
        if float(c.obj_impulse[0]) > 0:
            print(f"A kind {ka} (m {kinds.MASS[ka]}), B kind {kb} (m {kinds.MASS[kb]}): A {float(v_before[0]):.3f} -> {float(s.obj_vel[0,0]):+.3f}, "
                  f"B {float(v_before[1]):.3f} -> {float(s.obj_vel[1,0]):+.3f}; impulses A {float(c.obj_impulse[0]):.3f} B {float(c.obj_impulse[1]):.3f} "
                  f"(movie: A -> 0, B -> 1.5 times 0.94 decay, impulses 2.7, 2.7)")
            break
