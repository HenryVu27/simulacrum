"""What the training world's physics does when an inert object rolls into a wall at the movies' speed:
the impulse (sound) of the bounce per kind, the rebound speed, and friction before contact."""
import jax
import jax.numpy as jnp
import numpy as np
from umwelt import kinds
from umwelt.world import FAR, N_SLOTS, WORLDS, reset, step

for k in (0, 1, 3, 6):
    s = reset(jax.random.PRNGKey(0))
    r = float(kinds.RADIUS[k])
    walls = s.walls.at[4].set(jnp.array([6.0, 3.0, 6.0, 7.0])).at[5:].set(jnp.array([0.5, -1.0, 1.5, -1.0]))
    pos = jnp.full((N_SLOTS, 2), FAR).at[0].set(jnp.array([6.0 - r - 0.6, 5.0]))
    vel = jnp.zeros((N_SLOTS, 2)).at[0].set(jnp.array([1.5, 0.0]))
    s = s._replace(agent_pos=jnp.array([2.0, 1.0]), agent_vel=jnp.zeros(2), obj_pos=pos, obj_vel=vel,
                   obj_active=jnp.zeros(N_SLOTS, bool).at[0].set(True), obj_kind=s.obj_kind.at[0].set(k), walls=walls)
    xs, imps, vs = [], [], []
    for t in range(12):
        s, c = step(s, jnp.zeros(2), WORLDS["lawful"], jax.random.PRNGKey(t))
        xs.append(float(s.obj_pos[0, 0])); imps.append(float(c.obj_impulse[0])); vs.append(float(s.obj_vel[0, 0]))
    t_hit = int(np.argmax(np.abs(imps)))
    print(f"kind {k} mass {kinds.MASS[k]}: world impulse at the bounce {imps[t_hit]:.3f} (movie uses {(1.8) * 0.15 / 0.1 * kinds.MASS[k]:.3f}); "
          f"speed just before {vs[t_hit-1]:.3f}, just after {vs[t_hit]:.3f} (ratio {-vs[t_hit]/vs[t_hit-1]:.3f}); "
          f"per-step speed decay before contact {vs[1]/vs[0]:.3f} (movies: 1.000)")
