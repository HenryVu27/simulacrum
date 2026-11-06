"""Check 5: the tracking mind's own head in the permanence movies, rebuilt independently: true position from each
movie's oracle channel (allocentric, by kind) and the agent's pose there, hidden frames from the rendered rays
(object colour) and, separately, from geometry; plus the same head on held-out recorded episodes."""
import pickle
import jax, jax.numpy as jnp, numpy as np
from common import *  # noqa
from umwelt import learner, voe, kinds
from umwelt.world import ARENA

saved = pickle.load(open(REPO / "results/v3/stage0/track_params.pkl", "rb"))
P, mu, sd = saved["params"], saved["mu"], saved["sd"]
hp = presets.PRESETS["human"]

def head(h):
    return np.asarray(learner._lin(P["trk2"], jax.nn.gelu(learner._lin(P["trk1"], h))))

def ego(rel, th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]], -1)

rows = []
traj = {}
for k in voe.INERT:
    for j in (-0.5, 0.0, 0.5):
        movies, tv = voe.permanence(k, j)
        m = movies[0]
        Tm = len(m["oracle"])
        orc = m["oracle"]
        apos = orc[:, 0:2] * ARENA
        th = np.arctan2(orc[:, 2], orc[:, 3])
        opos = orc[:, 6 + 5 * k + 1: 6 + 5 * k + 3] * ARENA
        active = orc[:, 6 + 5 * k] > 0.5
        assert active.all()
        true = ego(opos - apos, th) / ARENA * 4
        # hidden by rendered colour on any ray
        col = m["vision"][:, :, :3]
        seen_col = (np.abs(col - kinds.COLOR[k]).max(-1) < 1e-6).any(-1)
        # hidden by geometry: no tangent-limited ray from the eye reaches the disc without crossing the screen
        eye = apos[0]
        scr = np.array([[5.0, 4.0 + j], [5.0, 6.0 + j]])
        r = kinds.RADIUS[k]
        geo_hidden = np.zeros(Tm, bool)
        for t in range(Tm):
            c = opos[t]
            # sample points across the disc's visible chord (perpendicular to the sight line)
            u = (c - eye) / np.linalg.norm(c - eye)
            nrm = np.array([-u[1], u[0]])
            pts = c[None] + np.linspace(-r, r, 41)[:, None] * nrm[None]
            def crosses(p):
                # does the segment eye->p cross the screen segment x=5, y in [4+j, 6+j]?
                if (p[0] - 5.0) * (eye[0] - 5.0) > 0:
                    return False
                y = eye[1] + (p[1] - eye[1]) * (5.0 - eye[0]) / (p[0] - eye[0])
                return scr[0, 1] <= y <= scr[1, 1]
            geo_hidden[t] = all(crosses(p) for p in pts)
        o = (presets.observe({f"obs_{kk}": m[kk][None] for kk in m} | {"action": np.zeros((1, Tm, 2), np.float32)}, hp) - mu) / sd
        h = learner.latents(P, jnp.asarray(o, jnp.float32), jnp.zeros((1, Tm, 2), jnp.float32))
        out = head(h)[0].reshape(Tm, 8, 2)[:, k]
        t = np.arange(Tm)
        hidden = ~seen_col & (t >= voe.PRE)
        last = np.maximum.accumulate(np.where(seen_col, t, -1))
        assert (last[hidden] >= 0).all()
        frozen = np.linalg.norm(true[last] - true, axis=-1)
        err = np.linalg.norm(out - true, axis=-1)
        # extrapolate at the last seen velocity (what a tracker of moving objects would do)
        vel = true[1:] - true[:-1]
        extrap = true[last] + (t - last)[:, None] * vel[np.maximum(last - 1, 0)]
        ext = np.linalg.norm(extrap - true, axis=-1)
        before = (t >= voe.PRE) & seen_col & (t < tv)
        after = seen_col & (t > tv)
        rows.append(dict(k=k, j=j, n_hidden=int(hidden.sum()), agree_geo=int((hidden == (geo_hidden & (t >= voe.PRE))).sum()), T=Tm,
                         head_hidden=err[hidden].mean(), frozen=frozen[hidden].mean(), extrap=ext[hidden].mean(),
                         head_seen_before=err[before].mean(), head_seen_after=err[after].mean(), head_pre=err[: voe.PRE].mean(),
                         head_x_hidden=out[hidden, 0].mean(), head_y_hidden=out[hidden, 1].mean()))
        traj[(k, j)] = (true, out, seen_col)
print(f"{'kind':>4} {'j':>5} {'T':>3} {'nhid':>4} {'geo=col':>7} | head err: pre-move  seen-before  HIDDEN  seen-after | frozen  extrap | head mean (x, y) while hidden")
for r in rows:
    print(f"{r['k']:>4} {r['j']:+5.1f} {r['T']:>3} {r['n_hidden']:>4} {r['agree_geo']:>4}/{r['T']} | {r['head_pre']:.3f} {r['head_seen_before']:.3f} "
          f"{r['head_hidden']:.3f} {r['head_seen_after']:.3f} | {r['frozen']:.3f} {r['extrap']:.3f} | ({r['head_x_hidden']:+.2f}, {r['head_y_hidden']:+.2f})")
w = np.array([r["n_hidden"] for r in rows])
for key in ("head_hidden", "frozen", "extrap", "head_seen_before", "head_seen_after", "head_pre"):
    v = np.array([r[key] for r in rows])
    print(f"{key:18s} frame-weighted mean {np.average(v, weights=w):.3f}" if key in ("head_hidden", "frozen", "extrap") else f"{key:18s} movie mean {v.mean():.3f}")
print("total hidden frames", w.sum())
true, out, seen = traj[(voe.INERT[0], 0.0)]
print("\nkind", voe.INERT[0], "j 0: t, seen, true (x,y), head (x,y)")
for t in range(0, len(true), 3):
    print(f"  {t:3d} {int(seen[t])} ({true[t,0]:+.2f},{true[t,1]:+.2f}) ({out[t,0]:+.2f},{out[t,1]:+.2f})")
pickle.dump(traj, open(SCR / "c5_traj.pkl", "wb"))
