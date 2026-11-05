"""Claim 4: the copy predictor's S on noise-free observations, my own S and (for human, vision_only) my own
observation code; the solidity movies without the screen; and perturbations of screen, eye and heading."""
import sys
import numpy as np
from common import OFF, ORIG_RENDER, Capture, voe
from umwelt import presets
from umwelt.senses import RAY_REL

MINDS = ("human", "vision_only", "dog", "bat", "chimera")
SPACING = 2 * np.pi / 128


def noise_free(p):
    return presets.Preset(p.name, tuple(presets.Sense(**{**x.__dict__, "noise": 0.0}) for x in p.senses), p.uses_action, p.recurrent)


def my_obs(m, name):
    """My own reading of the human / vision_only channels (presets.py), from one movie's float16 obs."""
    v = m["vision"].astype(np.float32)[:, np.abs(np.rad2deg(RAY_REL)) <= 90].reshape(len(v := m["vision"]), -1) if False else None
    vis = m["vision"].astype(np.float32)
    vis = vis[:, np.abs(np.rad2deg(RAY_REL)) <= 90, :].reshape(vis.shape[0], -1)
    if name == "vision_only":
        return vis
    return np.concatenate([vis, m["hearing"].astype(np.float32), m["touch"].astype(np.float32), m["proprio"].astype(np.float32),
                           np.log1p(2.0 * np.maximum(m["smell"].astype(np.float32), 0))], -1)


def my_S(errs, crit):
    """errs (n, 4, T-1) one-step errors; S = (I1+I2-P1-P2)/(P1+P2) over transitions into frames >= crit."""
    tot = np.array([e[:, c - 1:].sum(-1) for e, c in zip(errs, crit)])
    P, I = tot[:, 0] + tot[:, 1], tot[:, 2] + tot[:, 3]
    return (I.sum() - P.sum()) / P.sum(), (I - P) / P


def score(quads_movies, crit, minds=MINDS, mine=False):
    T = min(len(q[0]["oracle"]) for q in quads_movies)
    out = {}
    for name in minds:
        if mine and name in ("human", "vision_only"):
            errs = []
            for q in quads_movies:
                o = np.stack([my_obs({k: v[:T].astype(np.float16) for k, v in mv.items()}, name) for mv in q])
                errs.append(np.mean((o[:, 1:] - o[:, :-1]) ** 2, -1))
            out[name] = my_S(np.array(errs), crit)
        else:
            d = {f"obs_{k}": np.stack([np.stack([mv[k][:T] for mv in q]) for q in quads_movies]).astype(np.float16).reshape(-1, T, *quads_movies[0][0][k].shape[1:])
                 for k in quads_movies[0][0]}
            d["action"] = np.zeros((len(quads_movies) * 4, T, 2), np.float32)
            o = presets.observe(d, noise_free(presets.PRESETS[name]))
            e = np.mean((o[:, 1:] - o[:, :-1]) ** 2, -1).reshape(len(quads_movies), 4, -1)
            out[name] = my_S(e, crit)
    return out


def base_calls(test):
    calls, crit = [], []
    makers = {"permanence": lambda k, j: voe.permanence(k, j), "solidity": lambda k, j: voe.solidity(k, j)}
    if test == "causality":
        pairs = [((a, b), j) for a, b in zip(voe.INERT, voe.INERT[1:] + voe.INERT[:1]) for j in (-0.5, 0.0, 0.5)]
        for (a, b), j in pairs:
            with Capture(voe) as cap:
                _, tc = voe.causality(a, b, j)
            calls.append((cap.calls, j)); crit.append(tc)
    else:
        for k in voe.INERT:
            for j in (-0.5, 0.0, 0.5):
                with Capture(voe) as cap:
                    _, tc = makers[test](k, j)
                calls.append((cap.calls, j)); crit.append(tc)
    return calls, np.array(crit)


def rerender(calls, screen=None, eye_shift=(0.0, 0.0), dth=0.0, recompute_screen=False):
    out = []
    for cs, j in calls:
        y = 5.0 + j
        q = []
        for c in cs:
            internal = c["internal"].copy()
            agent = c["agent"] + np.array(eye_shift)
            if recompute_screen:
                internal[1] = voe._solidity_screen(agent, y)
            if screen is not None:
                internal[1] = screen(internal[1].astype(np.float64), y)
            q.append(ORIG_RENDER(agent, c["th"] + dth, internal, c["kind"], c["pos"].astype(np.float32), c["active"], c["impulse"]))
        out.append(q)
    return out


def fmt(res):
    return ", ".join(f"{k} {v[0]:+.4f}" for k, v in res.items())


if __name__ == "__main__":
    print("== new movies, as committed: copy S on noise-free float16 observations (presets.observe), and my own reading")
    for test in voe.TESTS:
        calls, crit = base_calls(test)
        movies = [[c["obs"] for c in cs] for cs, _ in calls]
        res = score(movies, crit)
        res_m = score(movies, crit, minds=("human", "vision_only"), mine=True)
        print(f"  {test:10} {fmt(res)} | mine: {fmt(res_m)}")
        if test == "solidity":
            print("    per-quadruplet S (human): " + " ".join(f"{x:+.4f}" for x in res["human"][1]))
            print("    per-quadruplet S (chimera): " + " ".join(f"{x:+.4f}" for x in res["chimera"][1]))
            sol_calls, sol_crit = calls, crit
    calls, crit = sol_calls, sol_crit
    print("\n== solidity without the screen (screen slot moved off the arena; same bystander)")
    print("  " + fmt(score(rerender(calls, screen=lambda s, y: OFF.astype(np.float64)), crit)))
    print("\n== solidity with the screen perturbed")
    def shift(dx, dy):
        return lambda s, y: s + np.array([dx, dy, dx, dy])
    def rotate(deg):
        def f(s, y):
            c = (s[:2] + s[2:]) / 2
            R = np.array([[np.cos(np.deg2rad(deg)), -np.sin(np.deg2rad(deg))], [np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg))]])
            return np.r_[c + R @ (s[:2] - c), c + R @ (s[2:] - c)]
        return f
    def scale(f_):
        return lambda s, y: np.r_[(s[:2] + s[2:]) / 2 + f_ * (s[:2] - (s[:2] + s[2:]) / 2), (s[:2] + s[2:]) / 2 + f_ * (s[2:] - (s[:2] + s[2:]) / 2)]
    def extend_right(dl):  # move the far end (first end, p2) along the screen's direction
        def f(s, y):
            d = (s[:2] - s[2:]) / np.linalg.norm(s[:2] - s[2:])
            return np.r_[s[:2] + dl * d, s[2:]]
        return f
    def extend_left(dl):  # move the near end (second end, p1)
        def f(s, y):
            d = (s[2:] - s[:2]) / np.linalg.norm(s[2:] - s[:2])
            return np.r_[s[:2], s[2:] + dl * d]
        return f
    perts = []
    for d in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
        perts += [(f"shift x {d:+.3f}", shift(d, 0)), (f"shift x {-d:+.3f}", shift(-d, 0)), (f"shift y {d:+.3f}", shift(0, d)), (f"shift y {-d:+.3f}", shift(0, -d))]
    for deg in (0.25, 0.5, 1.0, 2.0):
        perts += [(f"rotate {deg:+.2f} deg", rotate(deg)), (f"rotate {-deg:+.2f} deg", rotate(-deg))]
    for dl in (0.01, 0.05, 0.1):
        perts += [(f"near end +{dl}", extend_left(dl)), (f"near end -{dl}", extend_left(-dl)), (f"far end +{dl}", extend_right(dl)), (f"far end -{dl}", extend_right(-dl))]
    for label, f in perts:
        print(f"  {label:18} " + fmt(score(rerender(calls, screen=f), crit)), flush=True)
    print("\n== solidity with the agent moved or turned, screen rebuilt by voe._solidity_screen for the new eye (the design rule)")
    for frac in (0.1, 0.25, 0.5, 0.75):
        for sgn in (1, -1):
            print(f"  heading {sgn * frac:+.2f} ray spacings  " + fmt(score(rerender(calls, dth=sgn * frac * SPACING, recompute_screen=True), crit)), flush=True)
    for dx, dy in ((0.02, 0), (-0.02, 0), (0.05, 0), (-0.05, 0), (0, 0.02), (0, -0.02), (0, 0.05), (0, -0.05)):
        print(f"  eye shift ({dx:+.2f},{dy:+.2f})       " + fmt(score(rerender(calls, eye_shift=(dx, dy), recompute_screen=True), crit)), flush=True)
