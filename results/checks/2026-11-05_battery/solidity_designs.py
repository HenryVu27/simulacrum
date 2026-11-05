"""Solidity designs compared on the way to the second correction of 2026-11-05 (docs/preregistration.md). Copied
from the scratch script that produced the numbers quoted in the record (only the order in which sections run was
tidied), so the record has its source. Each row: the
human's rays on internal walls (or, for E, on the event wall alone), and the copy predictor's S on noise-free
observations (0 = balanced) for five vision minds. A: agent off the wall line, wall slid away for "no wall"; C and D:
an occluder only in the no-wall movies; E: a screen in all four movies (E3 is the design adopted in voe.py).

    uv run python results/checks/2026-11-05_battery/solidity_designs.py > results/checks/2026-11-05_battery/solidity_designs.txt
"""
"""Candidate solidity designs: copy-predictor surprise (noise-free inputs) per vision mind, and the human's rays on the
wall. Bystanders are off the arena here (static and non-occluding, they cannot change copy balance)."""
import sys
import numpy as np
from umwelt import kinds, presets, voe
from umwelt.senses import RAY_REL

OFF = np.array([0.5, -1.0, 1.5, -1.0], np.float32)
HUMAN = np.abs(np.rad2deg(RAY_REL)) <= 90
NAMES = ["human", "vision_only", "dog", "bat", "chimera"]

def clean(p):
    return presets.Preset(p.name, tuple(presets.Sense(**{**s.__dict__, "noise": 0.0}) for s in p.senses), p.uses_action, p.recurrent)

def quad(kind, j, eye_dx, eye_dy, h, off_fn):
    r = kinds.RADIUS[kind]
    speed, x0, y = 0.15, 2.5, 5.0 + j
    n_in = int(np.ceil((6.0 - r - x0) / speed)); tc = voe.PRE + n_in; T = tc + 24; t = np.arange(T)
    x_in = np.clip(x0 + speed * np.maximum(t - voe.PRE, 0), None, 6.0 - r)
    x_pass = x0 + speed * np.maximum(t - voe.PRE, 0)
    x_bounce = np.where(t < tc, x_in, (6.0 - r) - 0.8 * speed * (t - tc))
    imp = np.zeros((T, voe.N_SLOTS)); imp[tc, 0] = (1 + 0.8) * speed / voe.DT * kinds.MASS[kind]
    eye = np.array([6.0 + eye_dx, y + eye_dy])
    on = np.array([6.0, y - h, 6.0, y + h], np.float32)
    off = np.asarray(off_fn(eye, y, h), np.float32)
    combos = {"P1": (on, x_bounce, imp), "P2": (off, x_pass, 0 * imp), "I1": (on, x_pass, 0 * imp), "I2": (off, x_bounce, imp)}
    act = np.zeros((T, voe.N_SLOTS), bool); act[:, 0] = True
    th = np.arctan2(y - eye[1], 6.0 - eye[0]) if eye_dy < 0 else np.pi / 2
    th = np.pi / 2
    movies = []
    for v in voe.VARIANTS:
        w, x, im = combos[v]
        pos = np.zeros((T, voe.N_SLOTS, 2), np.float32); pos[:, 0] = np.stack([x, np.full(T, y)], -1)
        movies.append(voe._render(eye, th, np.r_[w[None], np.tile(OFF, (2, 1))], voe._slots(kind), pos, act, im))
    return movies, tc

def evaluate(label, eye_dx, eye_dy, h, off_fn):
    movies, crit = [], []
    for k in voe.INERT:
        for j in (-0.5, 0.0, 0.5):
            m, tc = quad(k, j, eye_dx, eye_dy, h, off_fn); movies.append(m); crit.append(tc)
    T = min(len(m[0]["oracle"]) for m in movies)
    d = {f"obs_{k}": np.stack([np.stack([mv[v][k][:T] for v in range(4)]) for mv in movies]).astype(np.float16).reshape(-1, T, *movies[0][0][k].shape[1:]) for k in voe.OBS_KEYS}
    d["action"] = np.zeros((len(movies) * 4, T, 2), np.float32)
    crit = np.array(crit)
    on_rays = np.mean([((np.abs(m[0]["vision"][voe.PRE:c, :, :3] - kinds.INTERNAL_COLOR).max(-1) < 1e-6) & HUMAN).sum(-1).mean() for m, c in zip(movies, crit)])
    off_rays = np.mean([((np.abs(m[1]["vision"][voe.PRE:c, :, :3] - kinds.INTERNAL_COLOR).max(-1) < 1e-6) & HUMAN).sum(-1).mean() for m, c in zip(movies, crit)])
    out = []
    for n in NAMES:
        o = presets.observe(d, clean(presets.PRESETS[n]))
        s, _ = voe.surprise(voe.copy_errors(o), crit, n_boot=1)
        out.append(f"{n} {s:+.4f}")
    print(f"{label:58} wall rays on {on_rays:4.1f} off {off_rays:4.1f} | " + "  ".join(out), flush=True)

slid_up = lambda eye, y, h: [6.0, y + 0.8, 6.0, y + 0.8 + 2 * h]
for dx in (-1.0, -0.7, -0.5, -0.3, -0.15):
    evaluate(f"A/B agent ({6+dx:.2f}, y-2) h=1 off=slid up", dx, -2.0, 1.0, slid_up)

def scaled(s, ext):
    """Off wall: the on-wall's lower part (below the path) scaled toward the agent by s, so it covers the same
    angles (hides the passing object at the same frames), extended along its line to length ext."""
    def f(eye, y, h):
        a = eye + s * (np.array([6.0, y - h]) - eye)
        b = eye + s * (np.array([6.0, y]) - eye)
        u = (b - a) / np.linalg.norm(b - a)
        a = b - u * max(ext, np.linalg.norm(b - a))
        return [a[0], a[1], b[0], b[1]]
    return f
for s in (0.5, 0.6):
    for h in (1.0, 1.5):
        for ext in (0.0, 1.5):
            evaluate(f"C occluder s={s} h={h} ext={ext} agent (5, y-2)", -1.0, -2.0, h, scaled(s, ext))

print("\nD: oblique full-length occluder below the path, between the rays of the on-wall's lower part")
def oblique(delta_deg, length, clear=0.8):
    def f(eye, y, h):
        th_hi = np.arctan2(y - eye[1], 6.0 - eye[0]) - np.deg2rad(delta_deg)
        th_lo = np.arctan2(y - h - eye[1], 6.0 - eye[0])
        d1 = (y - clear - eye[1]) / np.sin(th_hi)
        p1 = eye + d1 * np.array([np.cos(th_hi), np.sin(th_hi)])
        # d2 on the lower ray so that |p1 - p2| = length
        u = np.array([np.cos(th_lo), np.sin(th_lo)]); w = p1 - eye
        b = -2 * w @ u; c = w @ w - length ** 2
        d2 = (-b + np.sqrt(b * b - 4 * c)) / 2
        p2 = eye + d2 * u
        assert p2[1] <= y - clear + 1e-6, ("too close to the path", p2, y)
        return [p2[0], p2[1], p1[0], p1[1]]
    return f
for h in (1.5,):
    for delta in (0.0, 2.0, 4.0, 6.0, 8.0):
        evaluate(f"D oblique h={h} delta={delta} len=1.52 agent (5, y-2)", -1.0, -2.0, h, oblique(delta, 1.52))

def evaluate_common(label, delta, length=1.52):
    """E: D's oblique occluder in all four movies (so both pass movies hide the object the same way), the event wall
    on the path (h=1) or slid up clear of it."""
    occ = oblique(delta, length)
    keep = voe._render
    def render(eye, th, walls, *a):
        y = eye[1] + 2.0
        return keep(eye, th, np.r_[walls[:1], np.asarray(occ(eye, y, 1.5), np.float32)[None], walls[2:]], *a)
    voe._render = render
    try:
        evaluate(label, -1.0, -2.0, 1.0, slid_up)
    finally:
        voe._render = keep
print("\nE: a screen in all four movies (D's occluder), the wall 1 above the path, slid away for no wall")
for delta in (0.0, 2.0, 4.0, 6.0):
    evaluate_common(f"E common occluder delta={delta} on h=1 off=slid up", delta)

def quad_e(kind, j, ax, h_lo, h_up, h_occ, event_on=True):
    r = kinds.RADIUS[kind]
    speed, x0, y = 0.15, 2.5, 5.0 + j
    n_in = int(np.ceil((6.0 - r - x0) / speed)); tc = voe.PRE + n_in; T = tc + 24; t = np.arange(T)
    x_in = np.clip(x0 + speed * np.maximum(t - voe.PRE, 0), None, 6.0 - r)
    x_pass = x0 + speed * np.maximum(t - voe.PRE, 0)
    x_bounce = np.where(t < tc, x_in, (6.0 - r) - 0.8 * speed * (t - tc))
    imp = np.zeros((T, voe.N_SLOTS)); imp[tc, 0] = (1 + 0.8) * speed / voe.DT * kinds.MASS[kind]
    eye = np.array([ax, y - 2.0])
    on = np.array([6.0, y - h_lo, 6.0, y + h_up], np.float32)
    off = np.array([6.0, y + OFF_GAP, 6.0, y + OFF_GAP + h_lo + h_up], np.float32)
    occ = np.asarray(oblique(0.0, 1.52)(eye, y, h_occ), np.float32)
    if not event_on:
        on = off = OFF
    combos = {"P1": (on, x_bounce, imp), "P2": (off, x_pass, 0 * imp), "I1": (on, x_pass, 0 * imp), "I2": (off, x_bounce, imp)}
    act = np.zeros((T, voe.N_SLOTS), bool); act[:, 0] = True
    movies = []
    for v in voe.VARIANTS:
        w, x, im = combos[v]
        pos = np.zeros((T, voe.N_SLOTS, 2), np.float32); pos[:, 0] = np.stack([x, np.full(T, y)], -1)
        movies.append(voe._render(eye, np.pi / 2, np.r_[w[None], occ[None], OFF[None]], voe._slots(kind), pos, act, im))
    return movies, tc, occ

def evaluate_e(ax, h_lo, h_up, h_occ):
    movies, crit, seen_on, seen_off, occs = [], [], [], [], []
    for k in voe.INERT:
        for j in (-0.5, 0.0, 0.5):
            m, tc, occ = quad_e(k, j, ax, h_lo, h_up, h_occ); movies.append(m); crit.append(tc); occs.append(occ)
            bare, _, _ = quad_e(k, j, ax, h_lo, h_up, h_occ, event_on=False)
            seen_on.append(((m[0]["vision"][voe.PRE:tc] != bare[0]["vision"][voe.PRE:tc]).any(-1) & HUMAN).sum(-1).mean())
            seen_off.append(((m[1]["vision"][voe.PRE:tc] != bare[1]["vision"][voe.PRE:tc]).any(-1) & HUMAN).sum(-1).mean())
    T = min(len(m[0]["oracle"]) for m in movies)
    d = {f"obs_{k}": np.stack([np.stack([mv[v][k][:T] for v in range(4)]) for mv in movies]).astype(np.float16).reshape(-1, T, *movies[0][0][k].shape[1:]) for k in voe.OBS_KEYS}
    d["action"] = np.zeros((len(movies) * 4, T, 2), np.float32)
    crit = np.array(crit)
    occs = np.array(occs); mid = (occs[:, :2] + occs[:, 2:]) / 2; half = np.linalg.norm(occs[:, 2:] - occs[:, :2], axis=-1) / 2
    ok = ((mid >= 2.5) & (mid <= 7.5)).all() and ((half >= 0.75) & (half <= 1.5)).all()
    out = []
    for n in NAMES:
        o = presets.observe(d, clean(presets.PRESETS[n]))
        s, _ = voe.surprise(voe.copy_errors(o), crit, n_boot=1)
        out.append(f"{n} {s:+.4f}")
    print(f"E agent x={ax} wall [-{h_lo}, +{h_up}] occluder to -{h_occ}: event wall rays on {np.mean(seen_on):4.1f} off {np.mean(seen_off):4.1f}, "
          f"occluder in range {ok} | " + "  ".join(out), flush=True)

OFF_GAP = 0.8
print("\nE, other agent positions and wall heights (event wall rays: the event wall alone)")
for ax in (5.0, 4.5, 4.0):
    for h_up in (1.0, 1.5):
        try:
            evaluate_e(ax, 1.0, h_up, 1.5)
        except AssertionError as e:
            print(f"E agent x={ax} h_up={h_up}: {e}")

print("\nE as adopted in voe.py: wall 1.4 above the path, slid 0.75 clear of it for no wall")
OFF_GAP = 0.75
evaluate_e(5.0, 1.0, 1.4, 1.5)
