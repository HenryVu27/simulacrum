"""_solidity_screen says 'only arithmetic and square roots, so every machine gets the same wall'. It uses w @ u (a
BLAS dot on many builds, which may or may not fuse multiply-add). Recompute the screen with and without fused
multiply-adds in every dot and see whether the float32 wall differs; and check the bystanders' endpoints lie on the
1/256 grid in float32."""
import math
import numpy as np
from common import Capture, voe


def screen(eye, y, fma):
    dot = (lambda a, b: math.fma(a[1], b[1], a[0] * b[0])) if fma else (lambda a, b: a[0] * b[0] + a[1] * b[1])
    p1 = eye + (np.array([6.0, y]) - eye) * (y - 0.8 - eye[1]) / (y - eye[1])
    u = np.array([6.0, y - 1.5]) - eye
    u = u / math.sqrt(dot(u, u)) if fma else u / np.linalg.norm(u)
    w = p1 - eye
    wu = dot(w, u)
    p2 = eye + (wu + math.sqrt(wu ** 2 - dot(w, w) + 1.52 ** 2)) * u
    return np.array([*p2, *p1], np.float32), np.array([*p2, *p1], np.float64)


for j in (-0.5, 0.0, 0.5):
    y = 5.0 + j
    eye = np.array([5.0, y - 2.0])
    ref = voe._solidity_screen(eye, y)
    a32, a64 = screen(eye, y, False)
    b32, b64 = screen(eye, y, True)
    gap32 = []
    for v in a64:  # distance (in float32 ulps) from a float32 rounding boundary
        f = np.float32(v)
        nxt = np.nextafter(f, np.float32(np.inf)) if v > f else np.nextafter(f, np.float32(-np.inf))
        gap32.append(abs(v - (float(f) + float(nxt)) / 2) / abs(float(nxt) - float(f)))
    print(f"j {j:+.1f}: voe {ref.tolist()} | plain==voe {np.array_equal(a32, ref)} | fma==voe {np.array_equal(b32, ref)} | "
          f"float64 plain-vs-fma max diff {np.abs(a64 - b64).max():.2e} | nearest float32 tie, in float32 ulps: {min(gap32):.3f}")

off = []
for k in voe.INERT:
    for j in (-0.5, 0.0, 0.5):
        for test, make in (("permanence", voe.permanence), ("solidity", voe.solidity)):
            with Capture(voe) as cap:
                make(k, j)
            W = cap.calls[0]["internal"]
            bys = W[1:] if test == "permanence" else W[2:]
            off.append(np.abs(bys * 256 - np.round(bys * 256)).max())
print("bystander endpoints off the 1/256 grid by at most", max(off))
