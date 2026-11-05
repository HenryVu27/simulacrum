"""Do the M2-mutant bystanders (no clearance from eye-object sight lines) hide anything? All 54 quadruplets, same
criterion as claim 2, and which quadruplets the unit test (kind 6 and causality (6, 0), j = -0.5, 0.5) covers."""
import importlib.util
import numpy as np
from common import BYSTANDER_SLOTS, OFF, ORIG_RENDER, S, Capture, quads
src = (S / "repo_c35/src/umwelt/voe.py").read_text().replace("gap = [np.ones(1), r + 0.3, r + 0.1]", "gap = [np.ones(1), r + 0.3, 0 * r - 1]")
spec = importlib.util.spec_from_loader("voe_m2", loader=None)
m2 = importlib.util.module_from_spec(spec)
exec(compile(src, "voe_m2", "exec"), m2.__dict__)
from umwelt import kinds
bad = []
for key, make in quads(m2):
    with Capture(m2) as cap:
        make()
    for v, c in zip(m2.VARIANTS, cap.calls):
        wo = c["internal"].copy()
        wo[BYSTANDER_SLOTS[key[0]]] = OFF
        b = ORIG_RENDER(c["agent"], c["th"], wo, c["kind"], c["pos"].astype(np.float32), c["active"], c["impulse"])
        ch = (c["obs"]["vision"] != b["vision"]).any(-1)
        was = b["vision"][ch][:, :3]
        boundary = (np.abs(was[:, None] - np.asarray(kinds.BOUNDARY_COLOR)[None]).max(-1) < 1e-6).any(-1)
        other = [k for k in ("hearing", "smell", "touch", "proprio", "compass", "ray_tex") if not np.array_equal(c["obs"][k], b[k])]
        if (~boundary).any() or other:
            bad.append((key, v, int((~boundary).sum()), other))
print(f"M2 bystanders that hide something: {len(set(b[0] for b in bad))} quadruplets: " + "; ".join(f"{k} {v}: {n} ray-frames lost, senses {o}" for k, v, n, o in bad))
