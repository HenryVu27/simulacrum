"""Follow-up mutations: is M2 an equivalent mutant (same bystanders on all 54 quads)? M3b: screen replaced by an
in-range wall seen edge-on (hides nothing), so only the balance test can catch it. M7b: permanence object vanishes in
plain view inside the scoring window."""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent
BASE = S / "repo_c35"
sys.path.insert(0, str(S))
from common import Capture, quads, voe as voe0, BYSTANDER_SLOTS


def load(src, name):
    spec = importlib.util.spec_from_loader(name, loader=None)
    m = importlib.util.module_from_spec(spec)
    exec(compile(src, name, "exec"), m.__dict__)
    return m


orig = (BASE / "src/umwelt/voe.py").read_text()
m2 = load(orig.replace("gap = [np.ones(1), r + 0.3, r + 0.1]", "gap = [np.ones(1), r + 0.3, 0 * r - 1]"), "voe_m2")
same = differ = 0
for (key, make0), (_, make2) in zip(quads(voe0), quads(m2)):
    with Capture(voe0) as c0:
        make0()
    with Capture(m2) as c2:
        make2()
    eq = np.array_equal(c0.calls[0]["internal"], c2.calls[0]["internal"])
    same += eq
    differ += not eq
print(f"M2 (no clearance from eye-object sight lines): bystanders identical in {same} of 54 quadruplets, different in {differ}")

MUTS = {
    "M3b solidity screen replaced by an in-range wall seen edge-on": ("    screen = _solidity_screen(eye, y)\n    places", "    screen = np.array([6.2, y - 2.0, 7.72, y - 2.0], np.float32)\n    places"),
    "M7b permanence: object vanishes in plain view, inside the window (I1 = t < tv + 12)": ('"I1": t < tv,', '"I1": t < tv + 12,'),
}
for name, (a, b) in MUTS.items():
    d = S / "repo_mut"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(BASE / "src", d / "src"); shutil.copytree(BASE / "tests", d / "tests"); shutil.copy(BASE / "pyproject.toml", d / "pyproject.toml")
    f = d / "src/umwelt/voe.py"
    src = f.read_text()
    assert src.count(a) == 1, name
    f.write_text(src.replace(a, b))
    r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--tb=line", "tests/test_core.py",
                        "-k", "walls or bystander or solidity or permanence_quadruplet"], cwd=d, capture_output=True, text=True,
                       env={"PYTHONPATH": str(d / "src"), "PATH": "/usr/bin:/bin"})
    lines = [l for l in r.stdout.splitlines() if "passed" in l or "failed" in l or "Error" in l]
    print(f"{name}:\n    " + "\n    ".join(lines[-5:]), flush=True)
    if name.startswith("M7b"):  # what S does the test see, with and without noise?
        code = ("import numpy as np\nfrom umwelt import voe\nfrom umwelt.presets import PRESETS, observe\nfrom dataclasses import replace\n"
                "m, tc = voe.permanence(0, 0.0)\nd = {f'obs_{k}': np.stack([mv[k] for mv in m]) for k in m[0]}\n"
                "d['action'] = np.zeros((4, len(m[0]['oracle']), 2), np.float32)\n"
                "p = PRESETS['vision_only']\nq = replace(p, senses=tuple(replace(x, noise=0.0) for x in p.senses))\n"
                "print('    test S with noise', voe.surprise(voe.copy_errors(observe(d, p)), np.array([tc]), n_boot=1)[0], "
                "'| noise-free', voe.surprise(voe.copy_errors(observe(d, q)), np.array([tc]), n_boot=1)[0])\n")
        r = subprocess.run([sys.executable, "-c", code], cwd=d, capture_output=True, text=True, env={"PYTHONPATH": str(d / "src"), "PATH": "/usr/bin:/bin"})
        print(r.stdout.strip(), r.stderr.strip()[-300:])
