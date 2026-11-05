"""Mutation tests: apply one defect to a copy of the committed voe.py and see which tests catch it."""
import shutil
import subprocess
import sys
from pathlib import Path

S = Path(__file__).resolve().parent
BASE = S / "repo_c35"
MUTS = {
    "M1 bystanders unconstrained (may hide anything)": ("if (_seg_dist(w[:2], w[2:], A, B) >= gap).all():", "if True:"),
    "M2 bystanders ignore sight lines to objects (gap 0 on eye-object lines)": ("gap = [np.ones(1), r + 0.3, r + 0.1]", "gap = [np.ones(1), r + 0.3, 0 * r - 1]"),
    "M3 solidity screen removed (off the arena)": ("    screen = _solidity_screen(eye, y)\n    places", "    screen = np.array([0.5, -1.0, 1.5, -1.0], np.float32)\n    places"),
    "M4 bystander centres drawn from [1, 9], no range check": ("        if not ((WALL_CENTRE[0] <= mid).all()", "        if False and not ((WALL_CENTRE[0] <= mid).all()"),
    "M5 solidity eye back on the wall's line (x = 6)": ("    eye = np.array([5.0, y - 2.0])", "    eye = np.array([6.0, y - 3.0])"),
    "M6 bystanders differ between the movies of a quadruplet": ("        movies.append(_render(eye, 0.0, walls,", "        walls = np.r_[screen[None], _bystanders(2, np.random.default_rng(len(movies)), eye, 0.0, screen[None], places)]\n        movies.append(_render(eye, 0.0, walls,"),
    "M7 permanence: object vanishes in plain view in I1 (unbalanced)": ('acts = {"P1": t >= 0, "P2": t < 0, "I1": t < tv, "I2": t >= tv}', 'acts = {"P1": t >= 0, "P2": t < 0, "I1": t < PRE + 2, "I2": t >= tv}'),
}
extra = {"M4 bystander centres drawn from [1, 9], no range check": ("mid = rng.uniform(*WALL_CENTRE, 2)", "mid = rng.uniform(1.0, 9.0, 2)")}
for name, (a, b) in MUTS.items():
    d = S / "repo_mut"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(BASE / "src", d / "src")
    shutil.copytree(BASE / "tests", d / "tests")
    shutil.copy(BASE / "pyproject.toml", d / "pyproject.toml")
    f = d / "src" / "umwelt" / "voe.py"
    src = f.read_text()
    assert src.count(a) == 1, (name, src.count(a))
    src = src.replace(a, b)
    if name in extra:
        assert src.count(extra[name][0]) == 1
        src = src.replace(*extra[name])
    f.write_text(src)
    r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-x", "--tb=line", "tests/test_core.py",
                        "-k", "walls or bystander or solidity or permanence_quadruplet"],
                       cwd=d, capture_output=True, text=True, env={"PYTHONPATH": str(d / "src"), "PATH": "/usr/bin:/bin"})
    lines = [l for l in r.stdout.splitlines() if "passed" in l or "failed" in l or "Error" in l or "assert" in l.lower()]
    print(f"{name}:\n    " + "\n    ".join(lines[-4:]), flush=True)
