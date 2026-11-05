"""The redesigned surprise-test movies, checked on every quadruplet (docs/preregistration.md, second correction of
2026-11-05). The unit tests check a few quadruplets; this checks all 54 and measures what the record quotes:

1. every movie's three walls lie in the range world.reset draws from, and a quadruplet's movies share bystanders;
2. bystander walls hide nothing: rendered with them and with them off the arena, only rays that met a boundary
   wall change, and only to meet a nearer wall; no other sense changes, nor the oracle's non-wall inputs;
   two walls placed to block something on purpose must be caught by the same criterion;
3. solidity, previous movies (voe.py as of recompute_battery.BEFORE) and new: how many of the human's rays tell
   the wall-in-the-path movies from the wall-slid-away movies before the contact, and in how many frames after the
   contact the passing object is seen, with the wall in its path and without;
4. balance: S of the copy predictor (next frame = this frame) on noise-free observations, per mind and test,
   previous and new movies. A balanced quadruplet gives 0 for every mind whose senses do not see the hidden
   event (the oracle's input includes it).

    uv run python experiments/battery_stimuli.py > results/checks/2026-11-05_battery/stimuli.txt
"""

import importlib.util
from pathlib import Path

import numpy as np

from umwelt import kinds, presets, voe
from umwelt.senses import RAY_REL

spec = importlib.util.spec_from_file_location("recompute_battery", Path(__file__).with_name("recompute_battery.py"))
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)

OFF = np.array([0.5, -1.0, 1.5, -1.0], np.float32)  # the old marker: nothing reaches it
BOUNDARY = np.asarray(kinds.BOUNDARY_COLOR)
HUMAN = np.abs(np.rad2deg(RAY_REL)) <= 90
KEEP = voe._bystanders


def quads():
    for k in voe.INERT:
        for j in (-0.5, 0.0, 0.5):
            yield ("permanence", k, j), lambda k=k, j=j: voe.permanence(k, j)[0]
            yield ("solidity", k, j), lambda k=k, j=j: voe.solidity(k, j)[0]
    for a, b in zip(voe.INERT, voe.INERT[1:] + voe.INERT[:1]):
        for j in (-0.5, 0.0, 0.5):
            yield ("causality", a, b, j), lambda a=a, b=b, j=j: voe.causality(a, b, j)[0]


def walls(m):
    return m["oracle"][0, -12:].reshape(3, 4) * 10.0


def in_range(w):
    mid, half = (w[:, :2] + w[:, 2:]) / 2, np.linalg.norm(w[:, 2:] - w[:, :2], axis=-1) / 2
    return bool(((2.5 - 1e-5 <= mid) & (mid <= 7.5 + 1e-5)).all() and ((0.75 - 1e-5 <= half) & (half <= 1.5 + 1e-5)).all())


def hides_nothing(A, B):
    """A rendered with bystanders, B with them off the arena. Returns (ok, ray-frames the bystanders take)."""
    taken = 0
    for a, b in zip(A, B):
        for k in a:
            if k == "vision":
                ch = (a[k] != b[k]).any(-1)
                if not (np.abs(b[k][ch][:, None, :3] - BOUNDARY[None]).max(-1) < 1e-6).any(-1).all():
                    return False, 0
                if not (a[k][ch][:, 3] < b[k][ch][:, 3]).all():
                    return False, 0
                taken += int(ch.sum())
            elif k == "oracle":
                if not np.array_equal(a[k][:, :-12], b[k][:, :-12]):
                    return False, 0
            elif not np.array_equal(a[k], b[k]):
                return False, 0
    return True, taken


def noise_free(p):
    return presets.Preset(p.name, tuple(presets.Sense(**{**x.__dict__, "noise": 0.0}) for x in p.senses), p.uses_action, p.recurrent)


def off_arena(n, *a):
    return np.tile(OFF, (n, 1))


def main():
    print("1 and 2. every quadruplet: walls in the training range, shared bystanders, bystanders hide nothing")
    bad, unseen = [], 0
    for key, make in quads():
        voe._bystanders = KEEP
        A = make()
        voe._bystanders = off_arena
        B = make()
        shared = slice(1, 3) if key[0] == "solidity" else slice(0, 3)
        ok_range = all(in_range(walls(m)) for m in A)
        ok_shared = all(np.array_equal(walls(m)[shared], walls(A[0])[shared]) for m in A)
        ok_hide, taken = hides_nothing(A, B)
        unseen += taken == 0
        if not (ok_range and ok_shared and ok_hide):
            bad.append(key)
        print(f"  {str(key):32} walls in range {ok_range}, bystanders shared {ok_shared}, hide nothing {ok_hide}, "
              f"ray-frames on bystanders {taken}")
    print(f"  {54 - len(bad)} of 54 quadruplets pass; {unseen} have bystanders no ray meets (edge-on or in a shadow)")

    voe._bystanders = off_arena
    B = voe.permanence(0, 0.0)[0]
    for what, w in (("across the object's path", [6.0, 7.0, 7.0, 7.0]), ("between the agent and the screen", [3.5, 4.2, 3.5, 5.8])):
        voe._bystanders = lambda n, *a, w=w: np.r_[np.array([w], np.float32), np.tile(OFF, (n - 1, 1))]
        print(f"  a wall placed {what} on purpose is caught: {not hides_nothing(voe.permanence(0, 0.0)[0], B)[0]}")

    print("\n3. solidity: rays (of the human's) that differ between wall-in-path and wall-slid-away movies before the "
          "contact; frames after the contact in which the passing object is seen")
    old = rb.previous_voe()
    voe._bystanders = KEEP  # the mutation checks above left it patched
    for label, make in (("previous", old.solidity), ("new", voe.solidity)):
        differ, seen = [], []
        for k in voe.INERT:
            for j in (-0.5, 0.0, 0.5):
                (p1, p2, i1, i2), tc = make(k, j)
                differ.append(((p1["vision"][voe.PRE:tc] != i2["vision"][voe.PRE:tc]).any(-1) & HUMAN).sum(-1))
                obj = [(np.abs(m["vision"][tc:, :, :3] - kinds.COLOR[k]).max(-1) < 1e-6).any(-1).sum() for m in (i1, p2)]
                seen.append(obj)
        differ, seen = np.concatenate(differ), np.array(seen)
        print(f"  {label:8} rays that differ per frame: min {differ.min()}, mean {differ.mean():.1f}, max {differ.max()}; "
              f"frames the passing object is seen (of 24), wall in path / slid away: mean {seen[:, 0].mean():.1f} / "
              f"{seen[:, 1].mean():.1f}, equal in {int((seen[:, 0] == seen[:, 1]).sum())} of 18 movies")

    print("\n4. balance: copy predictor's S on noise-free observations (0 = balanced), previous -> new movies")
    suites = {"previous": old.build_suite(), "new": voe.build_suite()}
    for test in voe.TESTS:
        row = []
        for name in ("human", "vision_only", "dog", "bat", "chimera", "blind", "hearing_only", "smell_only", "oracle"):
            S = []
            for suite in suites.values():
                d, crit = suite[test]
                with np.errstate(invalid="ignore"):
                    S.append(voe.surprise(voe.copy_errors(presets.observe(d, noise_free(presets.PRESETS[name]))), crit, n_boot=1)[0])
            row.append(f"{name} {S[0]:+.4f} -> {S[1]:+.4f}")
        print(f"  {test:10} " + ", ".join(row))
    voe._bystanders = KEEP


if __name__ == "__main__":
    main()
