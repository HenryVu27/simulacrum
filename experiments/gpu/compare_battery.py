"""E7b after the surprise-battery redesign (docs/gpu.md; docs/preregistration.md, second correction of 2026-11-05).

Four checks on the GPU's rerun on the Mac's recording (results/hardware/e7/f32det-macdata-battery):
0. both scored the same movies: the GPU files' provenance names the stimuli the Mac's recomputed files name;
1. every mind's surprise scores, interval ends and copy baselines (trained and untrained networks) against the
   Mac's recomputed ones, number by number;
2. every number against the GPU's previous E7b run (f32det-macdata-wallfix): only surprise numbers may differ,
   since the redesign changed the movies and nothing else, and whole GPU runs at these settings repeat bit for bit;
3. every number against the Mac's stored run, mind by mind.
Timings are left out. Nothing is written; the output goes to results/hardware/compare_e7b_battery.log.
Later reruns name their own folders (the scores over noise draws: --new f32det-macdata-noise --old
f32det-macdata-battery, output compare_e7b_noise.log).

    uv run python experiments/gpu/compare_battery.py [--new NAME] [--old NAME]
"""

import argparse
import json
from pathlib import Path

MAC = Path("results/runs/main/lawful_s0")
NEW = Path("results/hardware/e7/f32det-macdata-battery/lawful_s0")
OLD = Path("results/hardware/e7/f32det-macdata-wallfix/lawful_s0")
SKIP = {"timing", "seconds", "provenance", "corrections", "voe_before_wall_fix", "voe_untrained_before_wall_fix",
        "voe_before_battery_fix", "voe_untrained_before_battery_fix", "voe_before_noise_draws", "voe_untrained_before_noise_draws"}
CHANGED = ("voe/", "voe_untrained/")
FIELDS = (("S", lambda s: s["S"]), ("ci/0", lambda s: s["ci"][0]), ("ci/1", lambda s: s["ci"][1]), ("S_copy", lambda s: s["S_copy"]))


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if not path and k in SKIP:
                continue
            yield from leaves(v, f"{path}{k}/")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}{i}/")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield path.rstrip("/"), float(x)


def table(f):
    return dict(leaves(json.loads(f.read_text())))


def minds(d):
    return sorted(f for f in d.glob("*.json") if not f.name.startswith("compare_"))  # not compare_e7.py's summary


def main():
    print("0. the movies each side scored on (provenance stimuli)")
    hashes = {(side, f.stem): json.loads(f.read_text())["provenance"].get("stimuli") for side, d in (("Mac", MAC), ("GPU", NEW))
              for f in minds(d) if f.stem != "translation"}  # the Mac's translation file predates provenance records
    print("  " + ", ".join(f"{side} {h}: {sum(v == h for (s, _), v in hashes.items() if s == side)} minds"
                           for side, h in sorted({(s, v) for (s, _), v in hashes.items()})))
    assert len(set(hashes.values())) == 1, "the two sides scored different movies"

    print("\n1. every surprise number, GPU rerun against the Mac's recomputed values")
    worst = {}
    for f in minds(NEW):
        mac, gpu = json.loads((MAC / f.name).read_text()), json.loads(f.read_text())
        for k in ("voe", "voe_untrained"):
            for test in gpu.get(k, {}):
                for field, get in FIELDS:
                    d = abs(get(gpu[k][test]) - get(mac[k][test]))
                    key = f"{k}/{field}"
                    if d >= worst.get(key, (-1,))[0]:
                        worst[key] = (d, f.stem, test, get(mac[k][test]), get(gpu[k][test]))
    for key, (d, mind, test, m, g) in sorted(worst.items()):
        print(f"  largest {key:24} difference {d:.1e}  ({mind} {test}: Mac {m:+.7f}, GPU {g:+.7f})")

    print("\n2. every number against the GPU's previous E7b run: only surprise numbers may differ")
    for f in minds(NEW):
        new, old = table(f), table(OLD / f.name)
        assert new.keys() == old.keys(), f"{f.stem}: different fields"
        diff = [k for k in new if new[k] != old[k]]
        allowed = [k for k in diff if k.startswith(CHANGED)]
        other = sorted(set(diff) - set(allowed))
        print(f"  {f.stem:15} {len(new):5} numbers, {len(allowed):3} surprise numbers differ, "
              f"{len(other)} others{': ' + ', '.join(other[:5]) if other else ''}")
        assert not other, f"{f.stem}: numbers changed that the redesign cannot reach"

    print("\n3. every number against the Mac's stored run")
    overall = 0.0
    for f in minds(NEW):
        new, mac = table(f), table(MAC / f.name)
        common = new.keys() & mac.keys()
        d = max(abs(new[k] - mac[k]) for k in common)
        k = max(common, key=lambda k: abs(new[k] - mac[k]))
        overall = max(overall, d)
        print(f"  {f.stem:15} {len(common):5} numbers, largest difference {d:.1e} ({k})")
    print(f"  largest difference over all minds: {overall:.1e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="f32det-macdata-battery")
    ap.add_argument("--old", default="f32det-macdata-wallfix")
    args = ap.parse_args()
    NEW = Path(f"results/hardware/e7/{args.new}/lawful_s0")
    OLD = Path(f"results/hardware/e7/{args.old}/lawful_s0")
    main()
