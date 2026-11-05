"""E7b after the surprise-test stimulus fix (docs/gpu.md; docs/preregistration.md, 2026-11-05).

Three checks on the GPU's rerun on the Mac's recording (results/hardware/e7/f32det-macdata-wallfix):
1. the dog's and the oracle's surprise scores, interval ends and copy baselines against the Mac's corrected
   ones, number by number (the PASS line of compare_e7.py judges families at their median and 95th
   percentile, so it could hide one wrong score);
2. every number against the GPU's earlier E7b run (f32det-macdata): only the scores whose movies changed may
   differ, since whole GPU runs at these settings repeat bit for bit;
3. every number against the Mac's stored run, mind by mind.
Timings are left out. Nothing is written; the output goes to results/hardware/compare_e7b_wallfix.log.

    uv run python experiments/gpu/compare_wallfix.py
"""

import json
from pathlib import Path

MAC = Path("results/runs/main/lawful_s0")
NEW = Path("results/hardware/e7/f32det-macdata-wallfix/lawful_s0")
OLD = Path("results/hardware/e7/f32det-macdata/lawful_s0")
SKIP = {"timing", "seconds", "provenance", "corrections", "voe_before_wall_fix", "voe_untrained_before_wall_fix"}
CHANGED = {"dog": ("voe/permanence", "voe_untrained/permanence"), "oracle": ("voe/", "voe_untrained/")}


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


def surprise(mind):
    mac, new, old = (json.loads((d / f"{mind}.json").read_text()) for d in (MAC, NEW, OLD))
    worst = 0.0
    for k in ("voe", "voe_untrained"):
        for test, v in new[k].items():
            for field, get in (("S", lambda s: s["S"]), ("ci/0", lambda s: s["ci"][0]), ("ci/1", lambda s: s["ci"][1]),
                               ("S_copy", lambda s: s["S_copy"])):
                m, g = get(mac[k][test]), get(v)
                before = get(mac[f"{k}_before_wall_fix"][test])
                worst = max(worst, abs(g - m))
                print(f"  {mind:7} {k + '/' + test + '/' + field:32} Mac {m:+.7f}  GPU {g:+.7f}  diff {g - m:+.1e}"
                      f"   (GPU before the fix {get(old[k][test]):+.7f}, Mac before {before:+.7f})")
    return worst


def minds(d):
    return sorted(f for f in d.glob("*.json") if not f.name.startswith("compare_"))  # not compare_e7.py's summary


def main():
    print("1. surprise scores of the two minds whose movies changed, GPU rerun against the Mac's corrected values")
    worst = max(surprise(m) for m in CHANGED)
    print(f"  largest difference: {worst:.1e}\n")

    print("2. every number against the GPU's earlier E7b run: only the changed movies' scores may differ")
    for f in minds(NEW):
        mind = f.stem
        new, old = table(f), table(OLD / f.name)
        assert new.keys() == old.keys(), f"{mind}: different fields"
        diff = [k for k in new if new[k] != old[k]]
        allowed = [k for k in diff if any(k.startswith(p) for p in CHANGED.get(mind, ()))]
        other = sorted(set(diff) - set(allowed))
        print(f"  {mind:15} {len(new):5} numbers, {len(allowed):3} differ where the movies changed, "
              f"{len(other)} elsewhere{': ' + ', '.join(other[:5]) if other else ''}")
        assert not other, f"{mind}: numbers changed that the stimulus fix cannot reach"

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
    main()
