"""E7 in docs/gpu.md: every score of a full run (14 minds, lawful world, learner seed 0) against the Mac's.

For each score, the hardware difference |other - Mac seed 0| is set against the learner-seed difference
|Mac seed 1 - Mac seed 0| of the same score. Scores are grouped into families. A family passes when both
its median and its 95th-percentile hardware difference are under a quarter of the same statistics of the
seed differences. S_copy (surprise computed from the stimuli alone, the same for every learner seed) has no
seed spread; it is reported as a check that the surprise-test movies match, not judged by the rule. Nor
are the controls (untrained network, raw senses): the stored seed-1 run skipped them, so they have no
seed yardstick and are reported only.

    uv run python experiments/gpu/compare_e7.py results/hardware/e7/f32det/lawful_s0
    uv run python experiments/gpu/compare_e7.py OTHER --ref results/runs/main/lawful_s0 --seed1 results/runs/main/lawful_s1
"""

import argparse
import json
from pathlib import Path

import numpy as np

FAMILIES = {  # a mind without a section (some runs skip the controls) contributes nothing to that family
    "test error": lambda m: {"test_error": m["test_error"]},
    "probes (MLP)": lambda m: m.get("probes", {}),
    "probes (linear)": lambda m: m.get("probes_linear", {}),
    "controls (untrained, raw)": lambda m: {**{f"untrained/{k}": v for k, v in m.get("probes_untrained", {}).items()},
                                            **{f"raw/{k}": v for k, v in m.get("probes_raw", {}).items()}},
    "surprise S": lambda m: {**{t: v["S"] for t, v in m.get("voe", {}).items()},
                             **{f"untrained/{t}": v["S"] for t, v in m.get("voe_untrained", {}).items()}},
}
BAR = 0.25


def scores(run):
    out = {fam: {} for fam in FAMILIES}
    for f in sorted(Path(run).glob("*.json")):
        if f.name.startswith("compare_"):  # this script's own reports
            continue
        if f.name == "translation.json":
            t = json.loads(f.read_text())
            names = t["names"]
            for key, mat in t.items():
                if key == "names":
                    continue
                m = np.asarray(mat, float)
                for i, a in enumerate(names):
                    for j, b in enumerate(names):
                        if i != j:
                            out.setdefault(f"translation ({key})", {})[f"{a}->{b}"] = float(m[i, j])
            continue
        m = json.loads(f.read_text())
        for fam, get in FAMILIES.items():
            for k, v in get(m).items():
                out[fam][f"{f.stem}/{k}"] = float(v)
    return out


def s_copy(run):
    return {f"{f.stem}/{t}": v["S_copy"] for f in sorted(Path(run).glob("*.json"))
            if f.name != "translation.json" and not f.name.startswith("compare_")
            for t, v in json.loads(f.read_text())["voe"].items()}


def main(other, ref, seed1):
    A, B, S1 = scores(ref), scores(other), scores(seed1)
    rep = {"ref": ref, "other": other, "seed1": seed1, "bar": BAR, "families": {}}
    print(f"{'family':28}{'n':>5}{'hw median':>11}{'seed median':>13}{'hw p95':>9}{'seed p95':>10}  verdict  (worst score)")
    ok_all, complete = True, True
    for fam in A:
        B.setdefault(fam, {})
        missing = sorted(set(A[fam]) ^ set(B[fam]))
        complete &= not (set(A[fam]) - set(B[fam]))  # a score the reference has and this run lacks
        keys = sorted(set(A[fam]) & set(B[fam]) & set(S1.get(fam, {})))
        if not keys:  # no seed-1 values to measure against (the seed-1 run skipped the controls): report only
            both = sorted(set(A[fam]) & set(B[fam]))
            if both:
                hw = np.array([abs(B[fam][k] - A[fam][k]) for k in both])
                w = int(np.argmax(hw))
                rep["families"][fam] = {"n": len(both), "missing": missing, "hw_median": float(np.median(hw)),
                                        "hw_p95": float(np.percentile(hw, 95)), "pass": None,
                                        "worst": {"score": both[w], "hw": float(hw[w])}, "identical": int((hw == 0).sum())}
                print(f"{fam:28}{len(both):>5}{np.median(hw):11.5f}{'-':>13}{np.percentile(hw, 95):9.5f}{'-':>10}  "
                      f"report   ({both[w]}: {hw[w]:.4f}; no seed yardstick)")
            continue
        hw = np.array([abs(B[fam][k] - A[fam][k]) for k in keys])
        sd = np.array([abs(S1[fam][k] - A[fam][k]) for k in keys])
        hm, sm, hp, sp = (float(np.median(hw)), float(np.median(sd)), float(np.percentile(hw, 95)), float(np.percentile(sd, 95)))
        ok = hm <= BAR * sm and hp <= BAR * sp
        ok_all &= ok
        w = int(np.argmax(hw))
        rep["families"][fam] = {"n": len(keys), "missing": missing, "hw_median": hm, "seed_median": sm, "hw_p95": hp,
                                "seed_p95": sp, "pass": ok, "worst": {"score": keys[w], "hw": float(hw[w]), "seed": float(sd[w])},
                                "identical": int((hw == 0).sum())}
        print(f"{fam:28}{len(keys):>5}{hm:11.5f}{sm:13.5f}{hp:9.5f}{sp:10.5f}  {'PASS' if ok else 'FAIL':7}  "
              f"({keys[w]}: {hw[w]:.4f} vs seed {sd[w]:.4f}){'  MISSING ' + str(len(missing)) if missing else ''}")
    ca, cb = s_copy(ref), s_copy(other)
    dc = max(abs(cb[k] - ca[k]) for k in ca if k in cb)
    rep["s_copy_max_diff"] = dc
    rep["complete"] = complete
    rep["pass"] = ok_all and complete
    print(f"S_copy (stimuli only), largest difference from the reference: {dc:.2e}")
    print("E7:", "PASS" if rep["pass"] else "FAIL" + ("" if complete else " (incomplete: scores missing from this run)"))
    path = Path(other) / f"compare_e7_vs_{Path(ref).parent.name}_{Path(ref).name}.json"
    path.write_text(json.dumps(rep, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("other")
    ap.add_argument("--ref", default="results/runs/main/lawful_s0")
    ap.add_argument("--seed1", default="results/runs/main/lawful_s1")
    a = ap.parse_args()
    main(a.other, a.ref, a.seed1)
