"""How much do surprise scores depend on where the bystander walls stand? (docs/preregistration.md, second
correction of 2026-11-05.)

The redesign put real walls where the off-arena marker was. Their places are a draw (voe.LAYOUT_SEED), so this
scores stored minds on suites drawn with other seeds and reports, per mind and test, the spread of S across
layouts beside the width of the 95% interval of S on the battery's own layout. The oracle's permanence surprise
ranged from 0 to 0.48 across ten off-arena markers (lawful seed 0); if the redesign works, its spread across real
layouts is far smaller than that.

    uv run python experiments/battery_layouts.py [--seeds 0 1 2 3 4] [--runs main/lawful_s0 v2/dark_s0]
"""

import argparse
import importlib.util
import json
import pickle
import time
from pathlib import Path

import numpy as np

from umwelt import voe
from umwelt.experiment import RESULTS

spec = importlib.util.spec_from_file_location("recompute_battery", Path(__file__).with_name("recompute_battery.py"))
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)


def main(seeds, runs, out):
    t0 = time.time()
    suites = {s: voe.build_suite(layout_seed=s) for s in seeds}
    print(f"{len(seeds)} suites in {time.time() - t0:.0f} s: " + ", ".join(f"seed {s} {voe.suite_hash(x)}" for s, x in suites.items()), flush=True)
    rows = []
    for run in runs:
        d = RESULTS / "runs" / run
        for name in rb.minds(d):
            rec = json.loads((d / f"{name}.json").read_text())
            with open(d / f"{name}_params.pkl", "rb") as fh:
                saved = pickle.load(fh)
            by_seed = {s: rb.scores(rec, saved, suite) for s, suite in suites.items()}
            for k in by_seed[seeds[0]]:
                for test in voe.TESTS:
                    S = [by_seed[s][k][test]["S"] for s in seeds]
                    ci = [by_seed[s][k][test]["ci"] for s in seeds]
                    rows.append({"run": run, "mind": name, "net": "trained" if k == "voe" else "untrained", "test": test,
                                 "S": S, "ci": ci, "spread": max(S) - min(S), "ci_width": ci[0][1] - ci[0][0],
                                 "excludes_0": [not (lo <= 0 <= hi) for lo, hi in ci]})
            print(f"{run} {name}: done ({time.time() - t0:.0f} s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": seeds, "stimuli": {s: voe.suite_hash(x) for s, x in suites.items()}, "rows": rows}, indent=1))
    print(f"\nS on layout seeds {seeds}; spread = max - min across layouts; CI width = 95% interval on seed {seeds[0]}")
    for r in rows:
        flag = "" if len(set(r["excludes_0"])) == 1 else "  interval-excludes-0 differs between layouts"
        print(f"{r['run']:16} {r['mind']:15} {r['net']:9} {r['test']:10} " + " ".join(f"{s:+.4f}" for s in r["S"])
              + f"  spread {r['spread']:.4f}  CI width {r['ci_width']:.4f}{flag}")
    print("\nlargest spread / CI width per test and net (all minds):")
    for net in ("trained", "untrained"):
        for test in voe.TESTS:
            sel = [r for r in rows if r["net"] == net and r["test"] == test]
            if sel:
                worst = max(sel, key=lambda r: r["spread"] / max(r["ci_width"], 1e-9))
                print(f"  {net:9} {test:10} median spread/CI {np.median([r['spread'] / max(r['ci_width'], 1e-9) for r in sel]):.2f}, "
                      f"largest {worst['spread'] / max(worst['ci_width'], 1e-9):.2f} ({worst['run']} {worst['mind']}: spread "
                      f"{worst['spread']:.4f}, CI width {worst['ci_width']:.4f})")
    print(f"wrote {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--runs", nargs="+", default=["main/lawful_s0", "v2/dark_s0"])
    ap.add_argument("--out", type=Path, default=RESULTS / "checks" / "2026-11-05_battery" / "layouts.json")
    a = ap.parse_args()
    main(a.seeds, a.runs, a.out)
