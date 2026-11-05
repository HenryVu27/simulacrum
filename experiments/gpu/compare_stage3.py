"""E4 in docs/gpu.md: calibration stage 3 rerun on another machine, against the Mac's stored run.

Each folder holds calibrate.py's outputs (dark_n8192.json and the per-step error arrays). Each machine's
M values use its own recorded episodes for the masks (dark steps, object returns and so on).

    uv run python experiments/gpu/compare_stage3.py results/hardware/mac-cpu/stage3_ref results/hardware/pc-cpu/stage3

Yardsticks, from the confirmatory run (docs/preregistration.md, Amendment 4; STATUS.md):
M at object returns 0.093 (learner seed 0) against 0.057 (learner seed 1), both on data seed 1;
0.106 (data seed 0) against 0.093 (data seed 1), both learner seed 0. Pass: every hardware difference
in M is under a quarter of the learner-seed spread, 0.009.
"""

import json
import sys
from pathlib import Path

import numpy as np

KEYS = ("all", "dark", "lights_on", "object_returns", "other")
SEED_SPREAD = 0.093 - 0.057
BAR = SEED_SPREAD / 4


def main(ref, other):
    a, b = Path(ref), Path(other)
    ja, jb = (json.loads((p / "dark_n8192.json").read_text()) for p in (a, b))
    out = {"ref": ref, "other": other, "bar": BAR, "M": {}, "errors": {}}
    print(f"{'':16}{'ref':>10}{'other':>10}{'diff':>11}")
    worst = 0.0
    for k in KEYS:
        x, y = ja[k]["M"], jb[k]["M"]
        out["M"][k] = {"ref": x, "other": y, "diff": y - x}
        worst = max(worst, abs(y - x))
        print(f"M {k:14}{x:10.4f}{y:10.4f}{y - x:+11.5f}")
    for m in ("human", "memoryless"):
        ea, eb = np.load(a / f"dark_n8192_{m}.npy"), np.load(b / f"dark_n8192_{m}.npy")
        xa, xb = ea[:, 16:], eb[:, 16:]
        d = {"ref": float(xa.mean()), "other": float(xb.mean()), "rel_diff": float(xb.mean() / xa.mean() - 1),
             "identical": bool(np.array_equal(ea, eb)), "corr_per_step": float(np.corrcoef(xa.ravel(), xb.ravel())[0, 1]),
             "corr_per_episode": float(np.corrcoef(xa.mean(1), xb.mean(1))[0, 1])}
        out["errors"][m] = d
        print(f"test error {m:11}{d['ref']:.5f} vs {d['other']:.5f} ({d['rel_diff']:+.2%}); identical {d['identical']}; "
              f"correlation per step {d['corr_per_step']:.4f}, per episode {d['corr_per_episode']:.4f}")
    out["max_abs_M_diff"] = worst
    out["pass"] = worst < BAR
    print(f"largest difference in M: {worst:.5f}; bar {BAR:.4f} (a quarter of the learner-seed spread): "
          + ("PASS" if out["pass"] else "FAIL"))
    # one file per reference, named after it: several references can be compared with the same run
    path = b / f"compare_stage3_vs_{a.parent.name}_{a.name}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
