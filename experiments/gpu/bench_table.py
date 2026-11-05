"""Benchmarks from every machine side by side, with speedup against the Mac (docs/gpu.md, E6).

    uv run python experiments/gpu/bench_table.py            # reads results/hardware/*/bench.json
"""

import json
from pathlib import Path

ROOT = Path("results/hardware")
REF = "mac-cpu"


def main():
    runs = {p.parent.name: json.loads(p.read_text()) for p in sorted(ROOT.glob("*/bench.json"))}
    tags = [REF] + [t for t in runs if t != REF]
    ref = runs[REF]

    def key(u):
        return (u["minds"], u["hidden"], u["batch"])

    rows = sorted({key(u) for t in tags for u in runs[t].get("update", [])}, key=lambda k: (k[0] > 1, k))

    print("One training update (ms; speedup against the Mac in brackets)")
    print(f"{'minds':>5} {'hidden':>6} {'batch':>5} " + " ".join(f"{t:>22}" for t in tags))
    for k in rows:
        r = {key(v): v for v in ref["update"]}.get(k, {})
        cells = []
        for t in tags:
            m = {key(v): v for v in runs[t].get("update", [])}.get(k)
            if not m:
                cells.append("-")
            elif "error" in m:
                cells.append("failed: " + m["error"].split(":")[-1].strip()[:13])
            elif "ms_per_update" in r:
                cells.append(f"{m['ms_per_update']:9.1f} ({r['ms_per_update'] / m['ms_per_update']:5.1f}x)")
            else:
                cells.append(f"{m['ms_per_update']:9.1f}")
        print(f"{k[0]:>5} {k[1]:>6} {k[2]:>5} " + " ".join(f"{c:>22}" for c in cells))

    print("\nlearner.train, current learner, host batches (ms per update)")
    for t in tags:
        if "train" in runs[t]:
            ms = runs[t]["train"]["ms_per_update"]
            print(f"  {t:>16}: {ms:7.1f}  ({ref['train']['ms_per_update'] / ms:4.1f}x)")

    print("\nOne full mind, end to end (seconds per stage)")
    for t in tags:
        if "e2e" in runs[t]:
            e = runs[t]["e2e"]
            stages = " ".join(f"{k} {v:.0f}" for k, v in e["timing"].items())
            print(f"  {t:>16}: total {e['seconds_total']:.0f} | {stages} | test error {e['test_error']:.4f}")


if __name__ == "__main__":
    main()
