"""v2 report: does object memory appear once the world makes it pay? Checks Amendment 2.

Reads the calibration runs (results/calibration), the confirmatory runs (results/runs/v2), their
permanence decay curves (results/posthoc_permanence_v2_*), and the v1 lawful runs for comparison.
Writes results/report_v2/REPORT.md, summary.json and figures.
"""

import json

import numpy as np

from umwelt import report as R
from umwelt.experiment import RESULTS
from umwelt.probes import CONCEPTS

BINS = ("in view", "1-5 steps", "6-20 steps", "21-60 steps", "over 60 steps")
RAW_MINDS = ("human", "vision_only", "dog", "smell_only", "memoryless")
BLUE, ORANGE, GRAY, GREEN = "#2a78d6", "#eb6834", "#8a8984", "#1baf7a"
CAL_WORLDS = ("lawful", "occluders", "lookback", "dark", "occlusion")
STAGES = ((4000, 1024), (16000, 1024), (4000, 8192))  # calibration stages 1, 2 and 3 (Amendments 2, 3, 4)
STAGE_COLOR = {(4000, 1024): "#9ec5f4", (16000, 1024): "#6da7ec", (4000, 8192): BLUE}


def _stage(u, n):
    return f"{u:,} updates, {n:,} episodes"


def calibration():
    rows = {}
    for p in sorted((RESULTS / "calibration").glob("*.json")):
        r = json.loads(p.read_text())
        rows[(r["world"], r.get("updates", 4000), r.get("episodes", 1024))] = r
    return rows


def curves(tag, world, seeds=(0, 1)):
    out = {}
    for s in seeds:
        p = RESULTS / f"posthoc_permanence_{tag}_{world}_s{s}.json"
        if p.exists():
            out[s] = json.loads(p.read_text())
    return out


def mean_curve(cs, mind, kind="trained"):
    rows = [c[mind][kind] for c in cs.values() if mind in c and kind in c[mind]]
    if not rows:
        return None
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if not k.endswith("(n)")}


def event_memory_value(world, seed, tag="v2"):
    """Amendment 4 manipulation check: memory value overall and at object-return steps, on held-out episodes."""
    from umwelt.data import load
    from umwelt.probes import steps_since_seen

    run = RESULTS / "runs" / tag / f"{world}_s{seed}"
    paths = {m: run / f"{m}_test_errors.npy" for m in ("human", "memoryless")}
    if not all(p.exists() for p in paths.values()):
        return None
    meta = json.loads((run / "human.json").read_text())
    n_train, data_seed = meta.get("n_train", 1024), meta.get("data_seed", 0)
    d = load(world, n_train + 256, 256, seed=data_seed)
    te = slice(n_train, n_train + 256)
    vis = np.asarray(d["gt_vis180"][te]) & np.asarray(d["gt_obj_active"][te])
    since = steps_since_seen(vis)
    returns = (vis[:, 1:] & ~vis[:, :-1] & (since[:, :-1] < 1000) & (since[:, :-1] >= 3)).any(-1)
    burn = np.arange(returns.shape[1])[None] >= 16
    e = {m: np.load(p).astype(np.float32) for m, p in paths.items()}
    m_all = 1 - e["human"][:, 16:].mean() / e["memoryless"][:, 16:].mean()
    m_ret = 1 - e["human"][returns & burn].mean() / e["memoryless"][returns & burn].mean()
    return {"M": float(m_all), "M_returns": float(m_ret), "share_returns": float((returns & burn).mean())}


def hypotheses(world, runs, cs, v1_runs):
    out = []
    M = {s: v for s in (0, 1) if (v := event_memory_value(world, s))}
    out.append(("M1 (Amendment 4) memory pays where the world rewards it: M at object returns >= 0.10 and overall M >= 0, "
                "each learner seed; the original bar, overall M >= 0.10, failed in calibration",
                len(M) == 2 and all(v["M_returns"] >= 0.10 and v["M"] >= 0 for v in M.values()),
                "; ".join(f"seed {s}: M at returns {v['M_returns']:.3f}, overall M {v['M']:.3f}" for s, v in M.items())
                or "missing"))
    hum, raw, mem = mean_curve(cs, "human"), mean_curve(cs, "human", "raw_senses"), mean_curve(cs, "memoryless")
    if hum and raw and mem:
        bins = ("1-5 steps", "6-20 steps")
        ok = all(hum[b] - raw[b] > 0.05 and hum[b] - mem[b] > 0.05 for b in bins)
        out.append(("P1 permanence needs memory", ok, "; ".join(
            f"{b}: human {hum[b]:.2f}, human raw senses {raw[b]:.2f}, memoryless {mem[b]:.2f}" for b in bins)))
    pn = mean_curve(cs, "passive_novest")
    if hum and pn:
        out.append(("P2 updating a hidden object needs knowledge of own motion", pn["6-20 steps"] < hum["6-20 steps"] - 0.05,
                    f"6-20 steps: human {hum['6-20 steps']:.2f}, passive_novest {pn['6-20 steps']:.2f}"))
    T = R.table(runs, world)
    if {"human", "memoryless"} <= set(T):
        h, z = T["human"]["number"], T["memoryless"]["number"]
        out.append(("P3 counting needs memory when objects hide", h - z > 0.05, f"number: human {h:.2f}, memoryless {z:.2f}"))
    V, V1 = R.voe_table(runs, world), R.voe_table(v1_runs, "lawful")
    hp, mp, v1p = (V.get("human", {}).get("permanence", []), V.get("memoryless", {}).get("permanence", []),
                   V1.get("human", {}).get("permanence", []))
    if hp and mp and v1p:
        above = all(v["ci"][0] > 0 for v in hp)
        null = all(v["ci"][0] <= 0 <= v["ci"][1] for v in mp)
        s2, s1 = np.mean([v["S"] for v in hp]), np.mean([v["S"] for v in v1p])
        out.append(("P4 surprise at impossible permanence", above and null and s2 > s1,
                    "human " + ", ".join(f"S {v['S']:.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]" for v in hp)
                    + "; memoryless " + ", ".join(f"S {v['S']:.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]" for v in mp)
                    + f"; human mean S {s2:.4f} here vs {s1:.4f} in v1 lawful (all three parts must hold)"))
    return out, M


def _style(plt):
    plt.rcParams.update({"figure.facecolor": R.SURFACE, "axes.facecolor": R.SURFACE, "font.family": "sans-serif",
                         "axes.edgecolor": R.INK2, "axes.labelcolor": R.INK2, "xtick.color": R.INK2, "ytick.color": R.INK2})


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=R.GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figures(world, cal, runs, trans, cs, v1c, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    paths = []

    if cal:
        keys = [(w, *s) for w in CAL_WORLDS for s in STAGES if (w, *s) in cal]
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(keys) + 1.4))
        vals = [cal[k]["M"] for k in keys]
        ax.barh(range(len(keys)), vals, color=[STAGE_COLOR[k[1:]] for k in keys], height=0.62)
        for i, v in enumerate(vals):
            ax.text(v + (0.004 if v >= 0 else -0.004), i, f"{v:.3f}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=9, color=R.INK)
        ax.set_xlim(min(min(vals), 0) - 0.09, max(max(vals), 0.10) + 0.05)  # room for labels beside the bars
        ax.axvline(0, color=R.INK2, linewidth=0.8)
        ax.axvline(0.10, color=R.INK2, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(0.10, -0.6, " preregistered bar, 0.10", fontsize=8, color=R.INK2, va="bottom")
        ax.set_yticks(range(len(keys)), [f"{w}, {_stage(u, n)}" for w, u, n in keys], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("memory value M: share of prediction error that memory removes", fontsize=9)
        ax.set_title("How much each world rewards remembering (human vs memoryless)", loc="left", fontsize=10, color=R.INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        p = out / "memory_value.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)

    dc = [(k, cal[k]) for k in sorted(cal) if "dark_course" in cal[k]]
    if dc:
        fig, axs = plt.subplots(1, len(dc), figsize=(3.6 * len(dc), 3.4), sharey=True)
        axs = np.atleast_1d(axs)
        for ax, ((w, u, n), c) in zip(axs, dc):
            for m, col in (("human", BLUE), ("memoryless", GRAY)):
                y = c["dark_course"][m] + c["after_lights_on"][m]
                ax.plot(range(1, 31), y[:30], color=col, linewidth=2, label=m)
                ax.plot(range(34, 45), y[30:], color=col, linewidth=2)
            ax.axvline(32, color=R.INK2, linewidth=0.8, linestyle=(0, (3, 3)))
            ax.set_xticks([1, 10, 20, 30, 34, 44], ["1", "10", "20", "30", "lit", "+10"], fontsize=8)
            ax.set_title(f"{w}\n{_stage(u, n)}", loc="left", fontsize=9, color=R.INK)
            _clean(ax)
        axs[0].set_ylabel("one-step prediction error", fontsize=9)
        axs[0].set_xlabel("steps into the dark, then steps after the lights return", fontsize=9)
        axs[0].legend(fontsize=8, frameon=False)
        fig.suptitle("Where memory helps and hurts: error through a dark spell", x=0.01, ha="left", fontsize=11, color=R.INK)
        fig.tight_layout()
        p = out / "dark_course.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)

    minds = [m for m in RAW_MINDS if mean_curve(cs, m)]
    if minds:
        fig, axs = plt.subplots(1, len(minds), figsize=(3.1 * len(minds), 3.6), sharey=True)
        axs = np.atleast_1d(axs)
        x = np.arange(len(BINS))
        for ax, m in zip(axs, minds):
            series = [(mean_curve(cs, m), BLUE, "-", f"trained, {world}"),
                      (mean_curve(cs, m, "raw_senses"), GRAY, (0, (4, 3)), f"raw senses, {world}"),
                      (v1c.get(m, {}).get("trained"), ORANGE, "-", "trained, v1 lawful")]
            for c, col, ls, lab in series:
                if c:
                    ax.plot(x, [c[b] for b in BINS], color=col, linestyle=ls, linewidth=2, marker="o", markersize=5,
                            label=lab)
            for s, c in cs.items():
                if m in c:
                    ax.scatter(x, [c[m]["trained"][b] for b in BINS], color=BLUE, s=10, alpha=0.5, zorder=3)
            ax.set_xticks(x, ["in view", "1-5", "6-20", "21-60", "60+"], fontsize=8)
            ax.set_title(m, loc="left", fontsize=10, color=R.INK)
            ax.axhline(0, color=R.INK2, linewidth=0.8)
            _clean(ax)
        axs[0].set_ylabel("skill locating the object\n(0 = knows nothing, 1 = perfect)", fontsize=9)
        axs[len(axs) // 2].set_xlabel("steps since the object was last in view", fontsize=9)
        axs[0].legend(fontsize=8, frameon=False, loc="lower left")
        fig.suptitle("Does knowledge of an object outlast seeing it?  Compare lines within a column; levels across "
                     "columns are not comparable (each column has its own no-information baseline).",
                     x=0.01, ha="left", fontsize=10, color=R.INK)
        fig.tight_layout()
        p = out / "permanence_curves.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)

    allm = [m for m in R.ORDER if mean_curve(cs, m)]
    if allm:
        fig, axs = plt.subplots(1, 2, figsize=(11, 0.32 * len(allm) + 1.6), sharey=True)
        for ax, b in zip(axs, ("1-5 steps", "6-20 steps")):
            y = np.arange(len(allm))
            v2 = [mean_curve(cs, m)[b] for m in allm]
            v1 = [v1c.get(m, {}).get("trained", {}).get(b, np.nan) for m in allm]
            ax.hlines(y, np.minimum(v1, v2), np.maximum(v1, v2), color=R.GRID, linewidth=2)
            ax.scatter(v1, y, color=ORANGE, s=40, zorder=3, label="v1 lawful")
            ax.scatter(v2, y, color=BLUE, s=40, zorder=3, label=world)
            ax.set_yticks(y, allm, fontsize=9)
            ax.axvline(0, color=R.INK2, linewidth=0.8)
            ax.set_title(f"object out of view {b}", loc="left", fontsize=10, color=R.INK)
            ax.set_xlabel("skill locating the hidden object", fontsize=9)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(axis="x", color=R.GRID, linewidth=0.8)
            ax.set_axisbelow(True)
        axs[0].invert_yaxis()  # the y axis is shared, so invert it once
        axs[0].legend(fontsize=8, frameon=False, loc="lower right")
        fig.tight_layout()
        p = out / "memory_by_mind.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)

    T = R.table(runs, world)
    if T:
        names = [m for m in R.ORDER if m in T]
        Mx = np.array([[T[m].get(c, np.nan) for c in CONCEPTS] for m in names])
        fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(names) + 1.6))
        R._heatmap(ax, Mx, names, CONCEPTS, f"What each mind knows ({world} world, probe score)")
        fig.tight_layout()
        p = out / f"concepts_{world}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)
    t = trans.get((world, 0))
    if t:
        names = [n for n in R.ORDER if n in t["names"]]
        idx = [t["names"].index(n) for n in names]
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
        R._heatmap(axs[0], np.array(t["translation_r2"])[np.ix_(idx, idx)], names, names,
                   f"Translation R^2, row mind -> column mind ({world})", 0.0, 1.0)
        R._heatmap(axs[1], np.array(t["cka"])[np.ix_(idx, idx)], names, names,
                   f"Representational similarity, linear CKA ({world})", 0.0, 1.0)
        fig.tight_layout()
        p = out / f"translation_{world}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)
    return paths


def build(world, tag="v2"):
    out = RESULTS / "report_v2"
    out.mkdir(parents=True, exist_ok=True)
    cal = calibration()
    runs, trans = R.load(tag)
    v1_runs, _ = R.load("main")
    cs = curves(tag, world)
    v1p = RESULTS / "posthoc_permanence_main_lawful_s0.json"
    v1c = json.loads(v1p.read_text()) if v1p.exists() else {}
    hyp, M = hypotheses(world, runs, cs, v1_runs)
    valid = [c for c in R.checks(runs, trans, base=world, alt=None, with_anova=False) if c[0].startswith("V")]
    v1style = [c for c in R.checks(runs, trans, base=world, alt=None, with_anova=False) if c[0].startswith("H")]
    figs = figures(world, cal, runs, trans, cs, v1c, out)

    L = ["# umwelt v2: a world where remembering pays", "",
         f"Chosen world: **{world}**. Preregistration: `docs/preregistration.md`, Amendments 2 to 4 (the design run "
         "here is Amendment 4's: 8,192 training episodes, 4,000 updates, and the replacement manipulation check M1). "
         "Confirmatory data: data seed 1 (fresh episodes), learner seeds 0 and 1, all minds.", "", "## Calibration (exploratory, picks the world)", "",
         "Memory value M = 1 - err(human) / err(memoryless): the share of one-step prediction error on held-out episodes "
         "that memory removes. Data seed 0, learner seed 0. Only prediction error was looked at. Stage 1 used 4,000 "
         "updates (Amendment 2); stage 2 used 16,000 (Amendment 3); stage 3 used 8,192 training episodes (Amendment 4).", "",
         "| world | M | err human | err memoryless | M on dark steps | M on first lit step | M when an object returns | M on other steps |",
         "|---|---|---|---|---|---|---|---|"]
    for k in [(w, *s) for s in STAGES for w in CAL_WORLDS]:
        if k in cal:
            c = cal[k]
            g = lambda key: R._fmt(c[key]["M"], 3) if c.get(key) and c[key]["M"] is not None else "n/a"
            L.append(f"| {k[0]}, {_stage(k[1], k[2])} | {c['M']:.3f} | {c['all']['err_human']:.4f} | {c['all']['err_memoryless']:.4f} | "
                     f"{g('dark')} | {g('lights_on')} | {g('object_returns')} | {g('other')} |")
    L += ["", "## Validation (v1 criteria, in the chosen world)", ""]
    L += [f"- {R._verdict(ok)} {name}: {detail}" for name, ok, detail in valid]
    L += ["", "## Preregistered hypotheses (Amendments 2 and 4)", ""]
    weak = not (hyp and hyp[0][1])
    if weak:
        L += ["The manipulation check M1 failed, so by Amendments 2 and 4 every hypothesis P1 to P4 below is tested "
              "under a weaker manipulation than preregistered.", ""]
    L += [f"- {R._verdict(ok)} {name}" + (" (weak manipulation)" if weak and name.startswith("P") else "") + f": {detail}"
          for name, ok, detail in hyp]
    L += ["", "## v1 hypotheses re-run in the chosen world (exploratory)", ""]
    L += [f"- {R._verdict(ok)} {name}: {detail}" for name, ok, detail in v1style]
    L += ["", "## Permanence decay curves, every mind (mean of learner seeds)", "",
          "Skill at locating each object relative to the agent, by steps since it was last in view. 0 is a mind that "
          "knows nothing (always guesses the kind's average place), 1 is perfect. Comparisons within a column share one "
          "baseline; comparisons across columns do not, because objects far from their average place inflate skill. "
          "Raw-sense rows do not depend on the learner seed (same data, same probe seed), so each is one measurement.", "",
          "| mind | in view | 1-5 | 6-20 | 21-60 | 60+ | 1-5 dark | 1-5 lit | 6-20 dark | 6-20 lit |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m in R.ORDER:
        c = mean_curve(cs, m)
        if c:
            ex = [c.get(f"{b}, {d}", np.nan) for b in ("1-5 steps", "6-20 steps") for d in ("dark", "lit")]
            L.append(f"| {m} | " + " | ".join(R._fmt(c[b]) for b in BINS) + " | " + " | ".join(R._fmt(v) for v in ex) + " |")
        r = mean_curve(cs, m, "raw_senses")
        if r:
            ex = [r.get(f"{b}, {d}", np.nan) for b in ("1-5 steps", "6-20 steps") for d in ("dark", "lit")]
            L.append(f"| {m}, raw senses | " + " | ".join(R._fmt(r[b]) for b in BINS) + " | " + " | ".join(R._fmt(v) for v in ex) + " |")
    T = R.table(runs, world)
    if T:
        L += ["", "## Probe scores (MLP probes, mean of learner seeds)", "",
              "| mind | test error | " + " | ".join(CONCEPTS) + " |", "|---|---|" + "---|" * len(CONCEPTS)]
        for m in R.ORDER:
            if m in T:
                te = np.mean([runs[k][m]["test_error"] for k in runs if k[0] == world and m in runs[k]])
                L.append(f"| {m} | {te:.4f} | " + " | ".join(R._fmt(T[m].get(c)) for c in CONCEPTS) + " |")
    V = R.voe_table(runs, world)
    if V:
        L += ["", "## Surprise (S, mean of learner seeds; v1 movies)", "", "| mind | permanence | solidity | causality |", "|---|---|---|---|"]
        for m in R.ORDER:
            if m in V:
                L.append(f"| {m} | " + " | ".join(R._fmt(np.mean([v["S"] for v in V[m].get(s, [])]) if V[m].get(s) else None, 3)
                                                  for s in R.SCENARIOS) + " |")
    L += ["", "## Figures", ""] + [f"![{p.stem}]({p.name})" for p in figs]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    summary = {"world": world, "calibration": {f"{w}_u{u}_n{n}": v for (w, u, n), v in cal.items()}, "M": M,
               "hypotheses": [{"name": n, "pass": bool(ok), "detail": d} for n, ok, d in hyp],
               "validation": [{"name": n, "pass": bool(ok), "detail": d} for n, ok, d in valid],
               "curves": {m: {"trained": mean_curve(cs, m), "raw_senses": mean_curve(cs, m, "raw_senses")} for m in R.ORDER}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("\n".join(L[: L.index("## Permanence decay curves, every mind (mean of learner seeds)")]))
    return out
