"""Aggregate run results into REPORT.md, a JSON summary and figures.

Reads results/runs/<tag>/<world>_s<seed>/<mind>.json and translation.json, checks the
preregistered validation criteria and hypotheses, and splits each concept's variance into a
mind share, a world share and their interaction.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from umwelt.experiment import RESULTS
from umwelt.probes import CONCEPTS

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#eb6834", "#f0efec", "#2a78d6"
SCENARIOS = ("permanence", "solidity", "causality")


def load(tag):
    runs = defaultdict(dict)  # (world, seed) -> mind -> record
    trans = {}
    for d in sorted((RESULTS / "runs" / tag).glob("*_s*")):
        world, seed = d.name.rsplit("_s", 1)
        for f in d.glob("*.json"):
            if f.name == "translation.json":
                trans[(world, int(seed))] = json.loads(f.read_text())
            else:
                runs[(world, int(seed))][f.stem] = json.loads(f.read_text())
    return runs, trans


def table(runs, world, field="probes"):
    """mind -> concept -> mean over seeds."""
    acc = defaultdict(lambda: defaultdict(list))
    for (w, _), minds in runs.items():
        if w != world:
            continue
        for m, r in minds.items():
            for c, v in r.get(field, {}).items():
                acc[m][c].append(v)
    return {m: {c: float(np.nanmean(v)) for c, v in cs.items()} for m, cs in acc.items()}


def voe_table(runs, world, field="voe"):
    acc = defaultdict(lambda: defaultdict(list))
    for (w, _), minds in runs.items():
        if w != world:
            continue
        for m, r in minds.items():
            for s, v in r.get(field, {}).items():
                acc[m][s].append(v)
    return acc


def anova_shares(runs, concept, exclude=("oracle",), normalized=False):
    runs = {k: v for k, v in runs.items() if not k[0].endswith("_mj")}
    """Two-way ANOVA (mind x world, seeds as replicates). Returns sums-of-squares shares."""
    cells = defaultdict(list)
    for (w, _), minds in runs.items():
        for m, r in minds.items():
            if m in exclude:
                continue
            v = r["probes"].get(concept, np.nan)
            if normalized:
                v = skill(v, w, concept)
            if np.isfinite(v):
                cells[(m, w)].append(v)
    minds = sorted({m for m, _ in cells})
    worlds = sorted({w for _, w in cells})
    complete = [m for m in minds if all((m, w) in cells for w in worlds)]
    if len(complete) < 2 or len(worlds) < 2:
        return None
    Y = [[np.array(cells[(m, w)]) for w in worlds] for m in complete]
    allv = np.concatenate([y for row in Y for y in row])
    gm = allv.mean()
    cm = np.array([[y.mean() for y in row] for row in Y])
    n = np.array([[len(y) for y in row] for row in Y])
    mind_m = (cm * n).sum(1) / n.sum(1)
    world_m = (cm * n).sum(0) / n.sum(0)
    ss_mind = float(np.sum(n.sum(1) * (mind_m - gm) ** 2))
    ss_world = float(np.sum(n.sum(0) * (world_m - gm) ** 2))
    ss_cells = float(np.sum(n * (cm - gm) ** 2))
    ss_inter = ss_cells - ss_mind - ss_world
    ss_resid = float(sum(np.sum((y - y.mean()) ** 2) for row in Y for y in row))
    tot = ss_cells + ss_resid + 1e-12
    return {"mind": ss_mind / tot, "world": ss_world / tot, "interaction": ss_inter / tot, "seed_noise": ss_resid / tot,
            "minds": complete, "worlds": worlds}


SUBSET = ("object_seen", "object_hidden", "object_allo_seen", "object_allo_hidden")
_BASE = {}


def object_baselines(world):
    """Score of a mind that knows nothing (always predicts the training mean) on each object subset.

    The subset scores share one scale (variance over all present samples), so a constant prediction can
    score above or below 0 on a subset whose objects sit closer to or farther from the mean. Skill above
    this baseline, (score - base) / (1 - base), is 0 for no information and 1 for perfect.
    """
    if world in _BASE:
        return _BASE[world]
    from umwelt import probes
    from umwelt.data import load

    try:
        d = load(world, 1280)
    except Exception:
        return None
    gt = {k: v for k, v in d.items() if k.startswith("gt_")}
    Ttr = probes.targets({k: np.asarray(v[:1024]) for k, v in gt.items()})
    Tte = probes.targets({k: np.asarray(v[1024:1280]) for k, v in gt.items()})
    out = {}
    for key, names in (("ego", ("object_seen", "object_hidden")), ("allo", ("object_allo_seen", "object_allo_hidden"))):
        ytr, ptr = probes._rows(Ttr, key, 8), probes._rows(Ttr, "present", 8)
        yte, pte, vte = probes._rows(Tte, key, 4), probes._rows(Tte, "present", 4), probes._rows(Tte, "vis", 4)
        mean = np.array([[ytr[ptr[:, k], k, j].mean() if ptr[:, k].any() else 0.0 for j in range(2)] for k in range(ptr.shape[1])])
        Y = yte.reshape(len(yte), -1)
        pred = np.broadcast_to(mean.reshape(1, -1), Y.shape)
        m2 = lambda m: np.repeat(m, 2, axis=1)
        out[names[0]] = probes._r2_cols(pred, Y, m2(pte & vte), m2(pte))
        out[names[1]] = probes._r2_cols(pred, Y, m2(pte & ~vte), m2(pte))
    _BASE[world] = out
    return out


def skill(score, world, concept):
    b = object_baselines(world) if concept in SUBSET else None
    if b is None or score is None or not np.isfinite(score):
        return score
    return (score - b[concept]) / (1 - b[concept])


def _fmt(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _verdict(ok):
    return "PASS" if ok else "FAIL"


def checks(runs, trans, base="lawful", alt="impermanent", with_anova=True):
    out = []
    L = table(runs, base)
    L0 = table(runs, base, "probes_untrained")
    raw = table(runs, base, "probes_raw")
    if not L:
        return out
    if "oracle" in L:
        bad = [(m, c) for c in ("object_seen", "object_hidden", "number", "self_location") for m in L
               if m != "oracle" and L[m].get(c, -9) > L["oracle"].get(c, -9) + 0.05]
        out.append(("V1 oracle is the ceiling", not bad, "exceeded by: " + (", ".join(f"{m}/{c}" for m, c in bad) or "none")))
    fails = []
    for m in L:
        if m not in L0:
            continue
        wins = sum(L[m].get(c, -9) > L0[m].get(c, -9) for c in CONCEPTS)
        if wins < 5:
            fails.append(f"{m} ({wins}/{len(CONCEPTS)})")
    out.append(("V2 trained beats untrained on 5+ concepts", not fails, "fails: " + (", ".join(fails) or "none")))
    V0 = voe_table(runs, base, "voe_untrained")
    loud = [f"{m}/{s}" for m, ss in V0.items() for s, vs in ss.items()
            if any(abs(v["S"]) >= 0.01 and not (v["ci"][0] <= 0 <= v["ci"][1]) for v in vs)]
    out.append(("V3 untrained minds show no surprise", not loud, "surprised: " + (", ".join(loud) or "none")))
    loud_x = [x for x in loud if not x.startswith("oracle/")]
    out.append(("V3 excluding the oracle (not preregistered: the oracle sees the hidden event directly)", not loud_x,
                "surprised: " + (", ".join(loud_x) or "none")))

    def g(m, c, t=L):
        return t.get(m, {}).get(c, np.nan)

    if {"human", "memoryless"} <= set(L):
        d = g("human", "object_hidden") - g("memoryless", "object_hidden")
        near_raw = abs(g("memoryless", "object_hidden") - g("human", "object_hidden", raw)) <= 0.05
        out.append(("H1 permanence needs memory", d > 0.05 and near_raw,
                    f"human {_fmt(g('human','object_hidden'))}, memoryless {_fmt(g('memoryless','object_hidden'))}, raw {_fmt(g('human','object_hidden',raw))}"))
    if {"smell_only", "vision_only"} <= set(L):
        b = {m: g(m, "object_past") - g(m, "object_now") for m in ("smell_only", "vision_only")}
        out.append(("H2 smell lives in the past", b["smell_only"] > 0 and b["smell_only"] > b["vision_only"],
                    f"past bias smell {_fmt(b['smell_only'], 3)}, vision {_fmt(b['vision_only'], 3)}; smell_only object_now "
                    f"{_fmt(g('smell_only', 'object_now'))}, so a near-zero bias may be noise"))
    if {"human", "passive", "passive_novest"} <= set(L):
        ok = (g("passive_novest", "self_location") < g("human", "self_location") - 0.05
              and g("passive_novest", "self_motion") < g("human", "self_motion") - 0.05
              and abs(g("passive", "object_seen") - g("human", "object_seen")) <= 0.05)
        out.append(("H3 Held and Hein: efference shapes self, not objects", ok,
                    f"self_location human {_fmt(g('human','self_location'))} vs passive_novest {_fmt(g('passive_novest','self_location'))}; "
                    f"self_motion {_fmt(g('human','self_motion'))} vs {_fmt(g('passive_novest','self_motion'))}; "
                    f"object_seen human {_fmt(g('human','object_seen'))} vs passive {_fmt(g('passive','object_seen'))}"))
    if {"vision_only", "bat", "smell_only", "hearing_only"} <= set(L):
        ok = min(g("vision_only", "number"), g("bat", "number")) > max(g("smell_only", "number"), g("hearing_only", "number"))
        out.append(("H4 counting needs a sense that separates objects", ok,
                    ", ".join(f"{m} {_fmt(g(m, 'number'))}" for m in ("vision_only", "bat", "smell_only", "hearing_only"))))
    tl = [t for (w, _), t in trans.items() if w == base]
    if tl:
        names = tl[0]["names"]
        M = np.mean([np.array(t["translation_r2"]) for t in tl], 0)
        pairs = [("human", "blind"), ("human", "vision_only"), ("dog", "smell_only")]
        res = []
        ok = True
        for a, b in pairs:
            if a in names and b in names:
                ab, ba = M[names.index(a), names.index(b)], M[names.index(b), names.index(a)]
                res.append(f"{a}->{b} {_fmt(ab)} vs {b}->{a} {_fmt(ba)}")
                ok &= ab > ba
        if res:
            out.append(("H5 translation runs from richer to poorer minds", ok, "; ".join(res)))
    VL = voe_table(runs, base)
    VI = voe_table(runs, alt)

    def perm(V, m):
        vs = V.get(m, {}).get("permanence", [])
        return vs

    if VL:
        above = [m for m in ("human", "vision_only") if perm(VL, m) and all(v["ci"][0] > 0 for v in perm(VL, m))]
        mem = perm(VL, "memoryless")
        mem_ok = bool(mem) and all(v["ci"][0] <= 0 <= v["ci"][1] for v in mem)
        lower = None
        if VI:
            common = [m for m in VL if m in VI and m != "oracle"]
            diffs = [np.mean([v["S"] for v in perm(VI, m)]) - np.mean([v["S"] for v in perm(VL, m)]) for m in common if perm(VI, m) and perm(VL, m)]
            lower = float(np.mean(np.array(diffs) < 0)) if diffs else None
        ok = len(above) == 2 and mem_ok and (lower is None or lower > 0.5)
        out.append(("H6 surprise at impossible permanence", ok,
                    f"interval above 0 for {above or 'none'}; memoryless null {mem_ok}; "
                    f"share of minds less surprised after impermanent training {_fmt(lower)}"))
    shares = {c: anova_shares(runs, c) for c in CONCEPTS}
    if with_anova and all(shares[c] for c in ("object_hidden", "cause", "object_seen", "number", "self_motion")):
        ok = all(shares[c]["world"] > shares[c]["mind"] for c in ("object_hidden", "cause")) and all(
            shares[c]["mind"] > shares[c]["world"] for c in ("object_seen", "number", "self_motion"))
        out.append(("H7 world forces permanence and cause; the mind decides seeing, counting, self-motion", ok,
                    "; ".join(f"{c} mind {_fmt(shares[c]['mind'])} world {_fmt(shares[c]['world'])}" for c in
                              ("object_hidden", "cause", "object_seen", "number", "self_motion"))))
    return out


def _heatmap(ax, M, rows, cols, title, vmin=-0.2, vmax=1.0, diverging=False, fmt="{:.2f}"):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    if diverging:
        cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
        norm = TwoSlopeNorm(0.0, vmin, vmax)
    else:
        cmap = LinearSegmentedColormap.from_list("seq", [SURFACE] + SEQ)
        norm = None
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)), rows, fontsize=8, color=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    for (i, j), v in np.ndenumerate(M):
        if np.isfinite(v):
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if (not diverging and v > 0.6 * vmax) else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def figures(runs, trans, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.family": "sans-serif"})
    worlds = sorted({w for w, _ in runs})
    paths = []
    for w in worlds:
        T = table(runs, w)
        T0 = table(runs, w, "probes_untrained")
        minds = [m for m in ORDER if m in T]
        M = np.array([[T[m].get(c, np.nan) for c in CONCEPTS] for m in minds])
        G = np.array([[T[m].get(c, np.nan) - T0.get(m, {}).get(c, np.nan) for c in CONCEPTS] for m in minds])
        fig, axs = plt.subplots(1, 2, figsize=(13, 0.32 * len(minds) + 1.6))
        _heatmap(axs[0], M, minds, CONCEPTS, f"What each mind knows ({w} world, probe score)")
        _heatmap(axs[1], G, minds, CONCEPTS, "What learning added (trained minus untrained)", -0.5, 0.5, diverging=True)
        fig.tight_layout()
        p = out / f"concepts_{w}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)

    shares = {c: anova_shares(runs, c) for c in CONCEPTS}
    shares = {c: s for c, s in shares.items() if s}
    if shares:
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(shares) + 1.2))
        cs = list(shares)
        parts = [("mind", "#2a78d6", "mind (body and senses)"), ("world", "#eb6834", "world (physics)"),
                 ("interaction", "#1baf7a", "mind x world"), ("seed_noise", "#c3c2b7", "seed noise")]
        left = np.zeros(len(cs))
        for key, col, lab in parts:
            v = np.array([max(shares[c][key], 0) for c in cs])
            ax.barh(range(len(cs)), v, left=left, color=col, label=lab, edgecolor=SURFACE, linewidth=2, height=0.62)
            left += v
        ax.set_yticks(range(len(cs)), cs, fontsize=9, color=INK2)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of variance in probe score", color=INK2, fontsize=9)
        ax.set_title("What the mind contributes and what the world forces", loc="left", fontsize=10, color=INK)
        ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        p = out / "kant_split.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)

    for (w, seed), t in sorted(trans.items()):
        if seed != 0:
            continue
        names = [n for n in ORDER if n in t["names"]]
        idx = [t["names"].index(n) for n in names]
        Tm = np.array(t["translation_r2"])[np.ix_(idx, idx)]
        C = np.array(t["cka"])[np.ix_(idx, idx)]
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
        _heatmap(axs[0], Tm, names, names, f"Translation R^2, row mind -> column mind ({w})", 0.0, 1.0)
        _heatmap(axs[1], C, names, names, f"Representational similarity, linear CKA ({w})", 0.0, 1.0)
        fig.tight_layout()
        p = out / f"translation_{w}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)
    return paths


def diagnostics(runs):
    """Not preregistered: does any learner change or more training give the human mind object memory?"""
    arms = [("preregistered learner, 4,000 updates (main sweep)", runs.get(("lawful", 0), {}).get("human"))]
    for tag, label in [("ab_h5", "A/B: 5-step head (rerun)"), ("ab_h15", "A/B: 5 and 15-step heads"),
                       ("ab_h30", "A/B: 5, 15 and 30-step heads"), ("ab_drop", "A/B: 5 and 15-step heads, 30% observational dropout"),
                       ("jepa", "latent prediction (JEPA style), 4,000 updates"), ("long", "preregistered learner, 16,000 updates")]:
        f = RESULTS / "runs" / tag / "lawful_s0" / "human.json"
        arms.append((label, json.loads(f.read_text()) if f.exists() else None))
    raw = runs.get(("lawful", 0), {}).get("human", {}).get("probes_raw")
    cols = ("object_seen", "object_hidden", "object_allo_hidden", "number", "self_location")
    lines = ["", "## Diagnostics (not preregistered): where does object memory come from?", "",
             "Human mind, lawful world. If memory were learned, object_hidden, object_allo_hidden and number would rise "
             "above the raw-senses row.", "", "| learner | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    if raw:
        lines.append("| raw current senses, no learning | " + " | ".join(_fmt(raw.get(c, np.nan)) for c in cols) + " |")
    for label, r in arms:
        if r:
            lines.append(f"| {label} | " + " | ".join(_fmt(r["probes"].get(c, np.nan)) for c in cols) + " |")
    f = RESULTS / "memory_headroom_human.json"
    if f.exists():
        h = json.loads(f.read_text())
        for k, v in h.items():
            lines.append(f"| probe on raw senses, {k.replace('_', ' ')} | " + " | ".join(_fmt(v.get(c, np.nan)) for c in cols) + " |")
    return lines


ORDER = ["oracle", "human", "dog", "vision_only", "bat", "chimera", "blind", "hearing_only", "smell_only",
         "deafblind", "touch_only", "passive", "passive_novest", "memoryless"]


def build(tag="main"):
    runs, trans = load(tag)
    out = RESULTS / f"report_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    figs = figures(runs, trans, out)
    chk = checks(runs, trans)
    worlds = sorted({w for w, _ in runs})
    seeds = sorted({s for _, s in runs})
    lines = [f"# umwelt results ({tag})", "",
             f"Worlds: {', '.join(worlds)}. Seeds: {', '.join(map(str, seeds))}. Minds per world: "
             f"{', '.join(str(len(m)) for m in runs.values())}.", "",
             "## Preregistered checks", "", "| check | verdict | numbers |", "|---|---|---|"]
    lines += [f"| {n} | {_verdict(ok)} | {txt} |" for n, ok, txt in chk]
    if any(w.endswith("_mj") for w in worlds):
        chk_mj = checks(runs, trans, "lawful_mj", "impermanent_mj", with_anova=False)
        lines += ["", "## Audit: the same checks in the independent MuJoCo engine", "", "| check | verdict | numbers |", "|---|---|---|"]
        lines += [f"| {n} | {_verdict(ok)} | {txt} |" for n, ok, txt in chk_mj]
        for w in ("lawful", "impermanent", "ghost"):
            A, B = table(runs, w), table(runs, w + "_mj")
            pairs = [(A[m][c], B[m][c]) for m in A if m in B for c in CONCEPTS
                     if np.isfinite(A[m].get(c, np.nan)) and np.isfinite(B[m].get(c, np.nan))]
            if len(pairs) > 3:
                a, b = np.array(pairs).T
                d = np.abs(a - b)
                lines.append(f"\nAgreement between engines, {w} world: Pearson r = {np.corrcoef(a, b)[0, 1]:.3f} over {len(pairs)} "
                             f"mind-concept scores (mostly driven by differences between minds); mean absolute difference "
                             f"{d.mean():.3f}, largest {d.max():.3f}.")
    for w in worlds:
        T, T0, R = table(runs, w), table(runs, w, "probes_untrained"), table(runs, w, "probes_raw")
        Tl = table(runs, w, "probes_linear")
        lines += ["", f"## {w} world: probe scores (MLP probe; untrained / raw senses / linear probe in brackets)", "",
                  "| mind | " + " | ".join(CONCEPTS) + " |", "|---" * (len(CONCEPTS) + 1) + "|"]
        for m in [m for m in ORDER if m in T]:
            cells = [f"{_fmt(T[m].get(c))} [{_fmt(T0.get(m, {}).get(c))} / {_fmt(R.get(m, {}).get(c))} / {_fmt(Tl.get(m, {}).get(c))}]"
                     for c in CONCEPTS]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        V = voe_table(runs, w)
        lines += ["", f"### {w} world: surprise at impossible events (S, 95% interval; copy-baseline S)", "",
                  "| mind | " + " | ".join(SCENARIOS) + " |", "|---|---|---|---|"]
        for m in [m for m in ORDER if m in V]:
            cells = []
            for s in SCENARIOS:
                vs = V[m].get(s, [])
                cells.append("; ".join(f"{v['S']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] ({v['S_copy']:.3f})" for v in vs))
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    shares = {c: anova_shares(runs, c) for c in CONCEPTS}
    if any(shares.values()):
        lines += ["", "## Mind versus world (two-way ANOVA shares, oracle excluded)", "",
                  "| concept | mind | world | interaction | seed noise |", "|---|---|---|---|---|"]
        for c, s in shares.items():
            if s:
                lines.append(f"| {c} | {s['mind']:.2f} | {s['world']:.2f} | {s['interaction']:.2f} | {s['seed_noise']:.2f} |")
    lines += ["", "## Object scores as skill above a no-information baseline (not preregistered; corrects a scale artifact)", "",
              "A mind that knows nothing scores the baseline in the first row; skill = (score - baseline) / (1 - baseline).", ""]
    for w in worlds:
        b = object_baselines(w)
        if not b:
            continue
        T = table(runs, w)
        lines += [f"### {w}", "", "| mind | " + " | ".join(SUBSET) + " |", "|---" * (len(SUBSET) + 1) + "|",
                  "| no-information baseline (raw scale) | " + " | ".join(_fmt(b[c]) for c in SUBSET) + " |"]
        for m in [m for m in ORDER if m in T]:
            lines.append(f"| {m} | " + " | ".join(_fmt(skill(T[m].get(c), w, c)) for c in SUBSET) + " |")
        lines.append("")
    lines += ["### Mind versus world on skill scores", "", "| concept | mind | world | interaction | seed noise |", "|---|---|---|---|---|"]
    for c in SUBSET:
        sh = anova_shares(runs, c, normalized=True)
        if sh:
            lines.append(f"| {c} | {sh['mind']:.2f} | {sh['world']:.2f} | {sh['interaction']:.2f} | {sh['seed_noise']:.2f} |")
    lines += diagnostics(runs)
    lines += ["", "## Figures", ""] + [f"![{p.stem}]({p.name})" for p in figs]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    summary = {"checks": [{"name": n, "pass": bool(ok), "numbers": t} for n, ok, t in chk],
               "shares": {c: s for c, s in shares.items() if s}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {out / 'REPORT.md'} with {len(figs)} figures")
    return out
