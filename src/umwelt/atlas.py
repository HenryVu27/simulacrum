"""Aporia Atlas: every coherent position on a classic philosophical problem, and who has held it.

An aporia is a small set of individually plausible claims that cannot all be true. Each nogood is a
minimal unsatisfiable subset (MUS). A coherent position is a minimal correction subset (MCS): a
smallest set of claims to give up so the rest are consistent. MCSes are exactly the minimal hitting
sets of the MUSes (Reiter 1987), so with a handful of claims they can be enumerated directly.

The atlas reports which positions history has occupied, which coherent positions nobody has defended,
which claims nobody ever gives up, and which umwelt measurements bear on which premise.
"""

import itertools
import json
from dataclasses import dataclass, field

from umwelt.experiment import RESULTS


@dataclass
class Aporia:
    name: str
    claims: dict  # id -> statement
    nogoods: list  # list of sets of claim ids that cannot all hold
    holders: dict  # position label -> (claims rejected, representatives)
    umwelt: dict = field(default_factory=dict)  # claim id -> which umwelt measurement bears on it


APORIAS = [
    Aporia(
        "Mind and body (Kim's exclusion problem)",
        {"D": "Mental states are not identical to physical states.",
         "M": "Mental states cause physical events.",
         "C": "Every physical event has a sufficient physical cause.",
         "X": "Events are not systematically caused twice over."},
        [{"D", "M", "C", "X"}],
        {"identity theory": ({"D"}, "Place, Smart, Lewis"),
         "epiphenomenalism": ({"M"}, "Huxley, Jackson (1982)"),
         "interactionist dualism": ({"C"}, "Descartes, Popper and Eccles"),
         "accepted overdetermination": ({"X"}, "Bennett (2003), some nonreductive physicalists"),
         "eliminativism": ({"D", "M"}, "the Churchlands")},
    ),
    Aporia(
        "Object permanence and access",
        {"A": "We know only what appears to us.",
         "R": "Knowing that something exists while unperceived requires access to it while unperceived.",
         "K": "We know that objects persist when unperceived."},
        [{"A", "R", "K"}],
        {"Humean fiction": ({"K"}, "Hume: continued existence is a fiction of the imagination"),
         "transcendental idealism": ({"R"}, "Kant: permanence is a condition of experience, known a priori"),
         "inference to the best explanation": ({"R"}, "Russell, structural realists"),
         "direct realism": ({"A"}, "Reid, Austin")},
        {"R": "object_hidden: a learner that only ever predicts what it senses can come to encode hidden objects, "
              "and the impermanent world shows how much of that is forced by the world (H1, H6, H7)",
         "K": "permanence surprise (voe.permanence) is the behavioural sign of expecting persistence"},
    ),
    Aporia(
        "Hume's problem of causation",
        {"E": "All knowledge of matters of fact comes from experience.",
         "J": "Experience shows only constant conjunction, never necessary connection.",
         "N": "Causation involves necessary connection.",
         "Q": "We have knowledge of causal relations."},
        [{"E", "J", "N", "Q"}],
        {"regularity theory": ({"N"}, "Hume (one reading), Lewis"),
         "causal skepticism": ({"Q"}, "Hume (skeptical reading)"),
         "causality a priori": ({"E"}, "Kant"),
         "interventionism and direct perception": ({"J"}, "Anscombe, Woodward, Pearl")},
        {"J": "passive versus active minds and the cause probe: does acting on the world give more than "
              "conjunction (H3)? The nonlocal world keeps constant conjunction but removes contact",
         "N": "the nonlocal world: consistent action at a distance, a regularity without a mechanism"},
    ),
    Aporia(
        "Free will",
        {"F1": "Every event is either determined or random.",
         "F2": "A determined act is not free.",
         "F3": "A random act is not free.",
         "F4": "Some acts are free."},
        [{"F1", "F2", "F3", "F4"}],
        {"hard incompatibilism": ({"F4"}, "Spinoza, Pereboom"),
         "compatibilism": ({"F2"}, "Hobbes, Hume, Frankfurt, Dennett"),
         "agent-causal libertarianism": ({"F1"}, "Reid, Chisholm, O'Connor"),
         "event-causal libertarianism": ({"F3"}, "Kane")},
    ),
]


def minimal_correction_sets(claims, nogoods):
    ids = sorted(claims)
    hitting = [set(c) for r in range(1, len(ids) + 1) for c in itertools.combinations(ids, r)
               if all(set(c) & ng for ng in nogoods)]
    return [h for h in hitting if not any(o < h for o in hitting)]


def build():
    out = RESULTS / "atlas"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Aporia Atlas", "",
             "Each table lists every minimal way to restore consistency (a coherent position), who holds it, "
             "and the claims nobody gives up.", ""]
    data = []
    for a in APORIAS:
        mcs = minimal_correction_sets(a.claims, a.nogoods)
        occupied = {frozenset(rej): label for label, (rej, _) in a.holders.items()}
        never = [c for c in a.claims if not any(c in rej for rej, _ in a.holders.values())]
        lines += [f"## {a.name}", "", "| claim | statement |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in a.claims.items()]
        lines += ["", "| give up | position | held by |", "|---|---|---|"]
        for m in mcs:
            labels = [lab for lab, (rej, _) in a.holders.items() if rej == m]
            who = "; ".join(a.holders[lab][1] for lab in labels)
            lines.append(f"| {', '.join(sorted(m))} | {', '.join(labels) or 'UNOCCUPIED'} | {who or 'nobody found'} |")
        extra = [lab for lab, (rej, _) in a.holders.items() if rej not in [m for m in mcs]]
        if extra:
            lines.append(f"\nNon-minimal positions (give up more than needed): {', '.join(extra)}.")
        if never:
            lines.append(f"\nClaims no listed position gives up: {', '.join(never)}.")
        for k, v in a.umwelt.items():
            lines.append(f"\numwelt bears on {k}: {v}.")
        lines.append("")
        data.append({"name": a.name, "claims": a.claims, "mcs": [sorted(m) for m in mcs],
                     "occupied": {", ".join(sorted(k)): v for k, v in occupied.items()}, "never_given_up": never})
    (out / "ATLAS.md").write_text("\n".join(lines) + "\n")
    (out / "atlas.json").write_text(json.dumps(data, indent=1))
    print(f"wrote {out / 'ATLAS.md'}")
