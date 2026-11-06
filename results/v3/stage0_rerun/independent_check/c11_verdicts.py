"""H1 (v1 hypotheses) and P1 (Amendment 2) recomputed from the per-run rescore files with the rules as worded in
docs/preregistration.md, and compared with verdicts.json; plus the same rules with the memory-cut reading where available."""
import json
from pathlib import Path
D = Path("/Users/henryvu/personal/simulacrum/results/v3/stage0_rescore")
J = {r: json.loads((D / f"{r}.json").read_text()) for r in ("main_lawful_s0", "main_lawful_s1", "v2_dark_s0", "v2_dark_s1")}
V = json.loads((D / "verdicts.json").read_text())
g = lambda r, m, part, k: J[r]["minds"][m][part][k]
mean = lambda rs, m, part, k: sum(g(r, m, part, k) for r in rs) / len(rs)
L, Dk = ("main_lawful_s0", "main_lawful_s1"), ("v2_dark_s0", "v2_dark_s1")
h, m, raw = mean(L, "human", "trained", "object_hidden"), mean(L, "memoryless", "trained", "object_hidden"), mean(L, "human", "raw senses", "object_hidden")
print(f"H1: human {h:.4f} memoryless {m:.4f} raw {raw:.4f}; human-memoryless {h-m:+.4f} (>0.05? {h-m>0.05}); |memoryless-raw| {abs(m-raw):.4f} (<=0.05? {abs(m-raw)<=0.05}) -> {'PASS' if h-m>0.05 and abs(m-raw)<=0.05 else 'FAIL'} | stored {V['H1']['human']:.4f} {V['H1']['memoryless']:.4f} {V['H1']['human raw senses']:.4f} passes={V['H1']['passes']}")
ok = True
for b in ("1-5 steps", "6-20 steps", "1-5 steps, dark", "1-5 steps, lit", "6-20 steps, dark", "6-20 steps, lit"):
    hh, mm, rr = mean(Dk, "human", "trained", b), mean(Dk, "memoryless", "trained", b), mean(Dk, "human", "raw senses", b)
    s = V["P1"][b]
    if b in ("1-5 steps", "6-20 steps"): ok &= (hh - rr > 0.05) and (hh - mm > 0.05)
    print(f"P1 {b:17} human {hh:+.4f} raw {rr:+.4f} memoryless {mm:+.4f}: -raw {hh-rr:+.4f} -memoryless {hh-mm:+.4f} | stored {s['human']:+.4f} {s['human raw senses']:+.4f} {s['memoryless']:+.4f}")
print(f"P1 by the rule (1-5 and 6-20 pooled over dark and lit): {'PASS' if ok else 'FAIL'}; stored passes={V['P1']['passes']}")
for r in Dk:
    a = J[r]["minds"]["human"]["trained minus raw"]; b = J[r]["human minus memoryless"]
    print(f"{r}: 1-5 dark human-raw {a['1-5 steps, dark'][0]:+.3f} {a['1-5 steps, dark'][1]}, human-memoryless {b['1-5 steps, dark'][0]:+.3f} {b['1-5 steps, dark'][1]}; "
          f"1-5 lit human-raw {a['1-5 steps, lit'][0]:+.3f} {a['1-5 steps, lit'][1]}; 6-20 lit human-raw {a['6-20 steps, lit'][0]:+.3f} {a['6-20 steps, lit'][1]}")
