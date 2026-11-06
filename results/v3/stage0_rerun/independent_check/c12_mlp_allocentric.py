"""Can the rejected MLP candidate (early stopping, 1,024 episodes) read the allocentric last-seen fixture that ridge
cannot (c9)? One pooled probe, seed 0."""
import importlib.util, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("c9", HERE / "c9_allocentric_fixture.py")
src = (HERE / "c9_allocentric_fixture.py").read_text().split("for label, prod in")[0]  # reuse c9's setup only
ns = {"__file__": str(HERE / "c9_allocentric_fixture.py")}; exec(compile(src, "c9_setup", "exec"), ns)
probes, s0 = ns["probes"], ns["s0"]
A, B = s0.embed(ns["allo"](ns["gf"], ns["ttr"]), ns["allo"](ns["gt_"], ns["tte"]))
t0 = time.time()
pr = probes.Probe(A.reshape(-1, 128), ns["ytr"], ns["m2"](ns["Rtr"]["present"]), "reg", 256, 0, ns["groups"], 60_000, early_stop=True)
pred = pr(B.reshape(-1, 128))
print(f"MLP (early stopping) on the allocentric last-seen fixture, best step {pr.best_step} ({time.time()-t0:.0f}s): " + "  ".join(f"{k} {v:+.3f}" for k, v in ns["curve"](pred).items()))
