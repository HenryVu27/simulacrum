"""Are draws 0..19 exchangeable with fresh draws 40..239? z = (S over 0..19 - S over 40..239) / its draw-only SE."""
import sys
from pathlib import Path
import numpy as np
T = Path(sys.argv[1])
def S(t):
    P = t[..., 0].sum(-1) + t[..., 1].sum(-1)
    return (t[..., 2].sum(-1) + t[..., 3].sum(-1) - P) / P
zs = []; rows = []
for f in sorted(T.glob("*__40_240.npz")):
    a, b, mind, _ = f.stem.split("__")
    F = np.load(f); A = np.load(T / f"{a}__{b}__{mind}__0_20.npz"); B = np.load(T / f"{a}__{b}__{mind}__20_40.npz")
    for net in ("trained", "untrained"):
        if f"{net}/permanence/t64" not in F: continue
        for test in ("permanence", "solidity", "causality"):
            fa, fb, ff = A[f"{net}/{test}/t64"], B[f"{net}/{test}/t64"], F[f"{net}/{test}/t64"]
            sd1 = S(np.concatenate([fa, fb, ff])).std(ddof=1)
            se = sd1 * np.sqrt(1 / 20 + 1 / 200)
            zA = (S(fa.mean(0)) - S(ff.mean(0))) / se; zB = (S(fb.mean(0)) - S(ff.mean(0))) / se
            zs.append((zA, zB)); rows.append((f"{a}/{b}", mind, net, test, zA, zB, S(fa.mean(0)), S(fb.mean(0)), S(ff.mean(0))))
z = np.array(zs)
print(f"{len(z)} scores. z(draws 0..19 vs 40..239): mean {z[:, 0].mean():+.2f}, SD {z[:, 0].std(ddof=1):.2f}, |z|>2 in {np.sum(np.abs(z[:, 0]) > 2)}; "
      f"z(draws 20..39 vs 40..239): mean {z[:, 1].mean():+.2f}, SD {z[:, 1].std(ddof=1):.2f}, |z|>2 in {np.sum(np.abs(z[:, 1]) > 2)}")
print("(exchangeable draws: mean 0, SD 1, about 3 of 66 beyond 2)")
for r in sorted(rows, key=lambda r: -abs(r[4]))[:8]:
    print(f"  {r[0]:15} {r[1]:12} {r[2]:9} {r[3]:10} z0-19 {r[4]:+.2f} z20-39 {r[5]:+.2f}  S(0-19) {r[6]:+.5f} S(20-39) {r[7]:+.5f} S(40-239) {r[8]:+.5f}")
