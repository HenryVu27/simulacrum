"""Old (a223c3c) voe.surprise vs new (35f1e5e) voe.surprise on identical one-draw inputs, 2D and 3D (1, ...)."""
import sys, types, numpy as np
def load(path, name):
    mod = types.ModuleType(name)
    exec(compile(open(path).read(), path, "exec"), mod.__dict__)
    return mod
old = load(sys.argv[1], "voe_old"); new = load(sys.argv[2], "voe_new")
rng = np.random.default_rng(1)
n_eq = 0; N = 300
for t in range(N):
    n = int(rng.integers(2, 30)); T = int(rng.integers(20, 70))
    crit = rng.integers(2, T - 2, n)
    dtype = [np.float32, np.float64][t % 2]
    err = (rng.gamma(2.0, 1.0, (n * 4, T - 1)) * rng.uniform(0.1, 10)).astype(dtype)
    nb = [500, 1, 37][t % 3]
    a = old.surprise(err, crit, n_boot=nb); b = new.surprise(err, crit, n_boot=nb); c = new.surprise(err[None], crit, n_boot=nb)
    n_eq += (a == b == c)
print(f"old == new == new(err[None]) exactly in {n_eq}/{N} random cases")
