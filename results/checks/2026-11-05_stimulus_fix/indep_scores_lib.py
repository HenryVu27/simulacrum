import numpy as np
import jax
def gelu(x):  # jax.nn.gelu default: tanh approximation
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def sig(x):
    return 1 / (1 + np.exp(-x))

def my_errors(p, obs, act):
    P = jax.tree_util.tree_map(lambda a: np.asarray(a, np.float64), p)
    lin = lambda q, x: x @ q["w"] + q["b"]
    e = gelu(lin(P["enc2"], gelu(lin(P["enc1"], np.concatenate([obs, act], -1)))))
    if "core" in P:
        h = np.tanh(lin(P["core"], e))
    else:
        H = P["gru_h"]["w"].shape[0]
        xs = lin(P["gru_x"], e)
        hh_, out = np.zeros((obs.shape[0], H)), []
        for t in range(obs.shape[1]):
            x, hh = xs[:, t], lin(P["gru_h"], hh_)
            z, r = sig(x[:, :H] + hh[:, :H]), sig(x[:, H:2 * H] + hh[:, H:2 * H])
            n = np.tanh(x[:, 2 * H:] + r * hh[:, 2 * H:])
            hh_ = (1 - z) * n + z * hh_
            out.append(hh_)
        h = np.stack(out, 1)
    a_next = np.concatenate([act[:, 1:], np.zeros_like(act[:, :1])], 1)
    near = lin(P["dec2"], gelu(lin(P["dec1"], np.concatenate([h, a_next], -1))))
    return np.mean((near[:, :-1] - obs[:, 1:]) ** 2, -1)

def my_S(err, crit, n_boot=500):
    n = len(crit)
    err = err.reshape(n, 4, -1)
    tot = np.stack([err[i, :, crit[i] - 1:].sum(-1) for i in range(n)])
    P, I = tot[:, 0] + tot[:, 1], tot[:, 2] + tot[:, 3]
    s = (I.sum() - P.sum()) / P.sum()
    rng = np.random.default_rng(0)
    b = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        b.append((I[i].sum() - P[i].sum()) / P[i].sum())
    return s, np.percentile(b, [2.5, 97.5])

