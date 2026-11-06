"""Shared loaders for the Stage 0 checks. Writes only to this scratch folder."""
import importlib.util
from pathlib import Path
import numpy as np

REPO = Path("/Users/henryvu/personal/umwelt")
SCR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stage0", REPO / "experiments/stage0.py")
s0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s0)
from umwelt import presets, probes  # noqa: E402
from umwelt.data import load  # noqa: E402

N_TRAIN, N_TEST = 8192, 256
TIMES_TR = np.arange(probes.PAST, 256, 8)
TIMES_TE = np.arange(probes.PAST, 256, 4)
KEYS = ("ego", "present", "vis", "since", "dark", "loc")


def get_data():
    d = load("dark", N_TRAIN + N_TEST, seed=0)
    gt = {k: v for k, v in d.items() if k.startswith("gt_")}
    T = probes.targets({k: np.asarray(v) for k, v in gt.items()})
    T["dark"] = np.repeat(~np.asarray(gt["gt_light"])[..., None], probes.K, -1)
    return d, gt, T


def get_raw(d):
    path = SCR / "raw_obs_human.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    presets.observe(d, presets.PRESETS["human"], dtype=np.float16, path=path)
    return np.load(path, mmap_mode="r")


def lagged(raw, eps, times, lags=(0,)):
    out = []
    for i in range(eps.start, eps.stop, 512):
        chunk = np.asarray(raw[i: min(i + 512, eps.stop)])
        out.append(np.concatenate([chunk[:, times - L].astype(np.float32) for L in lags], -1))
    return np.concatenate(out)


def rows_of(T, eps, stride):
    return {k: T[k][eps][:, probes.PAST::stride].reshape(-1, *T[k].shape[2:]) for k in KEYS}


def r2(pred, Y, M, ref):
    """Independent re-implementation of probes._r2_cols (count-weighted mean of per-column R^2 vs variance over ref)."""
    sc, w = [], []
    for c in range(Y.shape[1]):
        m, r = M[:, c], ref[:, c]
        if m.sum() < 30 or r.sum() < 30:
            continue
        sc.append(1 - ((pred[m, c] - Y[m, c]) ** 2).mean() / (Y[r, c].var() + 1e-9))
        w.append(m.sum())
    return float(np.average(sc, weights=w)) if sc else float("nan")


def m2(m):
    return np.repeat(m, 2, axis=1)


def my_scores(pred, Rtr, Rte, qpred=None):
    """object_seen/hidden/now + permanence skill bins, written independently of stage0.score."""
    Y = Rte["ego"].reshape(len(pred), -1)
    pres, vis, since, dark = Rte["present"], Rte["vis"], Rte["since"], Rte["dark"]
    out = {"object_seen": r2(pred, Y, m2(pres & vis), m2(pres)), "object_hidden": r2(pred, Y, m2(pres & ~vis), m2(pres)),
           "object_now": r2(pred, Y, m2(pres), m2(pres))}
    ytr, ptr = Rtr["ego"], Rtr["present"]
    const = np.zeros_like(Y)
    for k in range(ptr.shape[1]):
        for j in range(2):
            const[:, 2 * k + j] = ytr[ptr[:, k], k, j].mean() if ptr[:, k].any() else 0.0
    def skill(m):
        s, b = r2(pred, Y, m2(m), m2(pres)), r2(const, Y, m2(m), m2(pres))
        return (s - b) / (1 - b)
    for name, rg in s0.BINS:
        m = pres & vis if rg is None else pres & ~vis & (since >= rg[0]) & (since <= rg[1])
        out[name] = skill(m)
        if rg is not None and rg[1] <= 20:
            out[name + ", dark"] = skill(m & dark)
            out[name + ", lit"] = skill(m & ~dark)
    if qpred is not None:
        L = Rte["loc"]
        out["self_location"] = r2(qpred, L, np.ones_like(L, bool), np.ones_like(L, bool))
    return out
