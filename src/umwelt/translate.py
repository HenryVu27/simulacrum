"""Can one mind's inner world be translated into another's?

Two measures on hidden states recorded over the same held-out moments:
  linear CKA (centered kernel alignment): symmetric similarity of representational geometry
  translation R^2: a ridge map from mind A's hidden state to mind B's, fit on training
    episodes and scored on held-out ones. Asymmetric: A may contain B without B containing A.
"""

import numpy as np

class Ridge:
    """Ridge regression with the penalty picked on a held-out fifth of the fitting data."""

    def __init__(self, X, Y, rng, lambdas=(0.1, 1.0, 10.0, 100.0)):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.ym = Y.mean(0)
        idx = rng.permutation(len(Z))
        tr, va = idx[: int(0.8 * len(Z))], idx[int(0.8 * len(Z)) :]
        errs = [np.mean((Z[va] @ self._solve(Z[tr], Y[tr] - self.ym, lam) + self.ym - Y[va]) ** 2) for lam in lambdas]
        self.w = self._solve(Z, Y - self.ym, lambdas[int(np.argmin(errs))])

    @staticmethod
    def _solve(Z, Y, lam):
        return np.linalg.solve(Z.T @ Z + lam * len(Z) * 1e-3 * np.eye(Z.shape[1]), Z.T @ Y)

    def __call__(self, X):
        return ((X - self.mu) / self.sd) @ self.w + self.ym


def r2(pred, y):
    """Variance-weighted R^2, so near-constant hidden units cannot dominate."""
    return float(1 - np.sum((pred - y) ** 2) / (np.sum((y - y.mean(0)) ** 2) + 1e-9))


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return float(hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-12))


def matrices(train, test, n_fit=30_000, n_eval=8_000, seed=0):
    """train/test: name -> (E, T, H) hidden states over the same episodes for every mind."""
    rng = np.random.default_rng(seed)
    names = list(train)
    flat_tr = {k: v.reshape(-1, v.shape[-1]) for k, v in train.items()}
    flat_te = {k: v.reshape(-1, v.shape[-1]) for k, v in test.items()}
    i_fit = rng.choice(len(next(iter(flat_tr.values()))), n_fit, replace=False)
    i_ev = rng.choice(len(next(iter(flat_te.values()))), n_eval, replace=False)
    n = len(names)
    cka = np.zeros((n, n))
    trans = np.zeros((n, n))
    for a, na in enumerate(names):
        for b, nb in enumerate(names):
            cka[a, b] = linear_cka(flat_te[na][i_ev], flat_te[nb][i_ev])
            if a == b:
                trans[a, b] = 1.0
                continue
            model = Ridge(flat_tr[na][i_fit], flat_tr[nb][i_fit], rng)
            trans[a, b] = r2(model(flat_te[na][i_ev]), flat_te[nb][i_ev])
    return names, cka, trans
