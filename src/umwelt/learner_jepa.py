"""Latent-prediction variant of the shared learner (JEPA style: joint-embedding predictive architecture).

Same recurrent core and the same inputs as learner.py, but the mind predicts the embedding of its
future observations, not the observations themselves. Targets come from a slowly updated copy of
the encoder (an exponential moving average), and variance and covariance penalties on the online
embedding keep it from collapsing. Garrido et al. 2025 found that this kind of objective acquires
intuitive physics where pixel prediction does not; this variant tests that here.

Surprise for the violation-of-expectation tests is the one-step latent prediction error.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from umwelt.learner import LearnerConfig, _dense, _future_actions, _lin, core_scan

HORIZONS = (1, 5, 15)
EMA = 0.99


def init(key, obs_dim, cfg: LearnerConfig, recurrent=True):
    ks = jax.random.split(key, 6 + 2 * len(HORIZONS))
    H, E = cfg.hidden, cfg.embed
    p = {"jenc1": _dense(ks[0], obs_dim, E), "jenc2": _dense(ks[1], E, E), "jin": _dense(ks[2], E + 2, E)}
    if recurrent:
        p["gru_x"] = _dense(ks[3], E, 3 * H)
        w = jnp.concatenate([jax.random.orthogonal(k, H) for k in jax.random.split(ks[4], 3)], 1)
        p["gru_h"] = {"w": w, "b": jnp.zeros(3 * H)}
    else:
        p["core"] = _dense(ks[3], E, H)
    for i, k in enumerate(HORIZONS):
        p[f"jp{k}_1"] = _dense(ks[6 + 2 * i], H + 2 * k, cfg.dec)
        p[f"jp{k}_2"] = _dense(ks[7 + 2 * i], cfg.dec, E)
    p["tenc1"] = jax.tree.map(jnp.copy, p["jenc1"])
    p["tenc2"] = jax.tree.map(jnp.copy, p["jenc2"])
    return p


def _embed(e1, e2, obs):
    return _lin(e2, jax.nn.gelu(_lin(e1, obs)))


def _norm(z):
    return (z - z.mean(-1, keepdims=True)) / (z.std(-1, keepdims=True) + 1e-5)


def latents(p, obs, act):
    z = _embed(p["jenc1"], p["jenc2"], obs)
    e = jax.nn.gelu(_lin(p["jin"], jnp.concatenate([z, act], -1)))
    return jnp.swapaxes(core_scan(p, jnp.swapaxes(e, 0, 1)), 0, 1)


def _target(p, obs):
    t1, t2 = jax.lax.stop_gradient(p["tenc1"]), jax.lax.stop_gradient(p["tenc2"])
    return _norm(_embed(t1, t2, obs))


def _preds(p, h, act):
    return {k: _lin(p[f"jp{k}_2"], jax.nn.gelu(_lin(p[f"jp{k}_1"], jnp.concatenate([h, _future_actions(act, k)], -1))))
            for k in HORIZONS}


def loss_fn(p, obs, act, burn_in):
    h = latents(p, obs, act)
    zt = _target(p, obs)
    preds = _preds(p, h, act)
    lp = jnp.mean(jnp.stack([jnp.mean((preds[k][:, burn_in:-k] - zt[:, burn_in + k :]) ** 2) for k in HORIZONS]))
    z = _embed(p["jenc1"], p["jenc2"], obs).reshape(-1, p["jenc2"]["w"].shape[1])
    std = jnp.sqrt(z.var(0) + 1e-4)
    var_loss = jnp.mean(jax.nn.relu(1.0 - std))
    zc = z - z.mean(0)
    cov = zc.T @ zc / (z.shape[0] - 1)
    cov_loss = (jnp.sum(cov**2) - jnp.sum(jnp.diag(cov) ** 2)) / z.shape[1]
    return lp + var_loss + 0.04 * cov_loss, (lp, var_loss)


def step_errors(p, obs, act):
    h = latents(p, obs, act)
    pred = _preds(p, h, act)[1]
    return jnp.mean((pred[:, :-1] - _target(p, obs)[:, 1:]) ** 2, -1)


def train(obs, act, cfg: LearnerConfig, seed=0, recurrent=True, log_every=500, verbose=True):
    E, T, D = obs.shape
    key = jax.random.PRNGKey(seed)
    key, kinit = jax.random.split(key)
    p = init(kinit, D, cfg, recurrent)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(cfg.lr, weight_decay=1e-4))
    state = opt.init(p)
    obs_j, act_j = jnp.asarray(obs), jnp.asarray(act)

    @jax.jit
    def update(p, state, key):
        k1, k2 = jax.random.split(key)
        ep = jax.random.randint(k1, (cfg.batch,), 0, E)
        start = jax.random.randint(k2, (cfg.batch,), 0, T - cfg.window + 1)
        idx = start[:, None] + jnp.arange(cfg.window)[None]
        o = obs_j[ep[:, None], idx].astype(jnp.float32)
        a = act_j[ep[:, None], idx]
        (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, o, a, cfg.burn_in)
        upd, state = opt.update(g, state, p)
        t_old = (p["tenc1"], p["tenc2"])
        p = dict(optax.apply_updates(p, upd))
        p["tenc1"] = jax.tree.map(lambda t, o: EMA * t + (1 - EMA) * o, t_old[0], p["jenc1"])
        p["tenc2"] = jax.tree.map(lambda t, o: EMA * t + (1 - EMA) * o, t_old[1], p["jenc2"])
        return p, state, loss, aux

    log = []
    for i in range(cfg.steps):
        key, k = jax.random.split(key)
        p, state, loss, (lp, lv) = update(p, state, k)
        if i % log_every == 0 or i == cfg.steps - 1:
            log.append((i, float(loss), float(lp), float(lv)))
            if verbose:
                print(f"    step {i:5d} loss {float(loss):.4f} (latent {float(lp):.4f}, variance {float(lv):.4f})", flush=True)
    return p, log


@jax.jit
def _latents_jit(p, obs, act):
    return latents(p, obs, act)


@jax.jit
def _errors_jit(p, obs, act):
    return step_errors(p, obs, act)
