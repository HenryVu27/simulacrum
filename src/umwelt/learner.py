"""The one learning rule every mind shares: a recurrent world model that predicts its next sensation.

Plain JAX and optax. Input at time t is the observation o_t plus the action a_t that produced it
(an efference copy, zeroed for passive minds). The model predicts o_{t+1} given a_{t+1}, and
o_{t+5} given the next five actions, so it must carry structure forward in time.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

HORIZON = 5


@dataclass(frozen=True)
class LearnerConfig:
    hidden: int = 128
    embed: int = 128
    dec: int = 256
    steps: int = 4000
    batch: int = 32
    window: int = 96
    burn_in: int = 16
    lr: float = 3e-4


def _dense(key, n_in, n_out):
    return {"w": jax.random.normal(key, (n_in, n_out)) * jnp.sqrt(1.0 / n_in), "b": jnp.zeros(n_out)}


def init(key, obs_dim, cfg: LearnerConfig, recurrent=True):
    ks = jax.random.split(key, 8)
    H, E = cfg.hidden, cfg.embed
    p = {
        "enc1": _dense(ks[0], obs_dim + 2, E),
        "enc2": _dense(ks[1], E, E),
        "dec1": _dense(ks[2], H + 2, cfg.dec),
        "dec2": _dense(ks[3], cfg.dec, obs_dim),
        "far1": _dense(ks[4], H + 2 * HORIZON, cfg.dec),
        "far2": _dense(ks[5], cfg.dec, obs_dim),
    }
    if recurrent:
        p["gru_x"] = _dense(ks[6], E, 3 * H)
        w = jnp.concatenate([jax.random.orthogonal(k, H) for k in jax.random.split(ks[7], 3)], 1)
        p["gru_h"] = {"w": w, "b": jnp.zeros(3 * H)}
    else:
        p["core"] = _dense(ks[6], E, H)
    return p


def _lin(p, x):
    return x @ p["w"] + p["b"]


def encode(p, obs, act):
    x = jnp.concatenate([obs, act], -1)
    return jax.nn.gelu(_lin(p["enc2"], jax.nn.gelu(_lin(p["enc1"], x))))


def core_scan(p, e):
    """Run the core over time. e: (T, B, E) -> h: (T, B, H)."""
    if "core" in p:
        return jnp.tanh(_lin(p["core"], e))
    H = p["gru_h"]["w"].shape[0]
    xs = _lin(p["gru_x"], e)

    def cell(h, x):
        hh = _lin(p["gru_h"], h)
        z = jax.nn.sigmoid(x[..., :H] + hh[..., :H])
        r = jax.nn.sigmoid(x[..., H : 2 * H] + hh[..., H : 2 * H])
        n = jnp.tanh(x[..., 2 * H :] + r * hh[..., 2 * H :])
        h = (1 - z) * n + z * h
        return h, h

    h0 = jnp.zeros(xs.shape[1:-1] + (H,))
    _, hs = jax.lax.scan(cell, h0, xs)
    return hs


def latents(p, obs, act):
    """obs (B, T, D), act (B, T, 2) -> hidden states (B, T, H)."""
    e = encode(p, obs, act)
    return jnp.swapaxes(core_scan(p, jnp.swapaxes(e, 0, 1)), 0, 1)


def _future_actions(act):
    """For each t, the next HORIZON actions a_{t+1..t+HORIZON}, zero padded at the end."""
    T = act.shape[1]
    pad = jnp.pad(act, ((0, 0), (0, HORIZON), (0, 0)))
    return jnp.concatenate([pad[:, k : k + T] for k in range(1, HORIZON + 1)], -1)


def predict(p, h, act):
    """One-step and HORIZON-step predictions from hidden states."""
    a_next = jnp.pad(act[:, 1:], ((0, 0), (0, 1), (0, 0)))
    near = _lin(p["dec2"], jax.nn.gelu(_lin(p["dec1"], jnp.concatenate([h, a_next], -1))))
    far = _lin(p["far2"], jax.nn.gelu(_lin(p["far1"], jnp.concatenate([h, _future_actions(act)], -1))))
    return near, far


def step_errors(p, obs, act):
    """Per-step squared error of the one-step prediction, (B, T-1), averaged over dims."""
    h = latents(p, obs, act)
    near, _ = predict(p, h, act)
    return jnp.mean((near[:, :-1] - obs[:, 1:]) ** 2, -1)


def loss_fn(p, obs, act, burn_in):
    h = latents(p, obs, act)
    near, far = predict(p, h, act)
    l1 = jnp.mean((near[:, burn_in:-1] - obs[:, burn_in + 1 :]) ** 2)
    l5 = jnp.mean((far[:, burn_in:-HORIZON] - obs[:, burn_in + HORIZON :]) ** 2)
    return l1 + 0.5 * l5, (l1, l5)


def train(obs, act, cfg: LearnerConfig, seed=0, recurrent=True, log_every=500, verbose=True):
    """obs (E, T, D) standardized, act (E, T, 2). Returns trained params and a loss log."""
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
        o = obs_j[ep[:, None], idx]
        a = act_j[ep[:, None], idx]
        (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, o, a, cfg.burn_in)
        upd, state = opt.update(g, state, p)
        return optax.apply_updates(p, upd), state, loss, aux

    log = []
    for i in range(cfg.steps):
        key, k = jax.random.split(key)
        p, state, loss, (l1, l5) = update(p, state, k)
        if i % log_every == 0 or i == cfg.steps - 1:
            log.append((i, float(loss), float(l1), float(l5)))
            if verbose:
                print(f"    step {i:5d} loss {float(loss):.4f} (1-step {float(l1):.4f}, {HORIZON}-step {float(l5):.4f})", flush=True)
    return p, log


@jax.jit
def _latents_jit(p, obs, act):
    return latents(p, obs, act)


@jax.jit
def _errors_jit(p, obs, act):
    return step_errors(p, obs, act)


def batched(fn, p, obs, act, chunk=64):
    out = [np.asarray(fn(p, jnp.asarray(obs[i : i + chunk]), jnp.asarray(act[i : i + chunk]))) for i in range(0, len(obs), chunk)]
    return np.concatenate(out, 0)


def standardizer(obs_train):
    mu = obs_train.reshape(-1, obs_train.shape[-1]).mean(0)
    sd = obs_train.reshape(-1, obs_train.shape[-1]).std(0)
    sd = np.where(sd < 1e-3, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)
