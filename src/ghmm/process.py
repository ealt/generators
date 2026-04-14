"""Generalized Hidden Markov Model."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from src.utils import principal_ev


class Data(NamedTuple):
    """Typed dictionary."""

    Ts: jax.Array
    eta_0: jax.Array
    w: jax.Array


def validate(Ts: jax.Array) -> bool:
    """Validate a single transition matrix."""
    if len(Ts.shape) != 3:
        return False
    if any(dim == 0 for dim in Ts.shape):
        return False
    if Ts.shape[1] != Ts.shape[2]:
        return False
    if not jnp.all(jnp.isfinite(Ts)):
        return False
    if not jnp.all(Ts >= 0):
        return False
    T = jnp.sum(Ts, axis=0)
    norm = jnp.linalg.norm(T, ord=jnp.inf)
    return bool(jnp.isclose(norm, 1))


def init(Ts: jax.Array) -> Data:
    """Compute the data of a GHMM."""
    T = Ts.sum(axis=0)
    w = principal_ev(T)
    eta_0 = principal_ev(T.T)
    eta_0 /= eta_0 @ w
    return Data(Ts=Ts, eta_0=eta_0, w=w)


def obs_dist(data: Data, eta: jax.Array) -> jax.Array:
    """Compute the observation distribution of a GHMM."""
    return eta @ data.Ts @ data.w


def sample(data: Data, eta: jax.Array, key: jax.Array) -> jax.Array:
    """Sample a token from a GHMM."""
    probs = obs_dist(data, eta)
    logits = jnp.where(probs > 0, jnp.log(probs), -jnp.inf)
    return jax.random.categorical(key, logits)


def update(data: Data, eta: jax.Array, x: jax.Array) -> jax.Array:
    """Compute the belief update of a GHMM."""
    eta = eta @ data.Ts[x]
    return eta / (eta @ data.w)


def generate(data: Data, eta: jax.Array, keys: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Generate a sequence from a GHMM."""

    def fn(eta: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = sample(data, eta, key)
        eta = update(data, eta, x)
        return eta, x

    return jax.lax.scan(fn, eta, keys)


def seq_prob(data: Data, xs: jax.Array) -> jax.Array:
    """Compute the sequence probability of a GHMM."""

    def fn(eta, x):
        return eta @ data.Ts[x], None

    eta, _ = jax.lax.scan(fn, init=data.eta_0, xs=xs)
    return eta @ data.w
