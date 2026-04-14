"""Generalized Hidden Markov Model."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from src.utils import principal_ev


class Data(NamedTuple):
    """Typed dictionary."""

    Ts: jnp.ndarray
    eta_0: jnp.ndarray
    w: jnp.ndarray


def validate_transition_matrix(Ts: jnp.ndarray) -> bool:
    """Validate a single transition matrix."""
    if len(Ts.shape) != 3:
        return False
    if any(dim == 0 for dim in Ts.shape):
        return False
    if Ts.shape[1] != Ts.shape[2]:
        return False
    if not jnp.all(jnp.isfinite(Ts)) or not jnp.all(Ts >= 0):
        return False
    T = jnp.sum(Ts, axis=0)
    norm = jnp.linalg.norm(T, ord=jnp.inf)
    return bool(jnp.isclose(norm, 1))


def normalizing_ev(Ts: jnp.ndarray) -> jnp.ndarray:
    """Compute the normalizing eigenvector of a GHMM."""
    T = Ts.sum(axis=0)
    return principal_ev(T)


def stationary_dist(Ts: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Compute the stationary distribution of a GHMM."""
    T = Ts.sum(axis=0)
    eta_0 = principal_ev(T.T)
    return eta_0 / (eta_0 @ w)


def init(Ts: jnp.ndarray) -> Data:
    """Compute the data of a GHMM."""
    w = normalizing_ev(Ts)
    eta_0 = stationary_dist(Ts, w)
    return Data(Ts=Ts, eta_0=eta_0, w=w)


def obs_dist(data: Data, eta: jnp.ndarray) -> jnp.ndarray:
    """Compute the observation distribution of a GHMM."""
    return eta @ data.Ts @ data.w


def sample(data: Data, eta: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    """Sample a token from a GHMM."""
    probs = obs_dist(data, eta)
    logits = jnp.where(probs > 0, jnp.log(probs), -jnp.inf)
    return jax.random.categorical(key, logits)


def belief_update(data: Data, eta: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Compute the belief update of a GHMM."""
    eta = eta @ data.Ts[x]
    return eta / (eta @ data.w)


def generate(
    data: Data, n: int, key: jnp.ndarray, *, eta: jnp.ndarray | None = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a sequence from a GHMM."""
    if eta is None:
        eta = data.eta_0

    keys = jax.random.split(key, n)

    def fn(eta, key):
        x = sample(data, eta, key)
        return belief_update(data, eta, x), x

    return jax.lax.scan(fn, eta, keys)


def seq_prob(data: Data, xs: jnp.ndarray) -> jnp.ndarray:
    """Compute the sequence probability of a GHMM."""

    def fn(eta, x):
        return eta @ data.Ts[x], None

    eta, _ = jax.lax.scan(fn, init=data.eta_0, xs=xs)
    return eta @ data.w
