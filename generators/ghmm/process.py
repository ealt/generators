from typing import NamedTuple

import jax
import jax.numpy as jnp

from generators.utils import principal_ev


class Data(NamedTuple):
    Ts: jax.Array
    eta_0: jax.Array
    w: jax.Array


def validate(Ts: jax.Array) -> bool:
    """Check that Ts is a well-formed operator family in the spectrally normalized gauge.

    Irreducibility (SPEC 3.2 items 3-4) is deliberately not checked: a reducible tensor
    is still well formed, and `init` is where its missing initial condition becomes an
    error.

    Args:
        Ts: Transition matrices, shape (V, S, S).

    Returns:
        Whether Ts is well formed and spectrally normalized.
    """
    if len(Ts.shape) != 3:
        return False
    if any(dim == 0 for dim in Ts.shape):
        return False
    if Ts.shape[1] != Ts.shape[2]:
        return False
    if not jnp.all(jnp.isfinite(Ts)):
        return False
    T = jnp.sum(Ts, axis=0)
    norm = jnp.linalg.norm(T, ord=jnp.inf)
    return bool(jnp.isclose(norm, 1))


def init(Ts: jax.Array, eta_0: jax.Array | None = None, w: jax.Array | None = None) -> Data:
    """Derive the runtime data of a GHMM from its transition matrices.

    The omitted arguments default to eigenvectors of the net matrix T = sum_x Ts[x],
    which SPEC 3.2 determines uniquely only when T is irreducible. A reducible
    (nonergodic) T carries one unit eigenvalue per recurrent component and no
    eigenvector of it is the process's, so the defaults raise rather than pick one:
    state the initial condition by passing eta_0 (a mixture over the components), or
    model the components as a mixture with `generators.nonergodic`.

    Args:
        Ts: Transition matrices, shape (V, S, S).
        eta_0: Initial state representative, shape (S,). Rescaled to the canonical
            representative of SPEC 3.7. Defaults to the stationary state.
        w: Normalizing eigenvector, shape (S,). Defaults to ones when Ts is
            row-stochastic, where SPEC 3.2.1 makes that exact, and otherwise to the
            right Perron eigenvector of T.

    Returns:
        Runtime data.

    Raises:
        ValueError: If T is reducible and an argument it cannot determine was omitted.
    """
    T = Ts.sum(axis=0)
    if w is None:
        w = jnp.ones(T.shape[0], dtype=T.dtype) if jnp.allclose(T.sum(axis=1), 1) else principal_ev(T)
    if eta_0 is None:
        eta_0 = principal_ev(T.T)
    return Data(Ts=Ts, eta_0=eta_0 / (eta_0 @ w), w=w)


def obs_dist(data: Data, eta: jax.Array) -> jax.Array:
    return eta @ data.Ts @ data.w


def sample(data: Data, eta: jax.Array, key: jax.Array) -> jax.Array:
    probs = obs_dist(data, eta)
    logits = jnp.where(probs > 0, jnp.log(probs), -jnp.inf)
    return jax.random.categorical(key, logits)


def update(data: Data, eta: jax.Array, x: jax.Array) -> jax.Array:
    eta = eta @ data.Ts[x]
    return eta / (eta @ data.w)


def generate(data: Data, eta: jax.Array, keys: jax.Array) -> tuple[jax.Array, jax.Array]:
    def fn(eta: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = sample(data, eta, key)
        eta = update(data, eta, x)
        return eta, x

    return jax.lax.scan(fn, eta, keys)


def seq_prob(data: Data, xs: jax.Array) -> jax.Array:
    def fn(eta, x):
        return eta @ data.Ts[x], None

    eta, _ = jax.lax.scan(fn, init=data.eta_0, xs=xs)
    return eta @ data.w
