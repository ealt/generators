"""Factored process."""

import itertools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from src.ghmm.process import Data as FactorData
from src.ghmm.process import init as init_factor
from src.ghmm.process import sample as sample_factor
from src.ghmm.process import update as update_factor
from src.ghmm.process import validate as validate_factor
from src.utils import stack


class Data(NamedTuple):
    """Typed dictionary."""

    Ts: jax.Array
    eta_0: jax.Array
    w: jax.Array
    Vs: jax.Array
    Ss: jax.Array
    V: int


def validate(Ts_list: list[jax.Array]) -> bool:
    """Validate inputs for an independent factored process."""
    return len(Ts_list) > 0 and all(validate_factor(Ts) for Ts in Ts_list)


def compile(Ts_list: list[jax.Array]) -> jax.Array:
    """Compile a list of independent transition matrices into a single transition matrix."""
    matrices = []
    factor_vocabs = [range(Ts.shape[0]) for Ts in Ts_list]
    for x_factors_rev in itertools.product(*reversed(factor_vocabs)):
        x_factors = tuple(reversed(x_factors_rev))
        factors = [Ts[int(x_i)] for Ts, x_i in zip(Ts_list, x_factors, strict=True)]
        composite = factors[-1]
        for factor in reversed(factors[:-1]):
            composite = jnp.kron(composite, factor)
        matrices.append(composite)
    return jnp.stack(matrices)


def init(Ts_list: list[jax.Array]) -> Data:
    """Initialize an independent factored process."""
    factors = [init_factor(Ts) for Ts in Ts_list]
    Ts = stack([factor.Ts for factor in factors])
    eta_0 = stack([factor.eta_0 for factor in factors])
    w = stack([factor.w for factor in factors])
    Vs = jnp.array([Ts.shape[0] for Ts in Ts_list])
    Ss = jnp.array([Ts.shape[1] for Ts in Ts_list])
    V = int(jnp.prod(Vs))
    return Data(Ts=Ts, eta_0=eta_0, w=w, Vs=Vs, Ss=Ss, V=V)


def obs_dist(data: Data, eta: jax.Array, *, decode: Callable[[jax.Array], jax.Array]) -> jax.Array:
    """Compute the observation distribution of an independent factored process."""

    def factor_obs(Ts_i: jax.Array, w_i: jax.Array, eta_i: jax.Array) -> jax.Array:
        return eta_i @ Ts_i @ w_i

    factor_dists = jax.vmap(factor_obs, in_axes=0)(data.Ts, data.w, eta)
    factor_ids = jnp.arange(factor_dists.shape[0])

    def obs_prob(x: jax.Array) -> jax.Array:
        x_factors = decode(x)
        factor_probs = factor_dists[factor_ids, x_factors]
        return jnp.prod(factor_probs)

    obs = jnp.arange(data.V)
    return jax.vmap(obs_prob, in_axes=0)(obs)


def sample(data: Data, eta: jax.Array, key: jax.Array) -> jax.Array:
    """Sample observations from each factor.

    Encode the composite observation as a post-processing step if desired.
    """
    factor_data = FactorData(Ts=data.Ts, eta_0=data.eta_0, w=data.w)
    num_factors = int(data.Ts.shape[0])
    keys = jax.random.split(key, num_factors)
    return jax.vmap(sample_factor, in_axes=0)(factor_data, eta, keys)


def update(data: Data, eta: jax.Array, x_factors: jax.Array) -> jax.Array:
    """Compute the belief updates of a factored process.

    Pass per-factor observations as `x_factors`.
    May require decoding a composite observation as a prerequisite.
    """
    factor_data = FactorData(Ts=data.Ts, eta_0=data.eta_0, w=data.w)
    return jax.vmap(update_factor, in_axes=0)(factor_data, eta, x_factors)


def generate(
    data: Data,
    eta: jax.Array,
    keys: jax.Array,
    *,
    encode: Callable[[jax.Array], jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Generate a sequence from an independent factored process.

    Returns the final belief state and a sequence of composite observations.
    """

    def step(eta, key):
        x_factors = sample(data, eta, key)
        x = encode(x_factors)
        return update(data, eta, x_factors), x

    return jax.lax.scan(step, eta, keys)


def seq_prob(data: Data, xs: jax.Array, *, decode: Callable[[jax.Array], jax.Array]) -> jax.Array:
    """Compute the sequence probability of an independent factored process."""
    xs_factors = jax.vmap(decode)(xs)

    def unnorm_update_factor(Ts_i: jax.Array, eta_i: jax.Array, x_i: jax.Array) -> jax.Array:
        return eta_i @ Ts_i[x_i]

    def unnorm_update(eta: jax.Array, x_factors: jax.Array) -> tuple[jax.Array, None]:
        next_eta = jax.vmap(unnorm_update_factor, in_axes=(0, 0, 0))(data.Ts, eta, x_factors)
        return next_eta, None

    eta, _ = jax.lax.scan(unnorm_update, init=data.eta_0, xs=xs_factors)

    def factor_seq_prob(eta_i: jax.Array, w_i: jax.Array) -> jax.Array:
        return eta_i @ w_i

    return jnp.prod(jax.vmap(factor_seq_prob, in_axes=(0, 0))(eta, data.w))
