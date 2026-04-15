"""Factored process."""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from src.ghmm.process import Data as FactorData
from src.ghmm.process import init as init_variant
from src.ghmm.process import sample as sample_variant
from src.ghmm.process import update as update_variant
from src.ghmm.process import validate as validate_variant
from src.utils import stack


class Data(NamedTuple):
    """Typed dictionary."""

    Ts: jax.Array
    eta_0: jax.Array
    w: jax.Array
    Ks: jax.Array
    Vs: jax.Array
    Ss: jax.Array
    V: int
    sigma_emit: jax.Array
    sigma_trans: jax.Array


def validate(Ts_list: list[jax.Array], sigma_emit_list: list[jax.Array], sigma_trans_list: list[jax.Array]) -> bool:
    """Validate the data."""
    if not len(Ts_list) > 0:
        return False
    Ks = jnp.array([Ts_i.shape[0] for Ts_i in Ts_list])
    if not jnp.all(Ks > 0):
        return False
    if not all(
        all(validate_variant(Ts_i[k]) for k in range(int(Ks_i))) for Ts_i, Ks_i in zip(Ts_list, Ks, strict=True)
    ):
        return False

    Vs = jnp.array([Ts_i.shape[1] for Ts_i in Ts_list])
    Vs_prev = jnp.roll(Vs, 1).at[0].set(1)

    def validate_sigma(sigma_list: list[jax.Array]) -> bool:
        if len(sigma_list) != len(Ts_list):
            return False
        if not all(jnp.issubdtype(sigma_i.dtype, jnp.integer) for sigma_i in sigma_list):
            return False
        if not all(jnp.all(jnp.isfinite(sigma_i)) for sigma_i in sigma_list):
            return False
        if not all(jnp.all(sigma_i >= 0) for sigma_i in sigma_list):
            return False
        if not all(jnp.all(sigma_i < K_i) for sigma_i, K_i in zip(sigma_list, Ks, strict=True)):
            return False
        return all(sigma_i.shape == (V_prev,) for sigma_i, V_prev in zip(sigma_list, Vs_prev, strict=True))

    return validate_sigma(sigma_emit_list) and validate_sigma(sigma_trans_list)


def compile_matrices(
    Ts_list: list[jax.Array], sigma_emit_list: list[jax.Array], sigma_trans_list: list[jax.Array]
) -> jax.Array:
    """Compile a list of transition matrices into a single transition matrix."""
    ...  # TODO: implement this


def init(Ts_list: list[jax.Array], sigma_emit_list: list[jax.Array], sigma_trans_list: list[jax.Array]) -> Data:
    """Initialize the data of a factored process."""
    Ts = stack(Ts_list)
    factors = [jax.vmap(init_variant)(Ts_i) for Ts_i in Ts_list]
    eta_0 = stack([factor.eta_0 for factor in factors])
    w = stack([factor.w for factor in factors])
    Ks = jnp.array([Ts_i.shape[0] for Ts_i in Ts_list])
    Vs = jnp.array([Ts_i.shape[1] for Ts_i in Ts_list])
    Ss = jnp.array([Ts_i.shape[2] for Ts_i in Ts_list])
    V = int(jnp.prod(Vs))
    sigma_emit = stack(sigma_emit_list)
    sigma_trans = stack(sigma_trans_list)
    return Data(Ts=Ts, eta_0=eta_0, w=w, Ks=Ks, Vs=Vs, Ss=Ss, V=V, sigma_emit=sigma_emit, sigma_trans=sigma_trans)


def validate_eta(data: Data, eta: jax.Array) -> bool:
    """Validate a runtime factor-state array against the shared factor state spaces."""
    if eta.ndim != 2:
        return False
    if eta.shape != (data.Ts.shape[0], data.Ts.shape[-1]):
        return False
    if not jnp.all(jnp.isfinite(eta)):
        return False
    if not jnp.all(eta >= 0):
        return False
    return all(
        jnp.any(eta_i[: int(S_i)] > 0) and jnp.all(eta_i[int(S_i) :] == 0)
        for eta_i, S_i in zip(eta, data.Ss, strict=True)
    )


def obs_dist(data: Data, eta: jax.Array, *, decode: Callable[[jax.Array], jax.Array]) -> jax.Array:
    """Compute the observation distribution of a factored process."""
    factor_data = FactorData(Ts=data.Ts, eta_0=data.eta_0, w=data.w)

    def obs_prob(x: jax.Array) -> jax.Array:
        factor_xs = decode(x)

        def factor_prob(
            x_prev: jax.Array, args: tuple[FactorData, jax.Array, jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            factor_i, sigma_emit_i, eta_i, x_i = args
            k_i = sigma_emit_i[x_prev]
            variant_Ts = factor_i.Ts[k_i]
            variant_w = factor_i.w[k_i]
            prob = (eta_i @ variant_Ts[x_i] @ variant_w) / (eta_i @ variant_w)
            return x_i, prob

        scan_xs: Any = (factor_data, data.sigma_emit, eta, factor_xs)
        _, probs = jax.lax.scan(factor_prob, jnp.array(0), scan_xs)
        return jnp.prod(probs)

    obs = jnp.arange(data.V)
    return jax.vmap(obs_prob, in_axes=0)(obs)


def sample(data: Data, eta: jax.Array, key: jax.Array) -> jax.Array:
    """Sample a token from a factored process.

    Returns:
        tuple[jax.Array, jax.Array]: The composite token and the component tokens.
    """

    def sample_factor(
        x_prev: jax.Array, args: tuple[FactorData, jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        factor_i, sigma_emit_i, eta_i, key_i = args
        K_i = sigma_emit_i[x_prev]
        variant = FactorData(Ts=factor_i.Ts[K_i], eta_0=factor_i.eta_0[K_i], w=factor_i.w[K_i])
        x = sample_variant(variant, eta_i, key_i)
        return x, x

    factor_data = FactorData(Ts=data.Ts, eta_0=data.eta_0, w=data.w)
    num_factors = int(data.Ts.shape[0])
    keys = jax.random.split(key, num_factors)
    scan_xs: Any = (factor_data, data.sigma_emit, eta, keys)
    _, factor_xs = jax.lax.scan(sample_factor, jnp.array(0), scan_xs)
    return factor_xs


def update(data: Data, eta: jax.Array, factor_xs: jax.Array) -> jax.Array:
    """Compute the belief updates of a factored process."""

    def update_factor(
        factor_i: FactorData, sigma_trans_i: jax.Array, x_prev_i: jax.Array, eta_i: jax.Array, xs_i: jax.Array
    ) -> jax.Array:
        K_i = sigma_trans_i[x_prev_i]
        variant = FactorData(Ts=factor_i.Ts[K_i], eta_0=factor_i.eta_0[K_i], w=factor_i.w[K_i])
        return update_variant(variant, eta_i, xs_i)

    factor_data = FactorData(Ts=data.Ts, eta_0=data.eta_0, w=data.w)
    xs_prev = jnp.roll(factor_xs, 1).at[0].set(0)
    vmap_args: Any = (factor_data, data.sigma_trans, xs_prev, eta, factor_xs)
    return jax.vmap(update_factor, in_axes=0)(*vmap_args)


def generate(
    data: Data,
    eta: jax.Array,
    keys: jax.Array,
    *,
    encode: Callable[[jax.Array], jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Generate a sequence of tokens from a factored process."""

    def step(eta, key):
        factor_xs = sample(data, eta, key)
        x = encode(factor_xs)
        return update(data, eta, factor_xs), x

    return jax.lax.scan(step, eta, keys)


def seq_prob(data: Data, eta: jax.Array, xs: jax.Array, *, decode: Callable[[jax.Array], jax.Array]) -> jax.Array:
    """Compute the sequence probability of a factored process."""

    def unnorm_update_factor(
        factor_i: FactorData,
        sigma_emit_i: jax.Array,
        sigma_trans_i: jax.Array,
        x_prev_i: jax.Array,
        eta_i: jax.Array,
        x_i: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        k_emit_i = sigma_emit_i[x_prev_i]
        emit_variant = FactorData(Ts=factor_i.Ts[k_emit_i], eta_0=factor_i.eta_0[k_emit_i], w=factor_i.w[k_emit_i])
        prob = (eta_i @ emit_variant.Ts[x_i] @ emit_variant.w) / (eta_i @ emit_variant.w)

        k_trans_i = sigma_trans_i[x_prev_i]
        trans_variant = FactorData(Ts=factor_i.Ts[k_trans_i], eta_0=factor_i.eta_0[k_trans_i], w=factor_i.w[k_trans_i])
        next_eta = eta_i @ trans_variant.Ts[x_i]
        return next_eta, prob

    factor_data = FactorData(Ts=data.Ts, eta_0=data.eta_0, w=data.w)

    def step(eta: jax.Array, xs_t: jax.Array) -> tuple[jax.Array, jax.Array]:
        xs_prev = jnp.roll(xs_t, 1).at[0].set(0)
        vmap_args: Any = (factor_data, data.sigma_emit, data.sigma_trans, xs_prev, eta, xs_t)
        next_eta, probs = jax.vmap(unnorm_update_factor, in_axes=0)(*vmap_args)
        return next_eta, jnp.prod(probs)

    factor_xs = jax.vmap(decode)(xs)
    _, probs = jax.lax.scan(step, eta, factor_xs)
    return jnp.prod(probs)
