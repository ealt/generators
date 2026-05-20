from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from generators.factored.independent import Data as ComponentData
from generators.factored.independent import compile as compile_component
from generators.factored.independent import init as init_component
from generators.ghmm.process import validate as validate_factor
from generators.nonergodic.ghmm import compile as compile_nonergodic
from generators.utils import stack


class Data(NamedTuple):
    Ts: jax.Array
    eta_0: jax.Array
    w: jax.Array
    Vs: jax.Array
    Ss: jax.Array
    V_cs: jax.Array
    V: int
    phi: jax.Array
    phi_one_hot: jax.Array
    beta_0: jax.Array


def validate(Ts_lists: list[list[jax.Array]], phi_list: list[jax.Array], beta_0: jax.Array) -> bool:
    def validate_Ts_lists() -> bool:
        def validate_Ts_list(Ts_list: list[jax.Array]) -> bool:
            if len(Ts_list) == 0:
                return False
            return all(validate_factor(Ts) for Ts in Ts_list)

        return len(Ts_lists) > 0 and all(validate_Ts_list(Ts_list) for Ts_list in Ts_lists)

    def validate_phi_list() -> bool:
        def validate_phi(phi: jax.Array, Ts_list: list[jax.Array]) -> bool:
            Vs = jnp.array([Ts.shape[0] for Ts in Ts_list])
            V = jnp.prod(Vs)
            if phi.shape != (V,):
                return False
            if not jnp.issubdtype(phi.dtype, jnp.integer):
                return False
            if not jnp.all(jnp.isfinite(phi)):
                return False
            if jnp.any(phi < 0):
                return False
            return len(jnp.unique(phi)) == len(phi)

        if len(phi_list) != len(Ts_lists):
            return False
        if not all(validate_phi(phi_c, Ts_list_c) for phi_c, Ts_list_c in zip(phi_list, Ts_lists, strict=True)):
            return False
        vocab: jax.Array = jnp.unique(jnp.concatenate(phi_list))
        return int(vocab.max()) == len(vocab) - 1

    def validate_beta_0() -> bool:
        if beta_0.shape != (len(Ts_lists),):
            return False
        if not jnp.all(jnp.isfinite(beta_0)):
            return False
        if jnp.any(beta_0 < 0):
            return False
        return bool(jnp.isclose(jnp.sum(beta_0), 1))

    return validate_Ts_lists() and validate_phi_list() and validate_beta_0()


def compile(Ts_lists: list[list[jax.Array]], phi_list: list[jax.Array], beta_0: jax.Array) -> jax.Array:
    component_Ts = [compile_component(Ts_list_c) for Ts_list_c in Ts_lists]
    return compile_nonergodic(component_Ts, phi_list, beta_0)


def init(Ts_lists: list[list[jax.Array]], phi_list: list[jax.Array], beta_0: jax.Array) -> Data:
    components = [init_component(Ts_list_c) for Ts_list_c in Ts_lists]
    Ts = stack([component.Ts for component in components])
    eta_0 = stack([component.eta_0 for component in components])
    w = stack([component.w for component in components])
    Vs = stack([component.Vs for component in components])
    Ss = stack([component.Ss for component in components])
    V_cs = jnp.array([component.V for component in components])
    V = max(int(phi.max()) for phi in phi_list) + 1
    phi = stack(phi_list)
    phi_one_hot = stack([jax.nn.one_hot(phi_c, V, dtype=Ts.dtype) for phi_c in phi_list])
    return Data(
        Ts=Ts,
        eta_0=eta_0,
        w=w,
        Vs=Vs,
        Ss=Ss,
        V_cs=V_cs,
        V=V,
        phi=phi,
        phi_one_hot=phi_one_hot,
        beta_0=beta_0,
    )


def _component_obs_dist(
    Ts_c: jax.Array,
    w_c: jax.Array,
    eta_c: jax.Array,
    Ss_c: jax.Array,
    V_component_c: jax.Array,
    x_component_ids: jax.Array,
    *,
    decode: Callable[[jax.Array], jax.Array],
) -> jax.Array:

    def factor_obs(Ts_i: jax.Array, w_i: jax.Array, eta_i: jax.Array) -> jax.Array:
        return eta_i @ Ts_i @ w_i

    factor_dists = jax.vmap(factor_obs, in_axes=0)(Ts_c, w_c, eta_c)
    factor_ids = jnp.arange(Ts_c.shape[0])

    valid_factors = Ss_c > 0

    def obs_prob(x_c: jax.Array) -> jax.Array:
        x_factors = decode(x_c)
        factor_probs = factor_dists[factor_ids, x_factors]
        return jnp.prod(jnp.where(valid_factors, factor_probs, 1))

    probs = jax.vmap(obs_prob, in_axes=0)(x_component_ids)
    return jnp.where(x_component_ids < V_component_c, probs, 0)


def obs_dist(data: Data, eta: jax.Array, beta: jax.Array, *, decode: Callable[[jax.Array], jax.Array]) -> jax.Array:
    x_component_ids = jnp.arange(data.phi.shape[1])

    component_obs_dist = partial(_component_obs_dist, x_component_ids=x_component_ids, decode=decode)
    component_obs_dists = jax.vmap(component_obs_dist, in_axes=(0, 0, 0, 0, 0))(
        data.Ts, data.w, eta, data.Ss, data.V_cs
    )
    weighted_component_obs_dists = beta[:, None] * component_obs_dists
    return jnp.einsum("cv,cvx->x", weighted_component_obs_dists, data.phi_one_hot)


def sample(
    data: Data, eta: jax.Array, beta: jax.Array, key: jax.Array, *, decode: Callable[[jax.Array], jax.Array]
) -> jax.Array:
    probs = obs_dist(data, eta, beta, decode=decode)
    logits = jnp.where(probs > 0, jnp.log(probs), -jnp.inf)
    return jax.random.categorical(key, logits)


def update(
    data: Data, eta: jax.Array, beta: jax.Array, x: jax.Array, *, decode: Callable[[jax.Array], jax.Array]
) -> tuple[jax.Array, jax.Array]:
    x_component_ids = jnp.arange(data.phi.shape[1])

    component_obs_dist = partial(_component_obs_dist, x_component_ids=x_component_ids, decode=decode)
    component_obs_dists = jax.vmap(component_obs_dist, in_axes=(0, 0, 0, 0, 0))(
        data.Ts, data.w, eta, data.Ss, data.V_cs
    )
    valid_x_components = x_component_ids[None, :] < data.V_cs[:, None]
    matches = valid_x_components & (data.phi == x)
    probs = jnp.sum(jnp.where(matches, component_obs_dists, 0), axis=1)
    obs_prob = beta @ probs

    def update_component_state(
        Ts_c: jax.Array,
        w_c: jax.Array,
        eta_c: jax.Array,
        Ss_c: jax.Array,
        x_c: jax.Array,
    ) -> jax.Array:

        def update_factor(Ts_i: jax.Array, w_i: jax.Array, eta_i: jax.Array, x_i: jax.Array) -> jax.Array:
            eta_next = eta_i @ Ts_i[x_i]
            norm = eta_next @ w_i
            return jnp.where(norm > 0, eta_next / norm, eta_i)

        x_factors = decode(x_c)
        eta_next = jax.vmap(update_factor, in_axes=0)(Ts_c, w_c, eta_c, x_factors)
        valid_factors = Ss_c > 0
        return jnp.where(valid_factors[:, None], eta_next, eta_c)

    x_components = jnp.argmax(matches, axis=1)
    eta_next = jax.vmap(update_component_state, in_axes=0)(data.Ts, data.w, eta, data.Ss, x_components)
    eta_next = jnp.where(probs[:, None, None] > 0, eta_next, eta)
    return jax.lax.cond(
        obs_prob > 0,
        lambda _: (eta_next, beta * probs / obs_prob),
        lambda _: (eta, beta),
        operand=None,
    )


def sample_component(data: Data, eta: jax.Array, key: jax.Array) -> tuple[ComponentData, jax.Array, jax.Array]:
    logits = jnp.where(data.beta_0 > 0, jnp.log(data.beta_0), -jnp.inf)
    c = jax.random.categorical(key, logits)
    component_data = ComponentData(
        Ts=data.Ts[c],
        eta_0=data.eta_0[c],
        w=data.w[c],
        Vs=data.Vs[c],
        Ss=data.Ss[c],
        V=int(data.V_cs[c]),
    )
    eta_c = eta[c]
    phi_c = data.phi[c]
    return component_data, eta_c, phi_c


def generate(
    data: Data,
    eta: jax.Array,
    beta: jax.Array,
    keys: jax.Array,
    *,
    decode: Callable[[jax.Array], jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    def step(carry, key):
        eta, beta = carry
        x = sample(data, eta, beta, key, decode=decode)
        eta, beta = update(data, eta, beta, x, decode=decode)
        return (eta, beta), x

    (eta, beta), xs = jax.lax.scan(step, (eta, beta), keys)
    return eta, beta, xs


def seq_prob(data: Data, xs: jax.Array, *, decode: Callable[[jax.Array], jax.Array]) -> jax.Array:
    max_V_component = data.phi.shape[1]
    x_component_ids = jnp.arange(max_V_component)

    def component_seq_prob(
        Ts_c: jax.Array,
        eta_0_c: jax.Array,
        w_c: jax.Array,
        Ss_c: jax.Array,
        V_component_c: jax.Array,
        phi_c: jax.Array,
    ) -> jax.Array:
        valid_factors = Ss_c > 0
        valid_x_components = x_component_ids < V_component_c

        def step(eta_c: jax.Array, x: jax.Array) -> tuple[jax.Array, None]:
            matches = valid_x_components & (phi_c == x)
            x_c = jnp.argmax(matches)

            def update_factor_state(Ts_i: jax.Array, eta_i: jax.Array, x_i: jax.Array) -> jax.Array:
                return eta_i @ Ts_i[x_i]

            x_factors = decode(x_c)
            eta_next = jax.vmap(update_factor_state, in_axes=0)(Ts_c, eta_c, x_factors)
            eta_next = jnp.where(valid_factors[:, None], eta_next, eta_c)
            eta_next = jnp.where(jnp.any(matches), eta_next, jnp.zeros_like(eta_c))
            return eta_next, None

        eta_c, _ = jax.lax.scan(step, eta_0_c, xs)

        def factor_seq_prob(eta_i: jax.Array, w_i: jax.Array) -> jax.Array:
            return eta_i @ w_i

        factor_probs = jax.vmap(factor_seq_prob, in_axes=0)(eta_c, w_c)
        return jnp.prod(jnp.where(valid_factors, factor_probs, 1))

    probs = jax.vmap(component_seq_prob, in_axes=0)(data.Ts, data.eta_0, data.w, data.Ss, data.V_cs, data.phi)
    return data.beta_0 @ probs
