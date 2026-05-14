from typing import NamedTuple

import jax
import jax.numpy as jnp

from generators.ghmm.process import Data as FactorData
from generators.ghmm.process import init as init_factor
from generators.ghmm.process import validate as validate_factor
from generators.utils import stack


class Data(NamedTuple):
    Ts: jax.Array
    eta_0: jax.Array
    w: jax.Array
    Vs: jax.Array
    Ss: jax.Array
    V: int
    phi: jax.Array
    phi_one_hot: jax.Array
    beta_0: jax.Array


def validate(Ts_list: list[jax.Array], phi_list: list[jax.Array], beta_0: jax.Array) -> bool:
    if len(Ts_list) == 0:
        return False
    if not all(validate_factor(Ts) for Ts in Ts_list):
        return False

    def validate_phi(phi: jax.Array, Ts: jax.Array) -> bool:
        if phi.shape != (Ts.shape[0],):
            return False
        if not jnp.issubdtype(phi.dtype, jnp.integer):
            return False
        if not jnp.all(jnp.isfinite(phi)):
            return False
        if jnp.any(phi < 0):
            return False
        return len(jnp.unique(phi)) == len(phi)

    if len(phi_list) != len(Ts_list):
        return False
    if not all(validate_phi(phi, Ts_i) for phi, Ts_i in zip(phi_list, Ts_list, strict=True)):
        return False
    vocab: jax.Array = jnp.unique(jnp.concatenate(phi_list))
    if vocab.max() != len(vocab) - 1:
        return False

    if beta_0.shape != (len(Ts_list),):
        return False
    if not jnp.all(jnp.isfinite(beta_0)):
        return False
    if jnp.any(beta_0 < 0):
        return False
    return bool(jnp.isclose(jnp.sum(beta_0), 1))


def compile(Ts_list: list[jax.Array], phi_list: list[jax.Array], beta_0: jax.Array) -> jax.Array:
    components = [init_factor(Ts) for Ts in Ts_list]
    Ss = jnp.array([Ts.shape[1] for Ts in Ts_list])
    bounds = jnp.concatenate([jnp.array([0]), jnp.cumsum(Ss)]) + 1
    V = max(int(phi.max()) for phi in phi_list) + 1

    phi_inv_list = [{int(x): i for i, x in enumerate(phi.tolist())} for phi in phi_list]
    matrices = []
    for x in range(V):
        T_x = jnp.zeros((bounds[-1], bounds[-1]))
        for component, phi_inv, beta_c, start, stop in zip(
            components, phi_inv_list, beta_0, bounds[:-1], bounds[1:], strict=True
        ):
            if x not in phi_inv:
                continue
            x_c = phi_inv[x]
            T_x = T_x.at[0, start:stop].set(beta_c * (component.eta_0 @ component.Ts[x_c]))
            T_x = T_x.at[start:stop, start:stop].set(component.Ts[x_c])
        matrices.append(T_x)

    return jnp.stack(matrices)


def init(Ts_list: list[jax.Array], phi_list: list[jax.Array], beta_0: jax.Array) -> Data:
    components = [init_factor(Ts) for Ts in Ts_list]
    Ts = stack([component.Ts for component in components])
    eta_0 = stack([component.eta_0 for component in components])
    w = stack([component.w for component in components])
    Vs = jnp.array([Ts.shape[0] for Ts in Ts_list])
    Ss = jnp.array([Ts.shape[1] for Ts in Ts_list])
    V = max(int(phi.max()) for phi in phi_list) + 1
    phi = stack(phi_list)
    phi_one_hot = stack([jax.nn.one_hot(phi_c, V, dtype=Ts.dtype) for phi_c in phi_list])
    return Data(Ts=Ts, eta_0=eta_0, w=w, Vs=Vs, Ss=Ss, V=V, phi=phi, phi_one_hot=phi_one_hot, beta_0=beta_0)


def obs_dist(data: Data, eta: jax.Array, beta: jax.Array) -> jax.Array:
    def component_obs_dist(Ts_c: jax.Array, w_c: jax.Array, eta_c: jax.Array) -> jax.Array:
        return eta_c @ Ts_c @ w_c

    component_obs_dists = jax.vmap(component_obs_dist, in_axes=0)(data.Ts, data.w, eta)
    weighted_component_obs_dists = beta[:, None] * component_obs_dists
    return jnp.einsum("cv,cvx->x", weighted_component_obs_dists, data.phi_one_hot)


def sample(data: Data, eta: jax.Array, beta: jax.Array, key: jax.Array) -> jax.Array:
    probs = obs_dist(data, eta, beta)
    logits = jnp.where(probs > 0, jnp.log(probs), -jnp.inf)
    return jax.random.categorical(key, logits)


def update(data: Data, eta: jax.Array, beta: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    def component_obs_dist(Ts_c: jax.Array, w_c: jax.Array, eta_c: jax.Array) -> jax.Array:
        return eta_c @ Ts_c @ w_c

    component_obs_dists = jax.vmap(component_obs_dist, in_axes=0)(data.Ts, data.w, eta)
    x_component_ids = jnp.arange(data.Ts.shape[1])
    valid_x_components = x_component_ids[None, :] < data.Vs[:, None]
    matches = valid_x_components & (data.phi == x)
    probs = jnp.sum(jnp.where(matches, component_obs_dists, 0), axis=1)
    obs_prob = beta @ probs

    def update_component_state(Ts_c: jax.Array, w_c: jax.Array, eta_c: jax.Array, x_c: jax.Array) -> jax.Array:
        eta_next = eta_c @ Ts_c[x_c]
        norm = eta_next @ w_c
        return jnp.where(norm > 0, eta_next / norm, eta_c)

    x_components = jnp.argmax(matches, axis=1)
    eta_next = jax.vmap(update_component_state, in_axes=0)(data.Ts, data.w, eta, x_components)
    eta_next = jnp.where(probs[:, None] > 0, eta_next, eta)
    return jax.lax.cond(
        obs_prob > 0,
        lambda _: (eta_next, beta * probs / obs_prob),
        lambda _: (eta, beta),
        operand=None,
    )


def sample_component(data: Data, eta: jax.Array, key: jax.Array) -> tuple[FactorData, jax.Array, jax.Array]:
    logits = jnp.where(data.beta_0 > 0, jnp.log(data.beta_0), -jnp.inf)
    c = jax.random.categorical(key, logits)
    component_data = FactorData(Ts=data.Ts[c : c + data.Ss[c]], eta_0=data.eta_0[c], w=data.w[c])
    eta_c = eta[c]
    phi_c = data.phi[c]
    return component_data, eta_c, phi_c


def generate(
    data: Data,
    eta: jax.Array,
    beta: jax.Array,
    keys: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    def step(carry, key):
        eta, beta = carry
        x = sample(data, eta, beta, key)
        eta, beta = update(data, eta, beta, x)
        return (eta, beta), x

    (eta, beta), xs = jax.lax.scan(step, (eta, beta), keys)
    return eta, beta, xs


def seq_prob(data: Data, xs: jax.Array) -> jax.Array:
    def component_seq_prob(
        Ts_c: jax.Array, eta_0_c: jax.Array, w_c: jax.Array, phi_c: jax.Array, V_c: jax.Array
    ) -> jax.Array:
        x_component_ids = jnp.arange(Ts_c.shape[0])
        valid_x_components = x_component_ids < V_c

        def step(eta_c: jax.Array, x: jax.Array) -> tuple[jax.Array, None]:
            matches = valid_x_components & (phi_c == x)
            x_c = jnp.argmax(matches)
            eta_next = jnp.where(jnp.any(matches), eta_c @ Ts_c[x_c], jnp.zeros_like(eta_c))
            return eta_next, None

        eta_c, _ = jax.lax.scan(step, eta_0_c, xs)
        return eta_c @ w_c

    probs = jax.vmap(component_seq_prob, in_axes=0)(data.Ts, data.eta_0, data.w, data.phi, data.Vs)
    return data.beta_0 @ probs
