from functools import partial

import jax
import jax.numpy as jnp

from src.factored.independent import (
    compile,
    generate,
    init,
    obs_dist,
    seq_prob,
    validate,
)
from src.ghmm.process import validate as validate_factor
from src.utils import mixed_radix_decode, mixed_radix_encode, mixed_radix_weights
from tests.transition_matrices import cycle, zero_one


def test_compile():
    Ts_list = [
        jnp.array(zero_one()),
        jnp.array(cycle(3, 1.0)),
    ]
    assert validate(Ts_list)

    composite = compile(Ts_list)
    assert validate_factor(composite)
    expected = jnp.zeros((6, 6, 6))
    expected = expected.at[0, 0, 3].set(1)  # (0, 0) -> (1, 1)
    expected = expected.at[3, 3, 4].set(1)  # (1, 1) -> (0, 2)
    expected = expected.at[4, 4, 1].set(1)  # (0, 2) -> (1, 0)
    expected = expected.at[1, 1, 2].set(1)  # (1, 0) -> (0, 1)
    expected = expected.at[2, 2, 5].set(1)  # (0, 1) -> (1, 2)
    expected = expected.at[5, 5, 0].set(1)  # (1, 2) -> (0, 0)
    assert jnp.allclose(composite, expected)


def test_init():
    Ts_list = [
        jnp.array(zero_one()),
        jnp.array(cycle(3, 1.0)),
    ]
    data = init(Ts_list)
    assert data.Ts.shape == (2, 3, 3, 3)
    assert jnp.allclose(data.Ts[0, :2, :2, :2], jnp.array([zero_one()]))
    assert jnp.all(data.Ts[0, 2, :, :] == 0)
    assert jnp.all(data.Ts[0, :, 2, :] == 0)
    assert jnp.all(data.Ts[0, :, :, 2] == 0)
    assert jnp.allclose(data.Ts[1], jnp.array([cycle(3, 1.0)]))
    assert jnp.allclose(data.eta_0, jnp.array([[1 / 2, 1 / 2, 0], [1 / 3, 1 / 3, 1 / 3]]))
    assert jnp.allclose(data.w, jnp.array([[1, 1, 0], [1, 1, 1]]))
    assert jnp.allclose(data.Vs, jnp.array([2, 3]))
    assert jnp.allclose(data.Ss, jnp.array([2, 3]))
    assert data.V == 6


def test_obs_dist():
    Ts_list = [
        jnp.array(zero_one()),
        jnp.array(cycle(3, 1.0)),
    ]
    data = init(Ts_list)
    eta = jnp.array([[0.8, 0.2, 0.0], [0.1, 0.6, 0.3]])
    weights = mixed_radix_weights(data.Vs)
    decode = partial(mixed_radix_decode, Vs=data.Vs, weights=weights)
    actual = obs_dist(data, eta, decode=decode)
    expected = jnp.array([0.08, 0.02, 0.48, 0.12, 0.24, 0.06])
    assert jnp.allclose(actual, expected)


def test_generate():
    Ts_list = [
        jnp.array(zero_one()),
        jnp.array(cycle(3, 1.0)),
    ]
    data = init(Ts_list)
    keys = jax.random.split(jax.random.key(0), 12)
    weights = mixed_radix_weights(data.Vs)
    encode = partial(mixed_radix_encode, weights=weights)
    eta, xs = jax.jit(generate, static_argnames="encode")(data, data.eta_0, keys, encode=encode)
    assert eta.shape == (2, 3)
    assert jnp.allclose(jnp.sum(eta, axis=1), 1)
    assert jnp.allclose(jnp.max(eta, axis=1), 1)
    assert eta[0, 2] == 0
    expected = jnp.array([0, 3, 4, 1, 2, 5, 0, 3, 4, 1, 2, 5])
    assert jnp.any(jnp.array([jnp.all(jnp.roll(xs, i) == expected) for i in range(6)]))


def test_seq_prob():
    Ts_list = [
        jnp.array(zero_one()),
        jnp.array(cycle(3, 1.0)),
    ]
    data = init(Ts_list)
    xs = jnp.array([0, 3, 4, 1, 2, 5])
    weights = mixed_radix_weights(data.Vs)
    decode = partial(mixed_radix_decode, Vs=data.Vs, weights=weights)
    actual = jax.jit(seq_prob, static_argnames="decode")(data, xs, decode=decode)
    assert jnp.isclose(actual, 1 / 6)
