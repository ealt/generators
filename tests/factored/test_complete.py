from functools import partial

import jax
import jax.numpy as jnp

from generators.factored.complete import compile, generate, init, obs_dist, seq_prob, validate, validate_eta
from generators.ghmm.process import validate as validate_ghmm
from generators.utils import mixed_radix_decode, mixed_radix_encode, mixed_radix_weights
from transition_matrices.classical import cycle


def swapped_cycle(n: int) -> jax.Array:
    return cycle(n)[::-1]


def test_compile():
    variant0 = cycle(2)
    variant1 = swapped_cycle(2)
    Ts_list = [
        jnp.array([variant0]),
        jnp.array([variant0]),
        jnp.stack([variant0, variant1]),
    ]
    sigma_list = [
        jnp.array(0),
        jnp.array([0, 0]),
        jnp.array([[0, 1], [0, 1]]),
    ]

    composite = compile(Ts_list, sigma_list)
    assert validate_ghmm(composite)
    expected = jnp.zeros((8, 8, 8))
    expected = expected.at[0, 0, 7].set(1)
    expected = expected.at[1, 1, 6].set(1)
    expected = expected.at[2, 6, 1].set(1)
    expected = expected.at[3, 7, 0].set(1)
    expected = expected.at[4, 4, 3].set(1)
    expected = expected.at[5, 5, 2].set(1)
    expected = expected.at[6, 2, 5].set(1)
    expected = expected.at[7, 3, 4].set(1)
    assert jnp.allclose(composite, expected)


def test_init():
    variant0 = cycle(2)
    variant1 = swapped_cycle(2)
    Ts_list = [
        jnp.array([variant0]),
        jnp.array([variant0]),
        jnp.stack([variant0, variant1]),
    ]
    sigma_emit_list = [
        jnp.array(0),
        jnp.array([0, 0]),
        jnp.array([[0, 1], [0, 1]]),
    ]
    sigma_trans_list = [
        jnp.array(0),
        jnp.array([0, 0]),
        jnp.array([[1, 0], [1, 0]]),
    ]

    assert validate(Ts_list, sigma_emit_list, sigma_trans_list)

    data = init(Ts_list, sigma_emit_list, sigma_trans_list)

    assert jnp.allclose(data.weights, jnp.array([1, 2, 4]))
    assert jnp.allclose(data.sigma_emit, jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 1]]))
    assert jnp.allclose(data.sigma_trans, jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0]]))


def test_obs_dist():
    variant0 = cycle(2)
    variant1 = swapped_cycle(2)
    Ts_list = [
        jnp.array([variant0]),
        jnp.array([variant0]),
        jnp.stack([variant0, variant1]),
    ]
    sigma_emit_list = [
        jnp.array(0),
        jnp.array([0, 0]),
        jnp.array([[0, 1], [0, 1]]),
    ]
    sigma_trans_list = sigma_emit_list
    data = init(Ts_list, sigma_emit_list, sigma_trans_list)
    eta = jnp.array(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
        ]
    )
    weights = mixed_radix_weights(data.Vs)
    decode = partial(mixed_radix_decode, Vs=data.Vs, weights=weights)
    actual = obs_dist(data, eta, decode=decode)
    expected = jnp.array([0.336, 0.056, 0.144, 0.024, 0.224, 0.084, 0.096, 0.036])
    assert jnp.allclose(actual, expected)


def test_generate():
    variant0 = cycle(2)
    variant1 = swapped_cycle(2)
    Ts_list = [
        jnp.array([variant0]),
        jnp.array([variant0]),
        jnp.stack([variant0, variant1]),
    ]
    sigma_emit_list = [
        jnp.array(0),
        jnp.array([0, 0]),
        jnp.array([[0, 1], [0, 1]]),
    ]
    sigma_trans_list = sigma_emit_list
    data = init(Ts_list, sigma_emit_list, sigma_trans_list)
    eta = jnp.array(
        [
            [1 / 2, 1 / 2],
            [1 / 2, 1 / 2],
            [1 / 2, 1 / 2],
        ]
    )
    assert validate_eta(data, eta)
    keys = jax.random.split(jax.random.key(0), 10)
    weights = mixed_radix_weights(data.Vs)
    encode = partial(mixed_radix_encode, weights=weights)
    decode = partial(mixed_radix_decode, Vs=data.Vs, weights=weights)
    eta_final, xs = jax.jit(generate, static_argnames="encode")(data, eta, keys, encode=encode)
    assert eta_final.shape == (3, 2)
    assert validate_eta(data, eta_final)
    assert xs.shape == (10,)
    xs_factors = jax.vmap(decode)(xs)
    assert jnp.all(xs_factors < 2)


def test_seq_prob():
    variant0 = cycle(2)
    variant1 = swapped_cycle(2)
    Ts_list = [
        jnp.array([variant0]),
        jnp.array([variant0]),
        jnp.stack([variant0, variant1]),
    ]
    sigma_emit_list = [
        jnp.array(0),
        jnp.array([0, 0]),
        jnp.array([[0, 1], [0, 1]]),
    ]
    sigma_trans_list = sigma_emit_list
    data = init(Ts_list, sigma_emit_list, sigma_trans_list)
    eta = jnp.array(
        [
            [1 / 2, 1 / 2],
            [1 / 2, 1 / 2],
            [1 / 2, 1 / 2],
        ]
    )
    xs = jnp.array([0, 3])
    weights = mixed_radix_weights(data.Vs)
    decode = partial(mixed_radix_decode, Vs=data.Vs, weights=weights)
    actual = jax.jit(seq_prob, static_argnames="decode")(data, eta, xs, decode=decode)
    assert jnp.isclose(actual, 1 / 8)
