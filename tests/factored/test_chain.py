from functools import partial

import jax
import jax.numpy as jnp

from src.factored.chain import (
    generate,
    init,
    obs_dist,
    seq_prob,
    validate,
    validate_eta,
)
from src.utils import mixed_radix_decode, mixed_radix_encode, mixed_radix_weights
from tests.transition_matrices import cycle, zero_one

# def test_compile():
#     """Independent factors should compile into a single valid GHMM."""
#     Ts_list = [
#         jnp.array(zero_one()),
#         jnp.array([cycle(3, 0.75), cycle(3, 0.25)]),
#     ]
#     sigma_emit_list = [
#         jnp.array([0]),
#         jnp.array([0, 1]),
#     ]
#     sigma_trans_list = [
#         jnp.array([0]),
#         jnp.array([0, 1]),
#     ]
#     composite = compile_matrices(Ts_list, sigma_emit_list, sigma_trans_list)
#     assert validate_ghmm(composite)
#     expected = jnp.array([...])
#     assert jnp.allclose(composite, expected)


def test_init():
    """The init function should return a valid data object."""
    Ts_list = [
        jnp.array([zero_one()]),
        jnp.array([cycle(3, 0.75), cycle(3, 0.25)]),
    ]
    sigma_emit_list = [
        jnp.array([0]),
        jnp.array([0, 1]),
    ]
    sigma_trans_list = [
        jnp.array([0]),
        jnp.array([0, 1]),
    ]
    assert validate(Ts_list, sigma_emit_list, sigma_trans_list)
    data = init(Ts_list, sigma_emit_list, sigma_trans_list)
    expected_Ts = jnp.stack(
        [
            jnp.stack(
                [
                    jnp.zeros((3, 3, 3)).at[0, 0, 1].set(1).at[1, 1, 0].set(1),
                    jnp.zeros((3, 3, 3)),
                ]
            ),
            jnp.stack(
                [
                    jnp.array(cycle(3, 0.75)),
                    jnp.array(cycle(3, 0.25)),
                ]
            ),
        ]
    )
    assert jnp.allclose(data.Ts, expected_Ts)
    expected_eta_0 = jnp.stack(
        [
            jnp.stack(
                [
                    jnp.array([1, 1, 0]) / 2,
                    jnp.zeros(3),
                ]
            ),
            jnp.stack(
                [
                    jnp.ones(3) / 3,
                    jnp.ones(3) / 3,
                ]
            ),
        ]
    )
    assert jnp.allclose(data.eta_0, expected_eta_0)
    expected_w = jnp.stack(
        [
            jnp.stack(
                [
                    jnp.array([1, 1, 0]),
                    jnp.zeros(3),
                ]
            ),
            jnp.stack(
                [
                    jnp.ones(3),
                    jnp.ones(3),
                ]
            ),
        ]
    )
    assert jnp.allclose(data.w, expected_w)
    assert jnp.allclose(data.Ks, jnp.array([1, 2]))
    assert jnp.allclose(data.Vs, jnp.array([2, 3]))
    assert jnp.allclose(data.Ss, jnp.array([2, 3]))
    assert data.V == 6
    assert jnp.allclose(data.sigma_emit, jnp.array([[0, 0], [0, 1]]))
    assert jnp.allclose(data.sigma_trans, jnp.array([[0, 0], [0, 1]]))


def test_obs_dist():
    """The factored observation distribution should use the selected emission variants."""
    variant0 = jnp.array(zero_one())
    variant1 = jnp.array(
        [
            [
                [0, 0],
                [1, 0],
            ],
            [
                [0, 1],
                [0, 0],
            ],
        ]
    )
    Ts_list = [
        jnp.array([zero_one()]),
        jnp.stack([variant0, variant1]),
    ]
    sigma_emit_list = [
        jnp.array([0]),
        jnp.array([0, 1]),
    ]
    sigma_trans_list = [
        jnp.array([0]),
        jnp.array([0, 1]),
    ]
    data = init(Ts_list, sigma_emit_list, sigma_trans_list)
    eta = jnp.array(
        [
            [0.8, 0.2],
            [0.8, 0.2],
        ]
    )
    weights = mixed_radix_weights(data.Vs)
    decode = partial(mixed_radix_decode, Vs=data.Vs, weights=weights)
    actual = obs_dist(data, eta, decode=decode)
    expected = jnp.array([0.64, 0.04, 0.16, 0.16])
    assert jnp.allclose(actual, expected)


def test_generate():
    """The generate function should return a valid sequence and runtime belief state."""
    Ts_list = [
        jnp.array([zero_one()]),
        jnp.array([cycle(3, 0.75), cycle(3, 0.25)]),
    ]
    sigma_emit_list = [
        jnp.array([0]),
        jnp.array([0, 1]),
    ]
    sigma_trans_list = [
        jnp.array([0]),
        jnp.array([0, 1]),
    ]
    data = init(Ts_list, sigma_emit_list, sigma_trans_list)
    eta = jnp.array(
        [
            [1 / 2, 1 / 2, 0],
            [1 / 3, 1 / 3, 1 / 3],
        ]
    )
    assert validate_eta(data, eta)
    keys = jax.random.split(jax.random.key(0), 12)
    weights = mixed_radix_weights(data.Vs)
    encode = partial(mixed_radix_encode, weights=weights)
    decode = partial(mixed_radix_decode, Vs=data.Vs, weights=weights)
    eta_final, xs = jax.jit(generate, static_argnames="encode")(data, eta, keys, encode=encode)
    assert eta_final.shape == (2, 3)
    assert validate_eta(data, eta_final)
    assert xs.shape == (12,)
    factor_xs = jax.vmap(decode)(xs)
    assert jnp.all(factor_xs[:, 0] < 2)
    assert jnp.all(factor_xs[:, 1] < 3)
    assert jnp.all(factor_xs[1:, 0] != factor_xs[:-1, 0])


def test_seq_prob():
    """The sequence probability should use the explicit runtime initial state."""
    Ts_list = [
        jnp.array([zero_one()]),
        jnp.array([cycle(3, 1.0), cycle(3, 1.0)]),
    ]
    sigma_emit_list = [
        jnp.array([0]),
        jnp.array([0, 1]),
    ]
    sigma_trans_list = [
        jnp.array([0]),
        jnp.array([0, 1]),
    ]
    assert validate(Ts_list, sigma_emit_list, sigma_trans_list)
    data = init(Ts_list, sigma_emit_list, sigma_trans_list)
    eta = jnp.array(
        [
            [1 / 2, 1 / 2, 0],
            [1 / 3, 1 / 3, 1 / 3],
        ]
    )
    xs = jnp.array([0, 3, 4, 1, 2, 5])
    weights = mixed_radix_weights(data.Vs)
    decode = partial(mixed_radix_decode, Vs=data.Vs, weights=weights)
    actual = jax.jit(seq_prob, static_argnames="decode")(data, eta, xs, decode=decode)
    assert jnp.isclose(actual, 1 / 6)
