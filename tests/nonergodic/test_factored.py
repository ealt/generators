from functools import partial

import jax
import jax.numpy as jnp

from generators.factored.independent import compile as compile_component
from generators.nonergodic.factored import (
    compile,
    generate,
    init,
    obs_dist,
    seq_prob,
    validate,
)
from generators.nonergodic.ghmm import compile as compile_nonergodic
from generators.utils import mixed_radix_decode, mixed_radix_weights
from transition_matrices.classical import cycle


def test_compile():
    Ts_lists = [
        [
            cycle(2),
            cycle(3, 1.0),
        ],
        [
            cycle(2),
            cycle(3, 1.0),
        ],
    ]
    phi_list = [
        jnp.array([0, 1, 2, 3, 4, 5]),
        jnp.array([3, 4, 5, 6, 7, 8]),
    ]
    beta_0 = jnp.array([0.4, 0.6])

    assert validate(Ts_lists, phi_list, beta_0)

    actual = compile(Ts_lists, phi_list, beta_0)
    component_Ts = [compile_component(Ts_list_c) for Ts_list_c in Ts_lists]
    expected = compile_nonergodic(component_Ts, phi_list, beta_0)

    assert jnp.allclose(actual, expected)
    assert jnp.isclose(jnp.linalg.norm(jnp.sum(actual, axis=0), ord=jnp.inf), 1)
    assert jnp.min(actual) > -1e-6


def test_init():
    Ts_lists = [
        [
            cycle(2),
            cycle(3, 1.0),
        ],
        [
            cycle(2),
            cycle(3, 1.0),
        ],
    ]
    phi_list = [
        jnp.array([0, 1, 2, 3, 4, 5]),
        jnp.array([3, 4, 5, 6, 7, 8]),
    ]
    beta_0 = jnp.array([0.4, 0.6])

    data = init(Ts_lists, phi_list, beta_0)

    expected_Ts = jnp.array(
        [
            [
                [
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ],
                [
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                ],
            ],
            [
                [
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ],
                [
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                ],
            ],
        ]
    )
    assert jnp.allclose(data.Ts, expected_Ts)
    expected_eta_0 = jnp.array(
        [
            [
                [1 / 2, 1 / 2, 0],
                [1 / 3, 1 / 3, 1 / 3],
            ],
            [
                [1 / 2, 1 / 2, 0],
                [1 / 3, 1 / 3, 1 / 3],
            ],
        ]
    )
    assert jnp.allclose(data.eta_0, expected_eta_0)
    expected_w = jnp.array(
        [
            [
                [1, 1, 0],
                [1, 1, 1],
            ],
            [
                [1, 1, 0],
                [1, 1, 1],
            ],
        ]
    )
    assert jnp.allclose(data.w, expected_w)
    assert jnp.allclose(data.Vs, jnp.array([[2, 3], [2, 3]]))
    assert jnp.allclose(data.Ss, jnp.array([[2, 3], [2, 3]]))
    assert jnp.allclose(data.V_cs, jnp.array([6, 6]))
    assert data.V == 9

    expected_phi = jnp.array([[0, 1, 2, 3, 4, 5], [3, 4, 5, 6, 7, 8]])
    assert jnp.allclose(data.phi, expected_phi)
    expected_phi_one_hot = jnp.array(
        [
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
            ],
            [
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
        ]
    )
    assert jnp.allclose(data.phi_one_hot, expected_phi_one_hot)
    assert jnp.allclose(data.beta_0, beta_0)


def test_obs_dist():
    Ts_lists = [
        [
            cycle(2),
            cycle(3, 1.0),
        ],
        [
            cycle(2),
            cycle(3, 1.0),
        ],
    ]
    phi_list = [
        jnp.array([0, 1, 2, 3, 4, 5]),
        jnp.array([3, 4, 5, 6, 7, 8]),
    ]
    beta_0 = jnp.array([0.4, 0.6])

    data = init(Ts_lists, phi_list, beta_0)
    weights = mixed_radix_weights(data.Vs[0])
    decode = partial(mixed_radix_decode, Vs=data.Vs[0], weights=weights)
    _obs_dist = partial(obs_dist, data=data, decode=decode)
    actual = jax.jit(_obs_dist)(eta=data.eta_0, beta=data.beta_0)
    expected = jnp.array([2, 2, 2, 5, 5, 5, 3, 3, 3]) / 30
    assert jnp.allclose(actual, expected)


def test_generate():
    Ts_lists = [
        [
            cycle(2),
            cycle(3, 1.0),
        ],
        [
            cycle(2),
            cycle(3, 1.0),
        ],
    ]
    phi_list = [
        jnp.array([0, 1, 2, 3, 4, 5]),
        jnp.array([3, 4, 5, 6, 7, 8]),
    ]
    beta_0 = jnp.array([0.4, 0.6])

    data = init(Ts_lists, phi_list, beta_0)
    batch_size = 4
    num_steps = 6
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size * num_steps).reshape(batch_size, num_steps, 2)
    eta_0 = jnp.repeat(data.eta_0[None, :, :, :], batch_size, axis=0)
    beta_init = jnp.repeat(data.beta_0[None, :], batch_size, axis=0)
    weights = mixed_radix_weights(data.Vs[0])
    decode = partial(mixed_radix_decode, Vs=data.Vs[0], weights=weights)

    def generate_one(eta: jax.Array, beta: jax.Array, keys: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return generate(data, eta, beta, keys, decode=decode)

    eta, beta, xs = jax.jit(jax.vmap(generate_one, in_axes=(0, 0, 0)))(eta_0, beta_init, keys)

    def validate_one_hot(arr: jax.Array, size: int | jax.Array) -> jax.Array:
        return (
            jnp.isclose(jnp.sum(arr), 1)
            & jnp.isclose(jnp.max(arr), 1)
            & jnp.isclose(jnp.min(arr), 0)
            & (jnp.argmax(arr) < size)
        )

    assert beta.shape == (batch_size, 2)
    assert jnp.all(jax.vmap(validate_one_hot, in_axes=(0, None))(beta, 2))

    def validate_updated_eta(eta_c: jax.Array, S_c: jax.Array) -> jax.Array:
        state_ids = jnp.arange(eta_c.shape[1])
        valid_states = state_ids[None, :] < S_c[:, None]
        valid_entries = jnp.where(valid_states, eta_c, 0)
        padded_entries = jnp.where(valid_states, 0, eta_c)
        return jnp.all(jax.vmap(validate_one_hot, in_axes=(0, 0))(valid_entries, S_c)) & jnp.all(
            jnp.isclose(padded_entries, 0)
        )

    def validate_component_eta(eta_c: jax.Array, eta_0_c: jax.Array, S_c: jax.Array) -> jax.Array:
        return jnp.allclose(eta_c, eta_0_c) | validate_updated_eta(eta_c, S_c)

    def validate_eta(eta_run: jax.Array, beta_run: jax.Array) -> jax.Array:
        component_is_valid = jax.vmap(validate_component_eta, in_axes=(0, 0, 0))(eta_run, data.eta_0, data.Ss)
        component_is_updated = jax.vmap(validate_updated_eta, in_axes=(0, 0))(eta_run, data.Ss)
        return jax.lax.cond(
            jnp.isclose(beta_run[0], 1),
            lambda _: component_is_updated[0] & component_is_valid[1],
            lambda _: component_is_valid[0] & component_is_updated[1],
            operand=None,
        )

    assert eta.shape == (batch_size, 2, 2, 3)
    assert jnp.all(jax.vmap(validate_eta)(eta, beta))

    def validate_cycle(actual: jax.Array, expected: jax.Array, size: int) -> jax.Array:
        return jnp.any(jax.vmap(lambda i: jnp.all(actual == jnp.roll(expected, i)))(jnp.arange(size)))

    def validate_xs(beta_run: jax.Array, xs_run: jax.Array) -> jax.Array:
        return jax.lax.cond(
            jnp.isclose(beta_run[0], 1),
            lambda _: validate_cycle(xs_run, jnp.array([0, 3, 4, 1, 2, 5]), 6),
            lambda _: validate_cycle(xs_run, jnp.array([3, 6, 7, 4, 5, 8]), 6),
            operand=None,
        )

    assert xs.shape == (batch_size, num_steps)
    assert jnp.all(jax.vmap(validate_xs)(beta, xs))


def test_seq_prob():
    Ts_lists = [
        [
            cycle(2),
            cycle(3, 1.0),
        ],
        [
            cycle(2),
            cycle(3, 1.0),
        ],
    ]
    phi_list = [
        jnp.array([0, 1, 2, 3, 4, 5]),
        jnp.array([3, 4, 5, 6, 7, 8]),
    ]
    beta_0 = jnp.array([0.4, 0.6])

    data = init(Ts_lists, phi_list, beta_0)
    weights = mixed_radix_weights(data.Vs[0])
    decode = partial(mixed_radix_decode, Vs=data.Vs[0], weights=weights)
    seq_prob_one = partial(seq_prob, data=data, decode=decode)
    xs = jnp.array(
        [
            [0, 3, 4, 1, 2, 5],
            [3, 6, 7, 4, 5, 8],
        ]
    )
    actual = jax.jit(jax.vmap(seq_prob_one))(xs=xs)
    expected = jnp.array([2, 3]) / 30
    assert jnp.allclose(actual, expected)
