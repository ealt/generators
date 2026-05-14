import jax
import jax.numpy as jnp

from generators.factored.nonergodic.process import compile, generate, init, obs_dist, seq_prob
from generators.ghmm.process import validate as validate_ghmm
from tests.transition_matrices import cycle, zero_one


def test_compile():
    Ts_list = [
        jnp.array(zero_one()),
        jnp.array(zero_one()),
    ]
    phi_list = [
        jnp.array([0, 1]),
        jnp.array([1, 2]),
    ]
    beta_0 = jnp.array([0.25, 0.75])

    composite = compile(Ts_list, phi_list, beta_0)
    assert validate_ghmm(composite)

    expected = jnp.zeros((3, 5, 5))
    expected = expected.at[0, 0, 2].set(0.125)
    expected = expected.at[1, 0, 1].set(0.125)
    expected = expected.at[1, 0, 4].set(0.375)
    expected = expected.at[2, 0, 3].set(0.375)
    expected = expected.at[0, 1, 2].set(1)
    expected = expected.at[1, 2, 1].set(1)
    expected = expected.at[1, 3, 4].set(1)
    expected = expected.at[2, 4, 3].set(1)
    assert jnp.allclose(composite, expected)


def test_init():
    Ts_list = [
        jnp.array(zero_one()),
        jnp.array(cycle(3, 0.5)),
    ]
    phi_list = [
        jnp.array([0, 1]),
        jnp.array([1, 2, 3]),
    ]
    beta_0 = jnp.array([0.25, 0.75])

    data = init(Ts_list, phi_list, beta_0)
    expected_Ts = jnp.stack(
        [
            jnp.zeros((3, 3, 3)).at[0, 0, 1].set(1).at[1, 1, 0].set(1),
            jnp.array(cycle(3, 0.5)),
        ]
    )
    assert jnp.allclose(data.Ts, expected_Ts)
    assert jnp.allclose(data.eta_0, jnp.array([[3, 3, 0], [2, 2, 2]]) / 6)
    assert jnp.allclose(data.w, jnp.array([[1, 1, 0], [1, 1, 1]]))
    assert jnp.allclose(data.Vs, jnp.array([2, 3]))
    assert jnp.allclose(data.Ss, jnp.array([2, 3]))
    assert data.V == 4
    assert jnp.allclose(data.phi, jnp.array([[0, 1, 0], [1, 2, 3]]))
    expected_phi_one_hot = jnp.array(
        [
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        ]
    )
    assert jnp.allclose(data.phi_one_hot, expected_phi_one_hot)
    assert jnp.allclose(data.beta_0, jnp.array([0.25, 0.75]))


def test_obs_dist():
    Ts_list = [
        jnp.array(zero_one()),
        jnp.array(cycle(3, 0.5)),
    ]
    phi_list = [
        jnp.array([0, 1]),
        jnp.array([1, 2, 3]),
    ]
    beta_0 = jnp.array([0.25, 0.75])

    data = init(Ts_list, phi_list, beta_0)

    expected = jnp.array([0.125, 0.375, 0.25, 0.25])
    actual = jax.jit(obs_dist)(data, data.eta_0, data.beta_0)
    assert jnp.allclose(actual, expected)


def test_generate():
    Ts_list = [
        jnp.array(zero_one()),
        jnp.array(cycle(3, 1.0)),
    ]
    phi_list = [
        jnp.array([0, 1]),
        jnp.array([1, 2, 3]),
    ]
    beta_0 = jnp.array([0.25, 0.75])

    data = init(Ts_list, phi_list, beta_0)

    batch_size = 4
    num_steps = 12
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size * num_steps).reshape(batch_size, num_steps, 2)
    eta_0 = jnp.repeat(data.eta_0[None, :, :], batch_size, axis=0)
    beta_0 = jnp.repeat(data.beta_0[None, :], batch_size, axis=0)
    eta, beta, xs = jax.jit(jax.vmap(generate, in_axes=(None, 0, 0, 0)))(data, eta_0, beta_0, keys)

    def validate_one_hot(arr: jax.Array, size: int | jax.Array) -> jax.Array:
        return (
            jnp.isclose(jnp.sum(arr), 1)
            & jnp.isclose(jnp.max(arr), 1)
            & jnp.isclose(jnp.min(arr), 0)
            & (jnp.argmax(arr) < size)
        )

    assert beta.shape == (4, 2)
    assert jnp.all(jax.vmap(validate_one_hot, in_axes=(0, None))(beta, 2))

    def validate_component_eta(eta_c: jax.Array, S_c: jax.Array) -> jax.Array:
        state_ids = jnp.arange(eta_c.shape[0])
        valid_state = state_ids < S_c
        valid_entries = jnp.where(valid_state, eta_c, 0)
        padded_entries = jnp.where(valid_state, 0, eta_c)
        return validate_one_hot(valid_entries, S_c) & jnp.all(jnp.isclose(padded_entries, 0))

    def validate_eta(eta: jax.Array) -> jax.Array:
        return jnp.all(jax.vmap(validate_component_eta)(eta, data.Ss))

    assert eta.shape == (4, 2, 3)
    assert jnp.all(jax.vmap(validate_eta)(eta))

    def validate_cycle(actual: jax.Array, expected: jax.Array, size: int) -> jax.Array:

        def validate_specific_cycle(i: jax.Array) -> jax.Array:
            return jnp.all(actual == jnp.roll(expected, i))

        return jnp.any(jax.vmap(validate_specific_cycle)(jnp.arange(size)))

    def validate_xs(beta: jax.Array, xs: jax.Array) -> jax.Array:
        return jax.lax.cond(
            jnp.isclose(beta[0], 1),
            lambda _: validate_cycle(xs, jnp.mod(jnp.arange(num_steps), 2), 2),
            lambda _: validate_cycle(xs, 1 + jnp.mod(jnp.arange(num_steps), 3), 3),
            operand=None,
        )

    assert xs.shape == (4, 12)
    assert jnp.all(jax.vmap(validate_xs)(beta, xs))


def test_seq_prob():
    Ts_list = [
        jnp.array(zero_one()),
        jnp.array(cycle(3, 1.0)),
    ]
    phi_list = [
        jnp.array([0, 1]),
        jnp.array([1, 2, 3]),
    ]
    beta_0 = jnp.array([0.25, 0.75])

    data = init(Ts_list, phi_list, beta_0)
    xs = jnp.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            [2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    actual = jax.jit(jax.vmap(seq_prob, in_axes=(None, 0)))(data, xs)
    expected = jnp.array([0.125, 0.25, 0.25, 0.125])
    assert jnp.allclose(actual, expected)
