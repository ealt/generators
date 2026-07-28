import jax
import jax.numpy as jnp
import pytest

from generators.ghmm.process import generate, init, seq_prob, validate
from transition_matrices.classical import zero_one_random


def block_diagonal_binary(p: float = 0.9, S: int = 4) -> jax.Array:
    """Two disjoint recurrent components over a shared binary alphabet."""
    T = jnp.zeros((2, S, S))
    h = S // 2
    for s in range(h):  # component 0: symbol-0 heavy
        T = T.at[0, s, (s + 1) % h].set(p).at[1, s, s].set(1 - p)
    for s in range(h, S):  # component 1: symbol-1 heavy
        T = T.at[1, s, h + (s + 1 - h) % h].set(p).at[0, s, s].set(1 - p)
    return T / T.sum(axis=(0, 2), keepdims=True)


def test_init():
    Ts = jnp.array(zero_one_random(0.5))
    assert validate(Ts)
    data = init(Ts)
    assert jnp.allclose(data.Ts, Ts)
    assert jnp.allclose(data.eta_0, jnp.ones(3) / 3)
    assert jnp.allclose(data.w, jnp.ones(3))


def test_init_reducible():
    Ts = block_diagonal_binary()
    assert validate(Ts)  # well formed and in gauge; reducibility is init's to refuse
    with pytest.raises(ValueError, match="reducible"):
        init(Ts)


def test_init_reducible_explicit_eta_0():
    Ts = block_diagonal_binary()
    data = init(Ts, eta_0=jnp.ones(4) / 4)
    assert jnp.allclose(data.eta_0, jnp.ones(4) / 4)
    assert jnp.allclose(data.w, jnp.ones(4))  # exact for a row-stochastic Ts

    # Every state is reachable from some component, so the stated mixture — not the
    # component the eigensolver happens to return — sets the sequence probability.
    xs = jnp.array([1, 0, 1, 1, 0, 1])
    component_0 = init(Ts[:, :2, :2])
    component_1 = init(Ts[:, 2:, 2:])
    assert not jnp.isclose(seq_prob(component_0, xs), seq_prob(component_1, xs))
    expected = (seq_prob(component_0, xs) + seq_prob(component_1, xs)) / 2
    assert jnp.isclose(seq_prob(data, xs), expected)


def test_generate_reducible():
    Ts = block_diagonal_binary()
    data = init(Ts, eta_0=jnp.ones(4) / 4)
    keys = jax.random.split(jax.random.key(0), (16, 32))
    eta, _ = jax.jit(jax.vmap(generate, in_axes=(None, None, 0)))(data, data.eta_0, keys)

    # Sequences must occupy both components. Under the collapsed eta_0 they all
    # occupied component 0, and every cheaper check still looked healthy.
    component_1_mass = jnp.sum(eta[:, 2:], axis=1)
    assert jnp.any(component_1_mass > 0.5)
    assert jnp.any(component_1_mass < 0.5)


def test_generate():
    Ts = jnp.array(zero_one_random(0.5))
    data = init(Ts)
    keys = jax.random.split(jax.random.key(0), 12)
    eta, xs = jax.jit(generate)(data, data.eta_0, keys)
    assert jnp.allclose(data.w, jnp.ones(3))
    assert jnp.allclose(data.eta_0, jnp.ones(3) / 3)
    assert eta.shape == (3,)
    assert jnp.isclose(jnp.sum(eta), 1)
    assert xs.shape == (12,)
    assert any([jnp.all(xs[i::3] == 0) for i in range(3)])
    assert any([jnp.all(xs[i::3] == 1) for i in range(3)])


def test_seq_prob():
    Ts = jnp.array(zero_one_random(0.5))
    data = init(Ts)
    xs = jnp.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0])
    expected = (0.5 ** (xs.size / 3)) / 3
    assert jnp.isclose(jax.jit(seq_prob)(data, xs), expected)
