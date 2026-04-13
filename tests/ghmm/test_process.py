import jax
import jax.numpy as jnp

from src.ghmm.process import generate, init, seq_prob, validate_transition_matrix
from tests.transition_matrices import zero_one, zero_one_random


def test_zero_one():
    """Test the generate function."""
    Ts = jnp.array(zero_one())
    assert validate_transition_matrix(Ts)
    data = init(Ts)
    eta, xs = generate(data, n=12, key=jax.random.key(0))
    assert jnp.allclose(data.w, jnp.ones(2))
    assert jnp.allclose(data.eta_0, jnp.array([0.5, 0.5]))
    assert eta.shape == (2,)
    assert jnp.isclose(jnp.sum(eta), 1)
    assert xs.shape == (12,)
    assert jnp.all(xs[1:] != xs[:-1])
    assert jnp.all(xs[2:] == xs[:-2])
    assert seq_prob(data, xs) == 0.5


def test_zero_one_random():
    """Test the generate function."""
    Ts = jnp.array(zero_one_random(0.5))
    assert validate_transition_matrix(Ts)
    data = init(Ts)
    eta, xs = generate(data, n=12, key=jax.random.key(0))
    assert jnp.allclose(data.w, jnp.ones(3))
    assert jnp.allclose(data.eta_0, jnp.ones(3) / 3)
    assert eta.shape == (3,)
    assert jnp.isclose(jnp.sum(eta), 1)
    assert xs.shape == (12,)
    assert any([jnp.all(xs[i::3] == 0) for i in range(3)])
    assert any([jnp.all(xs[i::3] == 1) for i in range(3)])
    assert jnp.isclose(seq_prob(data, xs), 0.5**4 / 3)
