import jax
import jax.numpy as jnp

from src.ghmm.process import generate, init, seq_prob, validate
from tests.transition_matrices import zero_one_random


def test_init():
    Ts = jnp.array(zero_one_random(0.5))
    assert validate(Ts)
    data = init(Ts)
    assert jnp.allclose(data.Ts, Ts)
    assert jnp.allclose(data.eta_0, jnp.ones(3) / 3)
    assert jnp.allclose(data.w, jnp.ones(3))


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
