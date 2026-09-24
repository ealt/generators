import jax.numpy as jnp
import pytest

from transition_matrices.classical import checksum


def test_checksum_rrxor():
    p = 0.3  # first bit
    q = 0.7  # second bit
    # States are (phase, running sum): S, "0", "1", F, T. The conventional RRXOR
    # basis {S, "0", "1", T, F} has the last two transposed.
    assert jnp.allclose(
        checksum(jnp.array([[p, 1 - p], [q, 1 - q]])),
        jnp.array(
            [
                [
                    [0, p, 0, 0, 0],
                    [0, 0, 0, q, 0],
                    [0, 0, 0, 0, q],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 1 - p, 0, 0],
                    [0, 0, 0, 0, 1 - q],
                    [0, 0, 0, 1 - q, 0],
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                ],
            ]
        ),
    )


def test_checksum_repeat():
    p = 0.3  # n = 1: emit a random symbol, then emit it again
    assert jnp.allclose(
        checksum(jnp.array([[p, 1 - p]])),
        jnp.array(
            [
                [
                    [0, p, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                ],
                [
                    [0, 0, 1 - p],
                    [0, 0, 0],
                    [1, 0, 0],
                ],
            ]
        ),
    )


def test_checksum_single_symbol():
    # m = 1: every sum mod 1 is 0, leaving a deterministic cycle of n + 1 states.
    assert jnp.allclose(checksum(jnp.ones((3, 1))), jnp.roll(jnp.eye(4), 1, axis=1)[None])


def test_checksum_ternary():
    # m = 3, n = 2: 7 states, and the checksum phase emits its residue.
    Ts = checksum(jnp.full((2, 3), 1 / 3))
    assert Ts.shape == (3, 7, 7)
    assert jnp.allclose(Ts.sum(axis=(0, 2)), 1)
    assert jnp.allclose(Ts.sum(axis=2)[:, 4:], jnp.eye(3))


def test_checksum_periodic():
    # Every state advances one phase per symbol, so the net matrix has n + 1
    # eigenvalues of modulus 1 and power iteration cycles instead of converging.
    T = checksum(jnp.full((2, 2), 0.5)).sum(axis=0)
    assert jnp.sum(jnp.abs(jnp.abs(jnp.linalg.eigvals(T)) - 1) < 1e-5) == 3
    seed = jnp.zeros(5).at[0].set(1.0)
    assert jnp.allclose(seed @ T @ T @ T, seed, atol=1e-5)


def test_checksum_entropy_rate():
    # n random symbols of log2(m) bits over a block of n + 1: 2/3 for RRXOR.
    Ts = checksum(jnp.full((2, 2), 0.5))
    emit = Ts.sum(axis=2).T
    h = -(emit * jnp.log2(jnp.where(emit > 0, emit, 1))).sum(axis=1)
    pi = jnp.array([2, 1, 1, 1, 1]) / 6  # steady state
    assert jnp.allclose(pi @ h, 2 / 3)


def test_checksum_invalid_probs():
    with pytest.raises(AssertionError):
        checksum(jnp.array([0.5, 0.5]))  # not 2-D
    with pytest.raises(AssertionError):
        checksum(jnp.array([[0.5, 0.4]]))  # rows must sum to 1
    with pytest.raises(AssertionError):
        checksum(jnp.zeros((0, 2)))  # n = 0 scatters to state index -1
