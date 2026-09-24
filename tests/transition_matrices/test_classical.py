import itertools

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


def test_checksum_block():
    n, m = 3, 3
    p = 1 / m
    Ts = checksum(jnp.full((n, m), p))
    seed = jnp.zeros(n * m + 1).at[0].set(1.0)
    # Any n random symbols, then their sum mod m, returns to the seed state.
    for block in itertools.product(range(m), repeat=n):
        eta = seed
        for x in (*block, sum(block) % m):
            eta = eta @ Ts[x]
        assert jnp.allclose(eta, p**n * seed)


def test_checksum_degenerate():
    p = 0.3
    # n = 1: emit a random symbol, then emit it again.
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
    # m = 1: every sum mod 1 is 0, leaving a deterministic cycle of n + 1 states.
    assert jnp.allclose(
        checksum(jnp.ones((3, 1))),
        jnp.array(
            [
                [
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 0, 0, 0],
                ]
            ]
        ),
    )


def test_checksum_invalid_probs():
    with pytest.raises(AssertionError):
        checksum(jnp.array([0.5, 0.5]))  # not 2-D
    with pytest.raises(AssertionError):
        checksum(jnp.array([[0.5, 0.4]]))  # rows must sum to 1
    with pytest.raises(AssertionError):
        checksum(jnp.zeros((0, 2)))  # n = 0 scatters to state index -1
