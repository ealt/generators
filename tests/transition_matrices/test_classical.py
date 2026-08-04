import jax.numpy as jnp
import pytest

from generators.utils import principal_ev
from transition_matrices.classical import checksum

# State permutation from this repo's (phase, residue) order to the conventional RRXOR
# basis {S: 0, "0": 1, "1": 2, T: 3, F: 4}. Only the two checksum states differ: T is
# residue 1 and F is residue 0, so they are transposed relative to residue order.
RRXOR_PERM = jnp.array([0, 1, 2, 4, 3])


def rrxor_reference(p1: float, p2: float) -> jnp.ndarray:
    """Verbatim transcription of simplexity's rrxor (commit 657aff7).

    simplexity/generative_processes/transition_matrices.py. Transcribed rather than
    imported so the check does not depend on simplexity being installed.
    """
    s = {"S": 0, "0": 1, "1": 2, "T": 3, "F": 4}
    Ts = jnp.zeros((2, 5, 5))
    Ts = Ts.at[0, s["S"], s["0"]].set(p1)
    Ts = Ts.at[1, s["S"], s["1"]].set(1 - p1)
    Ts = Ts.at[0, s["0"], s["F"]].set(p2)
    Ts = Ts.at[1, s["0"], s["T"]].set(1 - p2)
    Ts = Ts.at[0, s["1"], s["T"]].set(p2)
    Ts = Ts.at[1, s["1"], s["F"]].set(1 - p2)
    Ts = Ts.at[1, s["T"], s["S"]].set(1.0)
    Ts = Ts.at[0, s["F"], s["S"]].set(1.0)
    return Ts


def stationary(Ts: jnp.ndarray) -> jnp.ndarray:
    pi = principal_ev(Ts.sum(axis=0).T)
    return pi / pi.sum()


def entropy_rate(Ts: jnp.ndarray) -> jnp.ndarray:
    """Entropy rate in bits/symbol of a unifilar HMM: sum_s pi_s H(x | s)."""
    emit = Ts.sum(axis=2).T  # (S, V): P(x | s)
    terms = jnp.where(emit > 0, -emit * jnp.log2(jnp.where(emit > 0, emit, 1)), 0.0)
    return stationary(Ts) @ terms.sum(axis=1)


@pytest.mark.parametrize(("p1", "p2"), [(0.5, 0.5), (0.3, 0.7), (0.2, 0.2), (1.0, 0.4)])
def test_checksum_reproduces_rrxor(p1, p2):
    Ts = checksum(jnp.array([[p1, 1 - p1], [p2, 1 - p2]]))
    permuted = Ts[:, RRXOR_PERM][:, :, RRXOR_PERM]
    assert jnp.array_equal(permuted, rrxor_reference(p1, p2))


@pytest.mark.parametrize(("n", "m"), [(1, 2), (1, 5), (2, 2), (2, 3), (3, 2), (3, 4), (4, 5)])
def test_checksum_shape_and_stochasticity(n, m):
    Ts = checksum(jnp.full((n, m), 1 / m))
    assert Ts.shape == (m, n * m + 1, n * m + 1)
    assert jnp.all(Ts >= 0)
    assert jnp.allclose(Ts.sum(axis=(0, 2)), 1)


def test_checksum_n_1_repeats_the_random_symbol():
    # n = 1 degenerates to "emit a random symbol, then emit it again".
    p = 0.3
    Ts = checksum(jnp.array([[p, 1 - p]]))
    assert Ts.shape == (2, 3, 3)
    expected = jnp.array(
        [
            # symbol 0: seed -> residue-0 state with prob p; residue-0 state -> seed
            [
                [0, p, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            # symbol 1: seed -> residue-1 state with prob 1 - p; residue-1 state -> seed
            [
                [0, 0, 1 - p],
                [0, 0, 0],
                [1, 0, 0],
            ],
        ]
    )
    assert jnp.allclose(Ts, expected)


def test_checksum_deterministic_symbol_is_the_running_sum():
    # From a checksum state the emitted symbol is the residue, with probability 1.
    n, m = 3, 4
    Ts = checksum(jnp.full((n, m), 1 / m))
    for r in range(m):
        state = 1 + (n - 1) * m + r
        emit = Ts.sum(axis=2)[:, state]
        assert jnp.allclose(emit, jnp.eye(m)[r])


@pytest.mark.parametrize(("n", "m"), [(1, 2), (2, 2), (2, 3), (3, 2), (4, 5)])
def test_checksum_stationary_distribution(n, m):
    # Closed form: 1 / (n + 1) on the seed state, and P(sum of the first i symbols = r)
    # / (n + 1) on state (i, r) -- uniform rows make every residue equally likely.
    Ts = checksum(jnp.full((n, m), 1 / m))
    expected = jnp.concatenate([jnp.ones(1), jnp.full(n * m, 1 / m)]) / (n + 1)
    assert jnp.allclose(stationary(Ts), expected)


def test_checksum_stationary_distribution_biased():
    # RRXOR's published steady state [2, 1, 1, 1, 1] / 6 holds for uniform rows.
    Ts = checksum(jnp.array([[0.5, 0.5], [0.5, 0.5]]))
    assert jnp.allclose(stationary(Ts), jnp.array([2, 1, 1, 1, 1]) / 6)

    # With biased rows the residue masses follow the running-sum distribution.
    p1, p2 = 0.3, 0.7
    Ts = checksum(jnp.array([[p1, 1 - p1], [p2, 1 - p2]]))
    expected = jnp.array(
        [
            1,  # seed
            p1,  # phase 1, residue 0
            1 - p1,  # phase 1, residue 1
            p1 * p2 + (1 - p1) * (1 - p2),  # phase 2, residue 0
            p1 * (1 - p2) + (1 - p1) * p2,  # phase 2, residue 1
        ]
    ) / 3
    assert jnp.allclose(stationary(Ts), expected)


@pytest.mark.parametrize(("n", "m"), [(1, 2), (2, 2), (2, 3), (3, 2), (3, 4), (4, 5)])
def test_checksum_entropy_rate_uniform(n, m):
    # n random symbols of log2(m) bits each, spread over a block of n + 1 symbols.
    Ts = checksum(jnp.full((n, m), 1 / m))
    assert jnp.allclose(entropy_rate(Ts), n * jnp.log2(jnp.array(m)) / (n + 1))


def test_checksum_entropy_rate_rrxor_is_two_thirds():
    Ts = checksum(jnp.full((2, 2), 0.5))
    assert jnp.allclose(entropy_rate(Ts), 2 / 3)


def test_checksum_entropy_rate_biased():
    # sum_i H(probs[i]) / (n + 1), with per-row binary entropies.
    p1, p2 = 0.3, 0.7
    Ts = checksum(jnp.array([[p1, 1 - p1], [p2, 1 - p2]]))

    def h(p):
        return -p * jnp.log2(p) - (1 - p) * jnp.log2(1 - p)

    assert jnp.allclose(entropy_rate(Ts), (h(p1) + h(p2)) / 3)


def test_checksum_rejects_invalid_probs():
    with pytest.raises(AssertionError):
        checksum(jnp.array([0.5, 0.5]))  # not 2-D
    with pytest.raises(AssertionError):
        checksum(jnp.array([[0.5, 0.4]]))  # row does not sum to 1
    with pytest.raises(AssertionError):
        checksum(jnp.ones((2, 1)))  # m < 2
    with pytest.raises(AssertionError):
        checksum(jnp.zeros((0, 2)))  # n < 1
