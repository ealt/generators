import jax.numpy as jnp
import pytest

from transition_matrices.augmentations import compress_vocab, expand_vocab, leak_toward
from transition_matrices.classical import checksum, cycle, mess


def test_expand_vocab():
    x = 0.3  # y = 1 - ((s - 1) * x) = 0.7
    a = 0.6  # b = (1 - a) / (s - 1) = 0.4
    s = 2
    f = 2
    ax = 0.09  # a * x / f
    ay = 0.21  # a * y / f
    bx = 0.06  # b * x / f
    by = 0.14  # b * y / f
    assert jnp.allclose(
        expand_vocab(mess(x, a, s), f),
        jnp.array(
            [
                [
                    [ay, bx],
                    [ax, by],
                ],
                [
                    [ay, bx],
                    [ax, by],
                ],
                [
                    [by, ax],
                    [bx, ay],
                ],
                [
                    [by, ax],
                    [bx, ay],
                ],
            ]
        ),
    )


def test_compress_vocab():
    x = 0.3  # y = 1 - ((s - 1) * x) = 0.1
    a = 0.7  # b = (1 - a) / (s - 1) = 0.1
    s = 4
    f = 2
    ax = 0.24  # (a + (f - 1) * b) * x
    ay = 0.08  # (a + (f - 1) * b) * y
    bx = 0.06  # f * b * x
    by = 0.02  # f * b * y
    assert jnp.allclose(
        compress_vocab(mess(x, a, s), f),
        jnp.array(
            [
                [
                    [ay, ax, bx, bx],
                    [ax, ay, bx, bx],
                    [ax, ax, by, bx],
                    [ax, ax, bx, by],
                ],
                [
                    [by, bx, ax, ax],
                    [bx, by, ax, ax],
                    [bx, bx, ay, ax],
                    [bx, bx, ax, ay],
                ],
            ]
        ),
    )


def test_compress_vocab_inverts_expand_vocab():
    Ts = mess(0.15, 0.6, 4)
    assert jnp.allclose(compress_vocab(expand_vocab(Ts, 3), 3), Ts)


def leaky_rrxor_reference(p1: float, p2: float, epsilon: float) -> jnp.ndarray:
    """simplexity's leaky_rrxor leak formula (commit 657aff7), over our RRXOR base.

    The epsilon / 10 divisor hardcodes RRXOR's 2 symbols x 5 states. The uniform leak
    term is permutation-invariant, so the state ordering does not matter here; the base
    matrices are pinned to simplexity's rrxor in test_classical.py.
    """
    return (1 - epsilon) * checksum(jnp.array([[p1, 1 - p1], [p2, 1 - p2]])) + (epsilon / 10) * jnp.ones((2, 5, 5))


@pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.5, 1.0])
def test_leak_toward_reproduces_leaky_rrxor(epsilon):
    # The uniform default is exactly simplexity's leak at RRXOR's shape.
    p1, p2 = 0.3, 0.7
    Ts = checksum(jnp.array([[p1, 1 - p1], [p2, 1 - p2]]))
    assert jnp.allclose(leak_toward(Ts, epsilon), leaky_rrxor_reference(p1, p2, epsilon))


@pytest.mark.parametrize("epsilon", [0.0, 0.05, 0.3, 1.0])
@pytest.mark.parametrize(("n", "m"), [(1, 2), (2, 2), (2, 3), (3, 2), (4, 5)])
def test_leak_toward_preserves_row_sums(n, m, epsilon):
    # The bug carried over from leaky_rrxor: a hardcoded divisor only normalizes the one
    # shape it was written for. (2, 3), (3, 2) and (4, 5) are not 2 symbols x 5 states.
    Ts = leak_toward(checksum(jnp.full((n, m), 1 / m)), epsilon)
    assert jnp.allclose(Ts.sum(axis=(0, 2)), 1)
    assert jnp.all(Ts >= 0)


def test_leak_toward_hardcoded_divisor_would_unnormalize():
    # Guards the fix: RRXOR's literal epsilon / 10 breaks as soon as the shape changes.
    Ts = checksum(jnp.full((3, 2), 0.5))  # 2 symbols x 7 states
    epsilon = 0.2
    hardcoded = (1 - epsilon) * Ts + (epsilon / 10) * jnp.ones(Ts.shape)
    assert not jnp.allclose(hardcoded.sum(axis=(0, 2)), 1)
    assert jnp.allclose(leak_toward(Ts, epsilon).sum(axis=(0, 2)), 1)


def test_leak_toward_epsilon_zero_is_identity():
    Ts = checksum(jnp.array([[0.3, 0.7], [0.6, 0.4]]))
    assert jnp.allclose(leak_toward(Ts, 0.0), Ts)


def test_leak_toward_epsilon_one_is_pure_reset():
    Ts = checksum(jnp.full((2, 2), 0.5))
    v, s = Ts.shape[0], Ts.shape[1]
    assert jnp.allclose(leak_toward(Ts, 1.0), jnp.ones((v, s, s)) / (v * s))


def test_leak_toward_makes_deterministic_transitions_positive():
    # The point of a leak: nothing stays impossible.
    Ts = checksum(jnp.full((2, 2), 0.5))
    assert not jnp.all(Ts > 0)
    assert jnp.all(leak_toward(Ts, 0.1) > 0)


def test_leak_toward_stationary_target_preserves_the_stationary_state():
    # Leaking toward a process's own stationary state leaves that state stationary.
    Ts = checksum(jnp.array([[0.3, 0.7], [0.6, 0.4]]))
    eta = jnp.array([1, 0.3, 0.7, 0.3 * 0.6 + 0.7 * 0.4, 0.3 * 0.4 + 0.7 * 0.6]) / 3
    assert jnp.allclose(eta @ Ts.sum(axis=0), eta)
    leaked = leak_toward(Ts, 0.25, eta).sum(axis=0)
    assert jnp.allclose(eta @ leaked, eta)


def test_leak_toward_preserves_the_normalizing_eigenvector():
    # The gauge condition: Ts'.sum(0) @ w == w, so the leaked process stays a valid GHMM.
    Ts = cycle(4, 0.3)
    w = jnp.ones(4)
    leaked = leak_toward(Ts, 0.4).sum(axis=0)
    assert jnp.allclose(leaked @ w, w)


def test_leak_toward_rejects_invalid_arguments():
    Ts = checksum(jnp.full((2, 2), 0.5))
    with pytest.raises(AssertionError):
        leak_toward(Ts, -0.1)
    with pytest.raises(AssertionError):
        leak_toward(Ts, 1.5)
    with pytest.raises(AssertionError):
        leak_toward(Ts, 0.1, eta=jnp.ones(5))  # eta @ w == 5, not canonically normalized
