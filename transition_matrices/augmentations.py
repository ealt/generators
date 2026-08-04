"""Augmentations: transformations from one process's transition matrix to a related one.

`expand_vocab` / `compress_vocab` decouple emission-alphabet size from state-space size.
They are ported from simplex-research `src/transition_matrices/augmentations.py`
(commit 00312f5b).
"""

import jax
import jax.numpy as jnp


def expand_vocab(Ts: jax.Array, f: int) -> jax.Array:
    """Expand the vocabulary of a process.

    Each symbol of the original process is expanded into f symbols, splitting its mass
    evenly among them. The marginal state chain, Ts.sum(axis=0), is unchanged.

    Args:
        Ts: Transition matrix, shape (V, S, S).
        f: Factor to expand the vocabulary.

    Returns:
        Transition matrix, shape (V * f, S, S).
    """
    assert f > 0
    v = Ts.shape[0] * f
    v_idx = jnp.arange(v) // f
    return Ts[v_idx, :] / f


def compress_vocab(Ts: jax.Array, f: int) -> jax.Array:
    """Compress the vocabulary of a process.

    Each block of f consecutive symbols of the original process is compressed into 1
    symbol. The marginal state chain, Ts.sum(axis=0), is unchanged: coarse-graining the
    alphabet alters only what is observed, never the hidden dynamics. A compressed
    process and its uncompressed twin therefore differ in exactly one variable, leaving
    a large belief behind a small alphabet.

    Args:
        Ts: Transition matrix, shape (V, S, S). V must be divisible by f.
        f: Factor to compress the vocabulary.

    Returns:
        Transition matrix, shape (V // f, S, S).
    """
    assert f > 0
    assert Ts.shape[0] % f == 0
    return Ts.reshape((-1, f) + Ts.shape[1:]).sum(axis=1)


def leak_toward(
    Ts: jax.Array,
    epsilon: float,
    eta: jax.Array | None = None,
    w: jax.Array | None = None,
) -> jax.Array:
    """Leak a process toward a target state.

    With probability epsilon the process forgets its state: it emits a uniformly random
    symbol and resets to the target state eta. This smooths a process's sharp structure
    without changing its alphabet or state space, and in particular gives every
    deterministic transition positive probability.

    The leak operator is a rank-one reset, outer(w, eta) / V, so for each symbol

        Ts'[x] = (1 - epsilon) * Ts[x] + (epsilon / V) * outer(w, eta).

    The target is a parameter because the useful choices differ. Leaking toward the
    uniform state (the default) spreads mass evenly over all (symbol, to-state) pairs.
    Leaking toward the stationary state resets to the process's own equilibrium, which
    for a unital quantum GHMM is the fully mixed state.

    Both the V divisor and the uniform default's 1/S come from Ts.shape, which is what
    keeps row sums intact. Hardcoding their product — as a leak written against one
    process is apt to do — silently unnormalizes every other shape.

    Args:
        Ts: Transition matrix, shape (V, S, S).
        epsilon: Leak probability, in [0, 1]. 0 returns Ts unchanged.
        eta: Target state representative, shape (S,). Must satisfy eta @ w == 1, the
            canonical normalization of SPEC §3.7, which is what preserves the spectrally
            normalized gauge. Defaults to the uniform state, ones(S) / S.
        w: Normalizing right eigenvector of Ts.sum(axis=0), shape (S,). Defaults to
            ones(S), the HMM case.

    Returns:
        Transition matrix, shape (V, S, S).
    """
    assert 0 <= epsilon <= 1
    v, s = Ts.shape[0], Ts.shape[1]
    if w is None:
        w = jnp.ones(s)
    if eta is None:
        eta = jnp.ones(s) / s
    assert jnp.allclose(eta @ w, 1)
    return (1 - epsilon) * Ts + (epsilon / v) * jnp.outer(w, eta)
