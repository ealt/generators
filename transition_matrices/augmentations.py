"""Vocabulary augmentations: decouple emission-alphabet size from state-space size.

Ported from simplex-research `src/transition_matrices/augmentations.py` (commit 00312f5b).
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
