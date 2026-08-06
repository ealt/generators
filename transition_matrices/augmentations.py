import jax
import jax.numpy as jnp


def apply_symbol_map(Ts: jax.Array, C: jax.Array) -> jax.Array:
    assert C.shape[1] == Ts.shape[0]
    assert jnp.allclose(C.sum(axis=0), 1.0)
    return jnp.tensordot(C, Ts, axes=([1], [0]))


def expand_map(V: int, f: int) -> jax.Array:
    assert f > 0
    return jnp.repeat(jnp.eye(V) / f, f, axis=0)


def compress_map(V: int, f: int) -> jax.Array:
    assert f > 0
    assert V % f == 0
    return jnp.repeat(jnp.eye(V // f), f, axis=1)


def noise_map(V: int, eps: float) -> jax.Array:
    return (1.0 - eps) * jnp.eye(V) + (eps / V) * jnp.ones((V, V))


def confusion_map(V: int, pairs: list[tuple[int, int]], eps: float) -> jax.Array:
    """Asymmetric confusion: each (y, x) in pairs sends true y to observed x with probability eps.

    Columns without a listed confusion stay as the identity. Shape (V, V).
    """
    C = jnp.eye(V)
    for y, x in pairs:
        C = C.at[y, y].add(-eps)
        C = C.at[x, y].add(eps)
    return C
