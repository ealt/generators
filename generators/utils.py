import jax
import jax.numpy as jnp


def principal_ev(T: jax.Array) -> jax.Array:
    """Eigenvector of the Perron root of T, normalized to mean 1.

    The Perron root is selected by largest real part, not largest modulus: a periodic
    transition matrix has one eigenvalue of unit modulus per period and only the real
    one is Perron.

    Args:
        T: Square matrix.

    Returns:
        The Perron eigenvector, scaled so its entries average to 1.

    Raises:
        ValueError: If the Perron root is degenerate, in which case T alone does not
            determine an eigenvector.
    """
    eigenvalues, eigenvectors = jnp.linalg.eig(T)
    i = jnp.argmax(jnp.real(eigenvalues))
    multiplicity = int(jnp.sum(jnp.isclose(eigenvalues, eigenvalues[i])))
    if multiplicity > 1:
        raise ValueError(
            f"leading eigenvalue {eigenvalues[i]} has multiplicity {multiplicity}, so no eigenvector "
            "of it is determined by T alone; a transition matrix with a degenerate leading eigenvalue "
            "is reducible, with one such eigenvalue per recurrent component"
        )
    vector = jnp.real(eigenvectors[:, i])
    sign = jnp.where(jnp.sum(vector) < 0, -1.0, 1.0)
    vector = vector * sign
    return vector / jnp.mean(vector)


def pad(arr: jax.Array, target_shape: tuple[int, ...]) -> jax.Array:
    pad_width = [(0, target - current) for target, current in zip(target_shape, arr.shape, strict=True)]
    return jnp.pad(arr, pad_width, mode="constant", constant_values=0)


def stack(arrs: list[jax.Array]) -> jax.Array:
    shapes = [arr.shape for arr in arrs]
    max_shape = tuple(max(shape_i) for shape_i in zip(*shapes, strict=True))
    return jnp.stack([pad(arr, max_shape) for arr in arrs])


def mixed_radix_weights(Vs: jax.Array) -> jax.Array:
    return jnp.roll(jnp.cumprod(Vs), 1).at[0].set(1)


def mixed_radix_encode(x_factors: jax.Array, *, weights: jax.Array) -> jax.Array:
    return jnp.sum(x_factors * weights)


def mixed_radix_decode(x: jax.Array, *, Vs: jax.Array, weights: jax.Array) -> jax.Array:
    return (x // weights) % Vs
