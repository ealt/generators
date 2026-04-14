import jax
import jax.numpy as jnp


def principal_ev(T: jax.Array) -> jax.Array:
    """Compute the principal eigenvector of a transition matrix."""
    eigenvalues, eigenvectors = jnp.linalg.eig(T)
    i = jnp.argmax(jnp.abs(eigenvalues))
    vector = jnp.real(eigenvectors[:, i])
    sign = jnp.where(jnp.sum(vector) < 0, -1.0, 1.0)
    vector = vector * sign
    return vector / jnp.mean(vector)


def pad(arr: jax.Array, target_shape: tuple[int, ...]) -> jax.Array:
    """Pad an array to a given shape."""
    pad_width = [(0, target - current) for target, current in zip(target_shape, arr.shape, strict=True)]
    return jnp.pad(arr, pad_width, mode="constant", constant_values=0)


def stack(arrs: list[jax.Array]) -> jax.Array:
    """Stack a list of arrays."""
    max_shape = max(arr.shape for arr in arrs)
    return jnp.stack([pad(arr, max_shape) for arr in arrs])


def mixed_radix_weights(Vs: jax.Array) -> jax.Array:
    """Return little-endian mixed-radix place values for each factor."""
    return jnp.roll(jnp.cumprod(Vs), 1).at[0].set(1)


def mixed_radix_encode(xs: jax.Array, *, weights: jax.Array) -> jax.Array:
    """Encode local tokens into a global token using little-endian mixed radix."""
    return jnp.sum(xs * weights)


def mixed_radix_decode(x: jax.Array, *, Vs: jax.Array, weights: jax.Array) -> jax.Array:
    """Decode a global token into local tokens using little-endian mixed radix."""
    return (x // weights) % Vs
