import jax.numpy as jnp


def principal_ev(T: jnp.ndarray) -> jnp.ndarray:
    """Compute the principal eigenvector of a transition matrix."""
    eigenvalues, eigenvectors = jnp.linalg.eig(T)
    i = jnp.argmax(jnp.abs(eigenvalues))
    vector = jnp.real(eigenvectors[:, i])
    sign = jnp.where(jnp.sum(vector) < 0, -1.0, 1.0)
    vector = vector * sign
    return vector / jnp.mean(vector)
