import jax
import jax.numpy as jnp


def bloch_walk(
    measurement_angle: float,
    p_z_axis: float = 0.5,
) -> jax.Array:
    """Creates transition matrices for the Tom Quantum / Bloch Walk process.

    tom_quantum = lambda alpha, beta: bloch_walk(jnp.arctan(alpha / beta), p_z_axis=0.5)

    Internal state basis:
    --------------------
    The GHMM state vector is represented in the basis:

        [trace, bloch_x, bloch_z]

    A normalized state has the form:

        [1, bx, bz]

    corresponding to the x-z slice of a qubit Bloch vector.

    Symbols:
    -------
    The returned matrices are ordered as:

        0: z+
        1: z-
        2: x+
        3: x-

    Parameters:
    ----------
    measurement_angle:
        Controls the strength of the measurement.

        measurement_strength    = sin(measurement_angle)
        transverse_contraction  = cos(measurement_angle)

        Useful canonical range:

            0 <= measurement_angle <= pi / 2

        where:

            0       means no informative measurement;
            pi / 2  means projective measurement.

    p_z_axis:
        Probability of choosing the z-axis measurement.

        P(measure z-axis) = p_z_axis
        P(measure x-axis) = 1 - p_z_axis

    Returns:
    -------
    jax.Array of shape (4, 3, 3)

    Each symbol matrix is a linear, trace-weighted update on
    [trace, bloch_x, bloch_z].
    """
    if not 0 <= p_z_axis <= 1:
        raise ValueError(f"p_z_axis must be in [0, 1], got {p_z_axis}")

    m = jnp.sin(measurement_angle)  # measurement strength
    k = jnp.cos(measurement_angle)  # transverse contraction

    z_scale = p_z_axis / 2
    x_scale = (1 - p_z_axis) / 2

    z_plus = z_scale * jnp.array(
        [
            [1, 0, +m],
            [0, k, 0],
            [+m, 0, 1],
        ]
    )

    z_minus = z_scale * jnp.array(
        [
            [1, 0, -m],
            [0, k, 0],
            [-m, 0, 1],
        ]
    )

    x_plus = x_scale * jnp.array(
        [
            [1, +m, 0],
            [+m, 1, 0],
            [0, 0, k],
        ]
    )

    x_minus = x_scale * jnp.array(
        [
            [1, -m, 0],
            [-m, 1, 0],
            [0, 0, k],
        ]
    )

    return jnp.stack([z_plus, z_minus, x_plus, x_minus])
