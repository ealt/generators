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


def quantum_rrxor(phi: float, theta: float) -> jax.Array:
    r"""Creates transition matrices for the Quantum RRXOR process.

    A repeating quantum circuit with a persistent memory qubit and a measured ancilla:

        |psi>_mem --R_y(phi)--*----------- |psi'>_mem   (persistent)
                              |
        |0>_anc   ------------+--R_x(theta)--M-- x_t    (measured)

    The Kraus operators are K_x = D_x R_y(phi) where

        D_0 = diag(cos(theta / 2), -i sin(theta / 2))
        D_1 = diag(-i sin(theta / 2), cos(theta / 2))

    and they satisfy the completeness relation sum_x K_x^dagger K_x = I.

    Internal state basis:
    --------------------
    The GHMM state vector is represented in the 4-dimensional generalized Bloch basis

        (I / 2, sigma_x / 2, sigma_y / 2, sigma_z / 2)

    so the state vector is

        [trace, bloch_x, bloch_y, bloch_z]

    and a normalized state has the form [1, bx, by, bz], the Bloch vector of the memory
    qubit's density matrix. Matrices act on the right of these row vectors: eta @ Ts[x]
    is the extended Bloch vector of the subchannel output K_x rho K_x^dagger. The belief
    geometry is generically a 3-dimensional fractal in the Bloch ball.

    Unlike bloch_walk, no coordinate can be projected out in general: the memory rotation
    mixes y into the dynamics, so all four dimensions are used.

    Symbols:
    -------
    The two matrices are indexed by the ancilla measurement outcome, 0 or 1.

    Parameters:
    ----------
    phi:
        Memory rotation angle, in radians. R_y(phi) rotates the memory qubit before it
        entangles with the ancilla. When phi / pi is irrational and the readout is not
        projective, the process has no finite HMM — every distinct history induces a
        distinct belief. This is the property that makes the process genuinely quantum,
        shared with bloch_walk.

    theta:
        Readout angle, in radians, controlling how informative the measurement is.

            theta = 0 or pi   projective z-readout. The memory collapses onto a
                              computational basis state at every step, so the process is
                              classical with 2 states for any phi: exactly the symmetric
                              order-1 binary Markov chain mess(sin(phi / 2) ** 2, 1, 2),
                              which repeats a symbol with probability cos(phi / 2) ** 2.
            theta = pi / 2    uninformative. P(x) = 1/2 for every belief, so the process
                              is an IID fair coin and the memory carries nothing.

        theta = 0 and theta = pi swap D_0 and D_1 up to phase, so they swap the symbol
        labels; the chain they produce is symmetric under that swap, so both give the
        same word law.

    Naming:
    ------
    Inherited from the source. It is not descriptive: the classical limit above is a
    2-state Markov chain, not RRXOR, and nothing in the construction reproduces the
    checksum family at any parameter. See `classical.checksum` for RRXOR proper.

    Leakage:
    -------
    The source carried an epsilon parameter mixing each transition with a reset to the
    stationary (fully mixed) state. That is `augmentations.leak_toward`, whose target here
    is the fully mixed state e_0 = [1, 0, 0, 0], which is also this process's normalizing
    eigenvector:

        leak_toward(quantum_rrxor(phi, theta), epsilon, eta=e_0, w=e_0)

    Returns:
    -------
    jax.Array of shape (2, 4, 4)

    Entries may be negative — these are signed linear operators on the Bloch
    representation, not stochastic transition matrices over 4 latent states.

    References:
    ----------
    Riechers & Crutchfield (2021), Phys. Rev. Research 3, 013170.
    Riechers, Elliott & Shai (2025), arXiv:2507.07432, Appendix B.
    """
    c_theta, s_theta = jnp.cos(theta / 2), jnp.sin(theta / 2)
    c_phi, s_phi = jnp.cos(phi / 2), jnp.sin(phi / 2)

    R_y = jnp.array(
        [
            [c_phi, -s_phi],
            [s_phi, c_phi],
        ],
        dtype=complex,
    )
    D = jnp.stack(
        [
            jnp.diag(jnp.array([c_theta + 0j, -1j * s_theta])),
            jnp.diag(jnp.array([-1j * s_theta, c_theta + 0j])),
        ]
    )
    K = D @ R_y

    # Generalized Bloch basis, ordered (I / 2, sigma_x / 2, sigma_y / 2, sigma_z / 2).
    B = (
        jnp.stack(
            [
                jnp.eye(2, dtype=complex),
                jnp.array([[0, 1], [1, 0]], dtype=complex),
                jnp.array([[0, -1j], [1j, 0]], dtype=complex),
                jnp.array([[1, 0], [0, -1]], dtype=complex),
            ]
        )
        / 2
    )

    # Ts[x, i, j] = 2 tr(B_j K_x B_i K_x^dagger), the Bloch representation of the
    # subchannel rho -> K_x rho K_x^dagger. The factor of 2 is 1 / xi for a qubit.
    return 2 * jnp.real(jnp.einsum("xab,ibc,xdc,jda->xij", K, B, jnp.conj(K), B))
