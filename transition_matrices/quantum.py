import jax
import jax.numpy as jnp

from transition_matrices.classical import checksum


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


def _pauli_basis(num_qubits: int) -> jax.Array:
    """The 4 ** num_qubits Pauli products, identity first.

    A Hermitian, mutually orthogonal operator basis with tr(P_a P_b) = d * delta_ab, so a
    density matrix expands as rho = sum_a v_a P_a with real coefficients v_a = tr(P_a rho)
    / d. Index 0 is the identity, making v_0 the trace coordinate.
    """
    single = jnp.array(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
            [[1, 0], [0, -1]],
        ],
        dtype=complex,
    )
    basis = [jnp.ones((1, 1), dtype=complex)]
    for _ in range(num_qubits):
        basis = [jnp.kron(b, p) for b in basis for p in single]
    return jnp.stack(basis)


def _bloch_subchannel(K: jax.Array, basis: jax.Array) -> jax.Array:
    """Bloch representation of rho -> K rho K^dagger, acting on the right of row vectors.

    G[a, b] = tr(P_b K P_a K^dagger) / d, so that v @ G is the coefficient vector of the
    subchannel output. Same convention as quantum_rrxor.
    """
    d = basis.shape[1]
    return jnp.real(jnp.einsum("bpq,qr,ars,ps->ab", basis, K, basis, jnp.conj(K))) / d


def quantum_checksum(probs: jax.Array, rotor: jax.Array, tilt: float = 0.0) -> jax.Array:
    r"""Creates transition matrices for the Quantum Checksum process.

    A quantum generalization of `classical.checksum` that contains it exactly, rather than
    merely resembling it. The memory is a product of two registers:

        (phase, running sum mod m) accumulator  (x)  N-qubit rotor

    The accumulator is classical — the emitted symbol is measured before being added, so
    it is always in a definite residue and costs n * m + 1 dimensions rather than their
    square. The rotor is quantum, carries no classical record, and supplies the emission
    law. The two factors is why the classical and quantum structure coexist instead of
    competing.

    Each step applies the rotor unitary, then reads out a symbol whose law may depend on
    the rotor's computational basis state k:

        P(x | k) = (1 - tilt) * probs[phase, x] + tilt * [x == k mod m]
        K_x      = diag(sqrt(P(x | .))) @ rotor

    which satisfies sum_x K_x^dagger K_x = I for any tilt. At the checksum phase the
    accumulator emits its residue and resets, and the rotor is left untouched.

    The tilt:
    --------
        tilt = 0    The readout is uninformative: K_x = sqrt(probs[phase, x]) * rotor is a
                    scalar times a unitary, so P(x) is independent of the rotor state, the
                    rotor factor decouples, and the process is EXACTLY
                    checksum(probs) -- same word law, same 5-state belief simplex at
                    (n, m) = (2, 2), for any rotor and any number of rotor qubits.
        tilt = 1    Fully projective readout in the rotor's computational basis.
        otherwise   The readout is informative, so the rotor belief carries history and
                    the belief geometry becomes a fractal whose dimension grows with N.

    So probs is the same parameter on both sides: the classical family is the tilt = 0
    face of this one's parameter space.

    Internal state basis:
    --------------------
    The GHMM state vector is indexed as state * 4 ** N + a, where state indexes the
    accumulator in checksum's order (the seed state, then (phase, residue) pairs) and a
    indexes the Pauli products of `_pauli_basis`. A normalized state has

        eta[state * 4 ** N] = P(accumulator = state) / 2 ** N

    on the identity components, since v_0 = tr(rho) / d. The normalizing eigenvector is
    w[state * 4 ** N] = 2 ** N and 0 elsewhere.

    Args:
        probs: Row-stochastic emission probabilities, shape (n, m), exactly as for
            `classical.checksum`. Recovered exactly at tilt = 0.
        rotor: Unitary applied to the rotor each step, shape (2 ** N, 2 ** N). Supplied by
            the caller rather than built from an ansatz here; when it entangles the
            measured qubit as a CNOT target the projective readout stays non-classical,
            and when the measured qubit is the control it collapses.
        tilt: Readout informativeness, in [0, 1]. 0 is the classical limit.

    Returns:
        Transition matrix, shape (m, (n * m + 1) * 4 ** N, (n * m + 1) * 4 ** N).

    Entries may be negative, and w is not strictly positive, so like bloch_walk and
    quantum_rrxor this is outside SPEC 3.2 items 1 and 3 as written. See CONFORMANCE.md.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D, got shape {probs.shape}")
    n, m = probs.shape
    if n < 1:
        raise ValueError(f"probs must have at least one row, got shape {probs.shape}")
    if not jnp.allclose(probs.sum(axis=1), 1):
        raise ValueError(f"probs rows must sum to 1, got {probs.sum(axis=1)}")
    if not 0 <= tilt <= 1:
        raise ValueError(f"tilt must be in [0, 1], got {tilt}")
    if rotor.ndim != 2 or rotor.shape[0] != rotor.shape[1]:
        raise ValueError(f"rotor must be square, got shape {rotor.shape}")
    dim = rotor.shape[0]
    num_qubits = dim.bit_length() - 1
    if 2**num_qubits != dim:
        raise ValueError(f"rotor dimension must be a power of 2, got {dim}")
    if not jnp.allclose(jnp.conj(rotor).T @ rotor, jnp.eye(dim), atol=1e-6):
        raise ValueError("rotor must be unitary")

    basis = _pauli_basis(num_qubits)
    block = basis.shape[0]

    # P(x | rotor basis state k) at each phase, shape (n, dim, m).
    projective = (jnp.arange(dim)[:, None] % m) == jnp.arange(m)[None, :]
    conditional = (1 - tilt) * probs[:, None, :] + tilt * projective[None, :, :]

    # One Bloch subchannel per (phase, symbol).
    kraus = jnp.sqrt(conditional)[..., None] * rotor[None, :, None, :]
    subchannels = jnp.stack(
        [jnp.stack([_bloch_subchannel(kraus[i, :, x, :], basis) for x in range(m)]) for i in range(n)]
    )

    # The classical skeleton says which accumulator states connect; the subchannels say
    # what happens to the rotor along the way.
    size = (n * m + 1) * block
    Ts = jnp.zeros((m, size, size))
    for phase in range(n):
        residues = [0] if phase == 0 else list(range(m))
        for residue in residues:
            source = 0 if phase == 0 else 1 + (phase - 1) * m + residue
            for x in range(m):
                target = 1 + phase * m + (residue + x) % m
                Ts = jax.lax.dynamic_update_slice(Ts, subchannels[phase, x][None], (x, source * block, target * block))

    # The checksum phase emits its residue and resets, leaving the rotor alone.
    for residue in range(m):
        source = 1 + (n - 1) * m + residue
        Ts = jax.lax.dynamic_update_slice(Ts, jnp.eye(block)[None], (residue, source * block, 0))
    return Ts


def quantum_checksum_state(probs: jax.Array, num_qubits: int) -> jax.Array:
    """Stationary state of `quantum_checksum`: checksum's stationary, rotor fully mixed.

    Needed at tilt = 0. There the rotor decouples, so the net matrix has eigenvalue 1 with
    multiplicity 2 — one from the accumulator, one from the rotor's identity direction —
    and `generators.ghmm.process.init` rightly refuses to guess an initial state for it.
    Away from tilt = 0 the readout couples the two factors, eigenvalue 1 is simple, and
    `process.init` derives the same state on its own.

    Args:
        probs: The same rows passed to `quantum_checksum`.
        num_qubits: Number of rotor qubits.

    Returns:
        GHMM state representative, shape ((n * m + 1) * 4 ** num_qubits,), canonically
        normalized against w so that eta @ w = 1.
    """
    T = checksum(probs).sum(axis=0)
    n = probs.shape[0]
    # Cesaro average over one period: power iteration alone cycles. See checksum's note.
    iterates = [jnp.zeros(T.shape[0]).at[0].set(1.0)]
    for _ in range(n):
        iterates.append(iterates[-1] @ T)
    pi = jnp.stack(iterates).mean(axis=0)

    block = 4**num_qubits
    dim = 2**num_qubits
    return jnp.zeros(pi.size * block).at[jnp.arange(pi.size) * block].set(pi / dim)
