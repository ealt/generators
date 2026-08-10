import itertools

import jax.numpy as jnp
import pytest

from generators.ghmm import process
from transition_matrices.classical import checksum
from transition_matrices.quantum import (
    _bloch_subchannel,
    _pauli_basis,
    quantum_checksum,
    quantum_checksum_state,
)

RRXOR = jnp.array([[0.5, 0.5], [0.5, 0.5]])
BIASED = jnp.array([[0.3, 0.7], [0.6, 0.4]])
SINGLE = jnp.array([[0.4, 0.6]])
DEEP = jnp.array([[0.3, 0.7], [0.6, 0.4], [0.8, 0.2]])
TERNARY = jnp.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])


def ry(angle: float) -> jnp.ndarray:
    c, s = jnp.cos(angle / 2), jnp.sin(angle / 2)
    return jnp.array([[c, -s], [s, c]], dtype=complex)


def cnot(num_qubits: int, control: int, target: int) -> jnp.ndarray:
    dim = 2**num_qubits
    out = jnp.zeros((dim, dim), dtype=complex)
    for basis in range(dim):
        bits = [(basis >> (num_qubits - 1 - q)) & 1 for q in range(num_qubits)]
        if bits[control]:
            bits[target] ^= 1
        out = out.at[sum(b << (num_qubits - 1 - q) for q, b in enumerate(bits)), basis].set(1.0)
    return out


def rotor(*phis: float) -> jnp.ndarray:
    """Rotate every qubit, then couple with the measured qubit (0) as CNOT target."""
    out = jnp.array([[1.0 + 0j]])
    for phi in phis:
        out = jnp.kron(out, ry(phi))
    for q in range(len(phis) - 1, 0, -1):
        out = cnot(len(phis), q, 0) @ out
    return out


def normalizing_eigenvector(probs: jnp.ndarray, num_qubits: int) -> jnp.ndarray:
    """w[state * 4**N] = 2**N, zero elsewhere: the trace functional on each rotor block."""
    n, m = probs.shape
    block = 4**num_qubits
    size = (n * m + 1) * block
    return jnp.zeros(size).at[jnp.arange(n * m + 1) * block].set(2**num_qubits)


def word_probs(Ts: jnp.ndarray, eta: jnp.ndarray, w: jnp.ndarray, m: int, length: int) -> dict[str, float]:
    probs = {}
    for word in itertools.product(range(m), repeat=length):
        state = eta
        for x in word:
            state = state @ Ts[x]
        probs["".join(map(str, word))] = float(state @ w)
    return probs


def classical_word_probs(probs: jnp.ndarray, m: int, length: int) -> dict[str, float]:
    Ts = checksum(probs)
    n = probs.shape[0]
    # Cesaro average over one period -- the chain is periodic, so power iteration cycles.
    iterates = [jnp.zeros(Ts.shape[1]).at[0].set(1.0)]
    for _ in range(n):
        iterates.append(iterates[-1] @ Ts.sum(axis=0))
    pi = jnp.stack(iterates).mean(axis=0)
    out = {}
    for word in itertools.product(range(m), repeat=length):
        state = pi
        for x in word:
            state = state @ Ts[x]
        out["".join(map(str, word))] = float(state.sum())
    return out


def test_pauli_basis_is_orthogonal_and_identity_first():
    for num_qubits in (1, 2, 3):
        basis = _pauli_basis(num_qubits)
        dim = 2**num_qubits
        assert basis.shape == (4**num_qubits, dim, dim)
        assert jnp.allclose(basis[0], jnp.eye(dim))
        grams = jnp.einsum("apq,bqp->ab", basis, basis)
        assert jnp.allclose(grams, dim * jnp.eye(4**num_qubits), atol=1e-6)


def test_bloch_subchannel_matches_the_kraus_update():
    # v @ G must be the coefficient vector of K rho K^dagger, the same row convention
    # quantum_rrxor uses.
    basis = _pauli_basis(1)
    K = jnp.array([[0.6, -0.3], [0.3, 0.6]], dtype=complex)
    G = _bloch_subchannel(K, basis)
    rho = jnp.array([[0.7, 0.1 + 0.2j], [0.1 - 0.2j, 0.3]], dtype=complex)
    v = jnp.array([jnp.trace(p @ rho).real / 2 for p in basis])
    evolved = K @ rho @ jnp.conj(K).T
    expected = jnp.array([jnp.trace(p @ evolved).real / 2 for p in basis])
    assert jnp.allclose(v @ G, expected, atol=1e-6)


@pytest.mark.parametrize(
    "probs", [RRXOR, BIASED, SINGLE, DEEP, TERNARY], ids=["rrxor", "biased", "single", "deep", "ternary"]
)
@pytest.mark.parametrize("phis", [(1.0,), (1.0, 0.7)], ids=["N1", "N2"])
def test_tilt_zero_is_exactly_the_classical_checksum(probs, phis):
    # The headline property: at tilt = 0 the rotor decouples and the word law is exactly
    # checksum(probs), for any rotor and any number of rotor qubits.
    n, m = probs.shape
    Ts = quantum_checksum(probs, rotor(*phis), tilt=0.0)
    eta = quantum_checksum_state(probs, len(phis))
    w = normalizing_eigenvector(probs, len(phis))
    assert jnp.allclose(eta @ w, 1, atol=1e-6)

    for length in (1, 2, 3, 4):
        quantum = word_probs(Ts, eta, w, m, length)
        classical = classical_word_probs(probs, m, length)
        assert quantum.keys() == classical.keys()
        for word in quantum:
            assert abs(quantum[word] - classical[word]) < 1e-6, word


@pytest.mark.parametrize("phis", [(1.0,), (1.0, 0.7)], ids=["N1", "N2"])
def test_shape_and_gauge(phis):
    n, m = BIASED.shape
    num_qubits = len(phis)
    Ts = quantum_checksum(BIASED, rotor(*phis), tilt=0.4)
    block = 4**num_qubits
    assert Ts.shape == (m, (n * m + 1) * block, (n * m + 1) * block)

    # Trace preservation: the net matrix fixes w, so symbol probabilities sum to 1.
    w = normalizing_eigenvector(BIASED, num_qubits)
    assert jnp.allclose(Ts.sum(axis=0) @ w, w, atol=1e-6)


@pytest.mark.parametrize("tilt", [0.0, 0.3, 0.7, 1.0])
def test_symbol_probabilities_are_a_distribution(tilt):
    Ts = quantum_checksum(BIASED, rotor(1.0, 0.7), tilt=tilt)
    eta = quantum_checksum_state(BIASED, 2)
    w = normalizing_eigenvector(BIASED, 2)
    for length in (1, 2, 3):
        total = sum(word_probs(Ts, eta, w, 2, length).values())
        assert abs(total - 1) < 1e-5


def belief_rank(Ts: jnp.ndarray, eta: jnp.ndarray, w: jnp.ndarray, m: int, depth: int) -> int:
    beliefs = [eta]
    for _ in range(depth):
        successors = []
        for state in beliefs:
            for x in range(m):
                updated = state @ Ts[x]
                if updated @ w > 1e-9:
                    successors.append(updated / (updated @ w))
        beliefs = beliefs + successors
    stacked = jnp.stack(beliefs)
    return int(jnp.linalg.matrix_rank(stacked - stacked[0], tol=1e-6))


def test_tilt_controls_the_belief_dimension():
    # At tilt = 0 the rotor decouples and the geometry is checksum's 5-state simplex,
    # dimension 4, whatever the rotor size. Turning the readout informative grows it.
    for phis in [(1.0,), (1.0, 0.7)]:
        w = normalizing_eigenvector(RRXOR, len(phis))
        eta = quantum_checksum_state(RRXOR, len(phis))
        flat = belief_rank(quantum_checksum(RRXOR, rotor(*phis), 0.0), eta, w, 2, 5)
        assert flat == 4, phis

    tilted_one = belief_rank(
        quantum_checksum(RRXOR, rotor(1.0), 0.6),
        quantum_checksum_state(RRXOR, 1),
        normalizing_eigenvector(RRXOR, 1),
        2,
        5,
    )
    tilted_two = belief_rank(
        quantum_checksum(RRXOR, rotor(1.0, 0.7), 0.6),
        quantum_checksum_state(RRXOR, 2),
        normalizing_eigenvector(RRXOR, 2),
        2,
        5,
    )
    assert tilted_one > 4
    assert tilted_two > tilted_one


def test_initial_state_is_derivable_only_away_from_the_classical_face():
    # At tilt = 0 the rotor decouples, so eigenvalue 1 is doubly degenerate and process
    # .init refuses to guess -- which is why quantum_checksum_state exists. Once the
    # readout couples the factors, init derives the same state itself.
    flat = quantum_checksum(RRXOR, rotor(1.0), tilt=0.0)
    assert int(jnp.sum(jnp.abs(jnp.linalg.eigvals(flat.sum(axis=0)) - 1) < 1e-5)) == 2
    with pytest.raises(Exception, match="multiplicity"):
        process.init(flat)

    tilted = quantum_checksum(RRXOR, rotor(1.0), tilt=0.6)
    assert int(jnp.sum(jnp.abs(jnp.linalg.eigvals(tilted.sum(axis=0)) - 1) < 1e-5)) == 1
    data = process.init(tilted)
    # process.init normalizes w to sum to S; ours carries the raw trace functional.
    ours = normalizing_eigenvector(RRXOR, 1)
    assert jnp.allclose(data.w / data.w.sum(), ours / ours.sum(), atol=1e-6)


def test_rejects_invalid_arguments():
    good = rotor(1.0)
    with pytest.raises(ValueError, match="2-D"):
        quantum_checksum(jnp.array([0.5, 0.5]), good)
    with pytest.raises(ValueError, match="at least one row"):
        quantum_checksum(jnp.zeros((0, 2)), good)
    with pytest.raises(ValueError, match="sum to 1"):
        quantum_checksum(jnp.array([[0.5, 0.4]]), good)
    with pytest.raises(ValueError, match="tilt"):
        quantum_checksum(RRXOR, good, tilt=1.5)
    with pytest.raises(ValueError, match="square"):
        quantum_checksum(RRXOR, jnp.ones((2, 3), dtype=complex))
    with pytest.raises(ValueError, match="power of 2"):
        quantum_checksum(RRXOR, jnp.eye(3, dtype=complex))
    with pytest.raises(ValueError, match="unitary"):
        quantum_checksum(RRXOR, 2 * jnp.eye(2, dtype=complex))
