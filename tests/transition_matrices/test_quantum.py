import jax.numpy as jnp
import pytest

from generators.utils import principal_ev
from transition_matrices.augmentations import leak_toward
from transition_matrices.classical import checksum, mess
from transition_matrices.quantum import quantum_rrxor

E0 = jnp.array([1.0, 0.0, 0.0, 0.0])

SIGMA = jnp.stack(
    [
        jnp.array([[0, 1], [1, 0]], dtype=complex),
        jnp.array([[0, -1j], [1j, 0]], dtype=complex),
        jnp.array([[1, 0], [0, -1]], dtype=complex),
    ]
)

ANGLES = [(0.7, 1.1), (1.0, 0.4), (2.3, 2.9), (0.3, 1.9), (-0.8, 0.6)]


def kraus(phi: float, theta: float) -> jnp.ndarray:
    """Kraus operators K_x = D_x R_y(phi), built independently of the constructor."""
    c_t, s_t = jnp.cos(theta / 2), jnp.sin(theta / 2)
    c_p, s_p = jnp.cos(phi / 2), jnp.sin(phi / 2)
    R_y = jnp.array([[c_p, -s_p], [s_p, c_p]], dtype=complex)
    D = jnp.stack(
        [
            jnp.diag(jnp.array([c_t + 0j, -1j * s_t])),
            jnp.diag(jnp.array([-1j * s_t, c_t + 0j])),
        ]
    )
    return D @ R_y


def extended_bloch(rho: jnp.ndarray) -> jnp.ndarray:
    """[trace, bx, by, bz] of a 2x2 Hermitian operator."""
    return jnp.array([jnp.trace(rho).real, *[jnp.trace(rho @ s).real for s in SIGMA]])


def density_matrix(b: jnp.ndarray) -> jnp.ndarray:
    return jnp.eye(2, dtype=complex) / 2 + jnp.einsum("i,iab->ab", b + 0j, SIGMA) / 2


def word_probs(Ts: jnp.ndarray, length: int) -> dict[str, float]:
    """Stationary word law, in the GHMM gauge of SPEC §3.3."""
    w = principal_ev(Ts.sum(axis=0))
    pi = principal_ev(Ts.sum(axis=0).T)
    pi = pi / (pi @ w)
    probs = {}
    for k in range(Ts.shape[0] ** length):
        xs = []
        rest = k
        for _ in range(length):
            xs.append(rest % Ts.shape[0])
            rest //= Ts.shape[0]
        eta = pi
        for x in xs:
            eta = eta @ Ts[x]
        probs["".join(map(str, xs))] = float(eta @ w)
    return probs


@pytest.mark.parametrize(("phi", "theta"), ANGLES)
def test_quantum_rrxor_shape_and_realness(phi, theta):
    Ts = quantum_rrxor(phi, theta)
    assert Ts.shape == (2, 4, 4)
    assert jnp.all(jnp.isfinite(Ts))


@pytest.mark.parametrize(("phi", "theta"), ANGLES)
def test_kraus_completeness(phi, theta):
    # sum_x K_x^dagger K_x = I
    K = kraus(phi, theta)
    completeness = jnp.einsum("xba,xbc->ac", jnp.conj(K), K)
    assert jnp.allclose(completeness, jnp.eye(2), atol=1e-6)


@pytest.mark.parametrize(("phi", "theta"), ANGLES)
def test_trace_preservation(phi, theta):
    # Right eigenvector: T_sum @ e_0 = e_0.
    Ts = quantum_rrxor(phi, theta)
    assert jnp.allclose(Ts.sum(axis=0) @ E0, E0, atol=1e-6)


@pytest.mark.parametrize(("phi", "theta"), ANGLES)
def test_unitality(phi, theta):
    # Left eigenvector: e_0 @ T_sum = e_0, so the fully mixed state is stationary.
    Ts = quantum_rrxor(phi, theta)
    assert jnp.allclose(E0 @ Ts.sum(axis=0), E0, atol=1e-6)


@pytest.mark.parametrize(("phi", "theta"), ANGLES)
def test_nontrivial_eigenvalues_are_contractive(phi, theta):
    Ts = quantum_rrxor(phi, theta)
    eigvals = jnp.linalg.eigvals(Ts.sum(axis=0)[1:, 1:])
    assert jnp.all(jnp.abs(eigvals) < 1 + 1e-6)


@pytest.mark.parametrize(("phi", "theta"), ANGLES)
def test_row_vector_convention_matches_the_kraus_update(phi, theta):
    # The load-bearing convention check: eta @ Ts[x] must be the extended Bloch vector of
    # K_x rho K_x^dagger. The transpose would pass both eigenvector checks above, since
    # this GHMM is both trace-preserving and unital.
    Ts = quantum_rrxor(phi, theta)
    K = kraus(phi, theta)
    for b in [
        jnp.array([0.3, -0.5, 0.2]),
        jnp.array([0.0, 0.0, 0.9]),
        jnp.array([-0.6, 0.1, -0.3]),
    ]:
        rho = density_matrix(b)
        eta = jnp.concatenate([jnp.ones(1), b])
        for x in range(2):
            expected = extended_bloch(K[x] @ rho @ jnp.conj(K[x]).T)
            assert jnp.allclose(eta @ Ts[x], expected, atol=1e-6)


@pytest.mark.parametrize(("phi", "theta"), ANGLES)
def test_symbol_probabilities_are_a_distribution(phi, theta):
    # P(x | eta) = eta @ Ts[x] @ w with w = e_0, over physical belief states.
    Ts = quantum_rrxor(phi, theta)
    for b in [jnp.zeros(3), jnp.array([0.3, -0.5, 0.2]), jnp.array([0.0, 0.0, 1.0])]:
        eta = jnp.concatenate([jnp.ones(1), b])
        probs = jnp.array([eta @ Ts[x] @ E0 for x in range(2)])
        assert jnp.allclose(probs.sum(), 1, atol=1e-6)
        assert jnp.all(probs >= -1e-9)


@pytest.mark.parametrize("theta", [0.0, jnp.pi])
@pytest.mark.parametrize("phi", [0.7, 1.0, 2.3, 0.3])
def test_projective_readout_is_a_two_state_classical_chain(phi, theta):
    # theta in {0, pi} collapses the memory onto a basis state every step. The resulting
    # word law is exactly the symmetric order-1 binary Markov chain that repeats a symbol
    # with probability cos(phi / 2) ** 2, which this repo already has as mess(x, 1, 2).
    quantum = word_probs(quantum_rrxor(phi, theta), 6)
    classical = word_probs(mess(jnp.sin(phi / 2) ** 2, 1.0, 2), 6)
    assert quantum.keys() == classical.keys()
    for word in quantum:
        assert abs(quantum[word] - classical[word]) < 1e-6


@pytest.mark.parametrize("phi", [0.7, 1.0, 2.3])
def test_theta_pi_over_two_is_an_iid_fair_coin(phi):
    # The uninformative readout: every length-L word is equally likely.
    probs = word_probs(quantum_rrxor(phi, jnp.pi / 2), 4)
    assert jnp.allclose(jnp.array(list(probs.values())), 1 / 16, atol=1e-6)


def test_projective_readout_is_not_the_checksum_family():
    # The classical limit is a different process from RRXOR, not a reparameterization of
    # it. RRXOR's block structure forbids words once they are long enough to rule out
    # every phase alignment -- 4 of the 32 length-5 words, such as 00100 -- while the
    # order-1 chain the quantum limit reduces to has full support at every length.
    rrxor = word_probs(checksum(jnp.full((2, 2), 0.5)), 5)
    assert sum(p < 1e-12 for p in rrxor.values()) == 4
    assert rrxor["00100"] < 1e-12

    quantum = word_probs(quantum_rrxor(0.7, jnp.pi), 5)
    assert all(p > 1e-9 for p in quantum.values())
    assert max(abs(quantum[word] - rrxor[word]) for word in quantum) > 0.01


def test_generic_readout_has_no_finite_hmm():
    # With phi / pi irrational and a non-projective readout, every history induces a
    # distinct belief, so the belief-state count grows as 2 ** depth. The projective
    # limits collapse to 2 beliefs no matter how deep.
    def distinct_beliefs(Ts, depth):
        w = principal_ev(Ts.sum(axis=0))
        pi = principal_ev(Ts.sum(axis=0).T)
        beliefs = [pi / (pi @ w)]
        for _ in range(depth):
            successors = []
            for eta in beliefs:
                for x in range(2):
                    updated = eta @ Ts[x]
                    if updated @ w > 1e-12:
                        successors.append(updated / (updated @ w))
            beliefs = []
            for eta in successors:
                if not any(bool(jnp.allclose(eta, seen, atol=1e-7)) for seen in beliefs):
                    beliefs.append(eta)
        return len(beliefs)

    assert distinct_beliefs(quantum_rrxor(1.0, 1.1), 6) == 2**6
    assert distinct_beliefs(quantum_rrxor(1.0, jnp.pi), 6) == 2
    assert distinct_beliefs(quantum_rrxor(1.0, 0.0), 6) == 2


@pytest.mark.parametrize(("phi", "theta"), [(0.7, 1.1), (1.0, 0.4), (2.3, 2.9)])
def test_belief_geometry_is_three_dimensional(phi, theta):
    # bloch_walk projects the y coordinate out; here the readout turns the memory's real
    # off-diagonal into an imaginary one, so beliefs fill three Bloch dimensions and all
    # four coordinates of the representation are load-bearing.
    Ts = quantum_rrxor(phi, theta)
    pi = principal_ev(Ts.sum(axis=0).T)
    beliefs = [pi / (pi @ E0)]
    for _ in range(5):
        successors = []
        for eta in beliefs:
            for x in range(2):
                updated = eta @ Ts[x]
                if updated @ E0 > 1e-12:
                    successors.append(updated / (updated @ E0))
        beliefs = beliefs + successors
    stacked = jnp.stack(beliefs)

    assert jnp.abs(stacked[:, 2]).max() > 0.1  # the y coordinate is used
    # Beliefs are normalized, so their differences span at most the 3 Bloch coordinates.
    assert jnp.linalg.matrix_rank(stacked - stacked[0], tol=1e-8) == 3


@pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.5, 1.0])
def test_leak_toward_reproduces_the_source_leakage(epsilon):
    # The source's epsilon branch: (1 - eps) * T[x] + (eps / 2) * outer(e_0, e_0).
    Ts = quantum_rrxor(0.7, 1.1)
    expected = (1 - epsilon) * Ts + (epsilon / 2) * jnp.outer(E0, E0)
    assert jnp.allclose(leak_toward(Ts, epsilon, eta=E0, w=E0), expected)


@pytest.mark.parametrize("epsilon", [0.0, 0.25, 1.0])
def test_leak_toward_preserves_the_quantum_gauge(epsilon):
    # The leaked process is still trace-preserving and unital.
    Ts = leak_toward(quantum_rrxor(0.7, 1.1), epsilon, eta=E0, w=E0)
    assert jnp.allclose(Ts.sum(axis=0) @ E0, E0, atol=1e-6)
    assert jnp.allclose(E0 @ Ts.sum(axis=0), E0, atol=1e-6)
