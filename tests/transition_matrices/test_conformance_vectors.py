"""Independent oracles for the checksum family's conformance vectors.

The vectors in conformance-tests.json exist to police consumers' forks, so they must not
merely echo this repo's reference modules. Everything here derives from the process
definition -- emit n random symbols, then their sum mod m -- through the generative rules
below. checksum() is called in exactly one test, to confirm it agrees; if its index
arithmetic were wrong, these oracles would disagree with it.

Two independent derivations are used wherever both apply:

    the analytic closed forms   stationary distribution, and word probabilities as a sum
                                over the n+1 phase alignments
    the rule-built tensor       the (V, S, S) array assembled from the emission and
                                successor rules, then propagated

Deliberately written in plain Python floats, with no array library: sharing no numerical
machinery with the code under test is the point, and double precision is what SPEC 7.2's
abs_tol of 1e-12 needs.

tests/test_conformance.py checks the same vectors from the other side, against
generators/ghmm/process.py. It is the portable runner and knows none of this.
"""

import itertools
import json
import pathlib

import jax.numpy as jnp
import pytest

from transition_matrices.augmentations import leak_toward
from transition_matrices.classical import checksum

VECTORS_PATH = pathlib.Path(__file__).parents[2] / "conformance-tests.json"

# The processes the vectors are built over, identified by matching serialized matrices.
PROCESSES = [
    ("rrxor", [[0.5, 0.5], [0.5, 0.5]], None),
    ("biased", [[0.3, 0.7], [0.6, 0.4]], None),
    ("single", [[0.3, 0.7]], None),
    ("ternary", [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]], None),
    ("deep", [[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]], None),
    ("rrxor_leaked", [[0.5, 0.5], [0.5, 0.5]], 0.1),
]

Matrix = list[list[float]]


# --- The process, as rules over (phase, running sum mod m) ------------------------------


def states(n: int, m: int) -> list[tuple[int, int]]:
    """States in the order checksum documents: the seed state, then (phase, residue)."""
    return [(0, 0), *[(i, r) for i in range(1, n + 1) for r in range(m)]]


def emission_dist(probs: Matrix, state: tuple[int, int]) -> list[float]:
    """P(x | state). Random phases emit from their row; the checksum phase emits residue."""
    n, m = len(probs), len(probs[0])
    phase, residue = state
    if phase < n:
        return list(probs[phase])
    return [1.0 if x == residue else 0.0 for x in range(m)]


def successor(probs: Matrix, state: tuple[int, int], x: int) -> tuple[int, int] | None:
    """The state after emitting x, or None if x cannot be emitted from this state."""
    n, m = len(probs), len(probs[0])
    phase, residue = state
    if phase < n:
        return (phase + 1, (residue + x) % m)
    return (0, 0) if x == residue else None


def rules_tensor(probs: Matrix, leak: float | None = None) -> list[Matrix]:
    """The (V, S, S) tensor assembled from the rules above, independent of checksum()."""
    n, m = len(probs), len(probs[0])
    keys = states(n, m)
    index = {s: i for i, s in enumerate(keys)}
    size = len(keys)
    Ts = [[[0.0] * size for _ in range(size)] for _ in range(m)]
    for j, state in enumerate(keys):
        for x in range(m):
            nxt = successor(probs, state, x)
            if nxt is not None:
                Ts[x][j][index[nxt]] += emission_dist(probs, state)[x]
    if leak:
        share = leak / (m * size)
        Ts = [[[(1 - leak) * value + share for value in row] for row in matrix] for matrix in Ts]
    return Ts


def propagate(weights: list[float], matrix: Matrix) -> list[float]:
    """Row vector times matrix."""
    return [sum(weights[j] * matrix[j][k] for j in range(len(weights))) for k in range(len(matrix[0]))]


# --- Analytic closed forms -------------------------------------------------------------


def running_sum_dist(probs: Matrix, i: int) -> list[float]:
    """Distribution of the sum of the first i random symbols, mod m."""
    m = len(probs[0])
    dist = [1.0 if r == 0 else 0.0 for r in range(m)]
    for row in probs[:i]:
        dist = [sum(dist[(r - x) % m] * row[x] for x in range(m)) for r in range(m)]
    return dist


def stationary(probs: Matrix) -> list[float]:
    """1 / (n + 1) on the seed state; P(sum of the first i symbols = r) / (n + 1) on (i, r)."""
    n = len(probs)
    weights = [1.0]
    for i in range(1, n + 1):
        weights.extend(running_sum_dist(probs, i))
    return [weight / (n + 1) for weight in weights]


def stationary_leaked(probs: Matrix, leak: float) -> list[float]:
    """A leak has no closed form, so iterate the fixed point it defines.

    T_leaked = (1 - leak) * T_block + leak / S, so a stationary pi satisfies

        pi = (1 - leak) * pi T_block + leak / S,

    a contraction with modulus 1 - leak on the simplex. Converges geometrically.
    """
    block = rules_tensor(probs)
    size = len(block[0])
    net = [[sum(matrix[j][k] for matrix in block) for k in range(size)] for j in range(size)]
    pi = [1.0 / size] * size
    for _ in range(2000):
        moved = propagate(pi, net)
        nxt = [(1 - leak) * value + leak / size for value in moved]
        if max(abs(a - b) for a, b in zip(nxt, pi, strict=True)) < 1e-16:
            return nxt
        pi = nxt
    raise AssertionError("stationary_leaked did not converge")


def word_probability_by_alignment(probs: Matrix, tokens: list[int], weights: list[float]) -> float:
    """P(tokens) as a sum over phase alignments, weighted by each initial state's prior."""
    keys = states(len(probs), len(probs[0]))
    total = 0.0
    for weight, state in zip(weights, keys, strict=True):
        path, current = weight, state
        for x in tokens:
            if current is None or path == 0:
                path = 0.0
                break
            path *= emission_dist(probs, current)[x]
            current = successor(probs, current, x)
        total += path
    return total


# --- Operations, propagating a distribution over states --------------------------------


def obs_dist(probs: Matrix, weights: list[float], leak: float | None = None) -> list[float]:
    """P(x) under a distribution over states."""
    return [sum(propagate(weights, matrix)) for matrix in rules_tensor(probs, leak)]


def state_update(probs: Matrix, weights: list[float], x: int, leak: float | None = None) -> list[float]:
    """Posterior over states after observing x, normalized to sum to 1."""
    posterior = propagate(weights, rules_tensor(probs, leak)[x])
    total = sum(posterior)
    assert total > 0, f"observation {x} has probability 0 under this state"
    return [value / total for value in posterior]


def sequence_probability(probs: Matrix, tokens: list[int], weights: list[float], leak: float | None = None) -> float:
    """P(tokens) by propagating the prior through one operator per token."""
    Ts = rules_tensor(probs, leak)
    eta = list(weights)
    for x in tokens:
        eta = propagate(eta, Ts[x])
    return sum(eta)


# --- Vector plumbing -------------------------------------------------------------------

with VECTORS_PATH.open() as _f:
    DOCUMENT = json.load(_f)

VECTORS = DOCUMENT["test_vectors"]
IDS = [v["id"] for v in VECTORS]

REQUIRED_FIELDS = {"id", "category", "description", "operation", "input", "expected"}
OPTIONAL_FIELDS = {"process", "tolerance", "abs_tolerance", "notes"}
OPERATIONS = {"observation_distribution", "state_update", "sequence_probability", "stationary_distribution"}


def identify(vector: dict) -> tuple[str, Matrix, float | None]:
    """Recover which process a vector is over by matching its serialized matrices."""
    Ts = vector["process"]["transition_matrices"]
    for name, probs, leak in PROCESSES:
        candidate = rules_tensor(probs, leak)
        if len(candidate) != len(Ts) or len(candidate[0]) != len(Ts[0]):
            continue
        flat = zip(_flatten(candidate), _flatten(Ts), strict=True)
        if all(abs(a - b) < 1e-12 for a, b in flat):
            return name, probs, leak
    raise AssertionError(f"{vector['id']}: serialized matrices match none of the known processes")


def _flatten(tensor) -> list[float]:
    return [value for matrix in tensor for row in matrix for value in row]


def prior_weights(vector: dict, probs: Matrix, leak: float | None) -> list[float]:
    """The state distribution a vector's operation starts from.

    An HMM's canonical state representative is exactly a distribution over states, so
    normalizing the serialized state recovers the weights; no extra field is needed.
    """
    state = vector["input"].get("state") or vector["process"].get("initial_state")
    if state is not None:
        return [value / sum(state) for value in state]
    return stationary_leaked(probs, leak) if leak else stationary(probs)


def close(actual, expected, tol: float = 1e-12) -> bool:
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(close(a, b, tol) for a, b in zip(actual, expected, strict=True))
    return abs(actual - expected) <= tol


# --- Tests -----------------------------------------------------------------------------


def test_document_is_well_formed():
    assert set(DOCUMENT) >= {"spec_version", "tolerance", "test_vectors"}
    assert DOCUMENT["tolerance"] == {"rel_tol": 1e-6, "abs_tol": 1e-12}  # SPEC 7.2
    seen = set()
    for vector in VECTORS:
        assert set(vector) >= REQUIRED_FIELDS, vector["id"]
        # No repo-internal metadata: consumers copy this file verbatim.
        assert set(vector) <= REQUIRED_FIELDS | OPTIONAL_FIELDS, f"{vector['id']} carries non-SPEC fields"
        assert vector["operation"] in OPERATIONS, vector["id"]
        assert vector["id"] not in seen, f"duplicate id {vector['id']}"
        assert vector["notes"], f"{vector['id']} has no derivation note"
        seen.add(vector["id"])


@pytest.mark.parametrize("vector", VECTORS, ids=IDS)
def test_serialized_matrices_match_the_rules(vector):
    # identify() raises unless the serialized tensor equals a rule-built one to 1e-12.
    assert identify(vector)[0]


@pytest.mark.parametrize(("name", "probs", "leak"), PROCESSES, ids=[p[0] for p in PROCESSES])
def test_checksum_agrees_with_the_rules(name, probs, leak):
    # The other direction: the reference constructor reproduces the rule-built tensor.
    Ts = checksum(jnp.array(probs))
    if leak:
        Ts = leak_toward(Ts, leak)
    assert close(_flatten(Ts.tolist()), _flatten(rules_tensor(probs, leak)), tol=1e-6)


@pytest.mark.parametrize("vector", VECTORS, ids=IDS)
def test_expected_values_match_the_oracle(vector):
    _, probs, leak = identify(vector)
    weights = prior_weights(vector, probs, leak)
    operation = vector["operation"]

    if operation == "stationary_distribution":
        expected = stationary_leaked(probs, leak) if leak else stationary(probs)
    elif operation == "observation_distribution":
        expected = obs_dist(probs, weights, leak)
    elif operation == "state_update":
        expected = state_update(probs, weights, vector["input"]["token"], leak)
    elif operation == "sequence_probability":
        expected = sequence_probability(probs, vector["input"]["tokens"], weights, leak)
    else:
        raise AssertionError(f"unhandled operation {operation}")

    assert close(vector["expected"], expected), (
        f"{vector['id']}: vector says {vector['expected']}, oracle derives {expected}"
    )


@pytest.mark.parametrize("vector", VECTORS, ids=IDS)
def test_sequence_probabilities_agree_across_both_derivations(vector):
    # Where the analytic phase-alignment sum applies it must agree with the propagated one:
    # two derivations of the same number, sharing only the emission and successor rules.
    if vector["operation"] != "sequence_probability":
        return
    _, probs, leak = identify(vector)
    if leak:
        return  # a leak breaks the block structure the alignment argument relies on
    weights = prior_weights(vector, probs, leak)
    tokens = vector["input"]["tokens"]
    by_alignment = word_probability_by_alignment(probs, tokens, weights)
    assert close(by_alignment, sequence_probability(probs, tokens, weights))


def test_stationary_closed_form_is_stationary():
    for _, probs, leak in PROCESSES:
        if leak:
            continue
        pi = stationary(probs)
        assert close(sum(pi), 1.0)
        block = rules_tensor(probs)
        size = len(block[0])
        net = [[sum(matrix[j][k] for matrix in block) for k in range(size)] for j in range(size)]
        assert close(propagate(pi, net), pi)


def test_leaked_stationary_fixed_point_is_stationary():
    probs, leak = [[0.5, 0.5], [0.5, 0.5]], 0.1
    pi = stationary_leaked(probs, leak)
    assert close(sum(pi), 1.0)
    leaked = rules_tensor(probs, leak)
    size = len(leaked[0])
    net = [[sum(matrix[j][k] for matrix in leaked) for k in range(size)] for j in range(size)]
    assert close(propagate(pi, net), pi, tol=1e-14)

    # A leak toward uniform pulls the stationary distribution toward uniform.
    unleaked = stationary(probs)
    uniform = 1 / size
    assert sum(abs(p - uniform) for p in pi) < sum(abs(p - uniform) for p in unleaked)


def test_word_probabilities_sum_to_one():
    for _, probs, leak in PROCESSES:
        m = len(probs[0])
        weights = stationary_leaked(probs, leak) if leak else stationary(probs)
        for length in (1, 2, 3):
            total = sum(
                sequence_probability(probs, list(w), weights, leak) for w in itertools.product(range(m), repeat=length)
            )
            assert close(total, 1.0, tol=1e-12)


def test_a_forbidden_word_is_forbidden_only_without_a_leak():
    probs, tokens = [[0.5, 0.5], [0.5, 0.5]], [0, 0, 1, 0, 0]
    weights = stationary(probs)
    assert sequence_probability(probs, tokens, weights) == 0
    assert word_probability_by_alignment(probs, tokens, weights) == 0
    assert sequence_probability(probs, tokens, stationary_leaked(probs, 0.1), 0.1) > 0
