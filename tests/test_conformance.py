"""Runs the SPEC section 7 conformance vectors against generators/ghmm/process.py.

This is the file USAGE.md step 2 tells consumers to copy: point PROCESS at your fork and
green means your fork still implements the same math. It reads conformance-tests.json and
knows nothing about how the vectors were produced.

The oracles behind the vectors are checked separately, in
tests/transition_matrices/test_conformance_vectors.py.
"""

import json
import pathlib

import jax.numpy as jnp
import pytest

from generators.ghmm import process
from generators.utils import principal_ev

VECTORS_PATH = pathlib.Path(__file__).parent.parent / "conformance-tests.json"

with VECTORS_PATH.open() as f:
    DOCUMENT = json.load(f)

VECTORS = DOCUMENT["test_vectors"]
REL_TOL = DOCUMENT["tolerance"]["rel_tol"]
ABS_TOL = DOCUMENT["tolerance"]["abs_tol"]


def tolerances(vector: dict) -> tuple[float, float]:
    return vector.get("tolerance", REL_TOL), vector.get("abs_tolerance", ABS_TOL)


def matches(actual, expected, rel_tol: float, abs_tol: float) -> bool:
    """SPEC 7.2: |a - b| <= abs_tol + rel_tol * |b|, elementwise."""
    actual, expected = jnp.asarray(actual, float), jnp.asarray(expected, float)
    if actual.shape != expected.shape:
        return False
    return bool(jnp.all(jnp.abs(actual - expected) <= abs_tol + rel_tol * jnp.abs(expected)))


def build(definition: dict) -> process.Data:
    """A GHMM from a section 7.1.1 process definition."""
    assert definition["type"] in {"ghmm", "hmm"}, f"unsupported process type {definition['type']}"
    Ts = jnp.array(definition["transition_matrices"])
    eta_0 = definition.get("initial_state")
    return process.init(Ts, eta_0=None if eta_0 is None else jnp.array(eta_0, float))


def run(vector: dict):
    data = build(vector["process"])
    inputs = vector["input"]
    operation = vector["operation"]

    if operation == "stationary_distribution":
        # SPEC 7.1.1: pi, the stationary left eigenvector normalized to sum to 1.
        pi = principal_ev(data.Ts.sum(axis=0).T)
        return pi / pi.sum()

    if operation == "sequence_probability":
        return process.seq_prob(data, jnp.array(inputs["tokens"], int))

    eta = jnp.array(inputs["state"], float) if "state" in inputs else data.eta_0

    if operation == "observation_distribution":
        # SPEC 3.4 divides by eta . w. process.obs_dist assumes a canonical state, so the
        # division belongs here -- that is what makes the result scale-invariant.
        return process.obs_dist(data, eta) / (eta @ data.w)

    if operation == "state_update":
        # SPEC 3.5 renormalizes, so this is already in the canonical form 7.1.2 wants.
        return process.update(data, eta, jnp.array(inputs["token"], int))

    raise AssertionError(f"unsupported operation {operation}")


@pytest.mark.parametrize("vector", VECTORS, ids=[v["id"] for v in VECTORS])
def test_conformance_vector(vector):
    rel_tol, abs_tol = tolerances(vector)
    actual = run(vector)
    assert matches(actual, vector["expected"], rel_tol, abs_tol), (
        f"{vector['id']} ({vector['category']}): expected {vector['expected']}, got {actual}"
    )


def test_every_documented_category_is_exercised():
    # Categories from SPEC 7.4 that base-GHMM vectors can cover. The factored and
    # nonergodic categories are still unshipped; see issue #8.
    covered = {v["category"] for v in VECTORS}
    assert covered == {
        "ghmm_observation_distribution",
        "ghmm_belief_update",
        "ghmm_sequence_probability",
        "ghmm_hmm_case",
        "ghmm_stationary_distribution",
        "ghmm_projective_invariance",
    }


def test_projective_invariance_vectors_share_one_expected_value():
    # The category only means anything if the rescaled inputs really do differ.
    scaled = [v for v in VECTORS if v["category"] == "ghmm_projective_invariance"]
    by_expected = {}
    for vector in scaled:
        if vector["operation"] != "observation_distribution":
            continue
        by_expected.setdefault(json.dumps(vector["expected"]), []).append(vector["input"]["state"])
    assert len(by_expected) == 1
    (inputs,) = by_expected.values()
    assert len(inputs) > 1
    assert any(a != b for a, b in zip(inputs, inputs[1:], strict=False))


def test_hmm_vectors_have_a_unit_normalizing_eigenvector():
    # SPEC 3.2.1: an HMM's rows sum to 1, so w = 1.
    for vector in VECTORS:
        if vector["process"]["type"] != "hmm":
            continue
        data = build(vector["process"])
        assert matches(data.w, jnp.ones_like(data.w), REL_TOL, 1e-9), vector["id"]
        assert matches(data.Ts.sum(axis=(0, 2)), jnp.ones_like(data.w), REL_TOL, 1e-9), vector["id"]


def test_serialized_matrices_are_valid_ghmms():
    for vector in VECTORS:
        Ts = jnp.array(vector["process"]["transition_matrices"])
        assert process.validate(Ts), vector["id"]
        assert jnp.all(Ts >= 0), vector["id"]  # SPEC 3.2 item 1
