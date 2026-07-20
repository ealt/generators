import math

import jax.numpy as jnp
import numpy as np
import pytest

from transition_matrices.postquantum import (
    _encoded_scalar_state,
    _multiply_encoded_scalar_by,
    _raw_moon_transition_matrices,
    _reset_to_one_matrix,
    moon,
)


def test_multiply_matrix_encodes_scalar_multiplication():
    x = 2.3
    multiplier = 1.7

    eta_x = _encoded_scalar_state(x)
    matrix = _multiply_encoded_scalar_by(multiplier)

    np.testing.assert_allclose(
        eta_x @ matrix,
        _encoded_scalar_state(multiplier * x),
        rtol=1e-6,
        atol=1e-6,
    )


def test_reset_matrix_resets_to_eta_one_with_expected_weight():
    x = 2.3

    eta_x = _encoded_scalar_state(x)
    reset = _reset_to_one_matrix()

    expected_weight = x - 1.0 - np.log(x)

    np.testing.assert_allclose(
        eta_x @ reset,
        expected_weight * _encoded_scalar_state(1.0),
        rtol=1e-6,
        atol=1e-6,
    )


def test_raw_identities_hold_before_normalization():
    alpha = math.e
    beta = 0.5
    x = 2.3

    raw = _raw_moon_transition_matrices(alpha, beta)
    eta_x = _encoded_scalar_state(x)

    np.testing.assert_allclose(
        eta_x @ raw[0],
        (x - 1.0 - np.log(x)) * _encoded_scalar_state(1.0),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        eta_x @ raw[1],
        _encoded_scalar_state(alpha * x),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        eta_x @ raw[2],
        _encoded_scalar_state(beta * x),
        rtol=1e-6,
        atol=1e-6,
    )


def test_normalized_matrices_scale_raw_identities_by_spectral_radius():
    alpha = math.e
    beta = 0.5
    x = 2.3

    raw = _raw_moon_transition_matrices(alpha, beta)
    normalized = moon(alpha=alpha, beta=beta, normalize=True, validate=True)
    spectral_radius = jnp.max(jnp.abs(jnp.linalg.eigvals(raw.sum(axis=0))))
    eta_x = _encoded_scalar_state(x)

    np.testing.assert_allclose(normalized, raw / spectral_radius, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        eta_x @ normalized[1],
        _encoded_scalar_state(alpha * x) / spectral_radius,
        rtol=1e-6,
        atol=1e-6,
    )


def test_normalization_sets_spectral_radius_to_one():
    matrices = moon(alpha=math.e, beta=0.5, normalize=True)
    net_transition = matrices.sum(axis=0)
    spectral_radius = jnp.max(jnp.abs(jnp.linalg.eigvals(net_transition)))

    np.testing.assert_allclose(spectral_radius, 1.0, rtol=1e-6, atol=1e-6)


def test_validate_rejects_near_integer_log_ratio():
    # alpha = e^2, beta = e^{-1} => log(alpha)/log(beta) = -2
    with pytest.raises(ValueError, match="close to an integer"):
        moon(alpha=math.e**2, beta=math.e**-1, validate=True)
