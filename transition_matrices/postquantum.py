from __future__ import annotations

import math

import jax
import jax.numpy as jnp


# Symbol ordering used by the returned array.
RESET_TO_ONE = 0
MULTIPLY_BY_ALPHA = 1
MULTIPLY_BY_BETA = 2


def moon(
    alpha: float | jax.Array,
    beta: float | jax.Array,
    *,
    normalize: bool = True,
    validate: bool = True,
) -> jax.Array:
    """Creates GHMM transition matrices for the Moon process.

    Named for its crescent-shaped belief geometry. It is one example in the
    postquantum family.

    Generative picture:
    ------------------
    Easiest to understand as a multiplicative walk on a hidden scalar x > 0.

    After a reset, x = 1. Between resets, the non-reset symbols multiply x:

        RESET_TO_ONE      -> x <- 1
        MULTIPLY_BY_ALPHA -> x <- alpha * x
        MULTIPLY_BY_BETA  -> x <- beta  * x

    The reset transition has raw weight

        x - 1 - log(x) >= 0.

    The actual symbol probabilities are obtained through the GHMM
    normalization/evaluation machinery; these matrices are not row-stochastic
    HMM transition matrices.

    Symbol semantics:
    ----------------
    Returned matrices are ordered by the module-level constants:

        RESET_TO_ONE, MULTIPLY_BY_ALPHA, MULTIPLY_BY_BETA

    Parameter regime:
    ----------------
    The usual theoretical conditions are:

        alpha > 1 > beta > 0
        alpha + beta != 2
        log(alpha) / log(beta) irrational

    Validation checks only the enforceable numerical conditions, plus a small
    legacy-dangerous subset of the irrationality condition (log-ratio near an
    integer). Irrationality itself must be guaranteed by construction/choice of
    parameters.

    Validation is Python-side and intended for concrete scalar parameters. Set
    validate=False when using this constructor inside JAX transformations.

    State encoding:
    --------------
    Sequence-conditioned states lie on a 1D family parameterized by x > 0. That
    family is embedded into R^3 as the row vector

        eta(x) = [x, -1 - log(x), -1]

    so that multiplies and resets become ordinary matrix multiplies. The third
    coordinate is a homogeneous/affine slot (constant -1), which lets affine
    updates on (x, log x) fit inside linear algebra. Coordinates are chosen so
    that, with row-vector convention,

        eta(x) @ multiply_by(alpha) = eta(alpha * x)
        eta(x) @ multiply_by(beta)  = eta(beta  * x)

    and

        eta(x) @ reset = reset_weight(x) * eta(1)

    where

        reset_weight(x) = x - 1 - log(x) >= 0

    is the nonnegative weight that makes reset a valid GHMM update onto eta(1).

    The identities above hold for the unnormalized primitive matrices. If
    normalize=True, every right-hand side is additionally scaled by the common
    factor 1 / spectral_radius.

    The convex/normalized belief geometry associated with that 1D curve has the
    crescent-like shape that gives the process its name.

    Returns:
    -------
    jax.Array of shape (3, 3, 3)

    The first axis indexes the emitted symbol.
    The last two axes are the GHMM update matrix for that symbol.

    Notes:
    -----
    These are not stochastic transition matrices over 3 latent states.
    They are signed linear operators acting on the predictive-state encoding.

    Legacy:
    ------
    The predecessor constructor took log_alpha rather than alpha:

        post_quantum(log_alpha, beta) == moon(exp(log_alpha), beta)
    """
    if validate:
        _validate_moon_process_parameters(alpha, beta)

    transition_matrices = _raw_moon_transition_matrices(alpha, beta)

    if normalize:
        transition_matrices = _normalize_by_spectral_radius(transition_matrices)

    return transition_matrices


def _raw_moon_transition_matrices(
    alpha: float | jax.Array,
    beta: float | jax.Array,
) -> jax.Array:
    """Unnormalized symbol matrices ordered as reset, multiply-alpha, multiply-beta."""
    return jnp.stack(
        [
            _reset_to_one_matrix(),
            _multiply_encoded_scalar_by(alpha),
            _multiply_encoded_scalar_by(beta),
        ]
    )


def _encoded_scalar_state(x: float | jax.Array) -> jax.Array:
    """Encodes scalar state x > 0 as eta(x) = [x, -1 - log(x), -1]."""
    x = jnp.asarray(x)

    return jnp.array(
        [
            x,
            -1.0 - jnp.log(x),
            -1.0,
        ]
    )


def _multiply_encoded_scalar_by(multiplier: float | jax.Array) -> jax.Array:
    """Matrix implementing x -> multiplier * x in the encoded state space.

    For row-vector states:

        eta(x) @ M(multiplier) = eta(multiplier * x)

    where

        eta(x) = [x, -1 - log(x), -1].

    This is why log(multiplier) appears in the matrix.
    """
    multiplier = jnp.asarray(multiplier)

    return jnp.array(
        [
            [multiplier, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, jnp.log(multiplier), 1.0],
        ]
    )


def _reset_to_one_matrix() -> jax.Array:
    """Matrix implementing the reset transition x -> 1.

    For encoded row-vector states:

        eta(x) @ R = reset_weight(x) * eta(1)

    where

        reset_weight(x) = x - 1 - log(x).

    Since x - 1 - log(x) >= 0 for x > 0, this reset weight is nonnegative.
    """
    # Dotting eta(x) with this feature vector gives:
    #
    #   [x, -1 - log(x), -1] @ [1, 1, 0]
    #       = x - 1 - log(x)
    #
    reset_weight_feature = jnp.array([1.0, 1.0, 0.0])

    # eta(1) = [1, -1, -1]
    reset_target_state = _encoded_scalar_state(1.0)

    # With row-vector convention:
    #
    #   eta(x) @ outer(reset_weight_feature, reset_target_state)
    #       = (eta(x) @ reset_weight_feature) * reset_target_state
    #
    return jnp.outer(reset_weight_feature, reset_target_state)


def _normalize_by_spectral_radius(transition_matrices: jax.Array) -> jax.Array:
    """Scales all symbol matrices so the net transition has spectral radius 1.

    Classical HMMs usually normalize rows to make a stochastic matrix.

    This is a GHMM, so the relevant normalization is different: the sum of the
    symbol-labeled transition operators should have leading eigenvalue 1.
    """
    net_transition = transition_matrices.sum(axis=0)
    eigenvalues = jnp.linalg.eigvals(net_transition)
    spectral_radius = jnp.max(jnp.abs(eigenvalues))

    return transition_matrices / spectral_radius


def _validate_moon_process_parameters(
    alpha: float | jax.Array,
    beta: float | jax.Array,
) -> None:
    """Validates concrete scalar parameter conditions for the Moon process.

    Does not prove the irrationality condition. Besides the enforceable
    numerical checks, it only rejects a small legacy-dangerous subset: log
    ratios numerically close to integers.
    """
    alpha = float(alpha)
    beta = float(beta)

    if not (alpha > 1.0 > beta > 0.0):
        raise ValueError(f"Expected alpha > 1 > beta > 0, but got alpha={alpha}, beta={beta}.")

    if math.isclose(alpha + beta, 2.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Expected alpha + beta != 2, but got alpha + beta = {alpha + beta}.")

    # Irrationality must be guaranteed by construction/choice of parameters.
    # No floating-point check can prove it. This only catches ratios numerically
    # close to integers, not general rationals such as 3/2 or 5/7.
    log_ratio = math.log(alpha) / math.log(beta)

    if math.isclose(log_ratio, round(log_ratio), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            "log(alpha) / log(beta) appears to be close to an integer. "
            "The theoretical Moon process assumes this ratio is irrational. "
            f"Got log-ratio {log_ratio}."
        )
