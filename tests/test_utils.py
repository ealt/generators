import jax.numpy as jnp

from src.utils import mixed_radix_decode, mixed_radix_encode, mixed_radix_weights, principal_ev, stack


def test_principal_ev():
    T = jnp.array(
        [
            [14, -6],
            [6, -1],
        ]
    )
    actual = principal_ev(T)
    expected = jnp.array([2, 1]) * (2 / 3)
    assert jnp.allclose(actual, expected)


def test_stack():
    arrs = [jnp.ones((1, 1, 2)), jnp.ones((1, 2, 1)), jnp.ones((2, 1, 1))]
    expected = jnp.array(
        [
            [
                [
                    [1, 1],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 0],
                ],
            ],
            [
                [
                    [1, 0],
                    [1, 0],
                ],
                [
                    [0, 0],
                    [0, 0],
                ],
            ],
            [
                [
                    [1, 0],
                    [0, 0],
                ],
                [
                    [1, 0],
                    [0, 0],
                ],
            ],
        ]
    )
    actual = stack(arrs)
    assert jnp.array_equal(actual, expected)


def test_mixed_radix_decode_encode():
    Vs = jnp.array([2, 3, 2])
    weights = mixed_radix_weights(Vs)

    expected_digits = jnp.array(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        ]
    )
    xs = jnp.arange(12)

    actual_digits = jnp.stack([mixed_radix_decode(x, Vs=Vs, weights=weights) for x in xs], axis=1)
    actual_xs = jnp.array(
        [mixed_radix_encode(expected_digits[:, i], weights=weights) for i in range(expected_digits.shape[1])]
    )

    assert jnp.array_equal(actual_digits, expected_digits)
    assert jnp.array_equal(actual_xs, xs)
