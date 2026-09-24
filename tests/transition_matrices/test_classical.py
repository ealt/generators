import itertools

import jax.numpy as jnp
import pytest

from transition_matrices.classical import checksum


def test_checksum_rrxor():
    p = 0.3
    q = 0.7
    assert jnp.allclose(
        checksum(jnp.array([[p, 1 - p], [q, 1 - q]])),
        jnp.array(
            [
                [
                    [0, p, 0, 0, 0],
                    [0, 0, 0, q, 0],
                    [0, 0, 0, 0, q],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 1 - p, 0, 0],
                    [0, 0, 0, 0, 1 - q],
                    [0, 0, 0, 1 - q, 0],
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                ],
            ]
        ),
    )
