import itertools

import jax.numpy as jnp
import pytest

from transition_matrices.classical import checksum, cycle, mess, periodic_grid


def test_checksum():
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

def test_cycle():
    assert jnp.allclose(
        cycle(3, 0.6),
        jnp.array(
            [
                [
                    [0, 0.6, 0.4],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                [
                    [0, 0, 0],
                    [0.4, 0, 0.6],
                    [0, 0, 0],
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0.6, 0.4, 0],
                ],
            ]
        )
    )


def test_mess():
    s = 3
    a = 0.6
    b = 0.2  # (1 - a) / (s-1)
    x = 0.15
    y = 0.7  # 1 - (s-1) * x
    ax = 0.09
    ay = 0.42
    bx = 0.03
    by = 0.14

    assert jnp.allclose(
        mess(x, a, s),
        jnp.array(
            [
                [
                    [ay, bx, bx],
                    [ax, by, bx],
                    [ax, bx, by],
                ],
                [
                    [by, ax, bx],
                    [bx, ay, bx],
                    [bx, ax, by],
                ],
                [
                    [by, bx, ax],
                    [bx, by, ax],
                    [bx, bx, ay],
                ],
            ]
        )
    )

def test_periodic_grid():
    shape = (3, 3)

    # 0: [0, 0]
    # 1: [0, 1]
    # 2: [0, 2]
    # 3: [1, 0]
    # 4: [1, 1]
    # 5: [1, 2]
    # 6: [2, 0]
    # 7: [2, 1]
    # 8: [2, 2]

    edges = jnp.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 6],
        [1, 2],
        [1, 0],
        [1, 4],
        [1, 7],
        [2, 0],
        [2, 1],
        [2, 5],
        [2, 8],
        [3, 4],
        [3, 5],
        [3, 6],
        [3, 0],
        [4, 5],
        [4, 3],
        [4, 7],
        [4, 1],
        [5, 3],
        [5, 4],
        [5, 8],
        [5, 2],
        [6, 7],
        [6, 8],
        [6, 0],
        [6, 3],
        [7, 8],
        [7, 6],
        [7, 1],
        [7, 4],
        [8, 6],
        [8, 7],
        [8, 2],
        [8, 5],
    ])
    source = edges[:, 0]
    dest = edges[:, 1]
    expected = jnp.zeros((9, 9, 9)).at[source, source, dest].set(0.25)
    assert jnp.allclose(periodic_grid(shape), expected)
