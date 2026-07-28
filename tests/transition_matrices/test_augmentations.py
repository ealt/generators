import jax.numpy as jnp

from transition_matrices.augmentations import compress_vocab, expand_vocab
from transition_matrices.classical import mess


def test_expand_vocab():
    x = 0.3  # y = 1 - ((s - 1) * x) = 0.7
    a = 0.6  # b = (1 - a) / (s - 1) = 0.4
    s = 2
    f = 2
    ax = 0.09  # a * x / f
    ay = 0.21  # a * y / f
    bx = 0.06  # b * x / f
    by = 0.14  # b * y / f
    assert jnp.allclose(
        expand_vocab(mess(x, a, s), f),
        jnp.array(
            [
                [
                    [ay, bx],
                    [ax, by],
                ],
                [
                    [ay, bx],
                    [ax, by],
                ],
                [
                    [by, ax],
                    [bx, ay],
                ],
                [
                    [by, ax],
                    [bx, ay],
                ],
            ]
        ),
    )


def test_compress_vocab():
    x = 0.3  # y = 1 - ((s - 1) * x) = 0.1
    a = 0.7  # b = (1 - a) / (s - 1) = 0.1
    s = 4
    f = 2
    ax = 0.24  # (a + (f - 1) * b) * x
    ay = 0.08  # (a + (f - 1) * b) * y
    bx = 0.06  # f * b * x
    by = 0.02  # f * b * y
    assert jnp.allclose(
        compress_vocab(mess(x, a, s), f),
        jnp.array(
            [
                [
                    [ay, ax, bx, bx],
                    [ax, ay, bx, bx],
                    [ax, ax, by, bx],
                    [ax, ax, bx, by],
                ],
                [
                    [by, bx, ax, ax],
                    [bx, by, ax, ax],
                    [bx, bx, ay, ax],
                    [bx, bx, ax, ay],
                ],
            ]
        ),
    )


def test_compress_vocab_inverts_expand_vocab():
    Ts = mess(0.15, 0.6, 4)
    assert jnp.allclose(compress_vocab(expand_vocab(Ts, 3), 3), Ts)
