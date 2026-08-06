import jax.numpy as jnp

from transition_matrices.augmentations import (
    apply_symbol_map,
    compress_map,
    confusion_map,
    expand_map,
    noise_map,
)
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
    Ts = mess(x, a, s)
    C = expand_map(Ts.shape[0], f)
    assert jnp.allclose(
        apply_symbol_map(Ts, C),
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
    Ts = mess(x, a, s)
    C = compress_map(Ts.shape[0], f)
    assert jnp.allclose(
        apply_symbol_map(Ts, C),
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


def test_noise_vocab():
    eps = 0.3
    Ts = mess(0.3, 0.7, 3)
    C = noise_map(Ts.shape[0], eps)
    Ts_noisy = apply_symbol_map(Ts, C)
    diff = jnp.abs(Ts_noisy - Ts)
    assert jnp.all(diff > 0)
    assert jnp.all(diff <= eps)


def test_confusion_vocab():
    eps = 0.25
    Ts = mess(0.3, 0.7, 3)
    C = confusion_map(Ts.shape[0], [(0, 1)], eps)
    Ts_conf = apply_symbol_map(Ts, C)
    assert jnp.allclose(Ts_conf[0], (1 - eps) * Ts[0])
    assert jnp.allclose(Ts_conf[1], Ts[1] + eps * Ts[0])
    assert jnp.allclose(Ts_conf[2], Ts[2])
