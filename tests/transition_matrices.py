def mess3(x: float, a: float) -> list[list[list[float]]]:
    """Creates a transition matrix for the Mess3 Process."""
    b = (1 - a) / 2
    y = 1 - 2 * x

    ay = a * y
    bx = b * x
    by = b * y
    ax = a * x

    return [
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


def zero_one() -> list[list[list[float]]]:
    """Generate the transition matrices for a zero-one process."""
    return [
        [
            [0, 1],
            [0, 0],
        ],
        [
            [0, 0],
            [1, 0],
        ],
    ]


def zero_one_random(p: float) -> list[list[list[float]]]:
    """Generate the transition matrices for a zero-one random process."""
    q = 1 - p
    return [
        [
            [0, 1, 0],
            [0, 0, 0],
            [q, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 1],
            [p, 0, 0],
        ],
    ]


def cycle(n: int, p: float) -> list[list[list[float]]]:
    """Generate the transition matrices for a cycle process."""
    assert n >= 3
    q = 1 - p
    Ts: list[list[list[float]]] = [[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        Ts[i][i][(i + 1) % n] = p
        Ts[i][i][(i - 1) % n] = q
    return Ts
