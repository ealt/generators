import jax
import jax.numpy as jnp


def zero_one_random(p: float = 0.5) -> jax.Array:
    """Zero-one random transition matrix.

    The zero-one random process is a simple random walk on a cycle of 3 states.
    A symbol is deterministically emitted based on the current state.

    Args:
        p: Transition probability to the next state.

    Returns:
        Transition matrix.
    """
    q = 1 - p
    return jnp.array(
        [
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
    )


def cycle(n: int, p: float = 0.5) -> jax.Array:
    """Cycle transition matrix.

    The cycle process is a simple random walk on a cycle of n states.
    A symbol is deterministically emitted based on the current state.

    Args:
        n: Number of states. Must be greater than 0.
        p: Transition probability to the next state.
            Ignored for n = 1 or 2.
            q = 1 - p: Transition probability to the previous state.

    Returns:
        Transition matrix.
    """
    assert n > 0
    if n == 1:
        return jnp.ones((1, 1, 1))
    if n == 2:
        return jnp.array(
            [
                [
                    [0, 1],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [1, 0],
                ],
            ]
        )
    assert p >= 0
    assert p <= 1
    q = 1 - p
    Ts: list[list[list[float]]] = [[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        Ts[i][i][(i + 1) % n] = p
        Ts[i][i][(i - 1) % n] = q
    return jnp.array(Ts)


def checksum(probs: jax.Array) -> jax.Array:
    """Checksum transition matrix.

    The checksum process emits a block of n random symbols over an alphabet of size m,
    then deterministically emits their sum mod m — the block's checksum — and repeats.
    RRXOR is the n = 2, m = 2 case: two random bits followed by their XOR.

    Internal state basis:
    --------------------
    A causal state is a (phase, running sum mod m) pair, where phase counts emissions
    into the current block. The prefix itself does not matter, only its residue, so
    there are n * m + 1 states ordered as

        0                       phase 0, the seed state (running sum is always 0)
        1 + (i - 1) * m + r     phase i in {1, ..., n}, running sum r

    Phases 0 through n - 1 emit a random symbol; phase n emits the checksum r and
    returns to the seed state. At n = 2, m = 2 this orders the states as

        S, "0", "1", F, T

    where "0"/"1" are the first bit and F/T are the XOR states, so it agrees with the
    conventional RRXOR basis {S, "0", "1", T, F} up to the final transposition.

    Derived quantities, all closed-form:

        states          n * m + 1
        stationary      1 / (n + 1) on the seed state; P(sum of first i symbols = r)
                        / (n + 1) on (i, r)
        entropy rate    sum_i H(probs[i]) / (n + 1) bits/symbol,
                        which is n * log2(m) / (n + 1) for uniform rows

    The chain is periodic with period n + 1, since every state advances one phase per
    symbol. Its net matrix therefore has n + 1 eigenvalues of modulus 1, and power
    iteration does not converge to the stationary distribution — it cycles. Use an
    eigensolver, or average the iterates over a full period.

    m = 1 degenerates to a deterministic cycle of n + 1 states over a single symbol.

    Args:
        probs: Row-stochastic emission probabilities, shape (n, m). Row i is the
            emission distribution of the i-th random symbol. Requires n >= 1.

    Returns:
        Transition matrix, shape (m, n * m + 1, n * m + 1).
            Ts[o, j, k] = P(x_t=o, s_t=k | s_{t-1}=j)
    """
    assert probs.ndim == 2
    n, m = probs.shape
    # n = 0 scatters to state index -1, which JAX silently clamps instead of raising.
    assert n >= 1
    assert jnp.allclose(probs.sum(axis=1), 1)

    # The states that emit a random symbol are the seed state followed by m states for
    # each of phases 1 .. n - 1. Their position in these arrays is their state index.
    phases = jnp.concatenate([jnp.zeros(1, int), jnp.repeat(jnp.arange(1, n), m)])
    residues = jnp.concatenate([jnp.zeros(1, int), jnp.tile(jnp.arange(m), n - 1)])

    # Each of them can emit any of the m symbols, carrying the running sum to the next
    # phase, whose states start at 1 + (phase + 1 - 1) * m.
    sources = jnp.repeat(jnp.arange(phases.size), m)
    symbols = jnp.tile(jnp.arange(m), phases.size)
    source_phases = jnp.repeat(phases, m)
    source_residues = jnp.repeat(residues, m)
    destinations = 1 + source_phases * m + (source_residues + symbols) % m

    Ts = jnp.zeros((m, n * m + 1, n * m + 1))
    Ts = Ts.at[symbols, sources, destinations].set(probs[source_phases, symbols])

    # The checksum states follow the random ones. Each emits its residue and resets.
    checksums = jnp.arange(m)
    return Ts.at[checksums, phases.size + checksums, 0].set(1.0)


def _mess_trans(x: float, s: int) -> jax.Array:
    r"""State transition matrix for Mess process.

    Args:
        x: Transition probability to each other state.
            x = P(s_t = s' \forall s' \in {0, ..., s-1} \ s')
            y: Probability of staying in the same state.
            y = P(s_t = s_t-1)
        s: Number of states

    Returns:
        State transition matrix
            T[i, j] = P(s_t = i | s_{t-1} = j)
            T[i, i] = y
            T[i, j] = x for i != j
    """
    assert x >= 0
    assert x <= 1
    y = 1 - (s - 1) * x
    assert y >= 0
    assert y <= 1
    return (x * (jnp.ones((s, s)) - jnp.eye(s))) + (y * jnp.eye(s))


def _mess_emit(a: float, s: int) -> jax.Array:
    r"""Emission matrix for Mess process.

    Args:
        a: Emission probability corresponding to the previous state.
            a = P(x_t = s_t-1)
            b: Emission probabilities for each other value.
            b = P(x_t = s' \forall s' \in S_{t-1})
        s: Number of states
        rd: Ratio difference
            If the vocab size to state size ratio, V : S
            is reduced, num : denom (V / S = num / denom)
            rd := num - denom

    Returns:
        Emission matrix
            Tv[i, j] = P(x_t = j | s_t = i)
            Tv[i, i] = a
            Tv[i, j] = b for i != j
    """
    assert a >= 0
    assert a <= 1
    b = (1 - a) / (s - 1)

    return (a * jnp.eye(s)) + (b * (jnp.ones((s, s)) - jnp.eye(s)))


def mess(x: float, a: float, s: int) -> jax.Array:
    r"""Mess process.

    Args:
        x: Transition probability to each other state.
        a: Emission probability corresponding to the previous state.
        s: Number of states
        rd: Ratio difference

    Returns:
        Transition matrix
            Ts[o, j, k] = P(x_t=o, s_t=k | s_{t-1}=j)
    """
    assert s > 0
    T_trans = _mess_trans(x, s)
    T_emit = _mess_emit(a, s)

    # Ts[o, j, k] = P(x_t=o, s_t=k | s_{t-1}=j) = Tv[k, o] * T[k, j]
    inner = T_emit[:, :, None] * T_trans[None, :, :]
    return jnp.transpose(inner, (0, 2, 1))
