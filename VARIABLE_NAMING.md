# Variable Naming Conventions

This document defines naming conventions for observation variables in this repository.

The goal is to keep two distinctions obvious in code:

- composite observations vs per-factor observations
- per-factor observations vs per-component observations
- single observations vs sequences of observations

## Core rule

Use:

- `x` / `xs` for composite observations
- `x_factors` / `xs_factors` for per-factor observations

Do not use `factor_xs` for a single-step tuple of factor observations. Reserve plural `xs` names for actual sequences.

## GHMM naming

For base GHMM code:

- `x`: one observation
- `xs`: sequence of observations

Examples:

- `update(data, eta, x)`
- `seq_prob(data, xs)`

## Factored-process naming

For factored-process code:

- `x`: one composite observation
- `xs`: sequence of composite observations
- `x_factors`: one time step of per-factor observations, shape `(F,)`
- `xs_factors`: sequence of per-factor observation tuples, shape `(T, F)`
- `x_i`: one local observation for factor `i`

Examples:

- `decode(x) -> x_factors`
- `jax.vmap(decode)(xs) -> xs_factors`
- `update(data, eta, x_factors)`

## Sequential-chain scan naming

Inside left-to-right sequential emission scans:

- `x_prev`: the previous factor's local observation in the same time step
- `x_i`: the current factor's local observation

Use `x_prev` only for this local scan meaning. Outside that context, prefer explicit tuple names such as `x_factors`.

## Runtime-state naming

Use:

- `eta`: current runtime factor states, shape `(F, S_max)`
- `eta_i`: runtime state for one factor
- `eta_0`: per-variant metadata stored on `Data`, not a generic default runtime state for multi-variant factored processes

## Nonergodic-process naming

For nonergodic-mixture code:

- `x`: one global observation
- `xs`: sequence of global observations
- `x_components`: one time step of per-component local observations, shape `(C,)`
- `xs_components`: sequence of per-component local observations, shape `(T, C)`
- `x_c`: one component-local observation
- `x_component_ids`: vector of candidate component-local observation ids used for masking or lookup
- `xs_c`: sequence of observations in one component's local vocabulary
- `eta_c`: runtime state for one component
- `phi_c`: local-to-global vocabulary map for one component
- `beta`: current component-belief vector, shape `(C,)`
- `beta_0`: initial component-belief vector stored on `Data`

Do not reuse factor-oriented `_i` names for component-local quantities in nonergodic code. Reserve `_i` for factor-local variables.

## Probability naming

Use:

- `prob`: one scalar probability
- `probs`: a vector or collection of probabilities
- `obs_prob`: probability of one composite observation
- `obs_dist`: full observation distribution

Do not introduce `likelihood` / `likelihoods` unless the distinction from probability is important to the algorithm being implemented.

## Scan helper naming

Use:

- `scan_inputs`: tuple of values passed as `xs` into `jax.lax.scan`

Avoid:

- `scan_xs`

because it overloads `xs`, which this repository reserves for observation sequences.

## Summary table

| Meaning | Preferred name |
| --- | --- |
| Single GHMM observation | `x` |
| GHMM observation sequence | `xs` |
| Single composite factored observation | `x` |
| Composite observation sequence | `xs` |
| Single-step per-factor observations | `x_factors` |
| Sequence of per-factor observation tuples | `xs_factors` |
| Single factor-local observation | `x_i` |
| Single-step per-component observations | `x_components` |
| Sequence of per-component observations | `xs_components` |
| Single component-local observation | `x_c` |
| Previous factor-local observation in sequential scan | `x_prev` |
| Runtime factor states | `eta` |
| Per-factor runtime state | `eta_i` |
| Per-component runtime state | `eta_c` |
| Scalar probability | `prob` |
| Collection of probabilities | `probs` |
| Inputs to `jax.lax.scan` | `scan_inputs` |
