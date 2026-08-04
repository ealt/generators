# Conformance Vectors

`conformance-tests.json` holds the first conformance vectors for [SPEC.md](SPEC.md) §7.
They cover the base GHMM/HMM categories over the classical checksum family
(`transition_matrices/classical.py`).

## How to use them

```sh
uv run --extra dev pytest tests/test_conformance.py
```

`tests/test_conformance.py` is the portable runner: it reads the JSON, applies §3.4/§3.5/
§3.6/§3.8 through `generators/ghmm/process.py`, and compares at §7.2 tolerances. It knows
nothing about how the vectors were produced. Copy it alongside the JSON and change its one
import — `from generators.ghmm import process` — to point at your vendored, modified copy;
green means your fork still implements the same math.

## Why this family

Issue #8 warns that deriving expected values from the reference implementation makes the
vectors a regression baseline rather than an independent check. The checksum family avoids
that, because its oracles are closed-form rather than captured:

| Quantity | Oracle |
| --- | --- |
| State count | `n * m + 1` |
| Stationary distribution | `1/(n+1)` on the seed state, `P(sum of first i symbols = r)/(n+1)` on `(i, r)` |
| Entropy rate | `sum_i H(probs[i]) / (n + 1)` bits/symbol |
| Word probability | sum over the `n + 1` phase alignments |
| The `(2, 2)` case | pinned elementwise against simplexity's published `rrxor` matrices |

`tests/transition_matrices/test_conformance_vectors.py` implements those oracles from the
generative rules — emit `n` random symbols, then their sum mod `m` — in plain Python
floats, with no array library and without calling `checksum()` except in one test that
confirms the two agree. Every expected value in the JSON comes from there. So the vectors
are checked from both sides: against the closed forms, and against the reference modules.

Double precision is deliberate. JAX defaults to float32, whose ~1e-7 relative epsilon sits
uncomfortably close to §7.2's `rel_tol` of 1e-6 and cannot represent its `abs_tol` of
1e-12 at all. An oracle in the same precision as the code under test would not be able to
detect an error of the size the spec cares about.

## What §7 leaves open

These are gaps in the spec, not extensions invented here. Each is a decision the vectors
had to make in order to exist; none is authorized by §7's text.

1. **`operation` values are never enumerated.** §7.4 requires an `operation` field —
   "which operation is being tested" — but no list of legal identifiers appears anywhere
   in the spec. Exactly one is named in passing: §7.1.1's "The `stationary_distribution`
   operation is defined in v1.0 conformance only for base `ghmm` and `hmm` process
   definitions". The other three here are snake_case of §2's operation headings, chosen to
   match that one known identifier's style:

   | Value | Spec section |
   | --- | --- |
   | `observation_distribution` | §2.1 Observation probability distribution |
   | `state_update` | §2.3 State update |
   | `sequence_probability` | §2.4 Sequence probability |
   | `stationary_distribution` | §2.5 Stationary objects, named in §7.1.1 |

   §2.2 (observation sampling) has no vectors: §7.3 excludes anything RNG-dependent.

2. **The spec uses three vocabularies for the same operations.** §2 calls it "State
   update"; §7.4's category is `ghmm_belief_update`; issue #8 uses a third set
   (`ghmm_obs_dist`, `ghmm_seq_prob`, `stationary_distribution`). This file follows §2 for
   `operation` and §7.4 verbatim for `category`, since §7.4 is the normative list. Worth
   reconciling in the spec.

3. **`input` has no schema.** §7.4 says only "operation-specific input". The field names
   used here:

   | Operation | `input` |
   | --- | --- |
   | `observation_distribution` | `{"state": [...]}`, or `{}` for the process default |
   | `state_update` | `{"state": [...], "token": 0}` |
   | `sequence_probability` | `{"tokens": [0, 1, 0]}` |
   | `stationary_distribution` | `{}` |

   `state` is a GHMM state representative per §7.1.2, so it need not be normalized — the
   `ghmm_projective_invariance` vectors rely on that.

4. **`expected` has no schema either.** Inferred per operation: a length-`V` distribution;
   a canonical state representative per §7.1.2; a scalar; and, per §7.1.1, `pi` normalized
   to sum to 1.

5. **The document's top-level shape and location are unspecified.** §7.4 names "the
   companion file `conformance-tests.json`" with no path and no description of what wraps
   the vectors. Chosen here: repository root, and an object with `spec_version`,
   `description`, `tolerance` (restating §7.2's defaults so a runner need not hardcode
   them), and `test_vectors`.

6. **Whether unknown fields are allowed is unstated.** These vectors carry only the fields
   §7.4 lists, and `test_document_is_well_formed` enforces that — repo-internal metadata
   would otherwise ride along into every consumer's copy.

7. **The tolerance override fields are named asymmetrically.** §7.4 offers `tolerance`
   ("relative tolerance override") and `abs_tolerance`. `rel_tolerance` would pair better.
   No vector here needs either.

## What is not covered

Of §7.4's 22 categories, these vectors cover 6:
`ghmm_observation_distribution`, `ghmm_belief_update`, `ghmm_sequence_probability`,
`ghmm_hmm_case`, `ghmm_stationary_distribution`, `ghmm_projective_invariance`.

Not covered, and why:

- **`ghmm_nontrivial_w` (category 5)** needs a GHMM whose normalizing eigenvector is not
  `1`. The checksum family cannot supply one — it is an HMM, so `w = 1` by §3.2.1 — and
  filling the gap runs into a tension between §3.2 and every non-classical process in the
  repo. Measured at `061fcf2`:

  | Process | §3.2.1 entries nonneg | `rho(T) = 1` | `w` strictly positive | `w` |
  | --- | --- | --- | --- | --- |
  | `checksum`, `cycle`, `mess` | yes | yes | yes | `1` |
  | `bloch_walk(0.9, 0.5)` | no, min `-0.196` | yes | no | `[3, 0, 0]` |
  | `quantum_rrxor(0.7, 1.1)` | no, min `-0.446` | yes | no | `[4, 0, 0, 0]` |
  | `moon(e, 0.5)` | no, min `-0.261` | yes | yes | `[2.159, 0.720, 0.121]` |

  So §3.2 item 1 (every entry nonnegative) is violated by all three signed-operator
  processes, and item 3 (a *strictly* positive `w`) additionally by both quantum ones,
  whose `w` is proportional to `e_0`. This is pre-existing and not introduced here, but it
  has to be resolved before category 5 can be filled: either §3.2's conditions are too
  strong for signed Bloch/Liouville representations, or those constructors need a
  different gauge. `moon` is the nearest candidate — its `w` is both nontrivial and
  strictly positive, so relaxing item 1 alone would unblock the category.
- **Factored categories (8–16)** and **nonergodic categories (17–21)** need composite
  process definitions; issue #8 sequences them after the base GHMM ones.
- **`generation_layout` (category 22)** covers BOS/EOS/PAD, which is unimplemented here
  (issue #9 §2).

Adding a vector currently means hand-editing the JSON, since the derivation lives in the
test module rather than in a committed generator. That is fine at this size and worth
revisiting if the count grows.
