# Usage Model

This repo is not a library. Do not add it to your dependencies.

Copy the modules you need into your codebase. Delete what you don't use.
Modify what remains until it fits your problem exactly. The generative
process code lives *in your project* — inspectable, debuggable, and
optimizable — not behind an import boundary.

## The problem with a library

A library must commit to one implementation and hope it is good enough
for every consumer. Every design decision is a compromise across use
cases the author can only guess at:

- **Speed vs. memory.** Precomputing the joint observation law over a
  composite vocabulary is fast per step and heavy in memory; recomputing
  per-factor terms on the fly is the reverse. A library picks one. Your
  workload knows which it needs.
- **Generality vs. simplicity.** Accommodating every user in one code
  path has a shape: class hierarchies whose subclasses each cover a case
  you don't have, function signatures that grow a flag for every variant,
  and branching logic to dispatch among them. Flexibility that serves
  everyone is, for any one project, mostly machinery for paths it will
  never take — and each option is a branch to read, test, and get wrong.
  A library can't decline this; it doesn't know which path is yours.
- **Stability vs. progress.** Once consumers import a library, its API
  is a contract. Improving it means deprecation cycles, compatibility
  shims, and version churn — cost paid by everyone, forever.

All of this is worst on a research team. Different people pursue
different research directions, and each direction inevitably wants new
kinds of processes. With everyone sharing a single library dependency,
that means everyone merging their process code into the same shared
codebase — a coordination point nobody wanted. One person's ideal
modification is another person's breaking change, so changes get
negotiated, watered down, or blocked. And because past research must
remain reproducible, even internal APIs become effectively frozen: the
backwards-compatibility obligation binds hardest exactly where
exploration should be cheapest.

The usual escape hatches are worse: forking an entire library to change
one function, or maintaining a patch queue against upstream releases.

## The model here

This project inverts the relationship. It ships two things:

1. **A specification.** [SPEC.md](SPEC.md) defines the mathematical
   objects and required behaviors, independent of any implementation,
   with conformance test vectors in
   [`conformance-tests.json`](conformance-tests.json) — see
   [CONFORMANCE.md](CONFORMANCE.md) for what they cover so far.
2. **Reference modules.** Small, human-crafted, self-contained
   implementations of the spec — correct starting points, written to be
   read and rewritten.

The unit of reuse is the module, and the mechanism is copying. The
invariant that survives your modifications is not the code — it is the
spec. After you have rewritten a module for your use case, run the
conformance vectors against your copy: if they pass, your fork still
implements the same math.

Everything else about the repo serves copyability:

- **No dependencies** beyond JAX — a vendored module must not drag a
  tree of transitive dependencies into your project.
- **Modules, not a package** — `__init__.py` files are empty; there is
  no registry, builder, or class hierarchy coupling modules together.
  `ghmm/process.py` + `factored/chain.py` + `utils.py` is a complete,
  liftable unit.
- **Small and notation-faithful** — each module is a few hundred lines,
  named to mirror the spec's math ([VARIABLE_NAMING.md](VARIABLE_NAMING.md)),
  so you can hold it in your head before you change it.
- **Process definitions are your data, not our code** — transition
  matrices are inputs you construct; the repo does not own a process zoo.

## What this buys you

- **Exact-fit optimization.** Tune the batching, memory layout, device
  placement, and JIT boundaries for your workload instead of inheriting
  someone else's tradeoff.
- **No black box.** When something surprises you, you read and instrument
  the actual code in your repo — not a pinned wheel in a site-packages
  directory.
- **Agents can read it.** Coding agents now do much of the
  implementation work, and they understand code they can see far better
  than code hidden behind a dependency. Generative processes are the
  central objects of this research — with their definitions vendored
  into the codebase, a fresh agent session reads a few hundred lines of
  notation-faithful code and understands not just the API but the
  research itself. That context is simply absent when the same
  definitions sit in site-packages.
- **Specialization is deletion.** A library's complexity is largely
  deferred decisions — the class hierarchy, the option flags, the
  `if variant == …` ladders exist so each consumer can choose *later*.
  Copying a module lets you choose *now*: one composition scheme, one
  dtype, one layout. The abstractions that existed only to keep the
  other choices open become dead weight you remove — the multi-level
  hierarchy collapses to the one class you use, the twelve-argument call
  to the two you pass, the dispatch to the branch you're on. What's left
  isn't a configured instance of a general system; it's just the specific
  thing your project does.
- **Freedom to change.** Rename, restructure, specialize, delete. There
  is no upstream API contract to respect and no fork to maintain.
- **Reproducibility.** Your experiment artifacts carry the exact process
  code that generated them. Upstream changes cannot silently alter your
  results; a frozen artifact stays frozen — and it stays frozen without
  freezing anyone else.
- **Research directions decouple.** Each project owns its copy and
  diverges freely. New process types don't need to be merged into a
  shared dependency, reviewed against everyone else's use cases, or
  reconciled with a compatibility policy. The team shares the spec and
  the starting points, not a merge queue.
- **No supply-chain surface.** Nothing to update, nothing to audit on a
  release cadence, no version resolution against the rest of your stack.
- **Upstream moves fast.** Because nothing imports this repo, it carries
  no compatibility burden. Modules here can be improved or reshaped
  freely, and you adopt changes by choice — by diffing and re-copying —
  never by surprise.

## How to consume

1. Copy the modules you need (and their module-level imports — check the
   top of each file; the dependency chains are shallow and explicit).
2. Copy the relevant conformance vectors and wire them into your test
   suite against your copy — `conformance-tests.json` plus
   `tests/test_conformance.py`, whose one import you re-point at your
   copy.
3. Delete unused functions and modules.
4. Adapt: add what your use case needs (belief-trajectory collection,
   batched generation, framing tokens, ...), optimize what it stresses.
5. Re-run conformance. Green means your fork still implements the spec.

To pick up upstream improvements later, diff your copy against the
current module and take what you want. The modules are small enough
that this is a read, not an archaeology project.

## Prior art

This is the [shadcn/ui](https://ui.shadcn.com/docs) distribution model
applied to scientific computing, and it shares a spirit with Go's
vendoring culture and header-only C libraries. The addition that makes
it safe for numerical code is the spec-plus-conformance layer: copies
may diverge arbitrarily in implementation while remaining verifiably
equivalent in mathematics.
