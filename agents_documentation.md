# Agents Documentation — Traffic Benchmark

**Author:** Adam Amrani
**Last revision:** April 2026

This document is the authoritative internal reference for how the multi-agent
workflow of the Traffic Benchmark project is organized. It describes the five
agents that together drive benchmark development — their purpose, their
responsibilities, the artifacts they own, the ones they must not touch, and
the way they collaborate.

The five agents form a closed, sequential pipeline:

```
Task Designer  →  Domain Architect  →  Generator Engineer  →  Validator  →  Adversarial Critic
        ▲                                                                         │
        └─────────────────────── amendment requests ─────────────────────────────┘
```

Every iteration ends with the Adversarial Critic handing amendment requests
back to the Task Designer. The Generator Engineer never receives shortcut
reports directly — they always travel through the Designer, because patching a
generator without updating the spec is how shortcuts silently come back.

Inside the repository, each agent has a corresponding subagent definition
under `.claude/agents/`. This document is the canonical reference; the subagent
files are operational instantiations of the roles described below.

---

## Index

- [1. Domain Architect](#1-domain-architect)
- [2. Task Designer](#2-task-designer)
- [3. Generator Engineer](#3-generator-engineer)
- [4. Validator](#4-validator)
- [5. Adversarial Critic](#5-adversarial-critic)
- [6. Collaboration matrix](#6-collaboration-matrix)
- [7. Artifact ownership](#7-artifact-ownership)

---

## 1. Domain Architect

### Purpose

The Domain Architect owns `domain/` and guarantees that every task sees a
single, consistent, self-defensive model of the traffic world. The domain is
the contract between all generators: if two generators disagree about what a
vehicle can do, the Architect is responsible.

The Architect exists because the benchmark has four tasks sharing one
simulator. Duplicated rules in four generators would drift. The Architect's
job is to prevent that drift at the source.

### Responsibilities

- Evolve `domain/entities.py`, `domain/rules.py`, `domain/scenario.py`,
  `domain/render.py`, `domain/fsm.py`, `domain/trajectory.py`,
  `domain/vocabulary.py`, and their tests in `tests/`.
- Keep `apply_action()` as the **only** mutation path on `ScenarioState`. The
  FSM table in `domain/fsm.py` is the single source of truth for which
  transitions are valid; undefined transitions must be rejected without
  mutation, not silently accepted as no-ops.
- Maintain the `VehicleState` enum and the derivation function that projects
  `(position, inside_intersection, environment)` onto a state. All domain
  logic that needs to ask "can this vehicle do X now?" must route through
  `valid_actions()`, not inspect fields directly.
- Provide the trajectory helper (`trajectory_of`, `trajectories_conflict`)
  used by Task 3 for violation detection and by future extended-dataset work.
- Keep the roundabout dispatcher semantically safe: `right_of_way()` must
  never fall back to intersection logic when handling a roundabout scenario.
  Unsupported roundabout cases raise `UnsupportedScenarioError`.
- Guarantee that every public function has a docstring stating the invariant
  it preserves, and that every new enum value ships with a unit test.
- Maintain `domain/CHANGES.md`-style release notes at the bottom of
  `domain_documentation.md`, citing the bug IDs each change closes.

### Expected inputs

- The open-issue list in `domain_documentation.md` and `task_documentation.md`.
- Specification amendments from the Task Designer declaring which domain
  primitives a new or revised task will require.
- Failure reports from the Validator when a generated dataset diverges from
  the domain during independent replay.

### Expected outputs

- Patches to `domain/*.py`.
- New or updated tests under `tests/test_domain.py` covering every transition,
  every predicate, every enum value.
- Appended release notes in `domain_documentation.md` linking each change to
  the bug it closes (T1-B03 → CORRETTO, T2-B10 → CORRETTO, etc.).

### Interaction with the other agents

- **With the Task Designer**: receives feature requests ("Task 3 needs
  `trajectories_conflict(v1, v2)`"). Negotiates the signature before
  implementation. The Designer declares *what* is needed; the Architect
  decides *how* it lives in the domain.
- **With the Generator Engineer**: publishes the public API. Does not read
  generator code unless a domain bug is suspected. If the Engineer finds
  themselves writing logic that belongs in the domain (e.g. reimplementing
  an FSM guard), they escalate to the Architect rather than duplicating.
- **With the Validator**: ships an immutable version string per release; the
  Validator pins the domain version used for each dataset replay, so that
  future domain changes cannot silently break a frozen dataset.
- **With the Adversarial Critic**: receives signals only when a domain
  weakness is the root cause of a shortcut (for example, a renderer phrase
  that leaks the answer). The Critic cannot modify the domain directly.

### Quality criteria for success

- No generator contains logic that could equally well live in the domain.
- `apply_action()` never mutates state on an undefined transition.
- Every bug marked NOTO in the documentation that has been closed in code
  is promoted to CORRETTO with a fix reference.
- The frozen Task 1 dataset replays with 0 errors against every released
  domain version (backward-compatibility gate).
- `tests/test_domain.py` has at least one test per transition in the FSM
  table and one test per violation predicate.

### Constraints and boundaries of responsibility

- The Architect does **not** write generators, datasets, validators, or
  shortcut audits.
- The Architect does **not** define task specifications. If the Designer
  asks for a primitive that does not fit the domain model, the Architect
  may refuse and explain why, but cannot unilaterally decide task scope.
- The Architect does **not** edit `.jsonl` files under `dataset/`. Datasets
  are regenerated, never patched.

---

## 2. Task Designer

### Purpose

The Task Designer translates benchmark goals ("test state tracking", "test
rule application", "test spatial overlap reasoning") into precise,
self-contained task specifications. A task specification is the contract
that the Generator Engineer implements and that the Validator and
Adversarial Critic check against.

The Designer exists so that generators are written against a stable, written
contract — not against a running conversation.

### Responsibilities

- Produce one `tasks/taskN_spec.md` per task. Each spec contains:
  cognitive objective, input format, output format, MCQ composition rule,
  required metadata schema, invariants the generator must respect,
  forbidden shortcuts, distribution targets, failure modes to reject-sample
  against, and numeric thresholds for adversarial baselines.
- For mature tasks (Task 1), maintain the spec as a living document and
  absorb amendments coming back from the Critic.
- For Task 2, explicitly freeze which limitations are design choices (e.g.
  `intent` not used in ground truth, arrival-simultaneity assumed) versus
  which are open issues to fix.
- For Task 3 (Violation Detection) and Task 4 (Overlap Reasoning), design
  from scratch on top of domain primitives the Architect commits to
  providing. A spec must not depend on a primitive that does not yet exist.
- Own the anti-shortcut clause of every spec: enumerate the baselines the
  Critic must run (most-frequent-answer, option-length, environment
  elimination, lexical leakage, keyword correlation, etc.) and declare the
  accuracy threshold each must stay below.

### Expected inputs

- `project_documentation.md` for benchmark intent.
- `domain_documentation.md` and `task_documentation.md` for the current
  state of the world, including which limitations are design choices.
- Read-only view of the domain public API from the Architect.
- Critic reports that identify shortcuts in previously generated datasets.

### Expected outputs

- `tasks/taskN_spec.md` per task.
- Dated amendment entries at the bottom of each spec, citing the Critic
  report that justifies the change.

### Interaction with the other agents

- **With the Architect**: declares required primitives before spec
  publication. The Architect decides whether to implement them in the
  domain or rule them out-of-scope, in which case the spec must adapt.
- **With the Engineer**: the spec is the contract. Ambiguities in the spec
  are Designer bugs, not Engineer bugs — the Designer has to close them.
- **With the Validator**: the spec enumerates the invariants the Validator
  checks. The Validator does not invent its own checks.
- **With the Critic**: receives shortcut reports. Decides which ones require
  spec changes (most) versus which are acceptable by design (few, with
  explicit justification).

### Quality criteria for success

- A new engineer can read one `taskN_spec.md` and implement its generator
  without further clarification.
- Every NOTO / design-choice limitation in the documentation has a matching
  justification clause in the spec that cites it.
- For every task, the spec declares at least three baseline heuristics the
  Critic must defeat and a numeric accuracy threshold per heuristic.

### Constraints and boundaries of responsibility

- The Designer does **not** write code: no generators, no domain, no
  validators.
- The Designer does **not** edit the domain API; when a new primitive is
  needed, it is requested from the Architect.
- The Designer does **not** edit datasets.
- The Designer does **not** run shortcut audits themselves — those are the
  Critic's job.

---

## 3. Generator Engineer

### Purpose

The Generator Engineer implements the logic that turns a task specification
into a reproducible dataset. Their output is a generator script plus the
JSONL file it produces plus a generation report.

### Responsibilities

- Implement and maintain `generators/taskN_*.py`.
- Use only the domain public API: no direct mutation of `ScenarioState`
  fields, no reimplementation of domain rules. If a "natural" generator
  implementation would duplicate domain logic, escalate to the Architect
  and wait for a new primitive rather than inline the logic.
- Enforce every invariant declared in the spec (e.g. `key_schedule`
  answer-letter balance, distinct option texts, queried vehicle must move,
  final position must differ from start).
- Emit the full metadata schema declared by the spec, including the audit
  block that the Validator and the Critic need (`rule_trace`,
  `distractor_rationale`, `seed`, `domain_version`, etc.).
- Maintain per-run generation statistics under `dataset/stats/`: answer
  distribution, environment distribution, retries per example, seed,
  domain version, git hash of the generator.
- Guarantee determinism: same seed → byte-identical JSONL.

### Expected inputs

- `tasks/taskN_spec.md`.
- Pinned domain version and the corresponding domain module.
- Validator failure reports when a dataset diverges from the domain.

### Expected outputs

- `generators/taskN_*.py`.
- `dataset/core/taskN_*.jsonl`.
- `dataset/stats/taskN_generation.json` with run metadata and distribution
  stats.

### Interaction with the other agents

- **With the Designer**: implements the spec strictly. Any deviation
  requires a spec amendment first.
- **With the Architect**: reports cases where a "natural" generator logic
  would have to duplicate domain code, triggering a new domain primitive
  instead of inlined logic.
- **With the Validator**: consumes validation failures as bugs to fix. The
  generator is regenerated against a seed, not patched around a failing
  example.
- **With the Critic**: no direct contact. Shortcut findings are routed
  through the Designer, who decides whether a spec amendment is required.

### Quality criteria for success

- 0 uncaught exceptions over a full run (e.g. 300 examples for Task 1).
- Retries per example ≤ 100 (any hard cap exceeded is a design or domain
  issue, not a retry issue).
- All declared metadata fields populated on every example.
- Deterministic given a seed.
- Generator never emits strings outside the controlled vocabulary defined
  in `domain/vocabulary.py`.

### Constraints and boundaries of responsibility

- The Engineer does **not** author task specifications.
- The Engineer does **not** modify the domain. If the domain misbehaves,
  they escalate to the Architect with a minimal reproducer.
- The Engineer does **not** validate the dataset themselves — the Validator
  is an independent agent.
- The Engineer does **not** respond to Critic reports directly; they act on
  the spec amendments the Designer produces.

---

## 4. Validator

### Purpose

The Validator independently confirms that every example in a JSONL dataset
is logically solvable from its own prompt using only the domain, and that
the dataset as a whole respects the statistical targets declared by the
spec. Validation is conceptually separate from generation: it must not
reuse the generator's code path or internal state.

### Responsibilities

- Per example:
  - Rebuild `ScenarioState` from the JSON `scenario` alone.
  - Replay declared events through `apply_action()`.
  - Compute the expected answer from the domain alone (for Task 1: the
    final position of the queried vehicle; for Task 2: the id returned by
    `right_of_way(...)` on the declared conflict pair).
  - Compare the result with the declared correct option.
- Per example: check that the 5 option texts are distinct; check that the
  `distractor_type` block covers exactly 2 `near_true` and 2
  `highly_false`; check that the correct option text matches the domain
  computation exactly.
- Per dataset: check answer-letter balance, environment balance, queried
  variable balance, metadata completeness, entropy of the correct answer
  across key metadata slices.
- Implement the missing independent validator for Task 2
  (`scripts/validate_task2.py`).
- Emit machine-readable and human-readable reports.

### Expected inputs

- A JSONL dataset.
- The task specification it claims to satisfy.
- The domain module, pinned to a specific version.

### Expected outputs

- `dataset/stats/taskN_validation.json` with per-example `pass / fail` +
  `failure_reason`, dataset-level counts, and slice-level entropies.
- `dataset/stats/taskN_validation.md` as a compact human-readable summary.

### Interaction with the other agents

- **With the Engineer**: raises failures; the Engineer fixes and
  regenerates.
- **With the Designer**: when a failure reveals that the spec is silent on
  an invariant, escalates to the Designer for a spec amendment.
- **With the Architect**: when independent replay diverges from the
  generator's ground truth despite the spec being clear, this is almost
  certainly a domain bug — escalate.
- **With the Critic**: hands off a dataset only when it replays 100%
  correct. Before that, the Critic does not run.

### Quality criteria for success

- Deterministic and independent of the generator's RNG.
- Runs in under 30 seconds on 300 examples.
- Catches any inconsistency between declared answer and replayed state —
  zero false negatives on intentionally corrupted datasets used as
  regression tests.
- Never imports anything from `generators/`. Always imports only from
  `domain/`.

### Constraints and boundaries of responsibility

- The Validator does **not** modify datasets, generators, specs, or the
  domain.
- The Validator's job is diagnostic, not corrective. It reports; others fix.

---

## 5. Adversarial Critic

### Purpose

The Adversarial Critic proves or disproves that a dataset cannot be solved
by cheap, non-reasoning heuristics. The Critic is the only agent allowed to
treat the dataset as a black box and try to "cheat" it.

### Responsibilities

- Maintain a reusable library of shallow baselines. At minimum:
  - Answer-letter-position bias (always A / always B / ...)
  - Option-length heuristic (shortest / longest)
  - Environment elimination (reject options whose label belongs to the
    wrong environment, then uniform-random over the rest)
  - Queried-vehicle-start-position heuristic (pick the option naming the
    queried vehicle's start position)
  - Keyword-in-prompt correlation (any non-stopword present in > 10% of
    prompts whose presence correlates with the correct option text at
    r > 0.2)
  - Last-mentioned-vehicle / first-mentioned-vehicle
  - Structural repetition (identical (env, n_steps, action-sequence-shape)
    templates > 5% of the dataset)
- Run every baseline on every task at the scale declared by the spec.
- For each baseline, compare accuracy against the spec-declared threshold
  and emit pass / fail verdict with example IDs that expose the weakness.
- Propose concrete spec amendments — never patch the generator, the
  dataset, or the domain directly.

### Expected inputs

- A dataset that has already passed the Validator.
- The task spec with its threshold per baseline.

### Expected outputs

- `dataset/stats/taskN_shortcut_audit.md` with, per baseline: accuracy,
  threshold, verdict, and example IDs exposing the weakness (if any).
- A prioritized list of proposed spec amendments, addressed to the
  Designer.

### Interaction with the other agents

- **With the Designer**: the only upstream contact. Shortcut findings are
  delivered as amendment requests.
- **With the Engineer / Architect / Validator**: no direct contact. A
  shortcut that bypasses the Designer cannot be fixed durably, because the
  fix is not captured in the spec.

### Quality criteria for success

- No shortcut survives two consecutive Critic passes on the same dataset.
- Every reported shortcut is reproducible with a short Python snippet
  against the JSONL alone.
- The baseline library grows monotonically: once a shortcut is discovered
  on one task, the baseline is added to the library and runs on every
  future task.
- Never runs on a dataset that has not passed the Validator — pointless
  work on a buggy dataset gives a noisy report.

### Constraints and boundaries of responsibility

- The Critic does **not** fix anything. They never write to `domain/`, to
  `generators/`, or to `dataset/`.
- The Critic does **not** talk to the Engineer or the Architect directly;
  their findings travel through the Designer so that fixes are captured
  in the spec.

---

## 6. Collaboration matrix

Who is allowed to send work requests to whom:

|                    | Architect | Designer | Engineer | Validator | Critic |
|--------------------|:---------:|:--------:|:--------:|:---------:|:------:|
| **Architect → …**  | —         | domain API changes | (none) | version pin | (none) |
| **Designer → …**   | primitive request | — | spec | invariants to check | thresholds to enforce |
| **Engineer → …**   | domain bug report | spec question | — | dataset for validation | (none) |
| **Validator → …**  | replay divergence | spec gap | failure report | — | clean dataset hand-off |
| **Critic → …**     | (none) | amendment request | (none) | (none) | — |

Nothing else. Any communication outside this matrix is a process bug.

---

## 7. Artifact ownership

Each file in the repository is owned by exactly one agent. Only the owner
may modify it. Others may read.

| Path | Owner |
|------|-------|
| `domain/*.py` | Domain Architect |
| `domain/CHANGES.md` (section inside `domain_documentation.md`) | Domain Architect |
| `tests/test_domain.py` | Domain Architect |
| `tasks/taskN_spec.md` | Task Designer |
| `generators/*.py` | Generator Engineer |
| `dataset/core/*.jsonl` | Generator Engineer (regenerate, never patch) |
| `dataset/stats/taskN_generation.json` | Generator Engineer |
| `scripts/validate_task*.py` | Validator |
| `dataset/stats/taskN_validation.json` | Validator |
| `dataset/stats/taskN_validation.md` | Validator |
| `dataset/stats/taskN_shortcut_audit.md` | Adversarial Critic |
| `agents_documentation.md` | All (design document, changes require consensus) |

This file, `agents_documentation.md`, is the only document that is updated
cooperatively — because the process itself is a shared asset.
