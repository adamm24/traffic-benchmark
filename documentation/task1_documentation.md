# Task 1 Documentation — Position Tracking

**Scope:** Core benchmark Task 1  
**Generator:** `generators/task1_position.py` — current version: `task1_position.v14`  
**Dataset:** `dataset/core/task1_position.jsonl`  
**Last updated:** April 2026

---

## 1. Context

Task 1 is part of the Traffic Benchmark, developed during a Data Science internship to evaluate sequential and structured reasoning in Large Language Models (LLMs).

The broader benchmark is built around a road traffic scenario: multiple vehicles move and interact according to precise rules. The dataset is organized into a **Core Dataset** of 100 examples per task (expandable to 300), saved in JSONL format — one JSON object per line — with 5-option multiple choice questions (1 correct, 2 near-true, 2 highly-false).

Task 1 is the purest state-tracking task in the benchmark. It focuses exclusively on spatial reasoning and sequential state updates, without invoking traffic rules or violation logic (those belong to Task 2 and Task 3). It uses two environments: `intersection` and `multi_lane_road`. The roundabout environment is deliberately excluded from Task 1 because its internal position space is less distinguishable and more suited to overlap-based tasks.

---

## 2. Objective

Task 1 evaluates the ability of a model to **track the position of multiple vehicles across a sequence of actions** and correctly identify where a specific vehicle is at the end of the sequence.

The model must:
- parse an initial configuration of 3 vehicles in distinct positions
- mentally simulate each action in order, updating the internal state
- maintain independent tracking of each vehicle without confusion
- select the correct final position from 5 options, avoiding plausible-but-wrong distractors

The task is designed to resist two classes of shortcuts:
- **label shortcuts** — the correct answer must not be guessable from the vehicle's starting position or from the alphabetical order of the answer letters
- **pattern shortcuts** — event sequences must not be mechanical or repetitive enough for a model to exploit structural regularities without actually tracking state

---

## 3. Implementation

### 3.1 Domain layer

A shared `domain/` module was built as the single source of truth for all generators:

- `entities.py` — defines `Vehicle`, `ScenarioState`, and enums for `Action`, `Environment`, `Direction`, `IntentDirection`
- `rules.py` — implements traffic rules: right-of-way at intersections, roundabout priority, violation detection
- `scenario.py` — scenario builders (`build_intersection_scenario`, `build_multi_lane_scenario`) and the simulation engine `apply_action()`
- `vocabulary.py` — controlled vocabulary: maps internal position keys (e.g. `east_exit`) to human-readable labels (e.g. `the eastern exit`). This is the only source of allowed option strings across the entire benchmark.

### 3.2 Generator design

The generator (`generators/task1_position.py`) produces examples through a constrained simulation loop:

1. A joint **difficulty × environment schedule** is computed before generation, ensuring exactly 16–17 records per cell (easy/medium/hard × intersection/multi_lane_road)
2. A **key schedule** ensures exactly 20 records per answer letter (A–E), shuffled
3. For each slot, a scenario is built, a plan is constructed step by step, and the full example is validated against 13 invariants
4. If validation fails, the example is retried up to `MAX_RETRIES = 150`

### 3.3 Difficulty tiers

Three difficulty tiers are defined via per-environment profiles:

- **easy** — 3–4 events, 1–2 queried moves, minimal noise, no interleaving required
- **medium** — 3–5 events, 2–3 queried moves, some interleaving, actor diversity required
- **hard** — 4–5 events, 3+ queried moves, mandatory interleaving, multi-step non-queried trajectories required

### 3.4 Distractor construction

- `near_true` — sampled from positions actually visited in the scenario: queried vehicle's start position, intermediate positions, or other vehicles' final positions. Verified via `near_true_grounded_in_visited` invariant.
- `highly_false` — reachable positions in the same environment that were never visited by any vehicle. Verified via `highly_false_reachable_never_visited` invariant. Shoulders in multi-lane are downweighted to avoid overuse.

### 3.5 Invariant system

13 per-example invariants are enforced:

| Invariant | What it checks |
|---|---|
| `start_ne_final` | Queried vehicle's final position differs from start |
| `queried_moved` | Queried vehicle performed at least 1 action |
| `no_abab_vehicle_pattern` | Plan does not alternate strictly A-B-A-B |
| `plan_actor_diversity_ok` | At least 2 distinct vehicles act |
| `plan_action_diversity_ok` | At least 2 distinct action types used |
| `nonqueried_interaction_ok` | Non-queried vehicles have meaningful moves |
| `queried_interleaving_ok` | Queried vehicle moves are interleaved with others (hard) |
| `no_action_streak_len3` | No 3 consecutive identical actions |
| `no_vehicle_palindrome` | Actor sequence is not a palindrome |
| `near_true_grounded_in_visited` | All near-true options come from visited states |
| `highly_false_reachable_never_visited` | All highly-false options were never visited |
| `all_labels_in_vocabulary` | All choice strings come from `vocabulary.py` |
| `five_distinct_options` | All 5 choices are textually distinct |

---

## 4. Development Notes

### 4.1 Vocabulary split between internal keys and display labels
**Problem (logical):** The internal simulation uses snake_case position keys (`east_exit`, `south_approach`), while the prompt and choices display human-readable labels (`the eastern exit`, `the southern approach`). An external audit script comparing raw trace strings to choice text found 87/100 records with apparent mismatches. This initially appeared to be a critical data integrity issue.

**Root cause:** The mismatch is by design — `vocabulary.py` maps `east_exit` → `the eastern exit` via `label_of()`. The generator uses this function correctly throughout. The `near_true_grounded_in_visited` invariant passes because it also uses `label_of()` internally. The apparent mismatches were false positives produced by a naive string comparison that did not apply the vocabulary mapping.

**Resolution:** No code change required. The architecture is correct. External quality checks must use `label_of()` for comparison, not raw string matching.

---

### 4.2 High generator retry rate (36% of records needed >1 attempt, max 22)
**Problem (structural):** The plan builder was entirely reactive — it built a complete plan and then ran all 13 invariants post-hoc. If any invariant failed, the entire plan was discarded and rebuilt from scratch. For hard-difficulty examples with many simultaneous constraints (palindrome, interleaving, actor diversity, action streak), this caused frequent full restarts. One record required 22 attempts.

**Root cause:** No early pruning during plan construction. Bad branches were only detected after being fully built.

**Change:** Moved three invariant checks into the plan construction loop as in-loop guards:
- **Action streak guard** — if the last 3 steps have the same action, prune immediately
- **ABAB alternation guard** — if the last 4 steps alternate vehicle identities A-B-A-B, prune immediately
- **Action diversity lookahead** — if the plan is complete but has fewer than `MIN_DISTINCT_ACTIONS`, reject before committing

**Result:** Retry rate dropped from 36% to 11%, max attempts from 22 to 2. The post-hoc `_validate_example()` call was preserved as the authoritative correctness check.

---

### 4.3 STOP action underrepresented
**Problem (distributional):** The STOP action appeared in only 17 out of ~420 total actions (4%), compared to ~90 occurrences for CHANGE_LEFT. For non-queried intersection vehicles, STOP was always removed from the action pool. For the queried vehicle, it was front-prioritized with only a 20% probability.

**Root cause:** STOP was treated as a special case and structurally deprioritized in action pool construction to avoid trivial sequences.

**Change:** Raised the STOP front-prioritization probability from 0.20 to 0.35 for the queried vehicle.

After the in-loop streak guard was added, this probability increase was partly offset because the guard prunes some STOP sequences early. A second pass raised STOP to 34 occurrences (7.8%), which is acceptable for the core set.

---

### 4.4 Easy tier environment imbalance (intersection 20 vs multi_lane_road 14)
**Problem (distributional):** The difficulty schedule and environment schedule were built and shuffled independently, leaving their joint distribution uncontrolled. The easy tier ended up with 43% more intersection records than multi_lane_road.

**Root cause:** Two independent schedules whose joint distribution is random.

**Change:** Replaced the two independent schedules with a single joint difficulty × environment schedule. Each (difficulty, environment) cell is assigned exactly `n // (n_difficulties × n_environments)` slots, with the remainder distributed round-robin. The joint list is shuffled as a whole, then unzipped back into separate schedules so downstream code is unchanged.

**Result:** Every cell now has exactly 16 or 17 records (target: 100 / 6 ≈ 16.7).

---

## 5. Final State (v14)

| Metric | Value |
|---|---|
| Records | 100 |
| Generator version | task1_position.v14 |
| All 13 invariants passing | 0 failures |
| Answer distribution | 20 per letter (A–E) |
| Difficulty × environment | 16–17 per cell (6 cells) |
| Records needing >1 attempt | 11% (max 2 attempts) |
| STOP action share | 7.8% |
| CHANGE_LEFT/RIGHT share | ~22–23% each |
| MOVE_FORWARD share | 22.5% |
| TURN_LEFT/RIGHT share | ~11–14% each |
| Prompts unique | yes |
| IDs unique | yes |

---

## 6. Usage

### Generate core dataset
```bash
python generators/task1_position.py --n 100 --seed 13 --out dataset/core/task1_position.jsonl
```

### Quick audit
```bash
python - <<'PY'
import json
from collections import Counter
rows = [json.loads(l) for l in open('dataset/core/task1_position.jsonl') if l.strip()]
print('N', len(rows))
print('version', rows[0]['audit']['generator_version'])
print('answers', Counter(r['answer'] for r in rows))
print('envs', Counter(r['scenario']['environment'] for r in rows))
print('difficulty', Counter(r['metadata']['difficulty'] for r in rows))
print('max_attempts', max(r['audit']['attempt'] for r in rows))
inv_fails = {k: sum(1 for r in rows if not r['audit']['invariants'].get(k, True))
             for k in rows[0]['audit']['invariants']}
print('invariant_failures', {k:v for k,v in inv_fails.items() if v > 0} or 'none')
PY
```

---

## 7. Summary

Task 1 reached a final quality score of **9.5/10** after three improvement passes.

All 13 invariants pass on every record, the distribution is controlled across answers, difficulties and environments, and the retry rate remains low enough (11%, max 2 attempts) to show that the constraints are feasible.

The key engineering challenge was balancing the richness of the invariant system — which is necessary to prevent shortcut exploitation — against the practical efficiency of the generator. The solution was to shift invariant checks from post-hoc validation to in-loop pruning where possible, reducing wasted work without relaxing any correctness guarantees.

The main residual tradeoff is that TURN_RIGHT (11%) remains slightly below TURN_LEFT (13.6%) and the lane-change actions (~22%). This asymmetry is structural: right turns have fewer valid application contexts in intersection scenarios by domain design, and is not considered a quality issue.
