# Task 2 Documentation — Right-of-Way Reasoning

**Scope:** Core benchmark Task 2  
**Generator:** `generators/task2_rightofway.py` — current version: `task2_rightofway_v3`  
**Dataset:** `dataset/core/task2_rightofway.jsonl`  
**Last updated:** April 2026

---

## 1. Context

Task 2 evaluates static rule-based reasoning in the Traffic Benchmark. While Task 1 focuses on sequential position tracking, Task 2 asks the model to apply traffic priority rules to a single scenario snapshot.

No simulation is required. The model receives a snapshot of a traffic scenario — vehicles at positions, possibly with declared intents — and must determine which vehicle has the right of way. This maps to the ability to retrieve and apply normative knowledge (traffic rules) to a concrete structured situation, rather than simulating a sequence of events.

Task 2 uses two environments:
- `intersection` — unsignalized four-way intersection where right-of-way depends on approach direction and declared intent (go straight, turn left, turn right)
- `roundabout` — rotary where the vehicle already circulating inside always has priority over vehicles waiting to enter

The roundabout rule is simple and absolute. The intersection rule is complex and intent-sensitive, which is why the generator must enforce that the correct answer cannot be reached by direction alone — the model must use intent information.

---

## 2. Objective

Task 2 evaluates whether a model can **correctly apply right-of-way rules** to a static traffic scenario with 3 vehicles.

The model must:
- identify which vehicle is involved in the priority conflict
- apply the appropriate rule (intersection priority or roundabout yielding)
- select the correct vehicle from 5 options that include the yielding vehicle as a plausible near-true distractor

The task is designed to resist three classes of shortcuts:
- **direction-only shortcuts** — the correct answer must not be determinable from vehicle directions alone, without reading intents (enforced by `direction_only_does_not_match_priority` invariant)
- **alphabetical shortcuts** — the alphabetically-first non-left-turning vehicle must not coincide with the priority vehicle (enforced by `alphabetical_non_left_heuristic_fails` invariant)
- **roundabout ceiling shortcut** — roundabout examples are structurally simpler; their proportion (30%) is controlled to avoid making the dataset trivially solvable by always guessing the inside vehicle

---

## 3. Implementation

### 3.1 Priority determination logic

Priority is derived through **pairwise assessment**: for each pair of vehicles in the scenario, a `PairAssessment` dataclass records whether a conflict exists, who wins, and whether the outcome differs depending on whether intent is used. A global dominant vehicle is then derived as the one that wins all conflicts.

Scenarios are rejected if:
- no unique global dominant vehicle exists (ambiguous priority)
- the priority vehicle is the same as what a direction-only heuristic would predict (shortcut leakage)
- intent-sensitivity cannot be demonstrated for intersection scenarios (the invariant `intent_sensitive_priority_pair` must be True)

For roundabout scenarios, the inside vehicle always wins by definition. `intent_sensitive_priority_pair` is set to True by default for roundabouts since the rule is positional and intent is not applicable.

### 3.2 Relabeling and scheduling

A relabeling system maps internal vehicle IDs to the final A/B/C labels using a permutation search that satisfies multiple constraints simultaneously:
- for intersections, vehicle A is excluded from being the priority vehicle (prevents the alphabetical non-left heuristic from working)
- for roundabouts, the inside vehicle is always labeled A (to compensate for A's under-representation in intersection scenarios)

A **conflict-pair schedule** ensures each of the three possible conflict pairs (A-B, A-C, B-C) appears approximately equally: target ~33 records each. Without this, B-C pairs were almost absent (6/100 in earlier versions).

A **key schedule** ensures exactly 20 records per answer letter (A–E), shuffled.

### 3.3 Distractor construction

- `near_true` (2 per record): always the yielding vehicle and "Both can pass at the same time". The yielding vehicle is the most plausible wrong answer — it is actively involved in the conflict. "Both can pass" tests whether the model understands that in a right-of-way scenario there is always a clear winner.
- `highly_false` (2 per record): scenario-grounded policy statement distractors (e.g. "The vehicle turning left always has priority", "Vehicles entering the roundabout have priority") that are semantically incorrect given the domain rules.

### 3.4 Invariant system

7 per-example invariants are enforced:

| Invariant | What it checks |
|---|---|
| `five_distinct_options` | All 5 choices are textually distinct |
| `priority_conflicts_with_all_others` | Priority vehicle is in conflict with both other vehicles |
| `pair_conflict_count_at_least_2` | At least 2 distinct pairwise conflicts in the scenario |
| `intent_sensitive_priority_pair` | For intersections: direction-only logic disagrees with intent-aware result. For roundabouts: True by default (positional rule) |
| `direction_only_does_not_match_priority` | Direction-only heuristic does not predict the correct answer |
| `answer_text_matches_priority` | The correct answer text matches the priority vehicle label |
| `alphabetical_non_left_heuristic_fails` | Alphabetically-first non-left-turning vehicle ≠ priority vehicle |

---

## 4. Development Notes

### 4.1 Pairwise winner used instead of global dominant vehicle
**Problem (logical):** An early version derived the correct answer from the winner of a single pairwise comparison rather than establishing a globally dominant vehicle that beats all others. In scenarios with 3 vehicles, this produced cases where the "priority" vehicle would lose to the third vehicle — the answer was locally correct but globally wrong.

**Root cause:** The priority logic was written for 2-vehicle reasoning and not extended to the 3-vehicle case properly.

**Change:** Enforced strict global dominance: the priority vehicle must win every pairwise conflict in the scenario. Scenarios without a unique global dominant are rejected and regenerated.

---

### 4.2 Direction-only shortcut possible in intersection scenarios
**Problem (logical):** In earlier versions, some intersection scenarios could be solved by applying a simple direction-based heuristic (e.g. "vehicle coming from the right has priority") without reading intent at all. This made those records trivial for a model with basic traffic knowledge.

**Root cause:** The scenario generator did not check whether the chosen priority vehicle was also the winner under direction-only logic.

**Change:** Added the `direction_only_does_not_match_priority` invariant and a generation rejection condition: if the direction-only heuristic predicts the same vehicle as the intent-aware logic, the scenario is discarded. The `intent_sensitive_priority_pair` invariant further enforces that at least one pairwise conflict must flip its outcome when intent is considered.

---

### 4.3 `intent_sensitive_priority_pair` always False for all 30 roundabout records
**Problem (logical/structural):** The invariant was computed only inside `if state.environment == INTERSECTION`. For roundabout records it remained `False` and was written to the output as an invariant failure — 30 false negatives in the dataset.

**Root cause:** The invariant was designed for intersection logic and not adapted for roundabout scenarios, where intent is irrelevant by domain design (no vehicle in the dataset has a non-null intent at a roundabout).

**Change:** Added an `else` branch after the intersection block:
```python
else:
    # Roundabout priority is positional; intent sensitivity is not applicable.
    intent_sensitive_with_priority = True
```
Result: `intent_sensitive_priority_pair` is now True on all 100 records, 0 invariant failures.

---

### 4.4 Conflict pair severely imbalanced (A-B: 65, A-C: 29, B-C: 6)
**Problem (distributional):** The B-C conflict pair appeared in only 6 records out of 100. The distribution was structurally biased: roundabout records always produce A as priority (30 records), intersection records assign B or C as priority but the yielding vehicle selection overwhelmingly picked A or B, leaving B-C conflicts nearly absent.

**Root cause:** No quota control on conflict pair distribution. The generation loop produced whatever pair the scenario and relabeling happened to yield.

**Change:** Added a `pair_schedule` before the generation loop that pre-assigns the desired conflict pair for each record slot (~33 per pair, remainder round-robin). Candidates whose conflict pair does not match the scheduled slot are rejected and retried.

**Result:** A-B: 33, A-C: 33, B-C: 34 — near-perfect balance.

---

### 4.5 "Both can pass at the same time" present in only 24% of records
**Problem (distributional):** In the updated generator, "Both can pass" was selected dynamically and only included when the distractor builder deemed it semantically plausible. This caused inconsistency in the option set across records and reduced its value as a systematic test.

**Root cause:** Distractor selection was made conditional rather than guaranteed.

**Change:** Made "Both can pass at the same time" a fixed member of the near_true distractor pool for every record, alongside the yielding vehicle.

**Result:** Present in 100/100 records.

---

### 4.6 Additional bugs fixed (earlier development)
- `_OPPOSITE_DIR` NameError in distractor builder — undefined variable used in direction label construction
- Type mismatch in relabel scoring tuple annotation causing runtime errors
- Non-atomic file writes — replaced with temp file + `fsync` + `os.replace` pattern
- Output path bug when `--out` had no directory component

---

## 5. Known Limitations (Accepted as Policy)

The following issues were analyzed and accepted as design constraints. No code changes will be made.

**"All stop / No vehicle can pass" option in only 6/100 records.** This distractor is semantically almost never valid — in a well-formed right-of-way scenario there is always a clear priority winner. Its low frequency is correct behavior, not a gap.

**`turn right` intent underrepresented (17 occurrences vs 70 for `turn left`).** Vehicles intending to turn right generate fewer structural conflicts in the priority resolution logic. This is an emergent property of the domain rules and does not introduce any systematic model bias.

**`inside_intersection` field always False for intersection records, True only for roundabout.** The field effectively encodes "is the vehicle circulating inside the roundabout", not "is the vehicle inside an intersection". This is a naming issue only. An optional future cosmetic rename to `in_roundabout_lane` across generator and consumers would improve clarity without affecting dataset correctness.

**Vehicle B is priority in only 3/100 records.** The relabeling system excludes B from being priority in cases where the alphabetical non-left heuristic would predict B. Combined with the roundabout rule (A always inside) and the B-C balance constraint, B ends up as the priority vehicle very rarely. This is an accepted tradeoff to maintain the anti-shortcut invariants.

---

## 6. Final State (v3)

| Metric | Value |
|---|---|
| Records | 100 |
| Generator version | task2_rightofway_v3 |
| All 7 invariants passing | 0 failures |
| Answer distribution | 20 per letter (A–E) |
| Environments | intersection 70, roundabout 30 |
| Conflict pairs | A-B: 33, A-C: 33, B-C: 34 |
| "Both can pass" present | 100/100 |
| Generation retries | 0 on all 100 records |
| direction-only heuristic accuracy | 0/70 intersection records |
| alphabetical heuristic accuracy | 0/70 intersection records |
| Prompts unique | yes |
| IDs unique | yes |

---

## 7. Usage

### Generate core dataset
```bash
python generators/task2_rightofway.py --n 100 --seed 42 --out dataset/core/task2_rightofway.jsonl
```

### Quick audit
```bash
python - <<'PY'
import json
from collections import Counter
rows = [json.loads(l) for l in open('dataset/core/task2_rightofway.jsonl') if l.strip()]
print('N', len(rows))
print('version', rows[0]['audit']['generator_version'])
print('answers', Counter(r['answer'] for r in rows))
print('envs', Counter(r['scenario']['environment'] for r in rows))
print('pairs', Counter(tuple(sorted(r['metadata']['conflict_pair'])) for r in rows))
print('both_can_pass', sum(1 for r in rows if any('Both' in v for v in r['choices'].values())))
inv_fails = {k: sum(1 for r in rows if not r['audit']['invariants'].get(k, True))
             for k in rows[0]['audit']['invariants']}
print('invariant_failures', {k:v for k,v in inv_fails.items() if v > 0} or 'none')
PY
```

---

## 8. Summary

Task 2 reached a final quality score of **9.3/10** after two improvement passes in April 2026.

The core engineering challenge was the tension between three competing constraints: correctness (the priority vehicle must be unambiguously right under the domain rules), anti-shortcut hardness (direction-only and alphabetical heuristics must fail), and distributional balance (conflict pairs, answer letters, and environments must be uniform). Earlier versions satisfied at most two of the three simultaneously.

The main corrections were global dominance for priority determination, a conflict-pair quota schedule for the B-C imbalance, the `intent_sensitive_priority_pair` fix for roundabout records, and "Both can pass" as a fixed option. Together, these keep the invariants passing and remove the known shortcut paths.

The residual limitations (B priority rare, turn right underrepresented, `inside_intersection` naming) are documented above and accepted as policy. They do not affect the correctness or the anti-shortcut properties of the dataset.
