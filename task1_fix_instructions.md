# Task 1 — Position Tracking: Remaining Problems and Solutions

These are the 6 remaining problems in `generators/task1_position.py` and their solutions.
Apply all fixes, then run the generator and validate the output quality.

**Important context:**
- The generator is at `generators/task1_position.py`
- The vocabulary module is at `domain/vocabulary.py`
- The domain logic is at `domain/scenario.py` and `domain/entities.py`
- Task 1 uses only two environments: `multi_lane_road` (3 lanes) and `intersection` (9 positions)
- Each example has 5 answer choices: 1 correct, 2 near_true, 2 highly_false
- The current code imports `cross_env_labels`, `is_valid_label`, `label_of`, `labels_for_env`, `positions_for_env` from `domain.vocabulary`

---

## Problem 1 — Cross-environment distractors still present

**What is wrong:**
`build_choices()` intentionally uses `cross_env_labels(env)` to pick `highly_false_1`. This means intersection examples can have distractors like "the left lane" or "the right lane", and multi_lane examples can have distractors like "the northern exit" or "inside the intersection". These are trivially eliminable by environment knowledge alone, without any state tracking.

When `highly_false_2` can't find a same-environment unvisited label, it also falls back to a second cross-environment label, making it even worse.

**Where in the code:**
- `build_choices()` in `generators/task1_position.py`, lines ~273-280 (`highly_false_1` section)
- `build_choices()`, lines ~299-312 (`highly_false_2` fallback section)

**Solution:**
1. Remove all use of `cross_env_labels()` from the generator. Every distractor must come from `labels_for_env(env)` only.
2. The problem is that `multi_lane_road` has only 3 labels in the vocabulary (`the left lane`, `the center lane`, `the right lane`), so you cannot fill 5 slots from same-environment labels alone. To fix this, **expand `domain/vocabulary.py`** by adding dedicated highly_false labels for multi_lane_road. These must be plausible road-context positions that are never reachable in the simulation but belong to the same domain. Add them as a new category in the vocabulary module.
   Suggested labels for multi_lane highly_false: `"the road shoulder"`, `"the emergency lane"`, `"the median strip"`, `"the oncoming lane"`.
   These should be registered in `vocabulary.py` as a separate pool (e.g. `HIGHLY_FALSE_LABELS_BY_ENV`) — not as regular positions — so they are part of the controlled vocabulary but never appear as correct answers.
3. For `intersection` (9 labels), there are always enough same-environment labels for all 5 slots. No new labels needed.
4. If the generator still cannot produce 5 distinct valid labels after using the expanded vocabulary, it must reject the example and retry. Never fall back to cross-environment labels.

---

## Problem 2 — Audit detects problems but does not block them

**What is wrong:**
The `audit.rationale_by_letter` field correctly tags cross-environment distractors with rationales like `"cross-environment label (not valid for intersection)"`. But this detection is purely informational — the example is still saved to the dataset. The audit is passive annotation, not an active gate.

**Where in the code:**
- `generate_example()` in `generators/task1_position.py`, lines ~463-508 (the audit block is built but never used for rejection)

**Solution:**
1. Add a `validate_example()` function that runs after the example is fully constructed but before it is returned.
2. This function must check all quality constraints. If any check fails, `generate_example()` must return `None` (triggering a retry), not save the example.
3. Minimum checks for `validate_example()`:
   - All 5 choice texts belong to `labels_for_env(env)` or to the new `HIGHLY_FALSE_LABELS_BY_ENV[env]` pool.
   - All 5 choice texts are distinct.
   - The correct answer text matches the replay-simulated final position of the queried vehicle.
   - No distractor text equals the correct answer text.
   - `start_position != final_position` for the queried vehicle.

---

## Problem 3 — Weak invariant (`all_labels_in_vocabulary`)

**What is wrong:**
The current invariant in the audit block is:
```python
"all_labels_in_vocabulary": all(
    is_valid_label(choices[L], env)
    or choices[L] in cross_env_labels(env)
    for L in LETTERS
)
```
This accepts ANY label from the global vocabulary, including cross-environment labels. It is always `True` and provides no real validation.

**Where in the code:**
- `generate_example()`, inside the `"invariants"` dict, lines ~500-504

**Solution:**
Replace the invariant with an environment-specific check:
```python
"all_labels_in_vocabulary": all(
    is_valid_label(choices[L], env)
    or choices[L] in highly_false_labels_for_env(env)
    for L in LETTERS
)
```
Where `highly_false_labels_for_env()` is a new function in `domain/vocabulary.py` that returns only the dedicated highly_false labels for the given environment (the expanded pool from Problem 1).
This invariant must be `True` for every generated example. If it is `False`, `validate_example()` (Problem 2) must reject the example.

---

## Problem 4 — Inconsistent distractor selection criteria

**What is wrong:**
`build_choices()` mixes two different semantic criteria without being explicit:
- `near_true_2` rationale alternates between `"final position of vehicle X"` (tracking-based) and `"same-environment label, not currently held by any vehicle"` (occupancy-based). These are different concepts.
- `highly_false_2` uses `"same-environment label not visited by any vehicle"` (trajectory-based), which is a third, different criterion.
The inconsistency makes the distractor categories less interpretable and harder to reason about.

**Where in the code:**
- `build_choices()`, lines ~249-316

**Solution:**
Define a fixed, prioritized hierarchy for distractor selection. Apply criteria in this exact order, stopping at the first that yields a valid label:

**near_true (2 distractors — plausible confusions):**
1. `near_true_1`: Starting position of the queried vehicle. Rationale: `"queried vehicle's start position"`.
2. `near_true_2`, try in order:
   a. An intermediate position the queried vehicle passed through (from the queried trace, excluding start and final). Rationale: `"intermediate position visited by queried vehicle"`.
   b. The final position of another vehicle in the scenario. Rationale: `"final position of vehicle {id}"`.
   c. Any same-environment label not yet used. Rationale: `"same-environment position"`.

**highly_false (2 distractors — implausible but same-environment):**
1. `highly_false_1`: A same-environment position never visited by ANY vehicle during the entire sequence. Rationale: `"same-environment position never visited by any vehicle"`.
2. `highly_false_2`: A same-environment position not occupied by any vehicle at the end of the sequence (but possibly visited during it). Rationale: `"same-environment position not occupied at end of sequence"`.
3. If same-environment positions are exhausted (multi_lane_road): Use labels from the `HIGHLY_FALSE_LABELS_BY_ENV` pool (from Problem 1). Rationale: `"road-context position not reachable in simulation"`.

If no valid label can be found at any level, reject the example and retry. Never mix criteria within the same distractor slot.

---

## Problem 5 — Repetitive and mechanical action patterns

**What is wrong:**
`generate_sequence()` has no semantic quality constraints on the action sequence. It picks a random vehicle, picks a random valid action, and applies it. This produces sequences with:
- The same `(vehicle, action)` pair appearing in consecutive steps (e.g. "Vehicle A changes to the left lane" twice in a row — which in a 3-lane setup means left→center then center→right, but reads confusingly).
- The same vehicle acting many times in a row while others do nothing.
- Immediate positional zigzag: a vehicle moves left then immediately moves right, returning to its original position.

**Where in the code:**
- `generate_sequence()`, lines ~113-190 (the inner loop that builds the plan)

**Solution:**
Add three soft constraints inside the step-by-step generation loop:

1. **No consecutive identical `(vehicle, action)` pairs.** After appending `(vid, act)` to the plan, the next step must not produce the same pair. If it would, try another action or another vehicle before giving up.

2. **Max 2 consecutive actions by the same vehicle.** Track a `consecutive_same_vehicle` counter. If the same `vid` has acted for the last 2 steps, force selection of a different vehicle for the current step. Use a constant `MAX_CONSEC = 2`.

3. **Anti-zigzag check.** Before applying an action, simulate it on a snapshot and check the resulting position. If the vehicle's new position equals its position from 2 moves ago (i.e., the pattern X → Y → X), reject that action and try another. Track position history per vehicle to detect this.

These constraints should cause the generator to pick alternative actions or vehicles, not to reject the entire sequence. Only if no valid action can be found for any vehicle should the sequence attempt be abandoned and retried.

---

## Problem 6 — Weak overall distractor quality

**What is wrong:**
Even when the correct answer is verified via replay, the distractor set is sometimes weak because:
- One or more options are trivially invalid for the environment (solved by Problem 1).
- Distractors are too predictable (the start position is always `near_true_1`).
- Rationale categories are applied inconsistently (solved by Problem 4).

**Where in the code:**
- `build_choices()` and `validate_example()` (after Problem 2 creates it)

**Solution:**
After implementing Problems 1–5, add a minimal quality gate in `validate_example()`:
1. All 5 choices must be environment-consistent (belong to `labels_for_env(env)` or `HIGHLY_FALSE_LABELS_BY_ENV[env]`).
2. Exactly 2 distractors must be typed `near_true` and exactly 2 must be typed `highly_false`.
3. No duplicate texts among the 5 choices.
4. The correct answer must match the replayed final position.

Do NOT add complex scoring or statistical checks. Keep validation simple and deterministic. If all 6 problems are fixed correctly, distractor quality follows as a consequence.

---

## Execution order

The problems have dependencies. Fix them in this order:
1. **Problem 1 first** (expand vocabulary) — this is the prerequisite for everything else.
2. **Problem 4** (rewrite `build_choices()` with the new hierarchy) — uses the expanded vocabulary.
3. **Problem 5** (add soft constraints to `generate_sequence()`) — independent of distractors.
4. **Problem 2** (add `validate_example()`) — needs the new `build_choices()` to validate against.
5. **Problem 3** (fix the invariant) — uses the new vocabulary functions.
6. **Problem 6** (quality gate) — integrates into `validate_example()` from Problem 2.

After all fixes: run `python generators/task1_position.py --n 100 --seed 42` and verify:
- Yield: 100/100 (or very close)
- 0 cross-environment distractors
- 0 duplicate choices
- All answers match replay
- Balanced distributions (answer keys, environments, step counts)
- No zigzag patterns, no consecutive identical actions
- All invariants pass
