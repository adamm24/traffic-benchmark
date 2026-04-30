# Task 4 Documentation — Overlap Reasoning

**Scope:** Core benchmark Task 4  
**Generator:** `generators/task4_overlap.py` — version `task4_overlap_v6`  
**Validators:** `scripts/validate_task4.py`, `scripts/validate_task4_distribution.py`  
**Dataset:** `dataset/core/task4_overlap.jsonl`  
**Last updated:** April 2026

---

## 1. Context

Task 4 evaluates **epistemic certainty under partial spatial ambiguity**: when vehicles overlap inside an intersection or a roundabout, or when relative order on a multi-lane road is unspecified, the model must separate what is certainly derivable from what is only plausible.

Unlike Tasks 1–3, options are full statements. Exactly one statement is certainly true; two are near-true (uncertain); two are highly-false (contradicted by replayed state).

---

## 2. Objective

Given scenario + event sequence, the model must answer:

`Which of the following statements is certainly true at the end of the sequence?`

Required behavior:
- track all vehicles through events
- identify one provable statement
- reject uncertain overlap-ordering statements
- reject contradictory state statements

---

## 3. Implementation

### 3.1 Generator design

`generators/task4_overlap.py` implements:
- deterministic generation (`--seed`) with fixed schedules
- 100-example core set with exact key balance (`20 x A/B/C/D/E`)
- scenario schedule with six structural types:
  - `two_overlap_one_outside` (20)
  - `two_overlap_third_exited` (10)
  - `one_inside_one_exited_one_approach` (20)
  - `roundabout_overlap` (20)
  - `roundabout_non_entry` (10)
  - `multi_lane_positioning` (20)
- environment split:
  - `intersection` 50
  - `roundabout` 30
  - `multi_lane_road` 20
- difficulty split:
  - `easy` 33
  - `medium` 33
  - `hard` 34
- category split:
  - `containment_overlap` 20
  - `containment_non_entry` 20
  - `exit_reached` 20
  - `roundabout_overlap` 20
  - `lane_position` 20

### 3.2 Statement taxonomy (v6)

Near-true statements use five epistemic types:
- **spatial_present**: `"Vehicle X is ahead of Vehicle Y."` / `"Vehicle X is to the left of Vehicle Y."` (current overlap-state)
- **moved_past**: `"Vehicle X has already moved past Vehicle Y."` (relative position during simultaneous presence)
- **past_overlap**: `"Vehicle X was ahead of Vehicle Y inside the intersection."` / `"was to the left of"` (past-tense overlap uncertainty; pair-specific replay-checked)
- **will_future**: `"Vehicle X will exit before Vehicle Y enters the intersection."` etc.
- **lane_order_unknown**: `"Vehicle X is ahead of Vehicle Y on the road."` / `"is behind"` / `"is directly behind"` / `"has already moved past"` / `"will change lanes before"` for `multi_lane_road`

Correct-answer phrasing was also widened to reduce exact-text reuse:
- pair-containment variants now include `"Both Vehicle X and Vehicle Y are inside the intersection."`
- single-vehicle containment variants now include `"Vehicle X remains inside the intersection."`
- equivalent variants exist for roundabout overlap

Scale-up safeguards:
- single-vehicle correct answers are balanced during generation with a soft gap cap
- all correct-answer vehicle mentions are balanced during generation with a hard gap cap of `10`
- `correct_text` cap scales with dataset size: `max(20, ceil(n/15))`
- event-signature cap scales with dataset size: `max(20, ceil(n/10))`
- statement-signature cap scales with dataset size: `max(6, ceil(n/25))`

### 3.3 Invariant system

Per-example hard gate includes:
1. correct statement is replay-true
2. both `near_true` are replay-uncertain
3. both `highly_false` are replay-false
4. five distinct statements
5. no cross-environment position labels
6. replay matches `audit.final_state`
7. overlap detected, except categories that are valid without overlap (`containment_non_entry`, `lane_position`)
8. at least two acting vehicles
9. no action streak length 3
10. no ABAB actor pattern
11. only canonical IDs `Vehicle A/B/C`
12. answer letter follows key schedule
13. full `audit.option_rationale` coverage

### 3.4 Statement anti-collapse control

A normalized statement-structure signature cap is enforced:
- max reuse `<= 6` for each `(environment, scenario_type, certainly_true_category, normalized_choices)`

Global dedup caps:
- base `EVENT_SIG_CAP = 20`, scaled as `max(20, ceil(n/10))`
- base `CORRECT_TEXT_CAP = 20`, scaled as `max(20, ceil(n/15))`
- These caps prevent dominant patterns while accommodating the small vocabulary of a 3-vehicle, 3-action space
- These caps are enforced both during generation and by the final quality gate / independent validator

### 3.5 Output and reproducibility

- atomic write (`tempfile + fsync + os.replace`)
- same seed => byte-identical output
- output file: `dataset/core/task4_overlap.jsonl`

---

## 4. Independent Validator

`scripts/validate_task4.py` is independent of `generators/` and performs:
1. scenario reconstruction from JSON
2. event replay via `apply_action()`
3. independent statement truth classification (`true/uncertain/false`) — past-tense overlap handlers require the referenced vehicle pair to have actually overlapped during replay; multi-lane lane labels are handled by a local Task 4 mapping
4. recomputation of unique certainly-true option
5. comparison with declared `answer`
6. quality counters (duplicate prompts, near-true certainty leaks, cross-env contamination)

`scripts/validate_task4_distribution.py` adds Task 4 distribution and statement-pattern checks.

---

## 5. Known Limitations

**Vocabulary gap for multi-lane positions.** `domain/scenario.py` uses `left_lane`, `center_lane`, and `right_lane` as position keys for `multi_lane_road`, but `domain/vocabulary.py` does not provide human-readable labels for these three strings. Task 4 handles this with a local mapping inside the generator: `left_lane → the left lane`, etc. No modifications were made to `domain/`. This gap is also documented in `domain_documentation.md`.

---

## 6. Final Validation Status (Core-100, v6)

Command:

```bash
python scripts/validate_task4.py --input dataset/core/task4_overlap.jsonl
python scripts/validate_task4_distribution.py --input dataset/core/task4_overlap.jsonl
```

Result:
- `total = 100`
- `wrong = 0`
- `invalid = 0`
- duplicate prompts: `0`
- near_true misclassified as true/false: `0`
- highly_false misclassified: `0`
- cross-env label contamination: `0`
- `event_sig_max = 4` (`cap = 20`)
- `correct_text_max = 4` (`cap = 20`)
- past-overlap pair mismatches: `0`
- vehicle mentions in correct answers: `A=34`, `B=33`, `C=35`

Distribution summary:
- answer letters: `A/B/C/D/E = 20 each`
- environments: `intersection 50`, `roundabout 30`, `multi_lane_road 20`
- difficulty: `easy 33`, `medium 33`, `hard 34`
- scenario types: `two_overlap_one_outside 20`, `one_inside_one_exited_one_approach 20`, `roundabout_overlap 20`, `roundabout_non_entry 10`, `two_overlap_third_exited 10`, `multi_lane_positioning 20`
- certainly-true categories: `containment_overlap 20`, `exit_reached 20`, `containment_non_entry 20`, `roundabout_overlap 20`, `lane_position 20`
- num events per example: `2: 30`, `3: 36`, `4: 34`
- near_true type distribution: `spatial_present ~35`, `moved_past ~38`, `will_future ~51`, `round_spatial ~34`, `road_spatial ~22`, `past_overlap 20`

Reproducibility check:

```bash
python generators/task4_overlap.py --seed 42 --out /tmp/run1_task4.jsonl
python generators/task4_overlap.py --seed 42 --out /tmp/run2_task4.jsonl
diff /tmp/run1_task4.jsonl /tmp/run2_task4.jsonl
```

`diff` is empty (deterministic output confirmed).

---

## 7. Summary

The core dataset consists of 100 examples, all validated to zero errors by the independent validator. The generator and validator are deterministic given a fixed seed. All balance, overlap, uncertainty, and anti-shortcut constraints are satisfied.

The main design challenge was constructing near-true statements that are genuinely uncertain rather than just plausible. The five epistemic statement types (spatial_present, moved_past, past_overlap, will_future, lane_order_unknown) capture distinct reasons why a statement about vehicle positions cannot be confirmed from the observable sequence alone.
