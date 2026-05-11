# Task 4 Documentation — Certainty Under Spatial Ambiguity

**Scope:** Core benchmark Task 4
**Generator:** `generators/task4_overlap.py` — version `task4_certainty_ambiguity_v1`
**Validators:** `scripts/validate_task4.py`, `scripts/validate_task4_distribution.py`
**Dataset:** `dataset/core/task4_overlap.jsonl`
**Last updated:** May 2026

---

## 1. Task Definition

Task 4 evaluates **epistemic certainty under spatial ambiguity**: given a scenario and an event sequence, the model must identify which of five statements about vehicle positions is *certainly* true — provable from the observable sequence alone — and reject statements that are plausible but unverifiable.

The core skill being tested is not positional tracking (Task 1) or rule compliance (Tasks 2–3), but **epistemic discrimination**: the ability to distinguish what can be proven from what can only be inferred or guessed. This requires reasoning about what the sequence does *not* tell you, not just what it does.

Spatial ambiguity arises from multiple sources in this benchmark:

- **Simultaneous presence in a shared zone** (intersection body, roundabout lane): when two vehicles are both inside, their relative positions — who is ahead, who is to the left — are not determined by the event log alone.
- **Unknown entry/exit ordering**: when a vehicle is known not to have entered a zone, the sequence does not specify *when* it will enter relative to another vehicle's exit.
- **Unspecified lane ordering on a multi-lane road**: a vehicle's lane is known, but its longitudinal position relative to other vehicles in other lanes is not observable from lane-change events alone.

Overlap between vehicles is one mechanism that produces spatial ambiguity, but it is not the defining feature of the task. The task is defined by the epistemic challenge — certainty under ambiguity — and overlap is simply one of several scenarios that create that challenge.

Unlike Tasks 1–3, options are full declarative statements rather than position labels or vehicle identifiers. Exactly one statement is certainly true; two are near-true (uncertain given the sequence); two are highly-false (contradicted by the replayed final state).

---

## 2. Objective

Given scenario + event sequence, the model must answer:

`Which of the following statements is certainly true at the end of the sequence?`

Required behavior:
- track all vehicles through events
- identify one statement that is provably true given the sequence
- reject statements that are uncertain due to spatial ambiguity
- reject statements that are contradicted by the replayed state

---

## 3. Implementation

### 3.1 Generator design

`generators/task4_overlap.py` implements:
- deterministic generation (`--seed`) with fixed schedules
- 100-example core set with exact key balance (`20 x A/B/C/D/E`)
- scenario schedule with six structural types, each corresponding to a distinct source of spatial ambiguity:
  - `two_overlap_one_outside` (20) — ambiguity from simultaneous presence in intersection
  - `two_overlap_third_exited` (10) — ambiguity from simultaneous presence, third vehicle exited
  - `one_inside_one_exited_one_approach` (20) — ambiguity from unknown entry/exit ordering
  - `roundabout_overlap` (20) — ambiguity from simultaneous presence in roundabout
  - `roundabout_non_entry` (10) — ambiguity from unknown roundabout entry timing
  - `multi_lane_positioning` (20) — ambiguity from unobservable longitudinal lane ordering
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

### 3.2 Statement taxonomy

Near-true statements cover five distinct epistemic types, each corresponding to a different reason why a statement about vehicle positions cannot be confirmed from the observable sequence:

- **spatial_present**: `"Vehicle X is ahead of Vehicle Y."` / `"Vehicle X is to the left of Vehicle Y."` — relative ordering inside a shared zone is unspecified
- **moved_past**: `"Vehicle X has already moved past Vehicle Y."` — relative position during simultaneous presence cannot be recovered from event order alone
- **past_overlap**: `"Vehicle X was ahead of Vehicle Y inside the intersection."` — past-tense ordering inside a shared zone; pair-specific, replay-checked
- **will_future**: `"Vehicle X will exit before Vehicle Y enters the intersection."` — future ordering cannot be derived from the given sequence
- **lane_order_unknown**: `"Vehicle X is ahead of Vehicle Y on the road."` / `"is behind"` / `"is directly behind"` / `"has already moved past"` / `"will change lanes before"` — longitudinal ordering on a multi-lane road is not determined by lane-change events

Correct-answer phrasing uses multiple variants to reduce exact-text reuse:
- pair-containment variants include both `"Vehicles X and Y are both inside the intersection."` and `"Both Vehicle X and Vehicle Y are inside the intersection."`
- single-vehicle containment variants include both `"Vehicle X is inside the intersection."` and `"Vehicle X remains inside the intersection."`
- equivalent variants exist for roundabout scenarios

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
7. spatial ambiguity condition met: either replayed overlap exists, or the example belongs to a non-overlap ambiguity category (`containment_non_entry`, `lane_position`)
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

## 5. Design Notes

**File and dataset naming.** The generator (`task4_overlap.py`) and dataset (`task4_overlap.jsonl`) retain names from an earlier version of the project. These names are preserved deliberately to maintain pipeline stability and avoid reference breakage across validators, scripts, and reproducibility records. The task definition and all semantic identifiers (`TASK_NAME`, `GENERATOR_VERSION`, question wording, category labels) use the current definition: certainty under spatial ambiguity.

**Multi-lane vocabulary.** `domain/scenario.py` uses `left_lane`, `center_lane`, and `right_lane` as position keys for `multi_lane_road`, but `domain/vocabulary.py` does not expose human-readable labels for these strings. Task 4 resolves this with a local mapping inside the generator (`left_lane → the left lane`, etc.). No modifications were made to `domain/`. This is documented in `domain_documentation.md`.

---

## 6. Final Validation Status (Core-100, v1)

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

Task 4 evaluates certainty under spatial ambiguity: the ability to identify what is provably true about vehicle positions when the event sequence leaves some spatial relationships undetermined. The task is designed around the epistemic gap between what is observable and what is certain — a property that arises in multiple distinct spatial configurations, of which simultaneous zone occupancy (overlap) is one.

The core dataset consists of 100 examples spanning six scenario types and three environments, all validated to zero errors by the independent validator. The generator and validator are deterministic given a fixed seed. All balance, uncertainty classification, and anti-shortcut constraints are satisfied.
