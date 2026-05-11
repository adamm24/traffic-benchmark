# Task 1 Documentation — Position Tracking

**Scope:** Core benchmark Task 1
**Generator:** `generators/task1_position.py` (version `task1_position.v16`)
**Dataset:** `dataset/core/task1_position.jsonl`
**Last updated:** May 2026

## 1. Objective

Task 1 measures sequential position tracking: the model must simulate a short action sequence and identify the queried vehicle's final position.

Environments used:
- `intersection`
- `multi_lane_road`

Roundabout is excluded from Task 1 generation.

## 2. Domain-Consistency Decision (v16)

The shared domain is now the only contract for Task 1 state simulation.

Implemented standard:
- `left_lane`
- `center_lane`
- `right_lane`
- `roundabout_lane` (domain label only; not a multi-lane simulated state)

Removed from Task 1:
- `left_shoulder`, `right_shoulder`
- `far_left_lane`, `left_center_lane`, `right_center_lane`, `far_right_lane`

Result:
- no custom multi-lane state labels remain in Task 1
- all simulated transitions run through shared `domain/scenario.py` + shared FSM

## 3. Generation Design

Core loop:
1. Joint schedule over `(difficulty, environment)` keeps balanced quotas.
2. Key schedule keeps answers balanced across `A..E`.
3. Per-slot generation with retries and validation.
4. Post-generation dataset quality gate.

Task 1 keeps fixed 5-option MCQ format:
- 1 correct
- 2 near_true
- 2 highly_false

### 3.1 Choice-space policy after 3-lane standardization

With a strict 3-lane multi-lane state space, only 3 environment labels exist; this is insufficient for 5 unique options.

To preserve fixed 5-option format **without introducing custom labels**, Task 1 uses:
- **simulation state space:** shared multi-lane FSM states (`left/center/right lane`)
- **multi-lane option space:** shared-domain labels only (`left/center/right lane`, `roundabout lane`, and exit labels)

Constraints preserved:
- `near_true` are still grounded in visited states
- `highly_false` are never visited in that example
- all option texts are real labels from `domain/vocabulary.py`

## 4. Validation

Per-example validation checks:
- replay correctness of the full plan
- option uniqueness
- distractor type balance (2 near_true / 2 highly_false)
- anti-pattern checks (ABAB actor pattern, action streaks, etc.)
- audit consistency (`plan`, `traces`, queried shape)
- choice-space closure under Task 1 policy

Dataset-level checks:
- 100 generated records
- balanced answer letters
- balanced environment split
- balanced difficulty split
- no quality-gate violations

## 5. Main Fixes Applied

1. Removed Task 1 local 6-lane/shoulder simulation model.
2. Switched multi-lane simulation to shared domain builder + shared `apply_action` transitions.
3. Replaced shoulder-specific weighting/caps and custom-lane logic.
4. Updated validation from strict env-label closure to Task 1 choice-space closure (still domain-label closed).
5. Updated multi-lane difficulty profile flags to avoid collapse under reduced 3-lane label entropy.

## 6. Current Status (v16)

Validated on regenerated core set (`n=100`):
- generation succeeds (`100/100`)
- answer distribution is balanced (`20` per letter)
- environment split is balanced (`50/50`)
- difficulty split is balanced (`34/34/32`)
- all examples replay correctly under shared FSM
- no legacy custom labels appear in scenarios, traces, or options

## 7. Residual Tradeoff

Because Task 1 enforces fixed 5 options while multi-lane simulation has only 3 native states, some multi-lane highly_false options come from other shared-domain labels (still domain-valid, never custom).

This is intentional in v16 to satisfy simultaneously:
- shared-domain consistency
- no custom lane vocabulary
- fixed 5-option benchmark format
