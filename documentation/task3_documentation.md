# Task 3 Documentation — Violation Detection

**Scope:** Core benchmark Task 3  
**Generator:** `generators/task3_violation.py` — current version: `task3_violation_v9`  
**Dataset:** `dataset/core/task3_violation.jsonl`  
**Last updated:** April 2026

---

## 1. Context

Task 3 evaluates violation detection in event sequences. It builds on Task 1 state tracking and Task 2 rule application, but the target is now the first illegal action in a short trace.

The model receives a scenario with 3 vehicles and a short sequence of 2–3 events, and must identify which vehicle committed the first illegal action — or determine that no violation occurred. This combines two sub-skills: applying domain traffic rules to individual actions (rule lookup) and correctly attributing the first violation to the right actor across a multi-event trace (temporal grounding).

Task 3 uses all three environments:
- `intersection` — FSM-based movement through approach → inside → exit; violations include turning without entering, moving forward from an exit lane, and right-of-way infractions
- `multi_lane_road` — three lateral lanes; violations are lane changes out of bounds (left from left_lane, right from right_lane)
- `roundabout` — rotary; violation is entering without yielding to a circulating vehicle

The fixed 5-option answer set has a specific semantic structure: three vehicle labels (A, B, C), one "no violation" option ("No vehicle can be determined"), and one deliberate distractor ("Another vehicle (not A, B, or C)") which is never correct. This fifth option tests whether the model confuses the possibility of unspecified actors with the observable facts in the scenario.

---

## 2. Objective

Task 3 evaluates whether a model can **correctly identify the first vehicle to commit a traffic violation** in a short event sequence, or correctly determine that no violation occurred.

The model must:
- parse the initial vehicle positions and environment
- mentally simulate each event in sequence, maintaining FSM state for all 3 vehicles
- detect which action, if any, constitutes a rule violation
- attribute the violation to the correct vehicle or return the no-violation answer

The task is designed to resist two classes of shortcuts:
- **surface pattern shortcuts** — the sequence of actions must not be repetitive enough to allow pattern matching without genuine rule application (enforced by `EVENT_SIGNATURE_REUSE_CAP` and `ACTION_PATTERN_REUSE_CAP`)
- **answer-letter shortcuts** — the correct answer must be uniformly distributed across the five answer positions (enforced by a key schedule of exactly 20 records per letter)

---

## 3. Implementation

### 3.1 Domain layer

Task 3 reuses the shared `domain/` module built for Task 1:

- `entities.py` — `Vehicle`, `ScenarioState`, `Action`, `Environment`, `Direction`
- `rules.py` — `detect_violation()` for FSM-based violations, `detect_right_of_way_violation()` for intersection priority, `is_valid_transition()` for FSM legality checks
- `scenario.py` — `apply_action()` simulation engine, environment builders
- `render.py` — `describe_scenario()` for natural language prompt construction

### 3.2 Violation classes

Seven violation classes are defined and enforced via an `ALLOWED_VIOLATION_TYPES` whitelist:

| Class | Environment | Rule |
|---|---|---|
| `turn_without_entering` | intersection | Vehicle turns without first entering the intersection |
| `forward_from_exit` | intersection | Vehicle moves forward from an exit lane |
| `intersection_right_of_way` | intersection | Vehicle enters when another has priority |
| `lane_change_out_of_bounds_left` | multi_lane_road | CHANGE_LEFT from left_lane |
| `lane_change_out_of_bounds_right` | multi_lane_road | CHANGE_RIGHT from right_lane |
| `roundabout_entry_no_yield` | roundabout | Vehicle enters while another is circulating inside |
| `no_violation` | all | No illegal action in the sequence |

### 3.3 Generator design

The generator produces examples through a constrained simulation loop:

1. A **key schedule** ensures exactly 20 records per answer letter (A–E), shuffled
2. A **difficulty schedule** targets ~33 records per tier (easy / medium / hard), built jointly with a slot-swap mechanism
3. A **difficulty × environment target matrix** guides candidate selection to maintain environment balance within each tier
4. For each slot, a scenario is built, a violation or no-violation plan is constructed, and a full replay audit is run to verify ground truth
5. If validation fails, up to `MAX_RETRIES = 600` retries are allowed per slot, with a soft fallback after `SLOT_EARLY_FALLBACK_ATTEMPTS = 180` attempts that relaxes the strict difficulty tier requirement

### 3.4 Difficulty tiers

Difficulty is derived from two structural properties: the total number of events and the position of the violation within the sequence.

- **easy** — 2 events; violation at step 2 (last), or no violation with 2 events. The violation is always the final action, requiring no forward lookahead.
- **medium** — 3 events; violation at step 1 or step 2. The violation occurs early, with legal noise after it.
- **hard** — 3 events; violation at step 3 (last), or no violation with 3 events. The model must process a full 3-event trace before determining the outcome.

### 3.5 Distractor construction

The 5-option answer set is fixed semantically across all records:
- `near_true` — the two non-violating vehicle labels. Since any of A, B, or C can be the violator, the other two are the most plausible wrong answers.
- `highly_false` — "No vehicle can be determined" (wrong when a violation exists) and "Another vehicle (not A, B, or C)" (always wrong — no fourth vehicle is ever present).

The key schedule permutes which letter maps to which semantic position, so that the correct answer is distributed uniformly.

### 3.6 Replay audit and invariant system

Every example is verified through a deterministic replay: the generator re-simulates the event sequence from the initial state and confirms the violation ground truth matches what the plan intended. Nine per-example invariants are enforced:

| Invariant | What it checks |
|---|---|
| `no_duplicate_options` | All 5 choices are textually distinct |
| `fixed_option_set` | The 5 choices are exactly the expected semantic labels |
| `answer_in_choices` | The correct answer letter is present in the choices |
| `correct_vehicle_not_missing_from_choices` | The violating vehicle label appears in the choices |
| `undetermined_correct_only_for_no_violation` | "No vehicle can be determined" is correct only for no_violation records |
| `violation_step_none_only_for_no_violation` | `violation_step` is None only for no_violation records |
| `all_events_valid_format` | All event entries have the required fields |
| `first_illegal_event_matches_metadata` | `metadata.violation_step` matches the replay's first illegal event index |
| `target_matches_replay` | The correct answer letter matches the replay's violation attribution |

---

## 4. Development Notes

### 4.1 `invalid_fsm_transition` appearing as a violation type

**Problem (correctness):** In early versions, when the `apply_action()` simulation encountered an action that could not be applied to the current FSM state (an impossible transition), the failure itself was returned as a violation type called `invalid_fsm_transition`. Two records in the dataset had this as their ground truth. The model would be asked to identify a vehicle that committed an "illegal action" where the illegality was a generator internal failure, not a domain rule violation.

**Root cause:** The replay function `_replay_first_violation()` did not distinguish between a genuine rule violation detected by `detect_violation()` and a failure of the FSM apply step. Both paths raised or returned an error signal that was treated as a violation.

**Change:** Patched `_replay_first_violation()` to detect FSM apply failures and return a no-violation sentinel instead of propagating the internal error as a violation type. Added an `ALLOWED_VIOLATION_TYPES` whitelist that batch-validates all 100 records on output: any record whose `violation_type` is not in the whitelist causes the batch to be rejected and regenerated.

**Result:** 0 records with `invalid_fsm_transition` in all subsequent versions.

---

### 4.2 Action-pattern reuse affecting 94/100 records (original cap=2 ineffective)

**Problem (distributional):** With `ACTION_PATTERN_REUSE_CAP = 2` set to limit how many times the same action sequence could appear in the dataset, the audit showed that 94 out of 100 records shared their action pattern with at least one other record. Specifically, 43 distinct patterns each appeared exactly 2 times — the cap was being saturated almost universally. The dataset had structural repetition despite the cap being nominally active.

**Root cause:** The action space for short 2–3 event sequences over 3 vehicles is not large. With only a handful of violation classes per environment and constrained FSM transitions, many scenarios naturally converge to similar action sequences. A cap of 2 allowed every pattern to appear twice, which was enough to affect the majority of records given the small action space.

**First correction attempt:** Lowered `ACTION_PATTERN_REUSE_CAP` to 1 (each action pattern can appear at most once). This drove reuse to 0/100 but caused the generator crash described in 4.3.

**Final correction:** After the crash (4.3), raised `ACTION_PATTERN_REUSE_CAP` back to 2, and added `MAX_ACTION_PATTERN_REUSE_TOTAL = 20` as a global ceiling on the total number of records affected by any pattern reuse. This dual-constraint approach prevents the saturation problem (no pattern dominates) without exhausting the easy-tier plan space.

**Result:** Action-pattern max repeat is 2; total affected records stay within the 20-record budget. This is a bounded and controlled form of reuse rather than the uncapped saturation seen in the original version.

---

### 4.3 Generator crash under strict difficulty + aggressive reuse cap

**Problem (structural):** After lowering the action-pattern reuse cap to 1 (first fix attempt in 4.2), the generator began crashing with:

```
RuntimeError: Unable to satisfy strict difficulty tier 'easy' at slot 98
and no same-tier unfilled slot is available for swap.
```

The crash occurred because the `easy` tier has a structurally narrow action space: 2-event sequences with the violation at the last step. With cap=1, the easy tier's pool of valid unique plans was exhausted before all easy slots were filled. The slot-swap mechanism — which attempts to trade an unfilled slot for a same-tier slot elsewhere in the schedule — found no remaining same-tier candidate and raised a hard error.

**Root cause:** `ACTION_PATTERN_REUSE_CAP = 1` was too aggressive for the easy tier. The easy-tier action space is inherently more constrained than medium or hard (shorter sequences, fewer valid violation positions), so it hits uniqueness exhaustion faster. The hard difficulty constraint (requiring an exact same-tier swap partner) turned what should be a soft degradation into a fatal crash.

**Change:** Three changes were applied together:
- Raised `ACTION_PATTERN_REUSE_CAP` back to 2 to restore headroom in the easy tier
- Added `MAX_ACTION_PATTERN_REUSE_TOTAL = 20` as a global budget to limit aggregate reuse even with cap=2 (preventing the original saturation problem from recurring)
- Made difficulty fallback soft: after `SLOT_EARLY_FALLBACK_ATTEMPTS = 180` failed attempts per slot, the slot accepts the nearest available difficulty tier rather than crashing

**Result:** Generator runs to completion on all seeds. Most slots resolve in 1 attempt (`slot_attempts = 1` for 85/100 records); 2 medium slots required up to 290 attempts before the soft fallback triggered. The example-level `attempt` field (tracking full example rebuild retries) stays at max 2.

---

### 4.4 Difficulty imbalance — partially resolved, residual gap accepted

**Problem (distributional):** The difficulty distribution across versions was consistently unstable. Early versions with no quota control produced arbitrary splits. After adding a quota-based difficulty schedule and slot-swap mechanism, v8 produced easy: 25, medium: 45, hard: 30 — far outside the ~33/33/34 target. After the soft fallback fix (v9), the distribution improved but stabilized at approximately easy: 30, medium: 37, hard: 33 across multiple seeds.

**Root cause:** The difficulty schedule starts with a quota target of ~33 per tier, but the three tiers are not equally achievable. Easy and hard tiers are structurally constrained: easy requires 2-event sequences with a late violation, hard requires 3-event sequences with a late violation or no-violation trace. Medium is the broadest tier (violation at step 1 or 2 in a 3-event sequence) and naturally attracts overflow from both directions. When easy or hard slots exhaust their candidate pool under the reuse cap, the soft fallback reassigns those slots to the nearest feasible tier — which is almost always medium.

**What was tried:**
- **Strict difficulty enforcement** (crash on exhaustion) — caused the v9 RuntimeError described in 4.3; not viable
- **Slot-swap with same-tier partner search** — works within a tier but cannot manufacture easy-tier candidates when the plan space is genuinely exhausted
- **Tier-specific reuse cap** (looser cap for easy only) — partially helped but introduced inconsistency across tiers and still did not reach exact balance
- **Raising `MAX_RETRIES` further** — reduced variance across seeds but did not eliminate the imbalance because the constraint is structural, not stochastic

**Accepted as policy:** The final range of easy: 30, medium: 37, hard: 33 is within a ±7 deviation from target. This reflects an inherent asymmetry in the difficulty tier action spaces: easy has a smaller plan vocabulary than hard, and medium is the natural overflow sink. No further fix will be applied. The distribution is documented as a structural limitation of the task design.

---

### 4.5 Additional bugs fixed (earlier development)

- `invalid_fsm_transition` NameError in replay — undefined variable in violation type string construction
- Duplicate `option_rationale_by_letter` appearing in both `audit` and `metadata` — removed from `metadata`, kept only in `audit`
- Non-atomic file writes — replaced with temp file + `fsync` + `os.replace` pattern
- `audit.run` field duplicated in output — removed the duplicate key
- Environment distribution imbalance (early versions had intersection overrepresented) — fixed by adding a difficulty × environment target matrix guiding candidate selection

---

## 5. Known Limitations (Accepted as Policy)

The following issues were analyzed and accepted as design constraints. No code changes will be made.

**`roundabout_entry_no_yield` overrepresented at ~25/80 violation records (31%).** This is a direct structural consequence of the environment design. The roundabout environment has exactly one violation class: a vehicle entering without yielding to a circulating one. There is no other illegal action possible in a roundabout under the current domain rules. Since roundabout records account for 33% of the dataset, and approximately 8 of those 33 are no-violation records, the remaining ~25 are all necessarily `roundabout_entry_no_yield`. The other environments (intersection and multi_lane_road) each support 3+ violation classes, so their violations are diluted across multiple types. Introducing artificial roundabout violation types not grounded in real traffic law would compromise domain integrity without improving benchmark quality. This is accepted as a property of the domain, analogous to `turn_right` underrepresentation in Task 2.

**Difficulty distribution easy: 30, medium: 37, hard: 33 (target ~33 each).** As documented in 4.4, the medium tier naturally absorbs overflow from easy and hard slots that exhaust their structurally constrained plan spaces. Multiple fix attempts were made — strict enforcement, slot-swap, tier-specific caps, higher retry budgets — without achieving stable exact balance. The current ±7 deviation is accepted as policy. It does not affect per-record correctness or the anti-shortcut properties of the dataset.

**Two slots required >200 `slot_attempts` before soft fallback.** The `slot_attempts` field records how many candidate plans were tried before a slot was filled. For 85/100 records this is 1; for 2 records it reached up to 290, at which point the soft difficulty fallback accepted medium-tier outcomes. This is expected behavior for constrained tiers and does not affect output correctness.

**`Another vehicle (not A, B, or C)` is never correct.** This fifth option is a deliberate highly-false distractor. It tests whether the model correctly scopes its answer to the observable scenario rather than speculating about unseen actors. Its permanent incorrectness is by design.

---

## 6. Final State (v9)

| Metric | Value |
|---|---|
| Records | 100 |
| Generator version | task3_violation_v9 |
| All 9 invariants passing | 0 failures |
| Answer distribution | 20 per letter (A–E) |
| Environments | intersection 34, roundabout 33, multi_lane_road 33 |
| Difficulty | easy 30, medium 37, hard 33 |
| `slot_attempts` | max 290 (2 medium slots); 85/100 slots resolved in 1 attempt |
| Example-level retries (`attempt`) | max 2 |
| Action-pattern max repeat | 2 (total affected records: 20) |
| Violation types | forward 10, intersection ROW 8, lane-left 15, lane-right 13, roundabout 25, turn 9, no_violation 20 |
| `invalid_fsm_transition` records | 0 |
| No-violation records | 20/100 |
| Prompts unique | yes |
| IDs unique | yes |
| Rows sorted by id | yes (`task3_0000` ... `task3_0099`) |
| violation_class/type mismatch | 0 |

---

## 7. Usage

### Generate core dataset
```bash
python generators/task3_violation.py --n 100 --seed 42 --out dataset/core/task3_violation.jsonl
```

### Quick audit
```bash
python - <<'PY'
import json
from collections import Counter
rows = [json.loads(l) for l in open('dataset/core/task3_violation.jsonl') if l.strip()]
print('N', len(rows))
print('version', rows[0]['audit']['generator_version'])
print('answers', Counter(r['answer'] for r in rows))
print('envs', Counter(r['scenario']['environment'] for r in rows))
print('difficulty', Counter(r['metadata']['difficulty'] for r in rows))
print('violation_types', Counter(r['metadata']['violation_type'] for r in rows))
inv_fails = {k: sum(1 for r in rows if not r['audit']['invariants'].get(k, True))
             for k in rows[0]['audit']['invariants']}
print('invariant_failures', {k:v for k,v in inv_fails.items() if v > 0} or 'none')
print('max_slot_attempts', max(r['audit']['slot_attempts'] for r in rows))
print('max_example_retries', max(r['audit']['attempt'] for r in rows))
PY
```

---

## 8. Summary

Task 3 reached a final quality score of **8.8/10** after three improvement passes in April 2026.

The core engineering challenge was managing three competing pressures simultaneously: correctness (every replay must confirm the ground truth), diversity (action-pattern reuse must be bounded), and feasibility (the generator must complete without exhausting the plan space in any tier). These three pressures are in direct tension — tighter reuse caps increase diversity but shrink the feasible plan space, which destabilizes the difficulty distribution and can cause the generator to crash.

The key structural fixes were: replacing `invalid_fsm_transition` with a no-violation sentinel and adding the `ALLOWED_VIOLATION_TYPES` whitelist, introducing the `ACTION_PATTERN_REUSE_CAP` + `MAX_ACTION_PATTERN_REUSE_TOTAL` dual-constraint system to bound reuse without exhausting the plan space, and converting difficulty enforcement from hard-crash to soft-fallback to prevent the easy-tier exhaustion crash.

The two residual limitations — difficulty ±7 from target and `roundabout_entry_no_yield` at 31% of violations — are documented above and accepted as policy. The difficulty imbalance was checked across multiple strategies; the current deviation follows from the easy-tier plan space. The roundabout concentration is a domain property.
