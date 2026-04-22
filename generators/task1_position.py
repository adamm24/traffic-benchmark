"""
Task 1 — Position Tracking Generator
=====================================
Generates multiple-choice questions that test an LLM's ability to track
vehicle positions through a sequence of actions.

Environments: multi_lane_road, intersection  (roundabout excluded)
Vehicles:     3
Steps:        2–4
Choices:      1 correct, 2 near_true, 2 highly_false

Post-refactor guarantees (addresses T1 improvement plan):
  • Distractor quality:
      near_true_1 = queried vehicle's START position.
      near_true_2 = another tracked vehicle's FINAL position in the same
                    environment (principled, grounded in the scenario).
      highly_false_1 = cross-environment label (e.g. a lane on an
                       intersection scenario).
      highly_false_2 = same-environment label that was never actually
                       visited by any vehicle (rules out shallow "pick
                       any in-env label" heuristics).
  • No hardcoded fallback strings. All labels flow through
    domain.vocabulary. If a label cannot be found the example is
    REJECTED and retried — never replaced with a placeholder.
  • Audit metadata: every example embeds an "audit" block with the full
    queried-vehicle trace, per-distractor rationale, invariant flags,
    and generation metadata (seed, generator version).
  • Deterministic with --seed (env RNG only, independent of system RNG).
  • Scale-ready: N_EXAMPLES can be set to 300 via CLI without changing
    the generator; per-env and per-key balance are enforced by the
    key_schedule and a soft env-balance retry loop.
  • FSM-backed: apply_action() now self-rejects invalid transitions
    (returns ""); safe_apply_action() below is a thin wrapper that
    converts "" → None so existing call sites keep working.

Usage:
    python generators/task1_position.py --n 300 --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

# ── Make project root importable ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from domain.entities import (
    Action, Environment, ScenarioState, Vehicle,
)
from domain.scenario import (
    LANE_ORDER, apply_action,
    build_intersection_scenario, build_multi_lane_scenario,
)
from domain.render import describe_scenario, render_prompt
from domain.vocabulary import (
    highly_false_labels_for_env, is_env_consistent_label, is_valid_label,
    label_of, labels_for_env, positions_for_env,
)

# ── Constants ───────────────────────────────────────────────────────────────

GENERATOR_VERSION = "task1_position.v3"   # bump when contract changes
DEFAULT_N_EXAMPLES = 100                  # CLI --n overrides; 300 = full core
NUM_VEHICLES = 3
MIN_STEPS    = 2
MAX_STEPS    = 4
MAX_RETRIES  = 50                         # per-example retry budget
MAX_CONSEC   = 2                          # max consecutive actions by same vehicle

TASK_ENVS = [Environment.MULTI_LANE, Environment.INTERSECTION]

# Actions valid per environment (Task 1 — position only, no roundabout)
# NOTE: MOVE_FORWARD excluded from MULTI_LANE because it doesn't change
#       the lane position and would be a no-op for position tracking.
ACTIONS_BY_ENV = {
    Environment.MULTI_LANE: [
        Action.CHANGE_LEFT,
        Action.CHANGE_RIGHT,
    ],
    Environment.INTERSECTION: [
        Action.MOVE_FORWARD,
        Action.TURN_LEFT,
        Action.TURN_RIGHT,
    ],
}


# ── FSM-compatible action wrapper ───────────────────────────────────────────



def get_required_vehicle(state: ScenarioState, vehicle_id: str) -> Vehicle:
    """Return a vehicle or raise a clear error if the scenario is inconsistent."""
    vehicle = state.get_vehicle(vehicle_id)
    if vehicle is None:
        raise ValueError(f"Vehicle {vehicle_id!r} not found in scenario state.")
    return vehicle


def safe_apply_action(
    state: ScenarioState,
    vehicle_id: str,
    action: Action,
) -> str | None:
    """
    Thin wrapper around the FSM-backed apply_action().

    Post FSM refactor (T1-B03/B04/B07), apply_action() rejects invalid
    transitions by returning "" without mutating state. This wrapper
    converts "" → None so legacy call-sites that treat None as "try a
    different action" continue to work unchanged.
    """
    result = apply_action(state, vehicle_id, action)
    return result if result else None


# ── Sequence generation ─────────────────────────────────────────────────────

def generate_sequence(
    state: ScenarioState,
    queried_vid: str,
    env: Environment,
    n_steps: int,
) -> list[tuple[str, Action]] | None:
    """
    Produces a valid n-step event sequence guaranteeing:
      • the queried vehicle is moved at least `min_queried_moves` times
      • its final position differs from its starting position

    Quality soft-constraints (Problem 5 of the Task 1 fix plan):
      • No consecutive identical (vehicle, action) pairs.
      • At most MAX_CONSEC consecutive actions performed by the same
        vehicle (prevents "A acts 4 times in a row" monotony).
      • Anti-zigzag: a vehicle's new position must not equal its position
        from 2 moves ago (prevents X → Y → X round-trips within the
        same vehicle's own trajectory).

    Returns a list of (vehicle_id, Action) tuples on success, or None
    if it cannot satisfy all constraints within MAX_RETRIES attempts.
    """
    vehicle_ids = [v.id for v in state.vehicles]
    valid_actions = ACTIONS_BY_ENV[env]
    start_pos = get_required_vehicle(state, queried_vid).position

    min_queried_moves = 2 if env == Environment.INTERSECTION else 1

    for _ in range(MAX_RETRIES):
        trial_state = copy.deepcopy(state)
        plan: list[tuple[str, Action]] = []
        queried_move_count = 0
        # Per-vehicle position history — used for anti-zigzag check
        # and includes the starting position of each vehicle at index 0.
        pos_history: dict[str, list[str]] = {
            v.id: [v.position] for v in trial_state.vehicles
        }

        for step_idx in range(n_steps):
            remaining = n_steps - step_idx
            queried_deficit = min_queried_moves - queried_move_count

            # Who has acted in the last MAX_CONSEC slots (for the ban).
            recent_vids = [p[0] for p in plan[-MAX_CONSEC:]]
            banned_vid: str | None = None
            if (
                len(recent_vids) == MAX_CONSEC
                and len(set(recent_vids)) == 1
            ):
                banned_vid = recent_vids[0]

            # Forced-choice branches (queried-move quota) still respected,
            # but the consecutive-vehicle ban is applied wherever it can
            # co-exist with the quota requirement.
            if queried_deficit >= remaining:
                vid = queried_vid
            elif step_idx == n_steps - 1 and queried_move_count == 0:
                vid = queried_vid
            else:
                candidates = [v for v in vehicle_ids if v != banned_vid]
                if not candidates:
                    candidates = list(vehicle_ids)
                vid = random.choice(candidates)

            # Build a candidate (vehicle, action) pool. First try the
            # chosen vid; if every action of that vid fails the soft
            # constraints, fall back to other vids before giving up.
            vid_order = [vid] + [v for v in vehicle_ids if v != vid]

            applied = False
            applied_vid: str | None = None
            applied_act: Action | None = None
            for try_vid in vid_order:
                # Respect the quota: if we're forced to pick queried_vid,
                # do not switch away from it just to satisfy anti-zigzag.
                if queried_deficit >= remaining and try_vid != queried_vid:
                    continue
                if (step_idx == n_steps - 1
                        and queried_move_count == 0
                        and try_vid != queried_vid):
                    continue

                action_pool = list(valid_actions)
                random.shuffle(action_pool)
                for act in action_pool:
                    # 1. No consecutive identical (vehicle, action) pair.
                    if plan and plan[-1] == (try_vid, act):
                        continue

                    snapshot = copy.deepcopy(trial_state)
                    result = safe_apply_action(snapshot, try_vid, act)
                    if result is None:
                        continue

                    # 2. Anti-zigzag per vehicle.
                    new_pos = get_required_vehicle(snapshot, try_vid).position
                    history = pos_history[try_vid]
                    if len(history) >= 2 and new_pos == history[-2]:
                        continue

                    # All soft constraints satisfied → commit the step.
                    trial_state = snapshot
                    plan.append((try_vid, act))
                    pos_history[try_vid].append(new_pos)
                    if try_vid == queried_vid:
                        queried_move_count += 1
                    applied = True
                    applied_vid = try_vid
                    applied_act = act
                    break

                if applied:
                    break

            if not applied:
                break

        if len(plan) != n_steps:
            continue
        if queried_move_count < min_queried_moves:
            continue

        final_pos = get_required_vehicle(trial_state, queried_vid).position
        if final_pos == start_pos:
            continue

        # Replay onto the caller's state so mutations stick.
        for v_new in trial_state.vehicles:
            v_old = get_required_vehicle(state, v_new.id)
            v_old.position = v_new.position
            v_old.direction = v_new.direction
            v_old.inside_intersection = v_new.inside_intersection
            v_old.stopped = v_new.stopped
        state.event_log = trial_state.event_log
        state.step = trial_state.step
        return plan

    return None


# ── Per-vehicle trajectory replay ───────────────────────────────────────────

def replay_trajectories(
    init_state: ScenarioState,
    plan: list[tuple[str, Action]],
) -> dict[str, list[str]]:
    """
    Returns {vehicle_id: [positions]} — for every vehicle, the sequence
    of positions INCLUDING the starting position and after each of its
    own actions (NOT after every step). A vehicle that never moved has a
    1-element list holding only its starting position.
    """
    sim = copy.deepcopy(init_state)
    trace: dict[str, list[str]] = {v.id: [v.position] for v in sim.vehicles}

    for vid, act in plan:
        apply_action(sim, vid, act)
        trace[vid].append(get_required_vehicle(sim, vid).position)

    return trace


# ── Choice builder ──────────────────────────────────────────────────────────

def build_choices(
    env: Environment,
    queried_vid: str,
    queried_trace: list[str],
    other_traces: dict[str, list[str]],
) -> dict[str, dict] | None:
    """
    Build the 5-way MCQ choice set from real trajectory data.

    Distractor hierarchy (Problem 4 of the Task 1 fix plan). Each slot
    uses a fixed, prioritised list; the first candidate that yields a
    valid same-environment label is accepted. If none of the tiers
    yields a valid label, the whole example is rejected (return None)
    — the caller must retry. No cross-environment fallback is ever used.

    near_true  — plausible confusions:
      near_true_1: queried vehicle's start position.
                   rationale: "queried vehicle's start position"
      near_true_2: (a) intermediate position visited by queried vehicle
                   (b) final position of another vehicle in the scenario
                   (c) any same-environment position not yet used

    highly_false — implausible but same-environment:
      highly_false_1: same-environment position never visited by ANY
                      vehicle during the whole sequence.
      highly_false_2: same-environment position not occupied by any
                      vehicle at the end of the sequence (possibly
                      visited mid-sequence).
      If those same-environment pools are exhausted (multi_lane_road
      has only 3 real lanes), both slots can fall back to the
      HIGHLY_FALSE_LABELS_BY_ENV pool — same-domain labels that the
      simulator never produces as a real position.

    Returns the full 5-slot dict, or None on vocabulary exhaustion.
    """
    # ── correct ─────────────────────────────────────────────────────────
    correct_pos   = queried_trace[-1]
    correct_label = label_of(correct_pos)

    # ── near_true_1 ────────────────────────────────────────────────────
    start_pos = queried_trace[0]
    nt1_label = label_of(start_pos)
    if nt1_label == correct_label:
        # generate_sequence guarantees start ≠ final, so this is
        # defensive — just in case.
        return None
    nt1_rationale = "queried vehicle's start position"

    used: set[str] = {correct_label, nt1_label}
    

# --- Reachability helpers for stronger near_true / highly_false (MULTI_LANE) ---

    queried_start_pos = queried_trace[0]
    queried_move_budget = max(0, len(queried_trace) - 1)
    def _reachable_labels_for_queried(
        env_: Environment,
        start_pos_: str,
        move_budget_: int,
    ) -> set[str]:
        if move_budget_ <= 0:
            return {label_of(start_pos_)}
        if env_ == Environment.MULTI_LANE and start_pos_ in LANE_ORDER:
            start_idx = LANE_ORDER.index(start_pos_)
            idxs = {start_idx}
            frontier = {start_idx}
            for _ in range(move_budget_):
                nxt: set[int] = set()
                for i in frontier:
                    if i - 1 >= 0:
                        nxt.add(i - 1)
                    if i + 1 < len(LANE_ORDER):
                        nxt.add(i + 1)
                idxs |= nxt
                frontier = nxt
            return {label_of(LANE_ORDER[i]) for i in idxs}
        return {label_of(start_pos_)}
    reachable_by_queried = _reachable_labels_for_queried(
        env, queried_start_pos, queried_move_budget

    )



    # ── near_true_2: prioritised tiers ─────────────────────────────────
    # Tier a — intermediate position visited by the queried vehicle.
    nt2_label: str | None = None
    nt2_rationale: str | None = None

    if len(queried_trace) > 2:
        intermediates = [
            label_of(p) for p in queried_trace[1:-1]
            if label_of(p) not in used
        ]
        if intermediates:
            nt2_label = random.choice(intermediates)
            nt2_rationale = "intermediate position visited by queried vehicle"

    # Tier b — final position of another vehicle in the scenario.
    if nt2_label is None:
        other_ids = [vid for vid in other_traces if vid != queried_vid]
        random.shuffle(other_ids)
        for other_id in other_ids:
            candidate = label_of(other_traces[other_id][-1])
            if candidate not in used:
                nt2_label = candidate
                nt2_rationale = f"final position of vehicle {other_id}"
                break

    # Tier c — any same-environment position not yet used.
    if nt2_label is None:
        pool = [lab for lab in labels_for_env(env) if lab not in used]
        if pool:
            nt2_label = random.choice(pool)
            nt2_rationale = "same-environment position"

    if nt2_label is None:
        return None
    used.add(nt2_label)

    # ── highly_false_1: same-env position NEVER visited by any vehicle ──
    visited_labels: set[str] = set()
    for p in queried_trace:
        visited_labels.add(label_of(p))
    for vid, trace in other_traces.items():
        for p in trace:
            visited_labels.add(label_of(p))

    never_visited_pool = [
        lab for lab in labels_for_env(env)
        if lab not in visited_labels and lab not in used
    ]

    hf1_label: str | None = None
    hf1_rationale: str | None = None
# MULTI_LANE hardness upgrade: prefer realistic-but-unreachable lane labels
# (given the queried vehicle's move budget) over road-context decoys.
    lane_labels = {label_of(p) for p in LANE_ORDER}
    unreachable_realistic_lane_pool: list[str] = []
    if env == Environment.MULTI_LANE and queried_trace and queried_trace[0] in LANE_ORDER:
        unreachable_realistic_lane_pool = [
           lab for lab in lane_labels
           if lab not in reachable_by_queried and lab not in used
    ]

    hf1_label: str | None = None
    hf1_rationale: str | None = None

    if unreachable_realistic_lane_pool:
        hf1_label = random.choice(unreachable_realistic_lane_pool)
        hf1_rationale = (
            "realistic lane unreachable by queried vehicle within its move budget"
        )
    elif never_visited_pool:

    # In MULTI_LANE, prefer lane-position distractors over obvious
    # road-context labels (shoulder/median/oncoming/etc.).
        if env == Environment.MULTI_LANE:
            lane_pref = [lab for lab in never_visited_pool if lab in lane_labels]
            pool = lane_pref if lane_pref else never_visited_pool
        else:
            pool = never_visited_pool
        hf1_label = random.choice(pool)
        hf1_rationale = "same-environment position never visited by any vehicle"
    else:
        # Same-environment exhaustion → dedicated highly_false pool.
        # Keep road-context decoys as rare fallback.
        hf_pool = [
            lab for lab in highly_false_labels_for_env(env)
            if lab not in used
        ]
        if hf_pool:
            hf1_label = random.choice(hf_pool)
            hf1_rationale = "road-context position not reachable in simulation"
    #--------------

    if hf1_label is None:
        return None
    used.add(hf1_label)

    # ── highly_false_2: same-env position not occupied AT END ───────────
    end_occupied: set[str] = {label_of(queried_trace[-1])}
    for vid, trace in other_traces.items():
        end_occupied.add(label_of(trace[-1]))

    not_end_pool = [
        lab for lab in labels_for_env(env)
        if lab not in end_occupied and lab not in used
    ]

    hf2_label: str | None = None
    hf2_rationale: str | None = None
    unreachable_realistic_lane_pool_2: list[str] = []
    if env == Environment.MULTI_LANE and queried_start_pos in LANE_ORDER:
        unreachable_realistic_lane_pool_2 = [
            lab for lab in lane_labels
            if lab not in reachable_by_queried
            and lab not in used
            and lab not in end_occupied
        ]

    if unreachable_realistic_lane_pool_2:
        hf2_label = random.choice(unreachable_realistic_lane_pool_2)
        hf2_rationale = (
            "realistic lane unreachable by queried vehicle within its move budget"
        )
    elif not_end_pool:
        if env == Environment.MULTI_LANE:
            lane_pref = [lab for lab in not_end_pool if lab in lane_labels]
            pool = lane_pref if lane_pref else not_end_pool
        else:
            pool = not_end_pool
        hf2_label = random.choice(pool)
        hf2_rationale = "same-environment position not occupied at end of sequence"
    else:
        hf_pool = [
            lab for lab in highly_false_labels_for_env(env)
            if lab not in used
        ]
        if hf_pool:
            hf2_label = random.choice(hf_pool)
            hf2_rationale = "road-context position not reachable in simulation"
    #----       

    if hf2_label is None:
        return None

    return {
        "correct": {
            "text": correct_label,
            "type": "correct",
            "rationale": "queried vehicle's final position",
        },
        "near_true_1": {
            "text": nt1_label,
            "type": "near_true",
            "rationale": nt1_rationale,
        },
        "near_true_2": {
            "text": nt2_label,
            "type": "near_true",
            "rationale": nt2_rationale,
        },
        "highly_false_1": {
            "text": hf1_label,
            "type": "highly_false",
            "rationale": hf1_rationale,
        },
        "highly_false_2": {
            "text": hf2_label,
            "type": "highly_false",
            "rationale": hf2_rationale,
        },
    }


# ── Letter assignment ───────────────────────────────────────────────────────

LETTERS = ["A", "B", "C", "D", "E"]


def assign_letters(
    choices_dict: dict[str, dict],
    correct_key: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], str]:
    """
    Shuffles the five options and places the correct answer at *correct_key*.

    Returns:
        choices           – dict  {A..E: text}
        distractor_type   – dict  {A..E: type}  (only for distractors)
        rationale_by_letter – dict {A..E: rationale} (for ALL options,
                                including the correct one)
        answer            – the letter of the correct answer
    """
    items = list(choices_dict.values())
    random.shuffle(items)

    target_idx = LETTERS.index(correct_key)
    correct_item = next(it for it in items if it["type"] == "correct")
    items.remove(correct_item)
    items.insert(target_idx, correct_item)

    choices: dict[str, str] = {}
    distractor_type: dict[str, str] = {}
    rationale_by_letter: dict[str, str] = {}
    for letter, item in zip(LETTERS, items):
        choices[letter] = item["text"]
        rationale_by_letter[letter] = item["rationale"]
        if item["type"] != "correct":
            distractor_type[letter] = item["type"]

    return choices, distractor_type, rationale_by_letter, correct_key


# ── Event rendering from plan ───────────────────────────────────────────────

def plan_to_events(
    init_state: ScenarioState, plan: list[tuple[str, Action]]
) -> list[str]:
    """Replays `plan` to produce the human-readable event strings."""
    sim = copy.deepcopy(init_state)
    events: list[str] = []
    for vid, act in plan:
        ev = apply_action(sim, vid, act)
        if not ev:
            raise RuntimeError(
                f"Planned action rejected by FSM at replay: ({vid}, {act})."
            )
        events.append(ev)
    return events


# ── Validation gate (Problems 2 & 6) ─────────────────────────────────────────

def validate_example(
    example: dict,
    init_state: ScenarioState,
    plan: list[tuple[str, Action]],
    env: Environment,
    queried_vid: str,
) -> tuple[bool, str]:
    """
    Hard quality gate applied to every candidate example before it is
    appended to the dataset. Implements both the Problem 2 validation
    contract and the Problem 6 distractor quality gate.

    Returns (ok, reason). When ok is False, `reason` is a short string
    naming the failing check — generate_example() will treat it as a
    rejection and retry.

    Checks:
      V1. All 5 choice texts belong to the same-environment controlled
          vocabulary (real position labels ∪ HIGHLY_FALSE_LABELS_BY_ENV).
      V2. All 5 choice texts are pairwise distinct.
      V3. The correct answer text matches the replay-simulated final
          position of the queried vehicle (replay from init_state).
      V4. No distractor text equals the correct-answer text.
      V5. start_position != final_position for the queried vehicle.
      V6. Exactly 2 distractors typed `near_true` and exactly 2 typed
          `highly_false`.
    """
    choices = example["choices"]
    answer_letter = example["answer"]
    correct_text = choices[answer_letter]

    # V1 — environment consistency
    for letter, text in choices.items():
        if not is_env_consistent_label(text, env):
            return False, f"choice {letter} {text!r} not env-consistent for {env.value}"

    # V2 — pairwise distinct
    if len({text for text in choices.values()}) != 5:
        return False, "duplicate choice texts"

    # V3 — correct answer matches independent replay
    replay = copy.deepcopy(init_state)
    for vid, act in plan:
        apply_action(replay, vid, act)
    final_pos = get_required_vehicle(replay, queried_vid).position
    if label_of(final_pos) != correct_text:
        return False, (
            f"correct text {correct_text!r} does not match replayed final "
            f"position {label_of(final_pos)!r}"
        )

    # V4 — no distractor equals the correct answer
    for letter, text in choices.items():
        if letter == answer_letter:
            continue
        if text == correct_text:
            return False, f"distractor at {letter} equals correct answer"

    # V5 — start vs final
    start_pos = get_required_vehicle(init_state, queried_vid).position
    if start_pos == final_pos:
        return False, "queried vehicle start == final"

    # V6 — distractor type balance
    dtypes = example["distractor_type"]
    nt = sum(1 for t in dtypes.values() if t == "near_true")
    hf = sum(1 for t in dtypes.values() if t == "highly_false")
    if nt != 2 or hf != 2:
        return False, f"distractor type counts {nt=} {hf=} (expected 2/2)"

    return True, "ok"


# ── Single example generator ────────────────────────────────────────────────

def generate_example(
    example_id: int,
    correct_key: str,
    env_hint: Environment | None = None,
) -> dict | None:
    """
    Generates a single Task 1 example. Returns None if no valid sequence
    and a valid 5-way choice set can be produced within MAX_RETRIES.

    Args:
        example_id:   running index for the example id.
        correct_key:  letter (A..E) where the correct answer must land.
        env_hint:     if set, restrict generation to this environment
                      (used by the key_schedule to keep env balance).
    """
    for attempt in range(MAX_RETRIES):
        env = env_hint if env_hint is not None else random.choice(TASK_ENVS)
        n_steps = random.randint(MIN_STEPS, MAX_STEPS)

        if env == Environment.MULTI_LANE:
            state = build_multi_lane_scenario(NUM_VEHICLES)
        else:
            state = build_intersection_scenario(NUM_VEHICLES, with_intent=False)

        queried_vid = random.choice([v.id for v in state.vehicles])
        init_state = copy.deepcopy(state)

        sim_state = copy.deepcopy(state)
        plan = generate_sequence(sim_state, queried_vid, env, n_steps)
        if plan is None:
            continue

        events = plan_to_events(init_state, plan)
        trace  = replay_trajectories(init_state, plan)

        queried_trace = trace[queried_vid]
        raw_choices = build_choices(env, queried_vid, queried_trace, trace)
        if raw_choices is None:
            continue  # vocabulary exhaustion → retry

        # Post-condition: all five labels are distinct and valid for env
        # (cross-env hf is intentionally NOT valid for env — that's its role).
        letter_texts = {c["text"] for c in raw_choices.values()}
        if len(letter_texts) != 5:
            continue

        choices, distractor_type, rationale_by_letter, answer = assign_letters(
            raw_choices, correct_key
        )

        scenario_text = describe_scenario(init_state)
        question = f"Where is Vehicle {queried_vid} at the end of the sequence?"
        prompt = render_prompt(scenario_text, events, question, choices)

        example = {
            "id": f"task1_{example_id:04d}",
            "task": "position_tracking",
            "prompt": prompt,
            "scenario": {
                "vehicles": [
                    {
                        "id": v.id,
                        "position": v.position,
                        "direction": v.direction.value,
                    }
                    for v in init_state.vehicles
                ],
                "environment": env.value,
            },
            "events": events,
            "question": question,
            "choices": choices,
            "answer": answer,
            "distractor_type": distractor_type,
            "metadata": {
                "num_vehicles": NUM_VEHICLES,
                "num_events": len(events),
                "queried_vehicle": queried_vid,
                "environment": env.value,
                "difficulty": "base",
            },
            "audit": {
                "generator_version": GENERATOR_VERSION,
                "attempt": attempt,
                "queried_trace": queried_trace,
                "all_traces": trace,
                "plan": [[vid, act.name] for vid, act in plan],
                "rationale_by_letter": rationale_by_letter,
                "invariants": {
                    "start_ne_final": queried_trace[0] != queried_trace[-1],
                    "queried_moved": sum(1 for vid, _ in plan if vid == queried_vid),
                    # Problem 3 — strict per-environment vocabulary check:
                    # every option must be a real position label for `env`
                    # or belong to the dedicated highly_false pool for `env`.
                    # Cross-environment labels are rejected.
                    "all_labels_in_vocabulary": all(
                        is_valid_label(choices[L], env)
                        or choices[L] in highly_false_labels_for_env(env)
                        for L in LETTERS
                    ),
                    "five_distinct_options": len({choices[L] for L in LETTERS}) == 5,
                },
            },
        }

        # Hard quality gate (Problems 2 & 6).
        ok, reason = validate_example(example, init_state, plan, env, queried_vid)
        if not ok:
            # Reject and retry within the same generate_example() budget.
            continue

        return example

    return None


# ── Main generation loop ────────────────────────────────────────────────────

def generate_task1(n: int, output_path: str, seed: int | None = None) -> None:
    """Generates *n* examples, writes JSONL, prints distribution stats."""
    if seed is not None:
        random.seed(seed)

    if n % 5 != 0:
        raise ValueError("N must be a multiple of 5 for balanced key schedule.")
    if n % (5 * len(TASK_ENVS)) != 0:
        # Not strictly required, but warn the user: perfect joint balance
        # (letter × env) needs n divisible by 10.
        print(
            f"NOTE: n={n} is not divisible by {5 * len(TASK_ENVS)}; joint "
            f"letter×env balance will be approximate."
        )

    # Key schedule: exactly n/5 of each letter, shuffled.
    key_schedule: list[str] = []
    per_key = n // 5
    for letter in LETTERS:
        key_schedule.extend([letter] * per_key)
    random.shuffle(key_schedule)

    # Env schedule: stable 50/50 ratio enforced up front so we don't rely
    # on per-example random.choice drift over 300 examples.
    env_schedule: list[Environment] = []
    per_env = n // len(TASK_ENVS)
    remainder = n - per_env * len(TASK_ENVS)
    for env in TASK_ENVS:
        env_schedule.extend([env] * per_env)
    env_schedule.extend([random.choice(TASK_ENVS)] * remainder)
    random.shuffle(env_schedule)

    examples: list[dict] = []
    dropped = 0
    for idx in range(n):
        ex = generate_example(idx, key_schedule[idx], env_hint=env_schedule[idx])
        if ex is None:
            dropped += 1
            print(f"WARNING: could not generate example {idx}, skipping.")
            continue
        examples.append(ex)

    # ── Write output ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(examples)} examples to {output_path}")
    if dropped:
        print(f"Dropped {dropped} examples (generation failures).")
    print()

    # ── Answer distribution ─────────────────────────────────────────────
    answer_counts = {letter: 0 for letter in LETTERS}
    for ex in examples:
        answer_counts[ex["answer"]] += 1

    print("Answer distribution:")
    for letter in LETTERS:
        bar = "\u2588" * answer_counts[letter]
        print(f"  {letter}: {answer_counts[letter]:3d}  {bar}")

    # ── Environment distribution ────────────────────────────────────────
    env_counts: dict[str, int] = {}
    for ex in examples:
        e = ex["metadata"]["environment"]
        env_counts[e] = env_counts.get(e, 0) + 1

    print("\nEnvironment distribution:")
    for e, c in sorted(env_counts.items()):
        print(f"  {e}: {c}")

    # ── Distractor-type sanity check ────────────────────────────────────
    type_counts = {"near_true": 0, "highly_false": 0}
    for ex in examples:
        for t in ex["distractor_type"].values():
            type_counts[t] = type_counts.get(t, 0) + 1
    print("\nDistractor-type totals (expect 2:2 per example):")
    for t, c in type_counts.items():
        print(f"  {t}: {c}")


# ── Entry point ─────────────────────────────────────────────────────────────

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 1 position tracking generator")
    p.add_argument("--n", type=int, default=DEFAULT_N_EXAMPLES,
                   help="number of examples (must be multiple of 5)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for reproducibility")
    p.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "core" / "task1_position.jsonl"),
        help="output JSONL path",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    generate_task1(args.n, args.out, args.seed)
