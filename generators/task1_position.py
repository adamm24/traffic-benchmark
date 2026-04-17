"""
Task 1 — Position Tracking Generator
=====================================
Generates multiple-choice questions that test an LLM's ability to track
vehicle positions through a sequence of actions.

Environments: multi_lane_road, intersection  (roundabout excluded)
Vehicles:     3
Steps:        2–4
Choices:      1 correct, 2 near_true, 2 highly_false

Usage:
    python generators/task1_position.py
"""

from __future__ import annotations

import copy
import json
import random
import sys
import os
from pathlib import Path

# ── Make project root importable ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from domain.entities import (
    Action, Environment, Lane, ScenarioState, Vehicle,
)
from domain.scenario import (
    apply_action, build_intersection_scenario, build_multi_lane_scenario,
    LANE_ORDER,
)
from domain.render import describe_scenario, render_prompt

# ── Constants ───────────────────────────────────────────────────────────────

N_EXAMPLES   = 100          # change to 300 for the full core dataset
NUM_VEHICLES = 3
MIN_STEPS    = 2
MAX_STEPS    = 4
MAX_RETRIES  = 50           # per-example retry budget

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


# ── Helpers ─────────────────────────────────────────────────────────────────

def _lane_index(position: str) -> int:
    """Index of a lane in LEFT-CENTER-RIGHT order (–1 if not a lane)."""
    try:
        return LANE_ORDER.index(position)
    except ValueError:
        return -1


def safe_apply_action(
    state: ScenarioState,
    vehicle_id: str,
    action: Action,
) -> str | None:
    """
    Wrapper around apply_action() that enforces physical constraints.

    Returns the event string on success, or None if the action is invalid
    (edge-of-road or lane collision).
    """
    v = state.get_vehicle(vehicle_id)
    if v is None:
        return None

    # ── Multi-lane edge guards ──────────────────────────────────────────
    if state.environment == Environment.MULTI_LANE:
        idx = _lane_index(v.position)

        if action == Action.CHANGE_LEFT and idx <= 0:
            return None                         # already leftmost
        if action == Action.CHANGE_RIGHT and idx >= len(LANE_ORDER) - 1:
            return None                         # already rightmost

        # Compute destination lane and check for collisions
        # NOTE: collision check removed — in a discrete multi-lane model,
        # vehicles at the same lane are at different points along the road,
        # so sharing a lane is physically plausible.  The old check made
        # it impossible to generate any valid multi-lane sequence when all
        # 3 lanes were occupied (deadlock).

    # ── Intersection guards ────────────────────────────────────────────
    if state.environment == Environment.INTERSECTION:
        # Block turn if not inside yet
        if action in (Action.TURN_LEFT, Action.TURN_RIGHT):
            if not v.inside_intersection:
                return None                     # must MOVE_FORWARD first
        # Block MOVE_FORWARD if already inside (no-op → wastes a step)
        if action == Action.MOVE_FORWARD and v.inside_intersection:
            return None
        # Block MOVE_FORWARD from an exit position (prevents re-entry loop)
        if action == Action.MOVE_FORWARD and v.position.endswith("_exit"):
            return None

    return apply_action(state, vehicle_id, action)


# ── Sequence generation ─────────────────────────────────────────────────────

def generate_sequence(
    state: ScenarioState,
    queried_vid: str,
    env: Environment,
    n_steps: int,
) -> list[str] | None:
    """
    Produces a valid n-step event sequence guaranteeing:
      • the queried vehicle is moved at least once
      • its final position differs from its starting position

    Returns None if it cannot satisfy both constraints within MAX_RETRIES
    inner attempts.
    """
    vehicle_ids = [v.id for v in state.vehicles]
    valid_actions = ACTIONS_BY_ENV[env]
    start_pos = state.get_vehicle(queried_vid).position

    # For intersection, queried vehicle needs ≥2 actions to enter + turn.
    # For multi_lane, 1 action suffices (each lane change moves position).
    min_queried_moves = 2 if env == Environment.INTERSECTION else 1

    for _ in range(MAX_RETRIES):
        trial_state = copy.deepcopy(state)
        events: list[str] = []
        queried_move_count = 0

        for step_idx in range(n_steps):
            remaining = n_steps - step_idx
            queried_deficit = min_queried_moves - queried_move_count

            # Reserve enough remaining steps for the queried vehicle
            if queried_deficit >= remaining:
                vid = queried_vid
            elif step_idx == n_steps - 1 and queried_move_count == 0:
                vid = queried_vid
            else:
                vid = random.choice(vehicle_ids)

            # Try to find a valid action for this vehicle
            action_pool = list(valid_actions)
            random.shuffle(action_pool)
            applied = False
            for act in action_pool:
                snapshot = copy.deepcopy(trial_state)
                result = safe_apply_action(snapshot, vid, act)
                if result is not None:
                    trial_state = snapshot
                    events.append(result)
                    if vid == queried_vid:
                        queried_move_count += 1
                    applied = True
                    break

            if not applied:
                break                           # dead end → retry sequence

        if len(events) != n_steps:
            continue                            # incomplete sequence

        # Queried vehicle must have moved at least min_queried_moves times
        if queried_move_count < min_queried_moves:
            continue

        # Final position must differ from starting position
        final_pos = trial_state.get_vehicle(queried_vid).position
        if final_pos == start_pos:
            continue

        # Copy the mutated state back so the caller sees final positions
        for v_new in trial_state.vehicles:
            v_old = state.get_vehicle(v_new.id)
            v_old.position = v_new.position
            v_old.direction = v_new.direction
            v_old.inside_intersection = v_new.inside_intersection
            v_old.stopped = v_new.stopped

        state.event_log = trial_state.event_log
        state.step = trial_state.step
        return events

    return None                                 # exhausted retries


# ── Intermediate position tracker ───────────────────────────────────────────

def _track_intermediate(
    init_state: ScenarioState,
    events: list[str],
    queried_vid: str,
    env: Environment,
) -> str | None:
    """
    Replays the event sequence and returns the first intermediate position
    of the queried vehicle that differs from start and final.
    Returns None if no such intermediate exists.
    """
    sim = copy.deepcopy(init_state)
    vehicle_ids = [v.id for v in sim.vehicles]
    valid_actions = ACTIONS_BY_ENV[env]
    start_pos = sim.get_vehicle(queried_vid).position

    # We need to re-simulate to find intermediate positions
    # Parse events to reconstruct vehicle-action pairs
    positions: list[str] = []
    replay = copy.deepcopy(init_state)

    for event_text in events:
        # Identify which vehicle acted
        for vid in vehicle_ids:
            prefix = f"Vehicle {vid} "
            if event_text.startswith(prefix):
                action_str = event_text[len(prefix):].rstrip(".")
                # Find matching action enum
                for act in Action:
                    if act.value == action_str:
                        apply_action(replay, vid, act)
                        if vid == queried_vid:
                            positions.append(replay.get_vehicle(queried_vid).position)
                        break
                break

    final_pos = replay.get_vehicle(queried_vid).position
    for p in positions[:-1]:                    # exclude final
        if p != start_pos and p != final_pos:
            return p
    return None


# ── Choice builder ──────────────────────────────────────────────────────────

# Positions that only exist in a different environment → highly false
_CROSS_ENV_FALSE = {
    Environment.MULTI_LANE: [
        "inside the intersection",
        "the roundabout lane",
        "the northern exit",
        "the southern exit",
    ],
    Environment.INTERSECTION: [
        "the left lane",
        "the right lane",
        "the roundabout lane",
        "the center lane",
    ],
}

# Human-readable labels for internal position strings
_POS_LABEL = {
    "left_lane":           "the left lane",
    "center_lane":         "the center lane",
    "right_lane":          "the right lane",
    "inside_intersection": "inside the intersection",
    "roundabout_lane":     "the roundabout lane",
    "north_approach":      "the northern approach",
    "south_approach":      "the southern approach",
    "east_approach":       "the eastern approach",
    "west_approach":       "the western approach",
    "north_exit":          "the northern exit",
    "south_exit":          "the southern exit",
    "east_exit":           "the eastern exit",
    "west_exit":           "the western exit",
}


def _label(pos: str) -> str:
    return _POS_LABEL.get(pos, pos.replace("_", " "))


def _adjacent_lane(pos: str) -> str | None:
    """Returns a neighbouring lane label, or None."""
    idx = _lane_index(pos)
    if idx == -1:
        return None
    if idx == 0:
        return _label(LANE_ORDER[1])
    if idx == len(LANE_ORDER) - 1:
        return _label(LANE_ORDER[-2])
    return _label(LANE_ORDER[random.choice([idx - 1, idx + 1])])


def build_choices(
    correct_pos: str,
    start_pos: str,
    intermediate_pos: str | None,
    env: Environment,
) -> dict[str, dict]:
    """
    Returns a dict with exactly 5 entries:
        "correct":       { "text": ..., "type": "correct" }
        "near_true_1":   { "text": ..., "type": "near_true" }
        "near_true_2":   { "text": ..., "type": "near_true" }
        "highly_false_1":{ "text": ..., "type": "highly_false" }
        "highly_false_2":{ "text": ..., "type": "highly_false" }
    """
    correct_label = _label(correct_pos)

    # ── near_true 1: starting position ──────────────────────────────────
    nt1 = _label(start_pos)

    # ── near_true 2: intermediate or adjacent lane ──────────────────────
    if intermediate_pos and _label(intermediate_pos) != correct_label:
        nt2 = _label(intermediate_pos)
    else:
        adj = _adjacent_lane(correct_pos) or _adjacent_lane(start_pos)
        nt2 = adj if adj and adj != correct_label and adj != nt1 else None

    # Fallback: pick any other same-env position
    if nt2 is None or nt2 == correct_label or nt2 == nt1:
        if env == Environment.MULTI_LANE:
            all_labels = [_label(l) for l in LANE_ORDER]
        else:
            all_labels = [
                _label(f"{d}_approach") for d in ("north", "south", "east", "west")
            ] + ["inside the intersection"] + [
                _label(f"{d}_exit") for d in ("north", "south", "east", "west")
            ]
        candidates = [l for l in all_labels if l != correct_label and l != nt1]
        nt2 = random.choice(candidates) if candidates else "unknown position"

    # ── highly_false 1 & 2: cross-environment positions ─────────────────
    false_pool = list(_CROSS_ENV_FALSE[env])
    # Remove any that accidentally match correct or near-trues
    false_pool = [f for f in false_pool
                  if f != correct_label and f != nt1 and f != nt2]
    random.shuffle(false_pool)
    hf1 = false_pool[0] if len(false_pool) > 0 else "off the road"
    hf2 = false_pool[1] if len(false_pool) > 1 else "unknown location"

    return {
        "correct":        {"text": correct_label,  "type": "correct"},
        "near_true_1":    {"text": nt1,             "type": "near_true"},
        "near_true_2":    {"text": nt2,             "type": "near_true"},
        "highly_false_1": {"text": hf1,             "type": "highly_false"},
        "highly_false_2": {"text": hf2,             "type": "highly_false"},
    }


# ── Letter assignment ───────────────────────────────────────────────────────

LETTERS = ["A", "B", "C", "D", "E"]


def assign_letters(
    choices_dict: dict[str, dict],
    correct_key: str,
) -> tuple[dict[str, str], dict[str, str], str]:
    """
    Shuffles the five options and places the correct answer at *correct_key*.

    Returns:
        choices        – dict  {A..E: text}
        distractor_type – dict  {A..E: type}  (only for distractors)
        answer         – the letter of the correct answer
    """
    items = list(choices_dict.values())
    random.shuffle(items)

    # Ensure the correct item lands at correct_key
    target_idx = LETTERS.index(correct_key)
    correct_item = next(it for it in items if it["type"] == "correct")
    items.remove(correct_item)
    items.insert(target_idx, correct_item)

    choices = {}
    distractor_type = {}
    for letter, item in zip(LETTERS, items):
        choices[letter] = item["text"]
        if item["type"] != "correct":
            distractor_type[letter] = item["type"]

    return choices, distractor_type, correct_key


# ── Single example generator ────────────────────────────────────────────────

def generate_example(example_id: int, correct_key: str) -> dict | None:
    """
    Generates a single Task 1 example.
    Returns None if no valid sequence is found after MAX_RETRIES attempts.
    """
    for _ in range(MAX_RETRIES):
        # 1. Choose environment and step count
        env = random.choice(TASK_ENVS)
        n_steps = random.randint(MIN_STEPS, MAX_STEPS)

        # 2. Build initial scenario
        if env == Environment.MULTI_LANE:
            state = build_multi_lane_scenario(NUM_VEHICLES)
        else:
            state = build_intersection_scenario(NUM_VEHICLES, with_intent=False)

        # 3. Choose queried vehicle
        queried_vid = random.choice([v.id for v in state.vehicles])

        # 4. Snapshot the initial state for rendering
        init_state = copy.deepcopy(state)
        start_pos = state.get_vehicle(queried_vid).position

        # 5. Generate event sequence (on a deep copy)
        sim_state = copy.deepcopy(state)
        events = generate_sequence(sim_state, queried_vid, env, n_steps)
        if events is None:
            continue

        # 6. Record final position
        final_pos = sim_state.get_vehicle(queried_vid).position

        # 7. Find intermediate position
        intermediate_pos = _track_intermediate(init_state, events, queried_vid, env)

        # 8. Build choices
        raw_choices = build_choices(final_pos, start_pos, intermediate_pos, env)

        # 9. Assign letters (placing correct answer at correct_key)
        choices, distractor_type, answer = assign_letters(raw_choices, correct_key)

        # 10. Render prompt
        scenario_text = describe_scenario(init_state)
        question = f"Where is Vehicle {queried_vid} at the end of the sequence?"
        prompt = render_prompt(scenario_text, events, question, choices)

        return {
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
        }

    return None                                 # exhausted retries


# ── Main generation loop ────────────────────────────────────────────────────

def generate_task1(n: int, output_path: str) -> None:
    """Generates *n* examples, writes JSONL, prints distribution stats."""

    # Key schedule: exactly n/5 of each letter, shuffled
    assert n % 5 == 0, "N_EXAMPLES must be a multiple of 5 for balanced keys"
    key_schedule: list[str] = []
    per_key = n // 5
    for letter in LETTERS:
        key_schedule.extend([letter] * per_key)
    random.shuffle(key_schedule)

    examples: list[dict] = []
    for idx in range(n):
        ex = generate_example(idx, key_schedule[idx])
        if ex is None:
            print(f"WARNING: could not generate example {idx}, skipping.")
            continue
        examples.append(ex)

    # ── Write output ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(examples)} examples to {output_path}\n")

    # ── Answer distribution ─────────────────────────────────────────────
    answer_counts = {l: 0 for l in LETTERS}
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


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_PATH = str(PROJECT_ROOT / "dataset" / "core" / "task1_position.jsonl")
    generate_task1(N_EXAMPLES, OUTPUT_PATH)
