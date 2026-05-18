"""Task 1 position-tracking generator."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from domain.entities import (
    Action, Direction, Environment, Lane, ScenarioState, Vehicle,
)
from domain.scenario import (
    apply_action,
    build_intersection_scenario,
    build_multi_lane_scenario,
)
from domain.render import render_prompt
from domain.vocabulary import (
    POSITION_LABELS,
    label_of, labels_for_env,
)


GENERATOR_VERSION = "task1_position.v16"   # bump when contract changes
DEFAULT_N_EXAMPLES = 100                  # CLI --n overrides; 300 = full core
NUM_VEHICLES = 3
MIN_STEPS    = 3
MAX_STEPS    = 5
MAX_RETRIES  = 150                        # per-example retry budget
MAX_CONSEC   = 2                          # max consecutive actions by same vehicle
MIN_DISTINCT_ACTORS  = 2                  # harder sequences: at least 2 vehicles act
MIN_DISTINCT_ACTIONS = 2                  # avoid monotone one-action plans
SEQUENCE_TRIES_PER_ATTEMPT = 8            # local retries before burning outer attempt
CHOICE_RETRIES_PER_PLAN = 6               # avoid losing attempt to one unlucky draw
INTERSECTION_INSIDE_START_PROB = 0.60
INTERSECTION_INSIDE_QUERY_BIAS = 0.45
MULTI_LANE_BALANCE_WARMUP = 18
MULTI_LANE_CORRECT_LABEL_GAP = 6
INTERSECTION_BALANCE_WARMUP = 18
INTERSECTION_CORRECT_LABEL_GAP = 7
TASK_SLOT_RETRIES = 240
PLAN_TEMPLATE_MAX_REUSE = 30
INTERSECTION_QUERIED_SHAPE_MAX_REUSE = 40
MULTI_LANE_QUERIED_SHAPE_MAX_REUSE = 40

INTERSECTION_ALLOWED_CHOICE_LABELS = {
    "inside the intersection",
    "the northern exit",
    "the southern exit",
    "the eastern exit",
    "the western exit",
}
INTERSECTION_CORRECT_LABELS = {
    "the northern exit",
    "the southern exit",
    "the eastern exit",
    "the western exit",
}

MULTI_LANE_CHOICE_LABELS = [
    label_of("left_lane"),
    label_of("center_lane"),
    label_of("right_lane"),
    label_of("roundabout_lane"),
    label_of("north_exit"),
    label_of("south_exit"),
    label_of("east_exit"),
    label_of("west_exit"),
]

ALL_DOMAIN_LABELS = set(POSITION_LABELS.values())
MULTI_LANE_ORDER = [Lane.LEFT.value, Lane.CENTER.value, Lane.RIGHT.value]

TASK_ENVS = [Environment.INTERSECTION, Environment.MULTI_LANE]

DIFFICULTIES = ["easy", "medium", "hard"]

# Difficulty profiles are tuned for benchmark-quality variance:
# - easy:  still multi-step, moderate interaction
# - medium: stronger interaction and less lexical shortcut room
# - hard: max interaction, stronger ambiguity pressure
DIFFICULTY_PROFILES: dict[str, dict] = {
    "easy": {
        "allowed_envs": [Environment.INTERSECTION, Environment.MULTI_LANE],
        "step_min": 3,
        "step_max": 4,
        "min_queried_moves": 1,
        "max_queried_moves": 2,
        "min_distinct_actors": 2,
        "min_distinct_actions": 1,
        "min_nonqueried_moves": 1,
        "min_vehicles_moved": 2,
        "require_queried_interleaving": False,
        "require_near_true_from_other": False,
        "require_other_vehicle_multi_step": False,
        "prefer_ambiguous_families": False,
    },
    "medium": {
        "allowed_envs": [Environment.INTERSECTION, Environment.MULTI_LANE],
        "step_min": 3,
        "step_max": 5,
        "min_queried_moves": 2,
        "max_queried_moves": 3,
        "min_distinct_actors": 2,
        "min_distinct_actions": 2,
        "min_nonqueried_moves": 2,
        "min_vehicles_moved": 2,
        "require_queried_interleaving": True,
        "require_near_true_from_other": True,
        "require_other_vehicle_multi_step": False,
        "prefer_ambiguous_families": False,
    },
    "hard": {
        "allowed_envs": [Environment.INTERSECTION, Environment.MULTI_LANE],
        "step_min": 4,
        "step_max": 5,
        "min_queried_moves": 2,
        "max_queried_moves": 4,
        "min_distinct_actors": 3,
        "min_distinct_actions": 2,
        "min_nonqueried_moves": 3,
        "min_vehicles_moved": 3,
        "require_queried_interleaving": True,
        "require_near_true_from_other": True,
        "require_other_vehicle_multi_step": True,
        "prefer_ambiguous_families": True,
    },
}

QUESTION_TEMPLATES = [
    "Where is Vehicle {vid} at the end of the sequence?",
    "After all events, what is Vehicle {vid}'s final position?",
    "Track Vehicle {vid}. Which position does it occupy at the end?",
    "Based on the full sequence, where does Vehicle {vid} end up?",
]

EVENT_HEADER_TEMPLATES = [
    "Sequence of events:",
    "Event sequence:",
    "Observed actions:",
    "Action log:",
]

QUESTION_HEADER_TEMPLATES = [
    "Question:",
    "Final question:",
    "Task:",
]

INTERSECTION_PROMPT_NOTE = (
    'Note: In intersection scenarios, a vehicle in the "X approach" is heading '
    'toward the X exit (its current heading is X). "moves forward" advances it '
    "one step in that heading. If a vehicle starts inside the intersection, its "
    "heading is stated explicitly in the scenario."
)

# Actions valid per environment (Task 1 — position only, no roundabout)
# MOVE_FORWARD excluded from MULTI_LANE: it doesn't change lane position,
# so it's a no-op for position tracking.
ACTIONS_BY_ENV = {
    Environment.MULTI_LANE: [
        Action.STOP,
        Action.CHANGE_LEFT,
        Action.CHANGE_RIGHT,
    ],
    Environment.INTERSECTION: [
        Action.STOP,
        Action.MOVE_FORWARD,
        Action.TURN_LEFT,
        Action.TURN_RIGHT,
    ],
}

MAX_QUERIED_MOVES_BY_ENV = {
    Environment.INTERSECTION: 2,
    Environment.MULTI_LANE: 4,
}





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


def _safe_apply_action_for_env(
    state: ScenarioState,
    vehicle_id: str,
    action: Action,
    env: Environment,
) -> str | None:
    before_pos = get_required_vehicle(state, vehicle_id).position
    event = safe_apply_action(state, vehicle_id, action)
    if event is None:
        return None
    if env != Environment.MULTI_LANE:
        return event
    if action not in {Action.CHANGE_LEFT, Action.CHANGE_RIGHT}:
        return event
    after_pos = get_required_vehicle(state, vehicle_id).position
    if after_pos == before_pos:
        return event
    try:
        lane_label = label_of(after_pos)
    except ValueError:
        return event
    return f"Vehicle {vehicle_id} changes to {lane_label}."


def _build_multi_lane_scenario(num_vehicles: int = 3) -> ScenarioState:
    """Task1 wrapper around the shared domain multi-lane builder."""
    return build_multi_lane_scenario(num_vehicles)


def _num_vehicles_for_env(env: Environment) -> int:
    if env == Environment.MULTI_LANE:
        return 2
    return NUM_VEHICLES


def _build_intersection_scenario_task1(num_vehicles: int = 3) -> ScenarioState:
    """
    Intersection builder with occasional in-progress traffic:
    one vehicle may start inside the intersection body.
    This increases valid queried-shape diversity while staying domain-valid.
    """
    state = build_intersection_scenario(num_vehicles, with_intent=False)
    if random.random() < INTERSECTION_INSIDE_START_PROB:
        v = random.choice(state.vehicles)
        v.position = "inside_intersection"
        v.inside_intersection = True
    return state


def _is_abab_vehicle_pattern(window: list[str]) -> bool:
    """True for a 4-step vehicle-id window shaped A-B-A-B with A != B."""
    return (
        len(window) == 4
        and window[0] == window[2]
        and window[1] == window[3]
        and window[0] != window[1]
    )


def _would_create_abab_vehicle_pattern(
    plan: list[tuple[str, Action]],
    next_vid: str,
) -> bool:
    """Checks whether appending next_vid would make the last 4 steps A-B-A-B."""
    recent_vids = [vid for vid, _ in plan[-3:]]
    return _is_abab_vehicle_pattern(recent_vids + [next_vid])


def _contains_abab_vehicle_pattern(plan: list[tuple[str, Action]]) -> bool:
    """True iff any 4-step window in plan is A-B-A-B on vehicle ids."""
    vids = [vid for vid, _ in plan]
    return any(
        _is_abab_vehicle_pattern(vids[i:i + 4])
        for i in range(0, max(0, len(vids) - 3))
    )


def _has_action_streak(plan: list[tuple[str, Action]], streak_len: int = 3) -> bool:
    """True if same action appears `streak_len` times consecutively."""
    if streak_len <= 1:
        return False
    acts = [act for _, act in plan]
    for i in range(0, len(acts) - streak_len + 1):
        window = acts[i:i + streak_len]
        if len(set(window)) == 1:
            return True
    return False


def _queried_moves_interleaved(plan: list[tuple[str, Action]], queried_vid: str) -> bool:
    """True iff queried vehicle's moves are separated by at least one other move."""
    idxs = [i for i, (vid, _) in enumerate(plan) if vid == queried_vid]
    if len(idxs) <= 1:
        return True
    return all((b - a) > 1 for a, b in zip(idxs, idxs[1:]))


def _has_vehicle_palindrome(plan: list[tuple[str, Action]]) -> bool:
    """Reject symmetric actor patterns like A-B-C-B-A."""
    vids = [vid for vid, _ in plan]
    return len(vids) >= 5 and vids == list(reversed(vids))


def _difficulty_profile(name: str) -> dict:
    if name not in DIFFICULTY_PROFILES:
        return DIFFICULTY_PROFILES["medium"]
    return DIFFICULTY_PROFILES[name]


def _profile_for_env(profile: dict, env: Environment, difficulty: str) -> dict:
    """
    Environment-aware profile tuning: keep correctness strict, relax only
    high-rejection pressure for multi-lane to avoid generation collapse.
    """
    tuned = copy.deepcopy(profile)
    if env == Environment.INTERSECTION:
        if difficulty == "easy":
            tuned["step_min"] = 4
            tuned["step_max"] = 5
            tuned["min_queried_moves"] = 2
            tuned["max_queried_moves"] = 2
            tuned["min_distinct_actors"] = 2
            tuned["min_distinct_actions"] = 1
            tuned["min_nonqueried_moves"] = 2
            tuned["min_vehicles_moved"] = 2
            tuned["require_queried_interleaving"] = False
            tuned["require_other_vehicle_multi_step"] = True
            tuned["prefer_ambiguous_families"] = False
        elif difficulty == "medium":
            tuned["step_min"] = 5
            tuned["step_max"] = 5
            tuned["min_queried_moves"] = 2
            tuned["max_queried_moves"] = 2
            tuned["min_distinct_actors"] = 2
            tuned["min_distinct_actions"] = 2
            tuned["min_nonqueried_moves"] = 2
            tuned["min_vehicles_moved"] = 2
            tuned["require_queried_interleaving"] = True
            tuned["require_other_vehicle_multi_step"] = True
            tuned["prefer_ambiguous_families"] = False
        else:  # hard
            tuned["step_min"] = 5
            tuned["step_max"] = 5
            tuned["min_queried_moves"] = 2
            tuned["max_queried_moves"] = 2
            tuned["min_distinct_actors"] = 3
            tuned["min_distinct_actions"] = 2
            tuned["min_nonqueried_moves"] = 3
            tuned["min_vehicles_moved"] = 3
            tuned["require_queried_interleaving"] = True
            tuned["require_other_vehicle_multi_step"] = True
            tuned["prefer_ambiguous_families"] = True
    elif env == Environment.MULTI_LANE:
        if difficulty == "easy":
            tuned["step_min"] = 3
            tuned["step_max"] = 3
            tuned["min_queried_moves"] = 1
            tuned["max_queried_moves"] = 2
            tuned["min_distinct_actors"] = 2
            tuned["min_distinct_actions"] = 1
            tuned["min_nonqueried_moves"] = 1
            tuned["min_vehicles_moved"] = 2
            tuned["require_queried_interleaving"] = False
            tuned["require_other_vehicle_multi_step"] = False
            tuned["prefer_ambiguous_families"] = False
        elif difficulty == "medium":
            tuned["step_min"] = 3
            tuned["step_max"] = 4
            tuned["min_queried_moves"] = 2
            tuned["max_queried_moves"] = 3
            tuned["min_distinct_actors"] = 2
            tuned["min_distinct_actions"] = 1
            tuned["min_nonqueried_moves"] = 1
            tuned["min_vehicles_moved"] = 2
            tuned["require_queried_interleaving"] = True
            tuned["require_other_vehicle_multi_step"] = False
            tuned["require_near_true_from_other"] = False
            tuned["prefer_ambiguous_families"] = False
        else:  # hard
            tuned["step_min"] = 5
            tuned["step_max"] = 5
            tuned["min_queried_moves"] = 2
            tuned["max_queried_moves"] = 3
            tuned["min_distinct_actors"] = 2
            tuned["min_distinct_actions"] = 2
            tuned["min_nonqueried_moves"] = 2
            tuned["min_vehicles_moved"] = 2
            tuned["require_queried_interleaving"] = True
            tuned["require_other_vehicle_multi_step"] = False
            tuned["require_near_true_from_other"] = False
            tuned["prefer_ambiguous_families"] = False
    return tuned


def _inc(counter: dict[str, int] | None, key: str) -> None:
    if counter is None:
        return
    counter[key] = counter.get(key, 0) + 1


def _choice_labels_for_env(env: Environment) -> list[str]:
    if env == Environment.INTERSECTION:
        labels = list(labels_for_env(env))
        return [lab for lab in labels if lab in INTERSECTION_ALLOWED_CHOICE_LABELS]
    if env == Environment.MULTI_LANE:
        return list(MULTI_LANE_CHOICE_LABELS)
    return list(labels_for_env(env))


def _plan_template_key(plan: list[tuple[str, Action]], queried_vid: str, env: Environment) -> tuple:
    return (
        env.value,
        tuple(
            ("Q" if vid == queried_vid else "O", act.name)
            for vid, act in plan
        ),
    )


def _plan_template_from_example(example: dict) -> tuple:
    env = Environment(example["metadata"]["environment"])
    queried = example["metadata"]["queried_vehicle"]
    plan_raw = example["audit"]["plan"]
    return (
        env.value,
        tuple(
            ("Q" if vid == queried else "O", act_name)
            for vid, act_name in plan_raw
        ),
    )


def _queried_shape_from_example(example: dict) -> tuple:
    env = Environment(example["metadata"]["environment"])
    queried = example["metadata"]["queried_vehicle"]
    plan_raw = example["audit"]["plan"]
    queried_actions = tuple(
        act_name for vid, act_name in plan_raw if vid == queried
    )
    return (env.value, queried_actions)


def _pick_question(queried_vid: str) -> str:
    return random.choice(QUESTION_TEMPLATES).format(vid=queried_vid)


def _render_prompt_with_variation(
    scenario_text: str,
    events: list[str],
    question: str,
    choices: dict[str, str],
    env: Environment,
) -> tuple[str, str, str]:
    """Vary prompt headers without changing task semantics."""
    base = render_prompt(scenario_text, events, question, choices)
    if env == Environment.INTERSECTION:
        base = f"{INTERSECTION_PROMPT_NOTE}\n\n{base}"
    ev_header = random.choice(EVENT_HEADER_TEMPLATES)
    q_header = random.choice(QUESTION_HEADER_TEMPLATES)
    prompt = base.replace("Sequence of events:", ev_header, 1)
    prompt = prompt.replace("Question:", q_header, 1)
    return prompt, ev_header, q_header


def _describe_task1_scenario(state: ScenarioState) -> str:
    """Task 1 scenario text using the controlled vocabulary."""
    count_word = {2: "Two", 3: "Three", 4: "Four", 5: "Five"}.get(
        len(state.vehicles), str(len(state.vehicles))
    )
    env_label = {
        Environment.INTERSECTION: "an intersection",
        Environment.MULTI_LANE: "a multi-lane road",
        Environment.ROUNDABOUT: "a roundabout",
    }.get(state.environment, state.environment.value)
    lines = [f"{count_word} vehicles are at {env_label}."]
    for v in state.vehicles:
        label = label_of(v.position)
        if v.position == "inside_intersection":
            desc = f"Vehicle {v.id} is inside the intersection, heading {v.direction.value}"
        else:
            desc = f"Vehicle {v.id} is in {label}"
        if v.intent:
            desc += f", intending to {v.intent.value}"
        lines.append(desc + ".")
    return "\n".join(lines)


def _label_family(label: str) -> str:
    ll = label.lower()
    if "approach" in ll:
        return "approach"
    if "exit" in ll:
        return "exit"
    if "inside the intersection" in ll:
        return "inside"
    if "lane" in ll or "strip" in ll:
        return "lane_like"
    return "other"


def _weighted_label_pick(labels: list[str], weights: list[float]) -> str:
    return random.choices(labels, weights=weights, k=1)[0]


def _weighted_pick_two_distinct(labels: list[str], weights: list[float]) -> tuple[str, str]:
    first = _weighted_label_pick(labels, weights)
    rem_labels = [lab for lab in labels if lab != first]
    if not rem_labels:
        raise ValueError("Need at least two distinct labels.")
    rem_weights = [weights[labels.index(lab)] for lab in rem_labels]
    second = _weighted_label_pick(rem_labels, rem_weights)
    return first, second


def _hf_weight(
    label: str,
    env: Environment,
    hf_label_usage: dict[str, int] | None,
) -> float:
    _ = env
    usage = 0 if hf_label_usage is None else hf_label_usage.get(label, 0)
    return 1.0 / (1.0 + usage)


def _example_signature(example: dict) -> tuple:
    """Dedup key: (env, init_state, plan, queried_vehicle)."""
    vehicles = tuple(sorted(
        (v["id"], v["position"], v["direction"])
        for v in example["scenario"]["vehicles"]
    ))
    plan = tuple(tuple(step) for step in example["audit"]["plan"])
    queried = example["metadata"]["queried_vehicle"]
    env = example["metadata"]["environment"]
    return (env, vehicles, queried, plan)


def _has_position_overlap(state: ScenarioState) -> bool:
    positions = [v.position for v in state.vehicles]
    return len(set(positions)) != len(positions)


def _would_overlap_on_action(
    state: ScenarioState,
    vehicle_id: str,
    action: Action,
    env: Environment,
) -> bool:
    vehicle = state.get_vehicle(vehicle_id)
    if vehicle is None:
        return False
    occupied = {
        v.position
        for v in state.vehicles
        if v.id != vehicle_id
    }

    if env == Environment.INTERSECTION:
        if action == Action.MOVE_FORWARD:
            return "inside_intersection" in occupied
        if action in (Action.TURN_LEFT, Action.TURN_RIGHT) and vehicle.position == "inside_intersection":
            order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
            idx = order.index(vehicle.direction)
            if action == Action.TURN_RIGHT:
                new_dir = order[(idx + 1) % 4]
            else:
                new_dir = order[(idx - 1) % 4]
            return f"{new_dir.value}_exit" in occupied
        return False

    if env == Environment.MULTI_LANE and action in (Action.CHANGE_LEFT, Action.CHANGE_RIGHT):
        if vehicle.position not in MULTI_LANE_ORDER:
            return False
        idx = MULTI_LANE_ORDER.index(vehicle.position)
        if action == Action.CHANGE_LEFT:
            if idx == 0:
                return False
            target = MULTI_LANE_ORDER[idx - 1]
        else:
            if idx == len(MULTI_LANE_ORDER) - 1:
                return False
            target = MULTI_LANE_ORDER[idx + 1]
        return target in occupied

    return False



def generate_sequence(
    state: ScenarioState,
    queried_vid: str,
    env: Environment,
    n_steps: int,
    *,
    min_queried_moves: int,
    max_queried_moves: int,
    min_distinct_actors: int,
    min_distinct_actions: int,
    min_nonqueried_moves: int,
    min_vehicles_moved: int,
    require_queried_interleaving: bool,
    require_other_vehicle_multi_step: bool,
    max_unique_positions: int | None = None,
    choice_label_cap: int | None = None,
    choice_label_vocab: set[str] | None = None,
) -> list[tuple[str, Action]] | None:
    """
    Produces a valid n-step event sequence guaranteeing:
      • the queried vehicle is moved at least `min_queried_moves` times
      • its final position differs from its starting position

    Quality soft-constraints (Problem 5 of the Task 1 fix plan):
      • No consecutive identical (vehicle, action) pairs.
      • At most MAX_CONSEC consecutive actions performed by the same
        vehicle (prevents "A acts 4 times in a row" monotony).
      • No 4-step alternating vehicle pattern A-B-A-B.
      • Anti-zigzag: a vehicle's new position must not equal its position
        from 2 moves ago (prevents X → Y → X round-trips within the
        same vehicle's own trajectory).

    Returns a list of (vehicle_id, Action) tuples on success, or None
    if it cannot satisfy all constraints within MAX_RETRIES attempts.
    """
    vehicle_ids = [v.id for v in state.vehicles]
    valid_actions = ACTIONS_BY_ENV[env]
    start_pos = get_required_vehicle(state, queried_vid).position

    min_queried_moves = max(1, min_queried_moves)
    max_queried_moves = max(min_queried_moves, max_queried_moves)
    min_distinct_actors = max(1, min_distinct_actors)
    min_distinct_actions = max(1, min_distinct_actions)
    min_nonqueried_moves = max(0, min_nonqueried_moves)
    min_vehicles_moved = max(1, min_vehicles_moved)
    if min_queried_moves > n_steps:
        return None
    if _has_position_overlap(state):
        return None

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
            nonqueried_moves_done = len(plan) - queried_move_count
            nonqueried_deficit = min_nonqueried_moves - nonqueried_moves_done
            moved_vehicles_so_far = {v_id for v_id, _ in plan}
            new_actor_deficit = min_vehicles_moved - len(moved_vehicles_so_far)
            unmoved_actors = [v for v in vehicle_ids if v not in moved_vehicles_so_far]

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
            force_queried = (
                queried_deficit >= remaining
                or (step_idx == n_steps - 1 and queried_move_count == 0)
            )
            force_nonqueried = nonqueried_deficit >= remaining and not force_queried
            force_new_actor = new_actor_deficit >= remaining and not force_queried
            queried_at_cap = queried_move_count >= max_queried_moves
            if force_queried:
                vid = queried_vid
            else:
                candidates = [v for v in vehicle_ids if not (queried_at_cap and v == queried_vid)]
                if force_nonqueried:
                    candidates = [v for v in candidates if v != queried_vid]
                if force_new_actor:
                    candidates = [v for v in candidates if v in unmoved_actors]
                if not candidates:
                    break
                candidates_wo_ban = [v for v in candidates if v != banned_vid]
                if candidates_wo_ban:
                    candidates = candidates_wo_ban
                if queried_at_cap:
                    candidates = [v for v in candidates if v != queried_vid]
                vid = random.choice(candidates)

            # Build a candidate (vehicle, action) pool. First try the
            # chosen vid; if every action of that vid fails the soft
            # constraints, fall back to other vids before giving up.
            vid_order = [vid] + [v for v in vehicle_ids if v != vid]
            if queried_at_cap and not force_queried:
                vid_order = [v for v in vid_order if v != queried_vid]
            if force_nonqueried:
                vid_order = [v for v in vid_order if v != queried_vid]
            if force_new_actor:
                vid_order = [v for v in vid_order if v in unmoved_actors]

            applied = False
            for try_vid in vid_order:
                # Respect the quota: if we're forced to pick queried_vid,
                # do not switch away from it just to satisfy anti-zigzag.
                if queried_deficit >= remaining and try_vid != queried_vid:
                    continue
                if (step_idx == n_steps - 1
                        and queried_move_count == 0
                        and try_vid != queried_vid):
                    continue
                if force_nonqueried and try_vid == queried_vid:
                    continue
                if force_new_actor and try_vid not in unmoved_actors:
                    continue
                if (
                    try_vid == queried_vid
                    and queried_move_count >= max_queried_moves
                    and not force_queried
                ):
                    continue

                action_pool = list(valid_actions)
                random.shuffle(action_pool)
                if env == Environment.INTERSECTION and try_vid != queried_vid:
                    # Keep STOP available (for distribution balance), but
                    # still prefer moving traffic most of the time.
                    if Action.STOP in action_pool:
                        if (
                            len(plan) >= 2
                            and plan[-1][1] == Action.STOP
                            and plan[-2][1] == Action.STOP
                        ):
                            # Avoid wasting trials on immediate STOP-streak dead ends.
                            action_pool = [a for a in action_pool if a != Action.STOP] + [Action.STOP]
                        elif random.random() < 0.18:
                            action_pool = [Action.STOP] + [a for a in action_pool if a != Action.STOP]
                        else:
                            non_stop_actions = [a for a in action_pool if a != Action.STOP]
                            if non_stop_actions:
                                action_pool = non_stop_actions
                if env == Environment.INTERSECTION and try_vid == queried_vid:
                    queried_actions_so_far = [a for v_id, a in plan if v_id == queried_vid]
                    q_left_including_this = min_queried_moves - len(queried_actions_so_far)
                    q_pos = get_required_vehicle(trial_state, queried_vid).position
                    if q_pos == "inside_intersection" and q_left_including_this <= 1:
                        # Last queried move from inside should usually exit now.
                        turn_actions = [a for a in action_pool if a in (Action.TURN_LEFT, Action.TURN_RIGHT)]
                        if turn_actions:
                            random.shuffle(turn_actions)
                            action_pool = turn_actions + [a for a in action_pool if a not in turn_actions]
                    # Encourage additional queried-action shapes beyond
                    # MOVE_FORWARD->TURN_* without forcing them.
                    if (
                        not queried_actions_so_far
                        and Action.STOP in action_pool
                        and random.random() < 0.40
                    ):
                        action_pool = [Action.STOP] + [a for a in action_pool if a != Action.STOP]
                    elif queried_actions_so_far == [Action.STOP] and Action.MOVE_FORWARD in action_pool:
                        action_pool = [Action.MOVE_FORWARD] + [a for a in action_pool if a != Action.MOVE_FORWARD]
                    elif (
                        queried_actions_so_far == [Action.MOVE_FORWARD]
                        and Action.STOP in action_pool
                        and random.random() < 0.22
                    ):
                        action_pool = [Action.STOP] + [a for a in action_pool if a != Action.STOP]
                if env == Environment.MULTI_LANE and try_vid == queried_vid:
                    qacts = [a for v_id, a in plan if v_id == queried_vid]
                    opposite = {
                        Action.CHANGE_LEFT: Action.CHANGE_RIGHT,
                        Action.CHANGE_RIGHT: Action.CHANGE_LEFT,
                    }
                    if qacts:
                        prev = qacts[-1]
                        if prev in opposite and opposite[prev] in action_pool:
                            # Reduce repeated LL / RR collapses by preferring
                            # the opposite lane change after a queried move.
                            if (len(qacts) >= 2 and qacts[-1] == qacts[-2]) or random.random() < 0.60:
                                pref = opposite[prev]
                                action_pool = [pref] + [a for a in action_pool if a != pref]
                for act in action_pool:
                    # 1. No consecutive identical (vehicle, action) pair.
                    if plan and plan[-1] == (try_vid, act):
                        continue

                    # 1b. No mechanical A-B-A-B alternation over last 4 steps.
                    if _would_create_abab_vehicle_pattern(plan, try_vid):
                        continue

                    # 1c. No monotone action streaks.
                    if len(plan) >= 2 and plan[-1][1] == act and plan[-2][1] == act:
                        continue
                    if _would_overlap_on_action(trial_state, try_vid, act, env):
                        continue

                    snapshot = copy.deepcopy(trial_state)
                    result = _safe_apply_action_for_env(snapshot, try_vid, act, env)
                    if result is None:
                        continue
                    if _has_position_overlap(snapshot):
                        continue

                    # 2. Anti-zigzag per vehicle.
                    new_pos = get_required_vehicle(snapshot, try_vid).position
                    history = pos_history[try_vid]
                    # Allow one local reversal for diversity, but block
                    # repetitive XYXY oscillations.
                    if (
                        len(history) >= 3
                        and new_pos == history[-2]
                        and history[-1] == history[-3]
                    ):
                        continue

                    # 2b. Optional cap on global explored positions
                    # (used in multi-lane to keep 2 unvisited labels available
                    # for highly_false distractors).
                    if max_unique_positions is not None:
                        unique_positions: set[str] = set()
                        for v_id, h in pos_history.items():
                            unique_positions.update(h)
                            if v_id == try_vid:
                                unique_positions.add(new_pos)
                        if len(unique_positions) > max_unique_positions:
                            continue

                    if (
                        choice_label_cap is not None
                        and choice_label_vocab is not None
                    ):
                        covered_choice_labels: set[str] = set()
                        for v_id, h in pos_history.items():
                            for pos in h:
                                lab = label_of(pos)
                                if lab in choice_label_vocab:
                                    covered_choice_labels.add(lab)
                            if v_id == try_vid:
                                lab = label_of(new_pos)
                                if lab in choice_label_vocab:
                                    covered_choice_labels.add(lab)
                        if len(covered_choice_labels) > choice_label_cap:
                            continue

                    # All soft constraints satisfied → tentative step commit.
                    # We append first so pattern guards evaluate the real
                    # post-step suffix, then undo on early rejection.
                    plan.append((try_vid, act))

                    # Early prune 1: action streak length-3.
                    if len(plan) >= 3 and len({a for _, a in plan[-3:]}) == 1:
                        plan.pop()
                        continue

                    # Early prune 2: A-B-A-B actor alternation on last 4 steps.
                    if (
                        len(plan) >= 4
                        and plan[-1][0] == plan[-3][0]
                        and plan[-2][0] == plan[-4][0]
                        and plan[-1][0] != plan[-2][0]
                    ):
                        plan.pop()
                        continue

                    # Early prune 3: final-step action diversity feasibility.
                    steps_remaining = n_steps - len(plan)
                    if steps_remaining == 0 and len({a for _, a in plan}) < min_distinct_actions:
                        plan.pop()
                        continue

                    # Early feasibility lookaheads for quota-style invariants.
                    queried_after = queried_move_count + (1 if try_vid == queried_vid else 0)
                    if queried_after > max_queried_moves:
                        plan.pop()
                        continue
                    if queried_after + steps_remaining < min_queried_moves:
                        plan.pop()
                        continue

                    nonqueried_after = sum(1 for v_id, _ in plan if v_id != queried_vid)
                    if nonqueried_after + steps_remaining < min_nonqueried_moves:
                        plan.pop()
                        continue

                    distinct_actors_after = {v_id for v_id, _ in plan}
                    max_possible_distinct_actors = min(
                        len(vehicle_ids),
                        len(distinct_actors_after) + steps_remaining,
                    )
                    if max_possible_distinct_actors < min_distinct_actors:
                        plan.pop()
                        continue
                    if max_possible_distinct_actors < min_vehicles_moved:
                        plan.pop()
                        continue

                    distinct_actions_after = {a for _, a in plan}
                    max_possible_distinct_actions = min(
                        len(valid_actions),
                        len(distinct_actions_after) + steps_remaining,
                    )
                    if max_possible_distinct_actions < min_distinct_actions:
                        plan.pop()
                        continue

                    if require_other_vehicle_multi_step:
                        other_counts: dict[str, int] = {}
                        for v_id, _ in plan:
                            if v_id == queried_vid:
                                continue
                            other_counts[v_id] = other_counts.get(v_id, 0) + 1
                        best_other_after = max(other_counts.values(), default=0)
                        if best_other_after < 2 and best_other_after + steps_remaining < 2:
                            plan.pop()
                            continue

                    trial_state = snapshot
                    pos_history[try_vid].append(new_pos)
                    if try_vid == queried_vid:
                        queried_move_count += 1
                    applied = True
                    break

                if applied:
                    break

            if not applied:
                break

        if len(plan) != n_steps:
            continue
        if queried_move_count < min_queried_moves:
            continue

        if queried_move_count > max_queried_moves:
            continue

        if _contains_abab_vehicle_pattern(plan):
            continue

        if _has_action_streak(plan, streak_len=3):
            continue

        if len({vid for vid, _ in plan}) < min_distinct_actors:
            continue

        if len({act for _, act in plan}) < min_distinct_actions:
            continue

        nonqueried_moves = sum(1 for vid, _ in plan if vid != queried_vid)
        if nonqueried_moves < min_nonqueried_moves:
            continue

        moved_vehicles = {vid for vid, _ in plan}
        if len(moved_vehicles) < min_vehicles_moved:
            continue

        if require_queried_interleaving and not _queried_moves_interleaved(plan, queried_vid):
            continue

        if require_other_vehicle_multi_step:
            other_counts: dict[str, int] = {}
            for vid, _ in plan:
                if vid == queried_vid:
                    continue
                other_counts[vid] = other_counts.get(vid, 0) + 1
            if not any(c >= 2 for c in other_counts.values()):
                continue

        if _has_vehicle_palindrome(plan):
            continue

        final_pos = get_required_vehicle(trial_state, queried_vid).position
        if final_pos == start_pos:
            continue
        if env == Environment.INTERSECTION and final_pos == "inside_intersection":
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
    env = sim.environment

    for vid, act in plan:
        result = _safe_apply_action_for_env(sim, vid, act, env)
        if not result:
            raise RuntimeError(
                f"replay_trajectories: FSM rejected planned action "
                f"({vid}, {act}) — plan was not validated before replay."
            )
        trace[vid].append(get_required_vehicle(sim, vid).position)

    return trace



def build_choices(
    env: Environment,
    queried_vid: str,
    queried_trace: list[str],
    other_traces: dict[str, list[str]],
    require_near_true_from_other: bool = False,
    prefer_ambiguous_families: bool = False,
    hf_label_usage: dict[str, int] | None = None,
) -> dict[str, dict] | None:
    """
    Build 5-way MCQ with closed state-space logic:
      • near_true  = visited states (queried/other trajectories)
      • highly_false = reachable env states never visited
    """
    correct_label = label_of(queried_trace[-1])
    env_labels = _choice_labels_for_env(env)
    env_label_set = set(env_labels)
    if correct_label not in env_label_set:
        return None
    if env == Environment.INTERSECTION and correct_label == "inside the intersection":
        return None

    visited_labels: set[str] = {
        label_of(pos)
        for trace in other_traces.values()
        for pos in trace
    }

    # near_true candidates (visited != correct), with provenance.
    start_label = label_of(queried_trace[0])
    nt_candidates: list[dict[str, str | int]] = []
    if start_label != correct_label and start_label in env_label_set:
        nt_candidates.append({
            "label": start_label,
            "rationale": "queried vehicle's start position",
            "source": "queried_start",
            "weight": 4,
        })
    for pos in queried_trace[1:-1]:
        lab = label_of(pos)
        if lab == correct_label or lab not in env_label_set:
            continue
        nt_candidates.append({
            "label": lab,
            "rationale": "intermediate position visited by queried vehicle",
            "source": "queried_intermediate",
            "weight": 5,
        })

    for vid, trace in other_traces.items():
        if vid == queried_vid:
            continue
        for i, pos in enumerate(trace):
            lab = label_of(pos)
            if lab == correct_label or lab not in env_label_set:
                continue
            if i == len(trace) - 1:
                nt_candidates.append({
                    "label": lab,
                    "rationale": f"final position of vehicle {vid}",
                    "source": "other_final",
                    "weight": 4,
                })
            else:
                nt_candidates.append({
                    "label": lab,
                    "rationale": "position visited by another vehicle",
                    "source": "other_visited",
                    "weight": 3,
                })

    # Deduplicate near_true by label (keep strongest weight).
    nt_by_label: dict[str, dict[str, str | int]] = {}
    for cand in nt_candidates:
        lab = str(cand["label"])
        if lab not in nt_by_label or int(cand["weight"]) > int(nt_by_label[lab]["weight"]):
            nt_by_label[lab] = cand
    nt_pool = list(nt_by_label.values())
    if len(nt_pool) < 2:
        return None

    selected_nt: list[dict[str, str | int]] = []
    while len(selected_nt) < 2:
        remain = [c for c in nt_pool if c["label"] not in {s["label"] for s in selected_nt}]
        if not remain:
            return None
        pick = random.choices(
            remain,
            weights=[int(c["weight"]) for c in remain],
            k=1,
        )[0]
        selected_nt.append(pick)

    if require_near_true_from_other and not any(
        str(c["source"]).startswith("other_") for c in selected_nt
    ):
        other_pool = [c for c in nt_pool if str(c["source"]).startswith("other_")]
        if not other_pool:
            return None
        # Prefer a label-different replacement; if not available, switch
        # provenance to an other-vehicle candidate with the same label.
        replacement_pool = [
            c for c in other_pool
            if c["label"] != selected_nt[0]["label"]
        ]
        if replacement_pool:
            repl = random.choice(replacement_pool)
            selected_nt[1] = repl
        else:
            label_to_replace = selected_nt[0]["label"]
            same_label_other = [c for c in other_pool if c["label"] == label_to_replace]
            if not same_label_other:
                return None
            selected_nt[0] = random.choice(same_label_other)

    nt1_label = str(selected_nt[0]["label"])
    nt1_rationale = str(selected_nt[0]["rationale"])
    nt2_label = str(selected_nt[1]["label"])
    nt2_rationale = str(selected_nt[1]["rationale"])

    used = {correct_label, nt1_label, nt2_label}

    hf_pool = [lab for lab in env_labels if lab not in visited_labels and lab not in used]
    if len(hf_pool) < 2:
        return None

    def _pick_hf_pair() -> tuple[str, str]:
        if prefer_ambiguous_families:
            same_family = [lab for lab in hf_pool if _label_family(lab) == _label_family(correct_label)]
            if len(same_family) >= 1:
                same_weights = [_hf_weight(lab, env, hf_label_usage) for lab in same_family]
                hf1 = _weighted_label_pick(same_family, same_weights)
                rem = [lab for lab in hf_pool if lab != hf1]
                rem_same = [lab for lab in rem if _label_family(lab) == _label_family(correct_label)]
                source = rem_same if rem_same else rem
                src_weights = [_hf_weight(lab, env, hf_label_usage) for lab in source]
                hf2 = _weighted_label_pick(source, src_weights)
                return hf1, hf2
            hf_weights = [_hf_weight(lab, env, hf_label_usage) for lab in hf_pool]
            return _weighted_pick_two_distinct(hf_pool, hf_weights)
        hf_weights = [_hf_weight(lab, env, hf_label_usage) for lab in hf_pool]
        return _weighted_pick_two_distinct(hf_pool, hf_weights)

    hf1_label, hf2_label = _pick_hf_pair()

    hf1_rationale = "reachable same-environment state never visited by any vehicle"
    hf2_rationale = "reachable same-environment state never visited by any vehicle"

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



LETTERS = ["A", "B", "C", "D", "E"]


def assign_letters(
    choices_dict: dict[str, dict],
    correct_key: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], str]:
    """Shuffle options and place the correct answer at `correct_key`."""
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



def plan_to_events(
    init_state: ScenarioState, plan: list[tuple[str, Action]]
) -> list[str]:
    """Replays `plan` to produce the human-readable event strings."""
    sim = copy.deepcopy(init_state)
    env = sim.environment
    events: list[str] = []
    for vid, act in plan:
        ev = _safe_apply_action_for_env(sim, vid, act, env)
        if not ev:
            raise RuntimeError(
                f"Planned action rejected by FSM at replay: ({vid}, {act})."
            )
        events.append(ev)
    return events



def validate_example(
    example: dict,
    init_state: ScenarioState,
    plan: list[tuple[str, Action]],
    env: Environment,
    queried_vid: str,
    difficulty_profile: dict,
) -> tuple[bool, str]:
    """
    Runs per-example checks before appending to the dataset.
    Covers Problems 2 and 6 from the T1 fix plan.

    Returns (ok, reason). When ok is False, `reason` names the failing
    check — generate_example() treats it as a rejection and retries.

    Checks:
      V1. All 5 choice texts belong to the same-environment state space
          (reachable position labels only).
      V2. All 5 choice texts are pairwise distinct.
      V3. The correct answer text matches the replay-simulated final
          position of the queried vehicle (replay from init_state).
      V4. No distractor text equals the correct-answer text.
      V5. start_position != final_position for the queried vehicle.
      V6. Exactly 2 distractors typed `near_true` and exactly 2 typed
          `highly_false`.
      V7. Queried vehicle meets the profile move quota.
      V8. The replay of every planned action must be FSM-valid.
      V9. No A-B-A-B vehicle alternation and no consecutive identical
          (vehicle, action) pairs.
      V10. At least 2 distinct acting vehicles and 2 distinct action types.
      V11. near_true distractors must be labels visited in scenario traces.
      V12. Audit fields (plan / traces / rationale map) are internally
           consistent with reconstructed trajectories.
      V13. Options should resist trivial elimination shortcuts:
           all 5 options must stay in closed state-space and at least one
           distractor must share label-family with the correct option.
    """
    choices = example["choices"]
    answer_letter = example["answer"]
    correct_text = choices[answer_letter]

    # V1 — task-level choice-space consistency
    allowed_choice_labels = set(_choice_labels_for_env(env))
    for letter, text in choices.items():
        if text not in allowed_choice_labels:
            return False, f"choice {letter} {text!r} not in allowed choice-space for {env.value}"

    if _has_position_overlap(init_state):
        return False, "overlap detected in initial state"

    # V2 — pairwise distinct
    if len({text for text in choices.values()}) != 5:
        return False, "duplicate choice texts"

    # V3 — correct answer matches independent replay
    replay = copy.deepcopy(init_state)
    replay_trace: dict[str, list[str]] = {
        v.id: [v.position] for v in replay.vehicles
    }
    if _has_position_overlap(replay):
        return False, "overlap detected at replay step 0"
    for vid, act in plan:
        ev = _safe_apply_action_for_env(replay, vid, act, env)
        if not ev:
            return False, f"invalid planned transition during replay: ({vid}, {act})"
        if _has_position_overlap(replay):
            return False, f"overlap detected during replay after ({vid}, {act.name})"
        replay_trace[vid].append(get_required_vehicle(replay, vid).position)
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

    # V7 — queried vehicle multi-step quota from profile
    queried_moves = sum(1 for vid, _ in plan if vid == queried_vid)
    target_from_audit = (
        example.get("audit", {})
        .get("profile_constraints", {})
        .get("target_queried_moves")
    )
    if isinstance(target_from_audit, int):
        if not (
            difficulty_profile["min_queried_moves"]
            <= target_from_audit
            <= difficulty_profile["max_queried_moves"]
        ):
            return False, "audit target_queried_moves outside difficulty bounds"
        if queried_moves != target_from_audit:
            return False, (
                f"queried vehicle moved {queried_moves}, expected exact target "
                f"{target_from_audit}"
            )
    elif queried_moves < difficulty_profile["min_queried_moves"]:
        return False, (
            f"queried vehicle moved {queried_moves}, expected at least "
            f"{difficulty_profile['min_queried_moves']}"
        )
    if queried_moves > difficulty_profile["max_queried_moves"]:
        return False, (
            f"queried vehicle moved {queried_moves}, expected at most "
            f"{difficulty_profile['max_queried_moves']}"
        )

    # V9 — sequence anti-mechanical checks
    if _contains_abab_vehicle_pattern(plan):
        return False, "plan contains A-B-A-B vehicle alternation"
    for idx in range(len(plan) - 1):
        if plan[idx] == plan[idx + 1]:
            return False, f"plan repeats identical pair at step {idx + 1}"

    # V10 — minimum actor/action diversity
    if len({vid for vid, _ in plan}) < difficulty_profile["min_distinct_actors"]:
        return False, "plan has insufficient acting-vehicle diversity"
    if len({act for _, act in plan}) < difficulty_profile["min_distinct_actions"]:
        return False, "plan has insufficient action diversity"
    nonqueried_moves = sum(1 for vid, _ in plan if vid != queried_vid)
    if nonqueried_moves < difficulty_profile["min_nonqueried_moves"]:
        return False, "insufficient non-queried interaction"
    if len({vid for vid, _ in plan}) < difficulty_profile["min_vehicles_moved"]:
        return False, "too few vehicles moved"
    if difficulty_profile["require_queried_interleaving"]:
        if not _queried_moves_interleaved(plan, queried_vid):
            return False, "queried moves not interleaved with other vehicles"

    if difficulty_profile["require_other_vehicle_multi_step"]:
        other_counts: dict[str, int] = {}
        for vid, _ in plan:
            if vid == queried_vid:
                continue
            other_counts[vid] = other_counts.get(vid, 0) + 1
        if not any(c >= 2 for c in other_counts.values()):
            return False, "no non-queried vehicle with multi-step trajectory"

    if _has_vehicle_palindrome(plan):
        return False, "plan has symmetric palindrome actor pattern"

    if _has_action_streak(plan, streak_len=3):
        return False, "plan has action streak of length 3"

    # V11 — near_true choices must be grounded in visited states
    audit = example.get("audit", {})
    visited_labels = {
        label_of(pos)
        for t in replay_trace.values()
        for pos in t
    }
    for letter, dty in dtypes.items():
        if dty == "near_true" and choices[letter] not in visited_labels:
            return False, f"near_true option {letter} is not visited in trace"
    if difficulty_profile["require_near_true_from_other"]:
        rationale_by_letter = audit.get("rationale_by_letter", {})
        near_true_reasons = [
            rationale_by_letter.get(letter, "")
            for letter, dty in dtypes.items()
            if dty == "near_true"
        ]
        if not any(
            ("another vehicle" in r) or ("final position of vehicle" in r)
            for r in near_true_reasons
        ):
            return False, "near_true set lacks other-vehicle grounded distractor"

    # V12 — audit consistency
    if audit.get("queried_trace") != replay_trace.get(queried_vid):
        return False, "audit queried_trace mismatch"
    if audit.get("all_traces") != replay_trace:
        return False, "audit all_traces mismatch"
    expected_plan_audit = [[vid, act.name] for vid, act in plan]
    if audit.get("plan") != expected_plan_audit:
        return False, "audit plan mismatch"
    expected_qshape = [act.name for vid, act in plan if vid == queried_vid]
    if audit.get("queried_action_shape") != expected_qshape:
        return False, "audit queried_action_shape mismatch"
    rationale_by_letter = audit.get("rationale_by_letter", {})
    if set(rationale_by_letter.keys()) != set(LETTERS):
        return False, "audit rationale_by_letter does not cover A..E"

    # V13 — anti-shortcut option checks
    real_choice_space_count = sum(1 for text in choices.values() if text in allowed_choice_labels)
    if real_choice_space_count != 5:
        return False, "non-closed-set label detected in choices"
    correct_family = _label_family(correct_text)
    family_domain_count = sum(
        1 for label in _choice_labels_for_env(env)
        if _label_family(label) == correct_family
    )
    family_distractors = sum(
        1
        for letter, text in choices.items()
        if letter != answer_letter and _label_family(text) == correct_family
    )
    if family_domain_count >= 2 and family_distractors < 1:
        return False, "correct label-family is uniquely identifiable"
    if difficulty_profile["prefer_ambiguous_families"] and family_domain_count >= 2:
        near_true_same_family = sum(
            1
            for letter, dty in dtypes.items()
            if dty == "near_true" and _label_family(choices[letter]) == correct_family
        )
        highly_false_same_family = sum(
            1
            for letter, dty in dtypes.items()
            if dty == "highly_false" and _label_family(choices[letter]) == correct_family
        )
        if near_true_same_family < 1 or highly_false_same_family < 1:
            return False, "hard profile lacks family-level distractor ambiguity"

    return True, "ok"



def generate_example(
    example_id: int,
    correct_key: str,
    difficulty_hint: str | None = None,
    env_hint: Environment | None = None,
    quality_tracker: dict | None = None,
    reject_stats: dict[str, int] | None = None,
) -> dict | None:
    """Build one Task 1 example, or None if retries are exhausted."""
    for attempt in range(MAX_RETRIES):
        difficulty = difficulty_hint if difficulty_hint in DIFFICULTIES else random.choice(DIFFICULTIES)
        base_profile = _difficulty_profile(difficulty)
        allowed_envs = base_profile["allowed_envs"]
        if env_hint is not None and env_hint in allowed_envs:
            env = env_hint
        else:
            env = random.choice(allowed_envs)
        profile = _profile_for_env(base_profile, env, difficulty)
        min_total_steps = profile["min_queried_moves"] + profile["min_nonqueried_moves"]
        step_low = max(profile["step_min"], min_total_steps)
        step_high = profile["step_max"]
        if step_low > step_high:
            _inc(reject_stats, "reject.step_bounds")
            continue
        n_steps = random.randint(step_low, step_high)

        num_vehicles = _num_vehicles_for_env(env)
        if env == Environment.MULTI_LANE:
            state = _build_multi_lane_scenario(num_vehicles)
        else:
            state = _build_intersection_scenario_task1(num_vehicles)
        if _has_position_overlap(state):
            _inc(reject_stats, f"reject.initial_overlap.{env.value}")
            continue

        max_env_qmoves = MAX_QUERIED_MOVES_BY_ENV.get(env, n_steps)
        interleave_cap = (
            (n_steps + 1) // 2
            if profile["require_queried_interleaving"]
            else n_steps
        )
        max_feasible_qmoves = min(
            profile["max_queried_moves"],
            max_env_qmoves,
            n_steps - profile["min_nonqueried_moves"],
            interleave_cap,
        )
        min_feasible_qmoves = profile["min_queried_moves"]
        if max_feasible_qmoves < min_feasible_qmoves:
            continue
        feasible_qmoves = list(range(min_feasible_qmoves, max_feasible_qmoves + 1))
        if quality_tracker is not None:
            if env == Environment.MULTI_LANE:
                if difficulty == "easy":
                    base_pref = {1: 1.20, 2: 1.00, 3: 0.50, 4: 0.10}
                elif difficulty == "medium":
                    base_pref = {1: 0.60, 2: 1.00, 3: 1.15, 4: 0.40}
                else:
                    # Hard multi-lane stays multi-step, but over-sampling
                    # 3 queried moves creates many dead-end plans.
                    base_pref = {1: 0.10, 2: 1.60, 3: 0.45, 4: 0.15}
                q_usage = quality_tracker.get("multi_lane_queried_move_counts", {})
                q_weights = [
                    base_pref.get(m, 1.0) / (1.0 + q_usage.get(m, 0))
                    for m in feasible_qmoves
                ]
                target_qmoves = random.choices(feasible_qmoves, weights=q_weights, k=1)[0]
            else:
                if difficulty == "easy":
                    base_pref = {2: 1.00}
                elif difficulty == "medium":
                    base_pref = {2: 1.00}
                else:
                    base_pref = {2: 1.00}
                q_usage = quality_tracker.get("intersection_queried_move_counts", {})
                q_weights = [
                    base_pref.get(m, 1.0) / (1.0 + q_usage.get(m, 0))
                    for m in feasible_qmoves
                ]
                target_qmoves = random.choices(feasible_qmoves, weights=q_weights, k=1)[0]
        else:
            target_qmoves = random.choice(feasible_qmoves)
        seq_min_nonqueried_moves = profile["min_nonqueried_moves"]
        seq_min_vehicles_moved = profile["min_vehicles_moved"]
        seq_min_distinct_actions = profile["min_distinct_actions"]
        seq_require_other_vehicle_multi_step = profile["require_other_vehicle_multi_step"]

        init_state = copy.deepcopy(state)
        queried_vid: str | None = None
        plan: list[tuple[str, Action]] | None = None
        events: list[str] | None = None
        trace: dict[str, list[str]] | None = None
        visited_labels: set[str] | None = None
        for _ in range(SEQUENCE_TRIES_PER_ATTEMPT):
            if env == Environment.INTERSECTION:
                inside_candidates = [
                    v.id for v in state.vehicles
                    if v.position == "inside_intersection"
                ]
                if inside_candidates and random.random() < INTERSECTION_INSIDE_QUERY_BIAS:
                    try_queried_vid = random.choice(inside_candidates)
                else:
                    try_queried_vid = random.choice([v.id for v in state.vehicles])
            else:
                try_queried_vid = random.choice([v.id for v in state.vehicles])

            sim_state = copy.deepcopy(state)
            try_plan = generate_sequence(
                sim_state,
                try_queried_vid,
                env,
                n_steps,
                min_queried_moves=target_qmoves,
                max_queried_moves=target_qmoves,
                min_distinct_actors=profile["min_distinct_actors"],
                min_distinct_actions=seq_min_distinct_actions,
                min_nonqueried_moves=seq_min_nonqueried_moves,
                min_vehicles_moved=seq_min_vehicles_moved,
                require_queried_interleaving=profile["require_queried_interleaving"],
                require_other_vehicle_multi_step=seq_require_other_vehicle_multi_step,
                max_unique_positions=None,
                choice_label_cap=(3 if env == Environment.INTERSECTION else None),
                choice_label_vocab=(
                    set(INTERSECTION_ALLOWED_CHOICE_LABELS)
                    if env == Environment.INTERSECTION
                    else None
                ),
            )
            if try_plan is None:
                continue

            try_events = plan_to_events(init_state, try_plan)
            try_trace = replay_trajectories(init_state, try_plan)
            try_visited_labels = {
                label_of(pos)
                for t in try_trace.values()
                for pos in t
            }
            if env == Environment.INTERSECTION:
                covered_choice_labels = {
                    lab for lab in try_visited_labels
                    if lab in INTERSECTION_ALLOWED_CHOICE_LABELS
                }
                if len(covered_choice_labels) < 3:
                    _inc(reject_stats, "reject.intersection.choice_coverage_low")
                    continue

            queried_vid = try_queried_vid
            plan = try_plan
            events = try_events
            trace = try_trace
            visited_labels = try_visited_labels
            break
        if plan is None:
            _inc(reject_stats, f"reject.sequence.{env.value}")
            continue
        assert queried_vid is not None
        assert events is not None
        assert trace is not None
        assert visited_labels is not None

        queried_trace = trace[queried_vid]
        raw_choices = None
        for _ in range(CHOICE_RETRIES_PER_PLAN):
            raw_choices = build_choices(
                env,
                queried_vid,
                queried_trace,
                trace,
                require_near_true_from_other=profile["require_near_true_from_other"],
                prefer_ambiguous_families=profile["prefer_ambiguous_families"],
                hf_label_usage=(
                    quality_tracker.get("multi_lane_hf_label_counts")
                    if (quality_tracker is not None and env == Environment.MULTI_LANE)
                    else None
                ),
            )
            if raw_choices is not None:
                break
        if raw_choices is None:
            _inc(reject_stats, f"reject.choices.{env.value}")
            continue  # vocabulary exhaustion → retry

        # Post-condition: all five labels are pairwise distinct.
        # Environment consistency is enforced by validate_example().
        letter_texts = {c["text"] for c in raw_choices.values()}
        if len(letter_texts) != 5:
            _inc(reject_stats, "reject.choices.not_unique")
            continue

        choices, distractor_type, rationale_by_letter, answer = assign_letters(
            raw_choices, correct_key
        )
        scenario_text = _describe_task1_scenario(init_state)
        question = _pick_question(queried_vid)
        prompt, event_header_used, question_header_used = _render_prompt_with_variation(
            scenario_text, events, question, choices, env
        )

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
                        "inside_intersection": v.inside_intersection,
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
                "num_vehicles": len(init_state.vehicles),
                "num_events": len(events),
                "queried_vehicle": queried_vid,
                "environment": env.value,
                "difficulty": difficulty,
            },
            "audit": {
                "generator_version": GENERATOR_VERSION,
                "attempt": attempt,
                "difficulty_profile": difficulty,
                "profile_constraints": {
                    "min_queried_moves": profile["min_queried_moves"],
                    "max_queried_moves": profile["max_queried_moves"],
                    "target_queried_moves": target_qmoves,
                    "min_distinct_actors": profile["min_distinct_actors"],
                    "min_distinct_actions": profile["min_distinct_actions"],
                    "min_nonqueried_moves": profile["min_nonqueried_moves"],
                    "min_vehicles_moved": profile["min_vehicles_moved"],
                    "require_queried_interleaving": profile["require_queried_interleaving"],
                    "require_near_true_from_other": profile["require_near_true_from_other"],
                    "require_other_vehicle_multi_step": profile["require_other_vehicle_multi_step"],
                    "prefer_ambiguous_families": profile["prefer_ambiguous_families"],
                },
                "prompt_style": {
                    "event_header": event_header_used,
                    "question_header": question_header_used,
                },
                "queried_trace": queried_trace,
                "queried_action_shape": [act.name for vid, act in plan if vid == queried_vid],
                "all_traces": trace,
                "plan": [[vid, act.name] for vid, act in plan],
                "rationale_by_letter": rationale_by_letter,
                "invariants": {
                    "start_ne_final": queried_trace[0] != queried_trace[-1],
                    "queried_moved": sum(1 for vid, _ in plan if vid == queried_vid),
                    "no_abab_vehicle_pattern": not _contains_abab_vehicle_pattern(plan),
                    "plan_actor_diversity_ok": (
                        len({vid for vid, _ in plan}) >= profile["min_distinct_actors"]
                    ),
                    "plan_action_diversity_ok": (
                        len({act for _, act in plan}) >= profile["min_distinct_actions"]
                    ),
                    "nonqueried_interaction_ok": (
                        sum(1 for vid, _ in plan if vid != queried_vid) >= profile["min_nonqueried_moves"]
                    ),
                    "queried_interleaving_ok": (
                        (not profile["require_queried_interleaving"])
                        or _queried_moves_interleaved(plan, queried_vid)
                    ),
                    "no_action_streak_len3": not _has_action_streak(plan, streak_len=3),
                    "no_vehicle_palindrome": not _has_vehicle_palindrome(plan),
                    "near_true_grounded_in_visited": all(
                        choices[L] in visited_labels
                        for L, dty in distractor_type.items()
                        if dty == "near_true"
                    ),
                    "highly_false_reachable_never_visited": all(
                        (choices[L] in _choice_labels_for_env(env)) and (choices[L] not in visited_labels)
                        for L, dty in distractor_type.items()
                        if dty == "highly_false"
                    ),
                    # All options must be valid domain labels.
                    "all_labels_in_vocabulary": all(
                        choices[L] in ALL_DOMAIN_LABELS
                        for L in LETTERS
                    ),
                    "five_distinct_options": len({choices[L] for L in LETTERS}) == 5,
                },
            },
        }

        # Hard quality gate (Problems 2 & 6).
        ok, reason = validate_example(
            example,
            init_state,
            plan,
            env,
            queried_vid,
            profile,
        )
        if not ok:
            _inc(reject_stats, f"reject.validation.{reason}")
            continue

        return example

    return None



def _passes_soft_distribution_balance(example: dict, quality_tracker: dict) -> bool:
    # Soft balance: don't let one correct label dominate,
    # but don't reject so aggressively that generation stalls.
    env_value = example["metadata"]["environment"]
    correct_label = example["choices"][example["answer"]]
    if env_value == Environment.MULTI_LANE.value:
        counts: dict[str, int] = quality_tracker["multi_lane_correct_counts"]
        if correct_label not in counts:
            return True
        total = sum(counts.values())
        if total < MULTI_LANE_BALANCE_WARMUP:
            return True
        min_count = min(counts.values())
        if counts[correct_label] > min_count + MULTI_LANE_CORRECT_LABEL_GAP:
            return False
        return True

    if env_value == Environment.INTERSECTION.value:
        counts = quality_tracker["intersection_correct_counts"]
        if correct_label not in counts:
            return True
        total = sum(counts.values())
        if total < INTERSECTION_BALANCE_WARMUP:
            return True
        min_count = min(counts.values())
        if counts[correct_label] > min_count + INTERSECTION_CORRECT_LABEL_GAP:
            return False
    return True


def _update_quality_tracker(example: dict, quality_tracker: dict) -> None:
    tmpl = _plan_template_from_example(example)
    plan_counts = quality_tracker["plan_template_counts"]
    plan_counts[tmpl] = plan_counts.get(tmpl, 0) + 1

    qshape = _queried_shape_from_example(example)
    qshape_counts = quality_tracker["queried_shape_counts"]
    qshape_counts[qshape] = qshape_counts.get(qshape, 0) + 1

    env = example["metadata"]["environment"]
    choices = example["choices"]
    answer = example["answer"]
    correct_label = choices[answer]

    if env == Environment.INTERSECTION.value:
        if correct_label in quality_tracker["intersection_correct_counts"]:
            quality_tracker["intersection_correct_counts"][correct_label] += 1
        qmoves = int(example["audit"]["invariants"]["queried_moved"])
        inter_qmove_counts = quality_tracker["intersection_queried_move_counts"]
        inter_qmove_counts[qmoves] = inter_qmove_counts.get(qmoves, 0) + 1
        return

    if env != Environment.MULTI_LANE.value:
        return

    dtypes = example["distractor_type"]
    quality_tracker["multi_lane_correct_counts"][correct_label] += 1

    hf_labels = [choices[L] for L, t in dtypes.items() if t == "highly_false"]
    quality_tracker["multi_lane_hf_total"] += len(hf_labels)
    for lab in hf_labels:
        quality_tracker["multi_lane_hf_label_counts"][lab] = (
            quality_tracker["multi_lane_hf_label_counts"].get(lab, 0) + 1
        )

    qmoves = int(example["audit"]["invariants"]["queried_moved"])
    qmove_counts = quality_tracker["multi_lane_queried_move_counts"]
    qmove_counts[qmoves] = qmove_counts.get(qmoves, 0) + 1


def _assert_dataset_quality(examples: list[dict]) -> None:
    """
    Dataset-level checks run after generation.
    Raises RuntimeError if any quality constraint is broken.
    """
    issues: list[str] = []

    # Policy: intersection "inside" must never be the correct label.
    inside_correct = 0
    approach_opts = 0
    for ex in examples:
        if ex["metadata"]["environment"] != Environment.INTERSECTION.value:
            continue
        correct_label = ex["choices"][ex["answer"]]
        if correct_label == "inside the intersection":
            inside_correct += 1
        approach_opts += sum("approach" in t for t in ex["choices"].values())
    if inside_correct > 0:
        issues.append(f"intersection inside-as-correct count={inside_correct} (expected 0)")
    if approach_opts > 0:
        issues.append(f"intersection approach options present={approach_opts} (expected 0)")

    # Shape/template collapse caps.
    queried_shape_counts: dict[tuple, int] = {}
    plan_template_counts: dict[tuple, int] = {}
    for ex in examples:
        qshape = _queried_shape_from_example(ex)
        queried_shape_counts[qshape] = queried_shape_counts.get(qshape, 0) + 1
        pt = _plan_template_from_example(ex)
        plan_template_counts[pt] = plan_template_counts.get(pt, 0) + 1

    inter_shape_max = max(
        (c for (env_name, _), c in queried_shape_counts.items()
         if env_name == Environment.INTERSECTION.value),
        default=0,
    )
    ml_shape_max = max(
        (c for (env_name, _), c in queried_shape_counts.items()
         if env_name == Environment.MULTI_LANE.value),
        default=0,
    )
    if inter_shape_max > INTERSECTION_QUERIED_SHAPE_MAX_REUSE:
        issues.append(
            f"intersection queried-shape max reuse={inter_shape_max} "
            f"> cap {INTERSECTION_QUERIED_SHAPE_MAX_REUSE}"
        )
    if ml_shape_max > MULTI_LANE_QUERIED_SHAPE_MAX_REUSE:
        issues.append(
            f"multi-lane queried-shape max reuse={ml_shape_max} "
            f"> cap {MULTI_LANE_QUERIED_SHAPE_MAX_REUSE}"
        )
    plan_max = max(plan_template_counts.values(), default=0)
    if plan_max > PLAN_TEMPLATE_MAX_REUSE:
        issues.append(
            f"plan-template max reuse={plan_max} > cap {PLAN_TEMPLATE_MAX_REUSE}"
        )

    # Difficulty should be structurally ordered, not just count-balanced.
    by_diff: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
    for ex in examples:
        d = ex["metadata"].get("difficulty", "")
        if d in by_diff:
            by_diff[d].append(ex)
    for d, arr in by_diff.items():
        if not arr:
            issues.append(f"difficulty {d} has zero examples")
    if not issues:
        mean_qmoves: dict[str, float] = {}
        mean_nonq: dict[str, float] = {}
        mean_actors: dict[str, float] = {}
        for d, arr in by_diff.items():
            qmoves_vals = []
            nonq_vals = []
            actors_vals = []
            for ex in arr:
                plan = ex["audit"]["plan"]
                qid = ex["metadata"]["queried_vehicle"]
                qmoves = sum(1 for vid, _ in plan if vid == qid)
                nonq = len(plan) - qmoves
                actors = len({vid for vid, _ in plan})
                qmoves_vals.append(qmoves)
                nonq_vals.append(nonq)
                actors_vals.append(actors)
            mean_qmoves[d] = sum(qmoves_vals) / len(qmoves_vals)
            mean_nonq[d] = sum(nonq_vals) / len(nonq_vals)
            mean_actors[d] = sum(actors_vals) / len(actors_vals)
        if mean_qmoves["hard"] < mean_qmoves["easy"] + 0.15:
            issues.append(
                "difficulty separation too weak on queried moves: "
                f"{mean_qmoves}"
            )
        if mean_nonq["hard"] < mean_nonq["easy"] + 0.15:
            issues.append(
                "difficulty separation too weak on non-queried moves: "
                f"{mean_nonq}"
            )
        if mean_actors["hard"] < mean_actors["easy"] + 0.10:
            issues.append(
                "difficulty separation too weak on actor diversity: "
                f"{mean_actors}"
            )

    # Reconstruct from serialized scenario and replay full plans.
    replay_invalid = 0
    replay_wrong = 0
    replay_trace_mismatch = 0
    replay_overlap_init = 0
    replay_overlap_during = 0
    for ex in examples:
        env = Environment(ex["metadata"]["environment"])
        vehicles: list[Vehicle] = []
        for v in ex["scenario"]["vehicles"]:
            inside_flag = bool(
                v.get("inside_intersection", v.get("position") == "inside_intersection")
            )
            vehicles.append(
                Vehicle(
                    id=v["id"],
                    position=v["position"],
                    direction=Direction(v["direction"]),
                    inside_intersection=inside_flag,
                )
            )
        sim = ScenarioState(vehicles=vehicles, environment=env)
        qid = ex["metadata"]["queried_vehicle"]
        trace: dict[str, list[str]] = {v.id: [v.position] for v in sim.vehicles}
        if _has_position_overlap(sim):
            replay_overlap_init += 1
            continue
        ok = True
        for vid, act_name in ex["audit"]["plan"]:
            act = Action[act_name]
            ev = _safe_apply_action_for_env(sim, vid, act, env)
            if not ev:
                replay_invalid += 1
                ok = False
                break
            if _has_position_overlap(sim):
                replay_overlap_during += 1
                ok = False
                break
            trace[vid].append(get_required_vehicle(sim, vid).position)
        if not ok:
            continue
        expected_text = ex["choices"][ex["answer"]]
        final_label = label_of(get_required_vehicle(sim, qid).position)
        if final_label != expected_text:
            replay_wrong += 1
        if trace != ex["audit"]["all_traces"]:
            replay_trace_mismatch += 1
    if (
        replay_invalid
        or replay_wrong
        or replay_trace_mismatch
        or replay_overlap_init
        or replay_overlap_during
    ):
        issues.append(
            "serialized replay mismatch: "
            f"invalid={replay_invalid}, wrong={replay_wrong}, "
            f"trace_mismatch={replay_trace_mismatch}, "
            f"overlap_init={replay_overlap_init}, "
            f"overlap_during={replay_overlap_during}"
        )

    if issues:
        raise RuntimeError(
            "Task1 dataset-level quality gate failed: " + " | ".join(issues)
        )



def generate_task1(n: int, output_path: str, seed: int | None = None) -> None:
    """Generate examples, write JSONL, and print distribution stats."""
    if seed is not None:
        random.seed(seed)

    if n % 5 != 0:
        raise ValueError("N must be a multiple of 5 for balanced key schedule.")

    # Key schedule: exactly n/5 of each letter, shuffled.
    key_schedule: list[str] = []
    per_key = n // 5
    for letter in LETTERS:
        key_schedule.extend([letter] * per_key)
    random.shuffle(key_schedule)

    # Joint schedule: balanced quotas per (difficulty, environment) cell.
    # Keep downstream compatibility by unzipping into separate schedules.
    joint_cells: list[tuple[str, Environment]] = [
        (difficulty, env)
        for difficulty in DIFFICULTIES
        for env in TASK_ENVS
    ]
    joint_schedule: list[tuple[str, Environment]] = []
    per_cell = n // len(joint_cells)
    rem = n - per_cell * len(joint_cells)
    for cell in joint_cells:
        joint_schedule.extend([cell] * per_cell)
    for i in range(rem):
        joint_schedule.append(joint_cells[i % len(joint_cells)])
    random.shuffle(joint_schedule)

    difficulty_schedule = [difficulty for difficulty, _ in joint_schedule]
    env_schedule = [env for _, env in joint_schedule]

    quality_tracker = {
        "intersection_correct_counts": {
            lab: 0 for lab in INTERSECTION_CORRECT_LABELS
        },
        "intersection_queried_move_counts": {2: 0},
        "multi_lane_correct_counts": {
            lab: 0 for lab in labels_for_env(Environment.MULTI_LANE)
        },
        "multi_lane_hf_label_counts": {
            lab: 0 for lab in _choice_labels_for_env(Environment.MULTI_LANE)
        },
        "multi_lane_hf_total": 0,
        "multi_lane_queried_move_counts": {1: 0, 2: 0, 3: 0, 4: 0},
        "plan_template_counts": {},
        "queried_shape_counts": {},
    }

    examples: list[dict] = []
    dropped = 0
    seen_signatures: set[tuple] = set()
    reject_stats: dict[str, int] = {}
    for idx in range(n):
        ex: dict | None = None
        for _ in range(TASK_SLOT_RETRIES):
            candidate = generate_example(
                idx,
                key_schedule[idx],
                difficulty_hint=difficulty_schedule[idx],
                env_hint=env_schedule[idx],
                quality_tracker=quality_tracker,
                reject_stats=reject_stats,
            )
            if candidate is None:
                _inc(reject_stats, "reject.generate_example.none")
                continue
            sig = _example_signature(candidate)
            if sig in seen_signatures:
                _inc(reject_stats, "reject.duplicate_signature")
                continue
            tmpl = _plan_template_from_example(candidate)
            tmpl_count = quality_tracker["plan_template_counts"].get(tmpl, 0)
            if tmpl_count >= PLAN_TEMPLATE_MAX_REUSE:
                _inc(reject_stats, "reject.plan_template_cap")
                continue
            qshape = _queried_shape_from_example(candidate)
            qshape_count = quality_tracker["queried_shape_counts"].get(qshape, 0)
            env_value = candidate["metadata"]["environment"]
            qshape_cap = (
                INTERSECTION_QUERIED_SHAPE_MAX_REUSE
                if env_value == Environment.INTERSECTION.value
                else MULTI_LANE_QUERIED_SHAPE_MAX_REUSE
            )
            if qshape_count >= qshape_cap:
                _inc(reject_stats, "reject.queried_shape_cap")
                continue
            if not _passes_soft_distribution_balance(candidate, quality_tracker):
                _inc(reject_stats, "reject.soft_balance")
                continue
            seen_signatures.add(sig)
            _update_quality_tracker(candidate, quality_tracker)
            ex = candidate
            break

        if ex is None:
            dropped += 1
            _inc(reject_stats, "reject.slot_exhausted")
            break
        examples.append(ex)

    if len(examples) != n:
        env_counts: dict[str, int] = {}
        for ex in examples:
            e = ex["metadata"]["environment"]
            env_counts[e] = env_counts.get(e, 0) + 1
        top_reasons = sorted(
            reject_stats.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:12]
        reason_summary = ", ".join(f"{k}={v}" for k, v in top_reasons) or "none"
        raise RuntimeError(
            "Task1 generation failed: produced "
            f"{len(examples)}/{n} examples. "
            f"env_counts={env_counts}. "
            f"top_rejections=[{reason_summary}]"
        )

    _assert_dataset_quality(examples)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(examples)} examples to {output_path}")
    if dropped:
        print(f"Dropped {dropped} examples (generation failures).")
    print()

    answer_counts = {letter: 0 for letter in LETTERS}
    for ex in examples:
        answer_counts[ex["answer"]] += 1

    print("Answer distribution:")
    for letter in LETTERS:
        bar = "\u2588" * answer_counts[letter]
        print(f"  {letter}: {answer_counts[letter]:3d}  {bar}")

    env_counts: dict[str, int] = {}
    for ex in examples:
        e = ex["metadata"]["environment"]
        env_counts[e] = env_counts.get(e, 0) + 1

    print("\nEnvironment distribution:")
    for e, c in sorted(env_counts.items()):
        print(f"  {e}: {c}")

    difficulty_counts: dict[str, int] = {}
    for ex in examples:
        d = ex["metadata"].get("difficulty", "unknown")
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
    print("\nDifficulty distribution:")
    for d, c in sorted(difficulty_counts.items()):
        print(f"  {d}: {c}")

    type_counts = {"near_true": 0, "highly_false": 0}
    for ex in examples:
        for t in ex["distractor_type"].values():
            type_counts[t] = type_counts.get(t, 0) + 1
    print("\nDistractor-type totals (expect 2:2 per example):")
    for t, c in type_counts.items():
        print(f"  {t}: {c}")

    top_reasons = sorted(
        reject_stats.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[:12]
    if top_reasons:
        print("\nTop rejection reasons:")
        for k, v in top_reasons:
            print(f"  {k}: {v}")



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
