"""Task 3 violation-detection generator."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from domain.entities import (
    Action,
    Direction,
    Environment,
    IntentDirection,
    ScenarioState,
    Vehicle,
)
from domain.render import describe_scenario
from domain.rules import (
    detect_right_of_way_violation,
    detect_violation,
    is_valid_transition,
)
from domain.scenario import (
    apply_action,
    build_intersection_scenario,
    build_multi_lane_scenario,
    build_roundabout_scenario,
)



GENERATOR_VERSION = "task3_violation_v9"
DEFAULT_N_EXAMPLES = 100
DEFAULT_SEED: int | None = None
VERBOSE = os.environ.get("TASK3_VERBOSE") == "1"
PROFILE_RUNTIME = False

RUNTIME_SECONDS: Counter[str] = Counter()
RUNTIME_COUNTS: Counter[str] = Counter()

NUM_VEHICLES = 3
MAX_RETRIES = 600
EXAMPLE_RETRIES = 3
EVENT_SIGNATURE_REUSE_CAP = 1
ACTION_PATTERN_REUSE_CAP = 2
MAX_ACTION_PATTERN_REUSE_TOTAL = 20
MAX_SLOT_ATTEMPTS_SOFT_CAP = 320
SLOT_EARLY_FALLBACK_ATTEMPTS = 180
HIGH_ATTEMPT_SLOT_THRESHOLD = 200

LETTERS = ["A", "B", "C", "D", "E"]
QUESTION = "Which vehicle performs the first illegal action?"

SEMANTIC_VEHICLE_A = "vehicle_a"
SEMANTIC_VEHICLE_B = "vehicle_b"
SEMANTIC_VEHICLE_C = "vehicle_c"
SEMANTIC_UNDETERMINED = "undetermined"
SEMANTIC_FIFTH_DISTRACTOR = "fifth_distractor"

SEMANTIC_CHOICES = {
    SEMANTIC_VEHICLE_A: "Vehicle A",
    SEMANTIC_VEHICLE_B: "Vehicle B",
    SEMANTIC_VEHICLE_C: "Vehicle C",
    SEMANTIC_UNDETERMINED: "No vehicle performed an illegal action",
    SEMANTIC_FIFTH_DISTRACTOR: "Another vehicle (not A, B, or C)",
}

ALLOWED_CORRECT = {
    SEMANTIC_VEHICLE_A,
    SEMANTIC_VEHICLE_B,
    SEMANTIC_VEHICLE_C,
    SEMANTIC_UNDETERMINED,
}

VIOLATION_CLASSES = [
    "turn_without_entering",
    "forward_from_exit",
    "lane_change_out_of_bounds_left",
    "lane_change_out_of_bounds_right",
    "intersection_right_of_way",
    "roundabout_entry_no_yield",
]

ALLOWED_VIOLATION_TYPES = {
    "lane_change_out_of_bounds_right",
    "lane_change_out_of_bounds_left",
    "intersection_right_of_way",
    "forward_from_exit",
    "turn_without_entering",
    "roundabout_entry_no_yield",
    "no_violation",
}

VIOLATION_ENV = {
    "turn_without_entering": Environment.INTERSECTION,
    "forward_from_exit": Environment.INTERSECTION,
    "lane_change_out_of_bounds_left": Environment.MULTI_LANE,
    "lane_change_out_of_bounds_right": Environment.MULTI_LANE,
    "intersection_right_of_way": Environment.INTERSECTION,
    "roundabout_entry_no_yield": Environment.ROUNDABOUT,
}

VIOLATION_ENVS = [
    Environment.INTERSECTION.value,
    Environment.MULTI_LANE.value,
    Environment.ROUNDABOUT.value,
]

DIFFICULTIES = ["easy", "medium", "hard"]

NO_VIOLATION_ENVS = [
    Environment.INTERSECTION,
    Environment.MULTI_LANE,
    Environment.ROUNDABOUT,
]




def _semantic_for_vehicle(vid: str) -> str:
    return {
        "A": SEMANTIC_VEHICLE_A,
        "B": SEMANTIC_VEHICLE_B,
        "C": SEMANTIC_VEHICLE_C,
    }[vid]


def _vehicle_for_semantic(semantic: str) -> str | None:
    return {
        SEMANTIC_VEHICLE_A: "A",
        SEMANTIC_VEHICLE_B: "B",
        SEMANTIC_VEHICLE_C: "C",
    }.get(semantic)


def _event_text(vehicle_id: str, action: Action) -> str:
    return f"Vehicle {vehicle_id} {action.value}."


def _serialize_state(state: ScenarioState) -> dict:
    return {
        "vehicles": [
            {
                "id": v.id,
                "position": v.position,
                "direction": v.direction.value,
                "intent": v.intent.value if v.intent else None,
                "inside_intersection": bool(v.inside_intersection),
                "stopped": bool(v.stopped),
            }
            for v in state.vehicles
        ],
        "environment": state.environment.value,
    }


def _parse_intent(value: str | None) -> IntentDirection | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    for it in IntentDirection:
        if it.value == normalized or it.name.lower() == normalized:
            return it
    raise ValueError(f"unknown intent string: {value!r}")


def _reconstruct_state(scenario_json: dict) -> ScenarioState:
    env = Environment(scenario_json["environment"])
    vehicles: list[Vehicle] = []
    for v in scenario_json["vehicles"]:
        vehicles.append(
            Vehicle(
                id=v["id"],
                position=v["position"],
                direction=Direction(v["direction"]),
                intent=_parse_intent(v.get("intent")),
                inside_intersection=bool(v.get("inside_intersection", False)),
                stopped=bool(v.get("stopped", False)),
            )
        )
    return ScenarioState(vehicles=vehicles, environment=env)


def _build_state(env: Environment) -> ScenarioState:
    if env == Environment.INTERSECTION:
        return build_intersection_scenario(NUM_VEHICLES, with_intent=False)
    if env == Environment.MULTI_LANE:
        return build_multi_lane_scenario(NUM_VEHICLES)
    if env == Environment.ROUNDABOUT:
        return build_roundabout_scenario(NUM_VEHICLES)
    raise ValueError(f"Unsupported environment: {env!r}")


def _perf_start() -> float:
    if not PROFILE_RUNTIME:
        return 0.0
    return time.perf_counter()


def _perf_end(key: str, started_at: float) -> None:
    if not PROFILE_RUNTIME:
        return
    RUNTIME_SECONDS[key] += (time.perf_counter() - started_at)


def _log(msg: str, *, verbose_only: bool = True) -> None:
    if verbose_only and not VERBOSE:
        return
    t0 = _perf_start()
    print(msg)
    _perf_end("printing_logging_reporting", t0)




def _build_key_schedule(n: int) -> list[str]:
    if n % 5 != 0:
        raise ValueError("N must be a multiple of 5 for balanced answer keys.")
    per_key = n // 5
    schedule: list[str] = []
    for letter in LETTERS:
        schedule.extend([letter] * per_key)
    random.shuffle(schedule)
    return schedule


def _build_semantic_schedule(n: int) -> list[str]:
    # ~20% no_violation (exact when n multiple of 5)
    no_violation_count = n // 5
    violating_count = n - no_violation_count

    per_vehicle = violating_count // 3
    rem = violating_count % 3

    schedule: list[str] = []
    schedule.extend([SEMANTIC_UNDETERMINED] * no_violation_count)

    vehicle_semantics = [SEMANTIC_VEHICLE_A, SEMANTIC_VEHICLE_B, SEMANTIC_VEHICLE_C]
    for semantic in vehicle_semantics:
        schedule.extend([semantic] * per_vehicle)
    for i in range(rem):
        schedule.append(vehicle_semantics[i])

    random.shuffle(schedule)
    return schedule


def _build_violation_class_schedule(n_violating: int) -> list[str]:
    schedule: list[str] = []
    base = n_violating // len(VIOLATION_CLASSES)
    rem = n_violating % len(VIOLATION_CLASSES)

    for cls in VIOLATION_CLASSES:
        schedule.extend([cls] * base)
    for i in range(rem):
        schedule.append(VIOLATION_CLASSES[i])

    random.shuffle(schedule)
    return schedule


def _build_no_violation_env_schedule(n: int) -> list[Environment]:
    schedule: list[Environment] = []
    base = n // len(NO_VIOLATION_ENVS)
    rem = n % len(NO_VIOLATION_ENVS)

    for env in NO_VIOLATION_ENVS:
        schedule.extend([env] * base)
    for i in range(rem):
        schedule.append(NO_VIOLATION_ENVS[i])

    random.shuffle(schedule)
    return schedule


def _build_difficulty_quota(n: int) -> dict[str, int]:
    # Under strict action-pattern controls, 30/35/35 is more stable than
    # 33/33/34 while still staying in the required [30, 37] range.
    if n == 100:
        return {"easy": 30, "medium": 35, "hard": 35}

    base = n // len(DIFFICULTIES)
    rem = n % len(DIFFICULTIES)
    quota = {d: base for d in DIFFICULTIES}
    for i in range(rem):
        # Round-robin remainder from the end, so n=100 -> 33/33/34.
        d = DIFFICULTIES[-(i + 1)]
        quota[d] += 1
    return quota


def _build_difficulty_schedule(n: int) -> tuple[list[str], dict[str, int]]:
    quota = _build_difficulty_quota(n)
    schedule: list[str] = []
    for d in DIFFICULTIES:
        schedule.extend([d] * quota[d])
    random.shuffle(schedule)
    return schedule, quota


def _align_difficulty_schedule_with_semantics(
    semantic_schedule: list[str],
    difficulty_schedule: list[str],
) -> None:
    """
    No-violation slots cannot be true medium under _assign_difficulty.
    Swap medium away from no-violation indices while preserving global counts.
    """
    no_violation_indices = [
        i for i, sem in enumerate(semantic_schedule)
        if sem == SEMANTIC_UNDETERMINED
    ]
    medium_on_no_violation = [
        i for i in no_violation_indices
        if difficulty_schedule[i] == "medium"
    ]
    if not medium_on_no_violation:
        return

    donor_indices = [
        i
        for i, sem in enumerate(semantic_schedule)
        if sem != SEMANTIC_UNDETERMINED and difficulty_schedule[i] in {"easy", "hard"}
    ]
    random.shuffle(donor_indices)
    donor_cursor = 0

    for idx in medium_on_no_violation:
        while donor_cursor < len(donor_indices):
            j = donor_indices[donor_cursor]
            donor_cursor += 1
            if j == idx:
                continue
            difficulty_schedule[idx], difficulty_schedule[j] = (
                difficulty_schedule[j],
                difficulty_schedule[idx],
            )
            break


def _build_difficulty_env_targets(
    difficulty_quota: dict[str, int],
    env_quota: dict[str, int],
) -> dict[tuple[str, str], int]:
    # Core-task fixed matrix for n=100: each cell in [8,16], with quotas:
    # easy=33, medium=33, hard=34 and env=34/33/33.
    if (
        difficulty_quota == {"easy": 33, "medium": 33, "hard": 34}
        and env_quota == {
            Environment.INTERSECTION.value: 34,
            Environment.MULTI_LANE.value: 33,
            Environment.ROUNDABOUT.value: 33,
        }
    ):
        return {
            ("easy", Environment.INTERSECTION.value): 13,
            ("easy", Environment.MULTI_LANE.value): 12,
            ("easy", Environment.ROUNDABOUT.value): 8,
            ("medium", Environment.INTERSECTION.value): 10,
            ("medium", Environment.MULTI_LANE.value): 11,
            ("medium", Environment.ROUNDABOUT.value): 12,
            ("hard", Environment.INTERSECTION.value): 11,
            ("hard", Environment.MULTI_LANE.value): 10,
            ("hard", Environment.ROUNDABOUT.value): 13,
        }

    targets: dict[tuple[str, str], int] = {
        (d, env): 0
        for d in DIFFICULTIES
        for env in VIOLATION_ENVS
    }
    remaining_env = dict(env_quota)
    for d in DIFFICULTIES:
        for _ in range(difficulty_quota[d]):
            eligible = [env for env in VIOLATION_ENVS if remaining_env[env] > 0]
            if not eligible:
                break
            max_remaining = max(remaining_env[env] for env in eligible)
            pool = [env for env in eligible if remaining_env[env] == max_remaining]
            min_row_fill = min(targets[(d, env)] for env in pool)
            pool = [env for env in pool if targets[(d, env)] == min_row_fill]
            random.shuffle(pool)
            chosen = pool[0]
            targets[(d, chosen)] += 1
            remaining_env[chosen] -= 1
    return targets


def _pick_violation_class_for_env(
    env_value: str,
    violation_class_usage: Counter[str],
) -> str:
    eligible = [
        cls
        for cls in VIOLATION_CLASSES
        if VIOLATION_ENV[cls].value == env_value
    ]
    if not eligible:
        eligible = list(VIOLATION_CLASSES)
    min_used = min(violation_class_usage[cls] for cls in eligible)
    # Keep rough balance without over-constraining late-slot search.
    pool = [cls for cls in eligible if violation_class_usage[cls] <= (min_used + 1)]
    random.shuffle(pool)
    return pool[0]


def _pick_env_for_example(
    semantic: str,
    desired_difficulty: str,
    env_counts: Counter[str],
    env_quota: dict[str, int],
    difficulty_env_counts: Counter[tuple[str, str]],
    difficulty_env_targets: dict[tuple[str, str], int],
    allow_overquota: bool = False,
) -> str:
    del semantic

    pool = list(VIOLATION_ENVS)
    if not allow_overquota:
        pool = [env for env in pool if env_counts[env] < env_quota.get(env, 0)]
    if not pool:
        pool = list(VIOLATION_ENVS)

    weights: list[float] = []
    for env in pool:
        env_remaining = max(0, env_quota.get(env, 0) - env_counts[env])
        cell_target = difficulty_env_targets.get((desired_difficulty, env), 0)
        cell_remaining = max(0, cell_target - difficulty_env_counts[(desired_difficulty, env)])
        if not allow_overquota and cell_target > 0 and cell_remaining == 0:
            weights.append(0.0)
            continue
        # Favor filling the scheduled difficulty x environment cell first, then
        # global environment quota.
        weights.append(float(max(1, cell_remaining + 1) * max(1, env_remaining + 1)))

    if any(w > 0 for w in weights):
        return random.choices(pool, weights=weights, k=1)[0]
    # Cell targets can be saturated in late slots; keep global env balance.
    weights = [max(1, env_quota.get(env, 0) - env_counts[env]) for env in pool]
    return random.choices(pool, weights=weights, k=1)[0]



INTERSECTION_LOSING_PAIRS = [
    (Direction.NORTH, Direction.EAST),
    (Direction.SOUTH, Direction.WEST),
    (Direction.EAST, Direction.SOUTH),
    (Direction.WEST, Direction.NORTH),
]


def _set_target_exit_intersection(state: ScenarioState, target_vid: str) -> None:
    v = state.get_vehicle(target_vid)
    if v is None:
        raise ValueError(f"Vehicle {target_vid} not found")
    # Keep direction, move to corresponding exit.
    v.inside_intersection = False
    v.position = f"{v.direction.value}_exit"


def _add_inside_vehicle_intersection(state: ScenarioState, exclude_vid: str) -> str | None:
    others = [v for v in state.vehicles if v.id != exclude_vid]
    if not others:
        return None
    random.shuffle(others)
    inside_v = others[0]
    inside_v.position = "inside_intersection"
    inside_v.inside_intersection = True
    inside_v.intent = None
    inside_v.stopped = False
    return inside_v.id


def _set_intersection_priority_against_target(state: ScenarioState, target_vid: str) -> dict[str, str]:
    """
    Force a deterministic priority setup where target loses:
      target approaches from north, another vehicle approaches from east.
    Under priority-to-the-right, east has priority over north.
    """
    vmap = {v.id: v for v in state.vehicles}
    if target_vid not in vmap:
        raise ValueError(f"Vehicle {target_vid} not found")

    others = [vid for vid in ["A", "B", "C"] if vid != target_vid]
    random.shuffle(others)
    blocker_vid, third_vid = others

    mode = random.choice(["approach_priority", "approach_priority", "inside_blocker"])
    if mode == "approach_priority":
        target_dir, blocker_dir = random.choice(INTERSECTION_LOSING_PAIRS)
        third_dir = random.choice([
            d for d in Direction
            if d not in {target_dir, blocker_dir}
        ])

        for vid, d in {
            target_vid: target_dir,
            blocker_vid: blocker_dir,
            third_vid: third_dir,
        }.items():
            v = vmap[vid]
            v.direction = d
            v.position = f"{d.value}_approach"
            v.inside_intersection = False
            v.intent = None
            v.stopped = False
        return {
            "mode": mode,
            "target": target_vid,
            "blocker": blocker_vid,
            "third": third_vid,
        }

    # inside_blocker mode
    target_dir = random.choice(list(Direction))
    third_dir = random.choice([d for d in Direction if d != target_dir])
    v_target = vmap[target_vid]
    v_target.direction = target_dir
    v_target.position = f"{target_dir.value}_approach"
    v_target.inside_intersection = False
    v_target.intent = None
    v_target.stopped = False

    v_blocker = vmap[blocker_vid]
    v_blocker.direction = random.choice(list(Direction))
    v_blocker.position = "inside_intersection"
    v_blocker.inside_intersection = True
    v_blocker.intent = None
    v_blocker.stopped = False

    v_third = vmap[third_vid]
    v_third.direction = third_dir
    v_third.position = f"{third_dir.value}_approach"
    v_third.inside_intersection = False
    v_third.intent = None
    v_third.stopped = False

    return {
        "mode": mode,
        "target": target_vid,
        "blocker": blocker_vid,
        "third": third_vid,
    }


def _set_multilane_target_lane(state: ScenarioState, target_vid: str, lane: str) -> None:
    vmap = {v.id: v for v in state.vehicles}
    if target_vid not in vmap:
        raise ValueError(f"Vehicle {target_vid} not found")

    remaining = ["left_lane", "center_lane", "right_lane"]
    remaining.remove(lane)
    random.shuffle(remaining)

    vmap[target_vid].position = lane
    vmap[target_vid].inside_intersection = False
    vmap[target_vid].stopped = False

    others = [vid for vid in ["A", "B", "C"] if vid != target_vid]
    for vid, assigned_lane in zip(others, remaining):
        vmap[vid].position = assigned_lane
        vmap[vid].inside_intersection = False
        vmap[vid].stopped = False


def _set_roundabout_target_entering(state: ScenarioState, target_vid: str) -> dict[str, str]:
    """
    Configure target outside at an approach in one of two deterministic modes:
      - inside_priority: another vehicle is already circulating.
      - pre_entry: ring starts empty; another vehicle can enter first.
    """
    vmap = {v.id: v for v in state.vehicles}
    if target_vid not in vmap:
        raise ValueError(f"Vehicle {target_vid} not found")

    others = [vid for vid in ["A", "B", "C"] if vid != target_vid]
    random.shuffle(others)

    # Variant: no vehicle is initially circulating; another vehicle can enter
    # legally first, then target enters illegally on the next step.
    if random.random() < 0.45:
        pioneer_vid = others[0]
        for vid, v in vmap.items():
            v.intent = None
            v.stopped = False
            v.inside_intersection = False
            v.position = f"{v.direction.value}_approach"
        return {
            "target": target_vid,
            "mode": "pre_entry",
            "pioneer": pioneer_vid,
        }

    inside_vid = others[0]
    inside2_vid: str | None = others[1] if random.random() < 0.40 else None

    for vid, v in vmap.items():
        v.intent = None
        v.stopped = False
        if vid == inside_vid or vid == inside2_vid:
            v.inside_intersection = True
            v.position = "roundabout_lane"
        else:
            v.inside_intersection = False
            v.position = f"{v.direction.value}_approach"
    out = {
        "target": target_vid,
        "mode": "inside_priority",
        "inside": inside_vid,
    }
    if inside2_vid is not None:
        out["inside2"] = inside2_vid
    return out


def _set_multilane_layout(
    state: ScenarioState,
    left_vid: str,
    center_vid: str,
    right_vid: str,
) -> None:
    vmap = {v.id: v for v in state.vehicles}
    assignments = {
        left_vid: "left_lane",
        center_vid: "center_lane",
        right_vid: "right_lane",
    }
    for vid, lane in assignments.items():
        v = vmap[vid]
        v.position = lane
        v.inside_intersection = False
        v.intent = None
        v.stopped = False




def _legal_actions(state: ScenarioState, vehicle_id: str) -> list[Action]:
    actions: list[Action] = []
    for action in Action:
        if not is_valid_transition(state, vehicle_id, action):
            continue
        if detect_right_of_way_violation(state, vehicle_id, action)["is_violation"]:
            continue
        actions.append(action)
    return actions


def _plan_is_legal(state: ScenarioState, plan: list[tuple[str, Action]]) -> bool:
    sim = copy.deepcopy(state)
    for vid, action in plan:
        violation = detect_violation(sim, vid, action)
        if violation["is_violation"]:
            return False
        if not apply_action(sim, vid, action):
            return False
    return True


def _vehicle_with_position(state: ScenarioState, position: str) -> str | None:
    for v in state.vehicles:
        if v.position == position:
            return v.id
    return None


def _is_suspicious_action(env: Environment, action: Action) -> bool:
    if action == Action.STOP:
        return True
    if env == Environment.INTERSECTION and action == Action.MOVE_FORWARD:
        return True
    if env == Environment.MULTI_LANE and action in (Action.CHANGE_LEFT, Action.CHANGE_RIGHT):
        return True
    if env == Environment.ROUNDABOUT and action == Action.ENTER_ROUNDABOUT:
        return True
    return False


def _build_no_violation_templates(state: ScenarioState) -> list[list[tuple[str, Action]]]:
    vids = [v.id for v in state.vehicles]
    random.shuffle(vids)
    a, b, c = vids

    if state.environment == Environment.INTERSECTION:
        yielder = a
        priority = b
        observer = c
        yielder_dir, priority_dir = random.choice(INTERSECTION_LOSING_PAIRS)
        observer_dir = random.choice([
            d for d in Direction if d not in {yielder_dir, priority_dir}
        ])
        vmap = {v.id: v for v in state.vehicles}
        for vid, d in {
            yielder: yielder_dir,
            priority: priority_dir,
            observer: observer_dir,
        }.items():
            v = vmap[vid]
            v.direction = d
            v.position = f"{d.value}_approach"
            v.inside_intersection = False
            v.intent = None
            v.stopped = False

        turn_action = random.choice([Action.TURN_LEFT, Action.TURN_RIGHT])
        other_turn = Action.TURN_LEFT if turn_action == Action.TURN_RIGHT else Action.TURN_RIGHT
        plans = [
            [(yielder, Action.STOP), (priority, Action.MOVE_FORWARD), (priority, turn_action)],
            [(priority, Action.MOVE_FORWARD), (priority, turn_action), (yielder, Action.STOP)],
            [(yielder, Action.STOP), (priority, Action.STOP), (priority, Action.MOVE_FORWARD)],
            [(observer, Action.STOP), (yielder, Action.STOP), (priority, Action.MOVE_FORWARD)],
            [(yielder, Action.STOP), (priority, Action.MOVE_FORWARD), (observer, Action.STOP)],
            [(priority, Action.MOVE_FORWARD), (priority, Action.STOP), (yielder, Action.STOP)],
            [(yielder, Action.STOP), (priority, Action.MOVE_FORWARD), (priority, other_turn)],
        ]

        if random.random() < 0.60:
            inside_actor = observer
            v_inside = vmap[inside_actor]
            v_inside.position = "inside_intersection"
            v_inside.inside_intersection = True
            v_inside.stopped = False
            plans.extend(
                [
                    [(inside_actor, Action.TURN_LEFT), (yielder, Action.STOP)],
                    [(inside_actor, Action.TURN_RIGHT), (priority, Action.STOP)],
                    [(yielder, Action.STOP), (inside_actor, Action.TURN_LEFT)],
                    [(priority, Action.STOP), (inside_actor, Action.TURN_RIGHT)],
                    [(inside_actor, Action.STOP), (inside_actor, Action.TURN_LEFT)],
                    [(inside_actor, Action.STOP), (inside_actor, Action.TURN_RIGHT)],
                ]
            )

        return plans

    if state.environment == Environment.MULTI_LANE:
        mover = a
        left_actor = b
        right_actor = c
        _set_multilane_layout(state, left_actor, mover, right_actor)
        return [
            [(mover, Action.CHANGE_LEFT), (mover, Action.CHANGE_RIGHT), (left_actor, Action.STOP)],
            [(mover, Action.CHANGE_RIGHT), (mover, Action.CHANGE_LEFT), (right_actor, Action.STOP)],
            [(left_actor, Action.STOP), (mover, Action.CHANGE_RIGHT), (mover, Action.CHANGE_LEFT)],
            [(right_actor, Action.STOP), (mover, Action.CHANGE_LEFT), (mover, Action.CHANGE_RIGHT)],
            [(mover, Action.CHANGE_LEFT), (right_actor, Action.STOP), (mover, Action.CHANGE_RIGHT)],
            [(mover, Action.CHANGE_RIGHT), (left_actor, Action.STOP), (mover, Action.CHANGE_LEFT)],
        ]

    # roundabout
    inside = a
    entering = b
    observer = c
    vmap = {v.id: v for v in state.vehicles}
    empty_start = random.random() < 0.55
    for vid, v in vmap.items():
        v.intent = None
        v.stopped = False
        if not empty_start and vid == inside:
            v.inside_intersection = True
            v.position = "roundabout_lane"
        else:
            v.inside_intersection = False
            v.position = f"{v.direction.value}_approach"

    if empty_start:
        return [
            [(entering, Action.ENTER_ROUNDABOUT), (entering, Action.EXIT_ROUNDABOUT)],
            [(entering, Action.ENTER_ROUNDABOUT), (entering, Action.STOP)],
            [(observer, Action.STOP), (entering, Action.ENTER_ROUNDABOUT)],
            [(entering, Action.STOP), (observer, Action.ENTER_ROUNDABOUT)],
            [(observer, Action.ENTER_ROUNDABOUT), (observer, Action.EXIT_ROUNDABOUT)],
            [(observer, Action.ENTER_ROUNDABOUT), (observer, Action.STOP)],
        ]

    if random.random() < 0.45:
        vmap[observer].inside_intersection = True
        vmap[observer].position = "roundabout_lane"
        return [
            [(inside, Action.EXIT_ROUNDABOUT), (observer, Action.EXIT_ROUNDABOUT)],
            [(inside, Action.STOP), (observer, Action.EXIT_ROUNDABOUT)],
            [(observer, Action.STOP), (inside, Action.EXIT_ROUNDABOUT)],
            [(entering, Action.STOP), (inside, Action.EXIT_ROUNDABOUT)],
            [(entering, Action.STOP), (observer, Action.EXIT_ROUNDABOUT)],
        ]

    return [
        [(entering, Action.STOP), (inside, Action.EXIT_ROUNDABOUT), (entering, Action.ENTER_ROUNDABOUT)],
        [(inside, Action.EXIT_ROUNDABOUT), (entering, Action.ENTER_ROUNDABOUT), (observer, Action.STOP)],
        [(observer, Action.STOP), (entering, Action.STOP), (inside, Action.EXIT_ROUNDABOUT)],
        [(inside, Action.STOP), (entering, Action.STOP), (inside, Action.EXIT_ROUNDABOUT)],
        [(entering, Action.STOP), (observer, Action.STOP), (inside, Action.EXIT_ROUNDABOUT)],
        [(observer, Action.STOP), (inside, Action.EXIT_ROUNDABOUT), (entering, Action.ENTER_ROUNDABOUT)],
    ]


def _build_random_no_violation_candidate(
    state: ScenarioState,
    desired_difficulty: str | None = None,
) -> list[tuple[str, Action]]:
    sim = copy.deepcopy(state)
    vehicle_ids = [v.id for v in sim.vehicles]
    if desired_difficulty == "easy":
        n_events = 2
    elif desired_difficulty == "hard":
        n_events = 3
    else:
        n_events = random.choice([2, 3, 3, 3, 3])
    plan: list[tuple[str, Action]] = []
    suspicious = 0
    last_actor: str | None = None

    for _ in range(n_events):
        actor_pool = [vid for vid in vehicle_ids if _legal_actions(sim, vid)]
        if not actor_pool:
            break
        if last_actor in actor_pool and len(actor_pool) > 1 and random.random() < 0.70:
            actor_pool = [vid for vid in actor_pool if vid != last_actor]
        actor = random.choice(actor_pool)
        actions = _legal_actions(sim, actor)
        suspicious_actions = [a for a in actions if _is_suspicious_action(sim.environment, a)]
        if suspicious_actions and random.random() < 0.45:
            action = random.choice(suspicious_actions)
        else:
            action = random.choice(actions)

        if detect_violation(sim, actor, action)["is_violation"]:
            continue
        if not apply_action(sim, actor, action):
            continue

        plan.append((actor, action))
        if _is_suspicious_action(sim.environment, action):
            suspicious += 1
        last_actor = actor

    min_actors = 1 if desired_difficulty == "easy" else 2
    if (
        len(plan) < 2
        or len({vid for vid, _ in plan}) < min_actors
        or suspicious == 0
    ):
        return []
    return plan[:3]


def _build_no_violation_plan(
    state: ScenarioState,
    desired_difficulty: str | None = None,
) -> list[tuple[str, Action]]:
    templates = _build_no_violation_templates(state)
    candidates: list[list[tuple[str, Action]]] = []
    for plan in templates:
        if len(plan) < 2:
            continue
        min_actors = 1 if desired_difficulty == "easy" else 2
        if len({vid for vid, _ in plan}) < min_actors:
            continue
        if _plan_is_legal(state, plan):
            candidates.append(plan[:3])

    for _ in range(24):
        plan = _build_random_no_violation_candidate(state, desired_difficulty=desired_difficulty)
        if not plan:
            continue
        if _plan_is_legal(state, plan):
            candidates.append(plan[:3])

    if candidates:
        if desired_difficulty == "easy":
            filtered = [plan for plan in candidates if len(plan) == 2]
            if filtered:
                candidates = filtered
        elif desired_difficulty == "hard":
            filtered = [plan for plan in candidates if len(plan) >= 3]
            if filtered:
                candidates = filtered
        random.shuffle(candidates)
        return candidates[0]

    # Hard fallback (still legal).
    vehicle_ids = [v.id for v in state.vehicles]
    random.shuffle(vehicle_ids)
    return [(vehicle_ids[0], Action.STOP), (vehicle_ids[1], Action.STOP), (vehicle_ids[0], Action.STOP)]


def _pick_pre_action(
    state: ScenarioState,
    target_vid: str,
    violation_class: str,
    context: dict[str, str] | None,
) -> tuple[str, Action] | None:
    if violation_class in {"turn_without_entering", "intersection_right_of_way", "roundabout_entry_no_yield"}:
        if random.random() < 0.55 and is_valid_transition(state, target_vid, Action.STOP):
            return (target_vid, Action.STOP)

    candidates: list[tuple[str, Action]] = []
    restricted_actors: set[str] = set()
    restricted_actions: set[tuple[str, Action]] = set()

    if violation_class == "roundabout_entry_no_yield" and context:
        inside = context.get("inside")
        inside2 = context.get("inside2")
        if inside and inside2 is None:
            restricted_actions.add((inside, Action.EXIT_ROUNDABOUT))
    if violation_class == "intersection_right_of_way" and context:
        mode = context.get("mode")
        blocker = context.get("blocker")
        if mode == "inside_blocker" and blocker:
            restricted_actions.add((blocker, Action.TURN_LEFT))
            restricted_actions.add((blocker, Action.TURN_RIGHT))

    for v in state.vehicles:
        for action in _legal_actions(state, v.id):
            if action == Action.STOP:
                continue
            if v.id in restricted_actors:
                continue
            if (v.id, action) in restricted_actions:
                continue
            candidates.append((v.id, action))
    if candidates:
        random.shuffle(candidates)
        non_target = [(vid, action) for vid, action in candidates if vid != target_vid]
        if non_target:
            return non_target[0]
        return candidates[0]

    # STOP fallback on non-target actor.
    others = [v.id for v in state.vehicles if v.id != target_vid]
    random.shuffle(others)
    for vid in others:
        if is_valid_transition(state, vid, Action.STOP):
            return (vid, Action.STOP)
    if is_valid_transition(state, target_vid, Action.STOP):
        return (target_vid, Action.STOP)
    return None


def _build_violation_templates(
    state: ScenarioState,
    target_vid: str,
    violation_class: str,
    violation_action: tuple[str, Action],
    context: dict[str, str] | None,
) -> list[list[tuple[str, Action]]]:
    others = [v.id for v in state.vehicles if v.id != target_vid]
    random.shuffle(others)
    o1 = others[0]
    o2 = others[1]
    templates: list[list[tuple[str, Action]]] = []

    if violation_class in {"turn_without_entering", "forward_from_exit"}:
        templates.extend([
            [(target_vid, Action.STOP), violation_action],
            [(o1, Action.STOP), violation_action],
            [(target_vid, Action.STOP), (o1, Action.STOP), violation_action],
            [(o1, Action.STOP), violation_action, (o2, Action.STOP)],
        ])
        return templates

    if violation_class in {"lane_change_out_of_bounds_left", "lane_change_out_of_bounds_right"}:
        center_vid = _vehicle_with_position(state, "center_lane")
        if center_vid and center_vid != target_vid:
            legal_center_move = random.choice([Action.CHANGE_LEFT, Action.CHANGE_RIGHT])
            templates.append([(center_vid, legal_center_move), violation_action])
            templates.append([(center_vid, legal_center_move), (target_vid, Action.STOP), violation_action])
        templates.extend([
            [(o1, Action.STOP), violation_action],
            [(target_vid, Action.STOP), violation_action],
            [(o1, Action.STOP), violation_action, (o2, Action.STOP)],
        ])
        return templates

    if violation_class == "intersection_right_of_way":
        blocker = context.get("blocker") if context else None
        third = context.get("third") if context else None
        b = blocker if blocker and blocker != target_vid else o1
        t = third if third and third not in {target_vid, b} else o2
        templates.extend([
            [(target_vid, Action.STOP), violation_action],
            [(b, Action.STOP), violation_action],
            [(t, Action.STOP), violation_action],
            [(target_vid, Action.STOP), (b, Action.STOP), violation_action],
            [(target_vid, Action.STOP), violation_action, (t, Action.STOP)],
            [(b, Action.STOP), violation_action, (t, Action.STOP)],
        ])
        return templates

    if violation_class == "roundabout_entry_no_yield":
        inside = context.get("inside") if context else None
        inside2 = context.get("inside2") if context else None
        outside_other = o1 if o1 != inside else o2
        templates.extend([
            [(target_vid, Action.STOP), violation_action],
            [violation_action, (outside_other, Action.STOP)],
            [(outside_other, Action.STOP), violation_action],
            [(target_vid, Action.STOP), (outside_other, Action.STOP), violation_action],
            [(target_vid, Action.STOP), violation_action, (outside_other, Action.STOP)],
        ])
        if inside and inside2:
            templates.append([(inside, Action.EXIT_ROUNDABOUT), violation_action])
            templates.append([(inside2, Action.EXIT_ROUNDABOUT), violation_action])
        if inside and inside != target_vid:
            templates.append([(inside, Action.STOP), violation_action])
            templates.append([(inside, Action.STOP), (target_vid, Action.STOP), violation_action])
        return templates

    templates.append([violation_action])
    return templates


def _build_violation_plan(
    state: ScenarioState,
    target_vid: str,
    violation_class: str,
    context: dict[str, str] | None = None,
    desired_difficulty: str | None = None,
) -> list[tuple[str, Action]]:
    def _sample_legal(sim_state: ScenarioState) -> tuple[str, Action] | None:
        legal_pairs: list[tuple[str, Action]] = []
        non_stop_pairs: list[tuple[str, Action]] = []
        for vv in sim_state.vehicles:
            for aa in _legal_actions(sim_state, vv.id):
                legal_pairs.append((vv.id, aa))
                if aa != Action.STOP:
                    non_stop_pairs.append((vv.id, aa))
        if non_stop_pairs and random.random() < 0.65:
            return random.choice(non_stop_pairs)
        if legal_pairs:
            return random.choice(legal_pairs)
        return None

    if violation_class == "turn_without_entering":
        action = random.choice([Action.TURN_LEFT, Action.TURN_RIGHT])
        violation_action = (target_vid, action)

    elif violation_class == "forward_from_exit":
        violation_action = (target_vid, Action.MOVE_FORWARD)

    elif violation_class == "lane_change_out_of_bounds_left":
        violation_action = (target_vid, Action.CHANGE_LEFT)

    elif violation_class == "lane_change_out_of_bounds_right":
        violation_action = (target_vid, Action.CHANGE_RIGHT)

    elif violation_class == "intersection_right_of_way":
        violation_action = (target_vid, Action.MOVE_FORWARD)

    elif violation_class == "roundabout_entry_no_yield":
        violation_action = (target_vid, Action.ENTER_ROUNDABOUT)

    else:
        raise ValueError(f"Unknown violation class: {violation_class!r}")

    templates = _build_violation_templates(
        state,
        target_vid,
        violation_class,
        violation_action,
        context,
    )

    # Extra randomized candidate with legal pre-action for variation.
    sim = copy.deepcopy(state)
    randomized: list[tuple[str, Action]] = []
    if random.random() < 0.90:
        pre = _pick_pre_action(sim, target_vid, violation_class, context)
        if pre is not None:
            vid, action = pre
            if not detect_violation(sim, vid, action)["is_violation"] and apply_action(sim, vid, action):
                randomized.append(pre)
    randomized.append(violation_action)
    if len(randomized) < 3 and random.random() < 0.45:
        others = [v.id for v in state.vehicles if v.id != target_vid]
        random.shuffle(others)
        if others:
            randomized.append((others[0], Action.STOP))
    templates.append(randomized[:3])

    # Extra stochastic templates to increase action-pattern diversity.
    for _ in range(80):
        sim = copy.deepcopy(state)
        if desired_difficulty == "easy":
            total_len = 2
            violation_step = 2
        elif desired_difficulty == "hard":
            total_len = 3
            violation_step = 3
        elif desired_difficulty == "medium":
            total_len = 3
            violation_step = random.choice([1, 2])
        else:
            total_len = random.choice([2, 3])
            if total_len == 2:
                violation_step = random.choice([1, 2])
            else:
                violation_step = random.choice([1, 2, 3])

        plan: list[tuple[str, Action]] = []
        failed = False
        for step_i in range(1, total_len + 1):
            if step_i == violation_step:
                plan.append(violation_action)
                continue
            if step_i > violation_step:
                # Replay stops at first violation, so post-violation actions are
                # free to increase action-pattern diversity.
                tail_actor = random.choice([v.id for v in sim.vehicles])
                tail_action = random.choice(list(Action))
                plan.append((tail_actor, tail_action))
                continue
            sampled = _sample_legal(sim)
            if sampled is None:
                failed = True
                break
            vid, action = sampled
            plan.append((vid, action))
            if step_i < violation_step:
                if not apply_action(sim, vid, action):
                    failed = True
                    break
        if not failed and plan:
            templates.append(plan[:3])

    valid: list[tuple[list[tuple[str, Action]], int, bool]] = []
    random.shuffle(templates)
    checked = 0
    for plan in templates:
        checked += 1
        if len(plan) < 1:
            continue
        replay = _replay_first_violation(state, plan)
        if not replay["has_violation"]:
            continue
        if replay["violation_vehicle"] != target_vid:
            continue
        step_raw = replay.get("violation_step")
        step = step_raw if isinstance(step_raw, int) else 1
        distinct = len({vid for vid, _ in plan})
        min_distinct = 1 if desired_difficulty == "easy" else 2
        if distinct < min_distinct:
            continue
        non_final = step < len(plan)
        valid.append((plan[:3], step, non_final))
        if len(valid) >= 64:
            break
        if checked >= 180 and valid:
            break

    if desired_difficulty in {"easy", "medium", "hard"}:
        filtered_valid = []
        for plan, step, non_final in valid:
            derived = _difficulty_for_violation_plan(len(plan), step)
            if derived == desired_difficulty:
                if desired_difficulty == "medium" and len(plan) < 3:
                    continue
                filtered_valid.append((plan, step, non_final))
        if filtered_valid:
            valid = filtered_valid

    if not valid:
        others = [v.id for v in state.vehicles if v.id != target_vid]
        random.shuffle(others)
        pre_actor = others[0] if others else target_vid
        return [(pre_actor, Action.STOP), violation_action]

    # Prefer legal-before-illegal and non-final violations, but keep variety.
    weighted: list[tuple[int, list[tuple[str, Action]]]] = []
    for plan, step, non_final in valid:
        score = 0
        if step >= 2:
            score += 3
        if non_final:
            score += 2
        if len(plan) >= 3:
            score += 1
        score += random.randint(0, 2)
        weighted.append((score, plan))
    plans = [plan for _, plan in weighted]
    weights = [max(1, score) for score, _ in weighted]
    return random.choices(plans, weights=weights, k=1)[0][:3]




def _replay_first_violation(
    state: ScenarioState,
    plan: list[tuple[str, Action]],
) -> dict[str, object]:
    replay_t0 = _perf_start()
    try:
        sim = copy.deepcopy(state)

        for idx, (vehicle_id, action) in enumerate(plan, start=1):
            violation = detect_violation(sim, vehicle_id, action)
            if violation["is_violation"]:
                return {
                    "has_violation": True,
                    "violation_step": idx,
                    "violation_vehicle": vehicle_id,
                    "violation_type": violation["violation_type"],
                    "reason": violation["reason"],
                }

            applied = apply_action(sim, vehicle_id, action)
            if not applied:
                return {
                    "has_violation": False,
                    "violation_step": None,
                    "violation_vehicle": None,
                    "violation_type": "no_violation",
                    "reason": "apply_failed_after_legal_check - treated as no violation.",
                }

        return {
            "has_violation": False,
            "violation_step": None,
            "violation_vehicle": None,
            "violation_type": "no_violation",
            "reason": "No violation detected.",
        }
    finally:
        _perf_end("replay_audit", replay_t0)




def _assign_options(correct_semantic: str, correct_key: str) -> tuple[dict[str, str], dict[str, str], str]:
    if correct_semantic not in ALLOWED_CORRECT:
        raise ValueError(f"Invalid correct semantic: {correct_semantic!r}")

    items = [{"semantic": k, "text": v} for k, v in SEMANTIC_CHOICES.items()]
    correct_item = next(item for item in items if item["semantic"] == correct_semantic)
    distractors = [item for item in items if item["semantic"] != correct_semantic]
    random.shuffle(distractors)

    target_idx = LETTERS.index(correct_key)
    ordered = distractors
    ordered.insert(target_idx, correct_item)

    choices: dict[str, str] = {}
    semantic_by_letter: dict[str, str] = {}
    for letter, item in zip(LETTERS, ordered):
        choices[letter] = item["text"]
        semantic_by_letter[letter] = item["semantic"]

    return choices, semantic_by_letter, correct_key


def _build_option_rationale_by_letter(
    semantic_by_letter: dict[str, str],
    replay: dict[str, object],
    plan: list[tuple[str, Action]],
) -> dict[str, str]:
    violation_vehicle_raw = replay.get("violation_vehicle")
    violation_vehicle: str | None = (
        str(violation_vehicle_raw) if isinstance(violation_vehicle_raw, str) else None
    )

    violation_type_raw = replay.get("violation_type")
    violation_type = (
        str(violation_type_raw) if isinstance(violation_type_raw, str) else "unknown_violation"
    )

    violation_step_raw = replay.get("violation_step")
    violation_step: int | None = (
        violation_step_raw if isinstance(violation_step_raw, int) else None
    )
    has_violation = bool(replay["has_violation"])

    acted_any = [vid for vid, _ in plan]
    acted_any_set = set(acted_any)
    acted_before_set: set[str] = set()
    if violation_step is not None:
        acted_before_set = {
            vid
            for idx, (vid, _) in enumerate(plan, start=1)
            if idx < violation_step
        }

    out: dict[str, str] = {}
    for letter in LETTERS:
        semantic = semantic_by_letter[letter]

        if semantic == SEMANTIC_FIFTH_DISTRACTOR:
            out[letter] = (
                "Incorrect: all scenario vehicles are explicitly A, B, and C; "
                "no other vehicle exists in the scenario."
            )
            continue

        if semantic == SEMANTIC_UNDETERMINED:
            if has_violation:
                out[letter] = (
                    f"Incorrect: a specific violator can be determined - Vehicle {violation_vehicle} "
                    f"committed the first illegal action ({violation_type}) at event {violation_step}."
                )
            else:
                out[letter] = (
                    "Correct: replay finds no illegal action in the full event sequence."
                )
            continue

        vid = _vehicle_for_semantic(semantic)
        if vid is None:
            out[letter] = "Incorrect: unsupported option semantic."
            continue

        if has_violation and vid == violation_vehicle:
            out[letter] = (
                f"Correct: Vehicle {vid} committed the first illegal action "
                f"({violation_type}) at event {violation_step}."
            )
            continue

        if vid in acted_before_set:
            prefix = (
                f"Incorrect: Vehicle {vid} acted before the first violation but those actions were legal."
            )
        elif vid in acted_any_set:
            prefix = (
                f"Incorrect: Vehicle {vid} acted only after the first violation point."
            )
        else:
            prefix = (
                f"Incorrect: Vehicle {vid} never acted in the event sequence."
            )

        if has_violation:
            out[letter] = prefix + f" The first violation was committed by Vehicle {violation_vehicle}."
        else:
            out[letter] = prefix + " No violation occurred in this scenario."

    return out




def _semantic_from_replay(replay: dict[str, object]) -> str:
    if not replay["has_violation"]:
        return SEMANTIC_UNDETERMINED
    return _semantic_for_vehicle(str(replay["violation_vehicle"]))


def _expected_ground_truth_label(semantic: str) -> str:
    vid = _vehicle_for_semantic(semantic)
    if vid is None:
        return "no_violation"
    return vid


def _replay_from_example(example: dict) -> dict[str, object]:
    state = _reconstruct_state(example["scenario"])
    plan: list[tuple[str, Action]] = []
    for ev in example["event_plan"]:
        vehicle_id = ev["vehicle"]
        action = Action[ev["action"]]
        plan.append((vehicle_id, action))
    return _replay_first_violation(state, plan)


def validate_example_contract(example: dict) -> tuple[bool, str]:
    choices = example["choices"]
    answer = example["answer"]
    metadata = example["metadata"]

    if sorted(choices.keys()) != LETTERS:
        return False, "choices must contain A..E"

    if len(set(choices.values())) != 5:
        return False, "duplicate choice texts"

    if set(choices.values()) != set(SEMANTIC_CHOICES.values()):
        return False, "choices must exactly match fixed semantic option set"

    if answer not in LETTERS:
        return False, "invalid answer key"

    if len(example.get("event_plan", [])) != len(example.get("events", [])):
        return False, "event_plan length mismatch"

    option_rationale = example.get("audit", {}).get("option_rationale_by_letter", {})
    if sorted(option_rationale.keys()) != LETTERS:
        return False, "audit.option_rationale_by_letter must cover A..E"
    if any(not str(option_rationale[k]).strip() for k in LETTERS):
        return False, "empty option rationale in audit.option_rationale_by_letter"
    if "option_rationale_by_letter" in metadata:
        return False, "option rationale must live only in audit"
    invariants = example.get("audit", {}).get("invariants", {})
    if not invariants:
        return False, "audit.invariants missing"
    if not all(isinstance(v, bool) and v for v in invariants.values()):
        return False, "audit.invariants must all be true"

    replay = _replay_from_example(example)
    replay_semantic = _semantic_from_replay(replay)

    answer_semantic = next(
        sem for sem, txt in SEMANTIC_CHOICES.items()
        if txt == choices[answer]
    )
    if answer_semantic != replay_semantic:
        return False, "answer text does not match replayed violation result"

    expected_gt = _expected_ground_truth_label(replay_semantic)
    if metadata.get("ground_truth") != expected_gt:
        return False, "metadata.ground_truth mismatch"

    if replay["has_violation"]:
        if metadata.get("violation_vehicle") != replay["violation_vehicle"]:
            return False, "metadata.violation_vehicle mismatch"
        if metadata.get("violation_type") != replay["violation_type"]:
            return False, "metadata.violation_type mismatch"
        if metadata.get("violation_step") != replay["violation_step"]:
            return False, "metadata.violation_step mismatch"
    else:
        if metadata.get("violation_vehicle") is not None:
            return False, "metadata.violation_vehicle must be null for no_violation"
        if metadata.get("violation_type") != "no_violation":
            return False, "metadata.violation_type must be no_violation"
        if metadata.get("violation_step") is not None:
            return False, "metadata.violation_step must be null for no_violation"

    return True, "ok"




def _build_prompt(scenario_text: str, events: list[str], choices: dict[str, str]) -> str:
    parts = [scenario_text, "", "Sequence of events:"]
    for i, ev in enumerate(events, start=1):
        parts.append(f"{i}. {ev}")
    parts.append("")
    parts.append(f"Question: {QUESTION}")
    for key in LETTERS:
        parts.append(f"{key}) {choices[key]}")
    return "\n".join(parts)


def _event_signature_from_plan(plan_struct: list[dict[str, str]]) -> str:
    return "|".join(f"{step['vehicle']}:{step['action']}" for step in plan_struct)


def _assign_difficulty(record: dict) -> str:
    metadata = record.get("metadata", {})
    num_events_raw = metadata.get("num_events", len(record.get("event_plan", [])))
    try:
        num_events = int(num_events_raw)
    except (TypeError, ValueError):
        num_events = len(record.get("event_plan", []))

    violation_step = metadata.get("violation_step")
    violation_type = str(metadata.get("violation_type", "no_violation"))

    if num_events == 2 and (violation_step == 2 or violation_type == "no_violation"):
        return "easy"
    if violation_type == "no_violation":
        return "hard"
    if isinstance(violation_step, int) and violation_step == num_events:
        return "hard"
    return "medium"


def _difficulty_for_violation_plan(num_events: int, violation_step: int) -> str:
    if num_events == 2 and violation_step == 2:
        return "easy"
    if violation_step == num_events:
        return "hard"
    return "medium"


def _shape_state_for_violation(
    state: ScenarioState,
    target_vid: str,
    violation_class: str,
) -> dict[str, str]:
    if violation_class == "forward_from_exit":
        _set_target_exit_intersection(state, target_vid)
        if random.random() < 0.65:
            inside_id = _add_inside_vehicle_intersection(state, target_vid)
            return {"mode": "target_exit", "inside": inside_id}
        return {"mode": "target_exit"}

    if violation_class == "lane_change_out_of_bounds_left":
        _set_multilane_target_lane(state, target_vid, "left_lane")
        return {"mode": "left_boundary"}

    if violation_class == "lane_change_out_of_bounds_right":
        _set_multilane_target_lane(state, target_vid, "right_lane")
        return {"mode": "right_boundary"}

    if violation_class == "intersection_right_of_way":
        return _set_intersection_priority_against_target(state, target_vid)

    if violation_class == "roundabout_entry_no_yield":
        return _set_roundabout_target_entering(state, target_vid)

    # turn_without_entering: sometimes add an inside vehicle to diversify legal prefixes.
    if random.random() < 0.55:
        inside_id = _add_inside_vehicle_intersection(state, target_vid)
        return {"mode": "approach_turn", "inside": inside_id}
    return {"mode": "approach_turn"}


def generate_example(
    example_id: int,
    correct_key: str,
    target_semantic: str,
    seed: int | None,
    violation_class: str | None,
    no_violation_env: Environment | None,
    desired_difficulty: str | None = None,
    forbidden_event_signatures: set[str] | None = None,
    forbidden_action_patterns: set[str] | None = None,
) -> dict | None:
    forbidden_event_signatures = forbidden_event_signatures or set()
    forbidden_action_patterns = forbidden_action_patterns or set()

    for attempt in range(EXAMPLE_RETRIES):
        violation_context: dict[str, str] | None = None
        if target_semantic == SEMANTIC_UNDETERMINED:
            if no_violation_env is None:
                raise ValueError("no_violation examples require an environment hint")
            env = no_violation_env
            state = _build_state(env)
            plan_t0 = _perf_start()
            plan = _build_no_violation_plan(state, desired_difficulty=desired_difficulty)
            _perf_end("plan_generation", plan_t0)
        else:
            target_vid = _vehicle_for_semantic(target_semantic)
            if target_vid is None:
                continue
            if violation_class is None:
                continue
            env = VIOLATION_ENV[violation_class]
            state = _build_state(env)
            violation_context = _shape_state_for_violation(state, target_vid, violation_class)
            plan_t0 = _perf_start()
            plan = _build_violation_plan(
                state,
                target_vid,
                violation_class,
                context=violation_context,
                desired_difficulty=desired_difficulty,
            )
            _perf_end("plan_generation", plan_t0)

        plan_struct = [
            {"vehicle": vid, "action": action.name}
            for (vid, action) in plan
        ]
        signature = _event_signature_from_plan(plan_struct)
        if signature in forbidden_event_signatures:
            continue
        action_pattern = "|".join(step["action"] for step in plan_struct)
        if action_pattern in forbidden_action_patterns:
            continue

        replay = _replay_first_violation(state, plan)
        if replay.get("has_violation") and replay.get("violation_type") not in ALLOWED_VIOLATION_TYPES:
            continue
        replay_semantic = _semantic_from_replay(replay)
        if replay_semantic != target_semantic:
            continue

        distinct_actors = len({vid for vid, _ in plan})
        if (
            target_semantic != SEMANTIC_UNDETERMINED
            and distinct_actors < 2
            and desired_difficulty != "easy"
        ):
            continue
        if (
            target_semantic == SEMANTIC_UNDETERMINED
            and distinct_actors < 2
            and desired_difficulty != "easy"
        ):
            continue

        build_prompt_t0 = _perf_start()
        choices, semantic_by_letter, answer = _assign_options(target_semantic, correct_key)
        option_rationale_by_letter = _build_option_rationale_by_letter(
            semantic_by_letter,
            replay,
            plan,
        )
        events = [_event_text(vid, action) for (vid, action) in plan]
        prompt = _build_prompt(describe_scenario(state), events, choices)
        _perf_end("build_prompts_choices", build_prompt_t0)

        ground_truth = _expected_ground_truth_label(replay_semantic)
        violation_vehicle = replay["violation_vehicle"] if replay["has_violation"] else None
        violation_type = replay["violation_type"] if replay["has_violation"] else "no_violation"
        is_no_violation = ground_truth == "no_violation"
        answer_text = choices.get(answer, "")
        answer_is_undetermined = answer_text == SEMANTIC_CHOICES[SEMANTIC_UNDETERMINED]
        all_events_valid_format = all(
            step.get("vehicle") in {"A", "B", "C"} and step.get("action") in Action.__members__
            for step in plan_struct
        )
        metadata_violation_vehicle = violation_vehicle
        metadata_violation_type = violation_type
        metadata_violation_step = replay["violation_step"]
        first_illegal_event_matches_metadata = (
            (
                replay["has_violation"]
                and metadata_violation_vehicle == replay["violation_vehicle"]
                and metadata_violation_type == replay["violation_type"]
                and metadata_violation_step == replay["violation_step"]
            )
            or (
                (not replay["has_violation"])
                and metadata_violation_vehicle is None
                and metadata_violation_type == "no_violation"
                and metadata_violation_step is None
            )
        )
        expected_correct_text = SEMANTIC_CHOICES[replay_semantic]

        distractor_type: dict[str, str] = {}
        for letter in LETTERS:
            if letter == answer:
                continue
            sem = semantic_by_letter[letter]
            if sem in {SEMANTIC_VEHICLE_A, SEMANTIC_VEHICLE_B, SEMANTIC_VEHICLE_C}:
                distractor_type[letter] = "near_true"
            else:
                distractor_type[letter] = "highly_false"

        example = {
            "id": f"task3_{example_id:04d}",
            "task": "violation_detection",
            "prompt": prompt,
            "scenario": _serialize_state(state),
            "events": events,
            "event_plan": plan_struct,
            "question": QUESTION,
            "choices": choices,
            "answer": answer,
            "distractor_type": distractor_type,
            "metadata": {
                "num_vehicles": NUM_VEHICLES,
                "num_events": len(plan),
                "environment": env.value,
                "ground_truth": ground_truth,
                "violation_vehicle": violation_vehicle,
                "violation_type": violation_type,
                "violation_step": replay["violation_step"],
                "target_semantic": target_semantic,
                "configured_violation_class": violation_class,
            },
            "audit": {
                "generator_version": GENERATOR_VERSION,
                "seed": seed,
                "attempt": attempt,
                "semantic_by_letter": semantic_by_letter,
                "option_rationale_by_letter": option_rationale_by_letter,
                "replay": replay,
                "violation_context": violation_context,
                "invariants": {
                    "no_duplicate_options": len(set(choices.values())) == 5,
                    "fixed_option_set": set(choices.values()) == set(SEMANTIC_CHOICES.values()),
                    "answer_in_choices": answer in choices,
                    "correct_vehicle_not_missing_from_choices": expected_correct_text in choices.values(),
                    "undetermined_correct_only_for_no_violation": answer_is_undetermined == is_no_violation,
                    "violation_step_none_only_for_no_violation": (metadata_violation_step is None) == is_no_violation,
                    "all_events_valid_format": all_events_valid_format,
                    "first_illegal_event_matches_metadata": first_illegal_event_matches_metadata,
                    "target_matches_replay": replay_semantic == target_semantic,
                },
            },
        }

        audit_t0 = _perf_start()
        ok, _ = validate_example_contract(example)
        _perf_end("replay_audit", audit_t0)
        if not ok:
            continue

        return example

    return None




def _assert_dataset_quality(examples: list[dict]) -> None:
    if not examples:
        raise RuntimeError("empty dataset")

    n = len(examples)
    no_violation_count = sum(1 for ex in examples if ex["metadata"]["ground_truth"] == "no_violation")
    no_violation_ratio = no_violation_count / n
    if not (0.15 <= no_violation_ratio <= 0.25):
        raise RuntimeError(
            f"no_violation ratio out of range: {no_violation_ratio:.3f} (count={no_violation_count}, n={n})"
        )

    undetermined_correct = sum(
        1
        for ex in examples
        if ex["choices"][ex["answer"]] == SEMANTIC_CHOICES[SEMANTIC_UNDETERMINED]
    )
    if undetermined_correct != no_violation_count:
        raise RuntimeError(
            "No vehicle performed an illegal action must be correct exactly in no_violation examples"
        )

    violating_examples = [
        ex for ex in examples
        if ex["metadata"]["ground_truth"] in {"A", "B", "C"}
    ]
    if violating_examples:
        violator_is_last_actor = 0
        for ex in violating_examples:
            violator = ex["metadata"]["violation_vehicle"]
            last_actor = ex["event_plan"][-1]["vehicle"]
            if violator == last_actor:
                violator_is_last_actor += 1
        if violator_is_last_actor == len(violating_examples):
            raise RuntimeError("violating vehicle is always the last actor")

    violator_counts = Counter(
        ex["metadata"]["violation_vehicle"]
        for ex in examples
        if ex["metadata"]["violation_vehicle"] is not None
    )
    if violator_counts and len([vid for vid in ["A", "B", "C"] if violator_counts.get(vid, 0) > 0]) < 3:
        raise RuntimeError("violations are not distributed across A/B/C")

    seq_len_counts = Counter(ex["metadata"]["num_events"] for ex in examples)
    if seq_len_counts[1] > max(1, int(0.10 * n)):
        raise RuntimeError("too many 1-event examples (trivial sequences)")

    acted_vehicle_counts = Counter(
        len({step["vehicle"] for step in ex["event_plan"]})
        for ex in examples
    )
    if acted_vehicle_counts[1] > max(1, int(0.08 * n)):
        raise RuntimeError("too many single-actor traces")

    no_violation_examples = [
        ex for ex in examples
        if ex["metadata"]["ground_truth"] == "no_violation"
    ]
    if no_violation_examples:
        no_v_short = sum(1 for ex in no_violation_examples if ex["metadata"]["num_events"] < 2)
        if no_v_short > 0:
            raise RuntimeError("no_violation examples must have at least 2 events")

    if violating_examples:
        with_legal_prefix = sum(
            1
            for ex in violating_examples
            if ex["metadata"]["violation_step"] is not None and ex["metadata"]["violation_step"] >= 2
        )
        if with_legal_prefix < int(0.50 * len(violating_examples)):
            raise RuntimeError("too few violations with a legal prefix action")

        non_final_violation = sum(
            1
            for ex in violating_examples
            if ex["metadata"]["violation_step"] is not None
            and ex["metadata"]["violation_step"] < ex["metadata"]["num_events"]
        )
        if non_final_violation < int(0.20 * len(violating_examples)):
            raise RuntimeError("violating action is final too often")

    vtype_counts = Counter(ex["metadata"]["violation_type"] for ex in examples)
    unknown = [
        ex["metadata"]["violation_type"]
        for ex in examples
        if ex["metadata"]["violation_type"] not in ALLOWED_VIOLATION_TYPES
    ]
    if unknown:
        raise RuntimeError(f"Unknown violation types in dataset: {set(unknown)}")
    row_min = max(1, int(0.08 * n))
    if vtype_counts["intersection_right_of_way"] < row_min:
        raise RuntimeError("insufficient intersection_right_of_way coverage")
    if vtype_counts["roundabout_entry_no_yield"] < row_min:
        raise RuntimeError("insufficient roundabout_entry_no_yield coverage")

    sequence_counts = Counter(
        _event_signature_from_plan(ex["event_plan"])
        for ex in examples
    )
    repeated_count = sum(1 for c in sequence_counts.values() if c > 1)
    max_repeat = max(sequence_counts.values()) if sequence_counts else 0
    if repeated_count > int(0.18 * n):
        raise RuntimeError("too many repeated event-sequence templates")
    if max_repeat > EVENT_SIGNATURE_REUSE_CAP:
        raise RuntimeError("event-sequence repeat cap exceeded")

    action_pattern_counts = Counter(
        "|".join(s["action"] for s in ex["event_plan"])
        for ex in examples
    )
    max_action_repeat = max(action_pattern_counts.values()) if action_pattern_counts else 0
    if max_action_repeat > ACTION_PATTERN_REUSE_CAP:
        raise RuntimeError("action-pattern single-pattern repeat cap exceeded")
    total_affected = sum(c for c in action_pattern_counts.values() if c > 1)
    if total_affected > MAX_ACTION_PATTERN_REUSE_TOTAL:
        raise RuntimeError(
            f"too many records share an action pattern: {total_affected} > "
            f"{MAX_ACTION_PATTERN_REUSE_TOTAL}"
        )

    difficulty_counts = Counter(ex["metadata"].get("difficulty") for ex in examples)
    if n == 100:
        for d in DIFFICULTIES:
            if not (30 <= difficulty_counts.get(d, 0) <= 37):
                raise RuntimeError(f"difficulty count out of range for {d}: {difficulty_counts.get(d, 0)}")

        diff_env_counts = Counter(
            (ex["metadata"].get("difficulty"), ex["metadata"].get("environment"))
            for ex in examples
        )
        bad_cells = [
            (d, env, diff_env_counts[(d, env)])
            for d in DIFFICULTIES
            for env in VIOLATION_ENVS
            if not (8 <= diff_env_counts[(d, env)] <= 14)
        ]
        if bad_cells:
            raise RuntimeError(f"difficulty x environment cell out of range: {bad_cells}")

    record_attempts = [
        int(ex.get("audit", {}).get("attempt", 0)) + 1
        for ex in examples
    ]
    retry_over_one = sum(1 for a in record_attempts if a > 1)
    if VERBOSE and retry_over_one / n >= 0.20:
        _log(
            f"[task3] warning: elevated retry rate: {retry_over_one}/{n} records needed >1 attempt"
        )
    if VERBOSE and record_attempts and max(record_attempts) >= 15:
        _log(
            f"[task3] warning: elevated max attempts: {max(record_attempts)}"
        )




def _nearest_unfilled_same_tier(
    idx: int,
    unfilled: set[int],
    difficulty_schedule: list[str],
    blocked_pairs: set[tuple[int, int]],
) -> int | None:
    tier = difficulty_schedule[idx]
    candidates = [
        j for j in unfilled
        if j != idx and difficulty_schedule[j] == tier
    ]
    if not candidates:
        return None
    for j in sorted(candidates, key=lambda x: (abs(x - idx), x)):
        pair = (min(idx, j), max(idx, j))
        if pair not in blocked_pairs:
            return j
    return None


def _action_pattern_total_affected(action_pattern_usage: Counter[str]) -> int:
    return sum(count for count in action_pattern_usage.values() if count > 1)


def _action_pattern_would_exceed_caps(
    action_pattern_usage: Counter[str],
    action_pattern: str,
) -> bool:
    current = action_pattern_usage[action_pattern]
    if current >= ACTION_PATTERN_REUSE_CAP:
        return True

    total_affected = _action_pattern_total_affected(action_pattern_usage)
    if current == 0:
        next_total = total_affected
    elif current == 1:
        # First duplicate turns both records in this pattern into "affected".
        next_total = total_affected + 2
    else:
        next_total = total_affected + 1

    return next_total > MAX_ACTION_PATTERN_REUSE_TOTAL


def _blocked_action_patterns(action_pattern_usage: Counter[str]) -> set[str]:
    blocked = {
        pattern
        for pattern, count in action_pattern_usage.items()
        if count >= ACTION_PATTERN_REUSE_CAP
    }
    if _action_pattern_total_affected(action_pattern_usage) >= MAX_ACTION_PATTERN_REUSE_TOTAL:
        blocked.update(action_pattern_usage.keys())
    return blocked


def _difficulty_range_feasible(
    difficulty_counts: Counter[str],
    accepted_tier: str,
    remaining_slots_after: int,
    n: int,
) -> bool:
    if n != 100:
        return True
    projected = Counter(difficulty_counts)
    projected[accepted_tier] += 1
    for tier in DIFFICULTIES:
        low = projected[tier]
        high = projected[tier] + remaining_slots_after
        if high < 30 or low > 37:
            return False
    return True


def generate_task3(n: int, output_path: str, seed: int | None = DEFAULT_SEED) -> None:
    total_t0 = _perf_start()
    if PROFILE_RUNTIME:
        RUNTIME_SECONDS.clear()
        RUNTIME_COUNTS.clear()

    if seed is not None:
        random.seed(seed)

    env_quota = {env: n // len(VIOLATION_ENVS) for env in VIOLATION_ENVS}
    for i, env in enumerate(VIOLATION_ENVS):
        if i < n % len(VIOLATION_ENVS):
            env_quota[env] += 1
    difficulty_quota = _build_difficulty_quota(n)
    difficulty_env_targets = _build_difficulty_env_targets(difficulty_quota, env_quota)

    max_batch_attempts = 40
    slot_retries = MAX_RETRIES
    slot_attempt_budget = min(slot_retries, MAX_SLOT_ATTEMPTS_SOFT_CAP)
    early_fallback_attempts = min(slot_attempt_budget, SLOT_EARLY_FALLBACK_ATTEMPTS)
    max_difficulty_fallback = max(1, n // 20)
    examples: list[dict] = []
    difficulty_fallback_used = 0
    last_error = "generation failed"
    profile_rejection_reasons: Counter[str] = Counter()

    for batch_attempt in range(max_batch_attempts):
        _log(f"[task3] batch {batch_attempt + 1}/{max_batch_attempts} start")
        key_schedule = _build_key_schedule(n)
        semantic_schedule = _build_semantic_schedule(n)
        difficulty_schedule, _ = _build_difficulty_schedule(n)
        _align_difficulty_schedule_with_semantics(semantic_schedule, difficulty_schedule)
        semantic_remaining: Counter[str] = Counter(semantic_schedule)

        examples = []
        seen_prompts: set[str] = set()
        event_signature_usage: Counter[str] = Counter()
        action_pattern_usage: Counter[str] = Counter()
        env_counts: Counter[str] = Counter()
        violation_class_usage: Counter[str] = Counter()
        difficulty_counts: Counter[str] = Counter()
        difficulty_env_counts: Counter[tuple[str, str]] = Counter()
        swap_history: set[tuple[int, int]] = set()
        slot_swap_counts: Counter[int] = Counter()
        difficulty_fallback_used = 0
        batch_rejection_reasons: Counter[str] = Counter()

        unfilled: set[int] = set(range(n))
        failed_idx: int | None = None
        difficulty_priority = {"easy": 0, "medium": 1, "hard": 2}

        slot_i = 0
        while unfilled:
            progressed = False
            slot_indices = list(unfilled)
            random.shuffle(slot_indices)
            slot_indices.sort(key=lambda i: difficulty_priority[difficulty_schedule[i]])
            for idx in slot_indices:

                if VERBOSE and slot_i % 10 == 0:
                    _log(
                        f"[task3] batch {batch_attempt + 1} slot={slot_i} idx={idx} "
                        f"env={dict(env_counts)} diff={dict(difficulty_counts)}",
                    )
                slot_i += 1

                correct_key = key_schedule[idx]
                desired_difficulty = difficulty_schedule[idx]
                ex: dict | None = None
                attempts_used = 0
                chosen_semantic: str | None = None
                fallback_candidate: dict | None = None
                fallback_semantic: str | None = None
                fallback_difficulty: str | None = None
                fallback_signature: str | None = None
                fallback_action_pattern: str | None = None
                fallback_env: str | None = None
                fallback_configured_class: str | None = None
                fallback_score: tuple[int, int] | None = None

                for _attempt_i in range(slot_attempt_budget):
                    attempts_used += 1
                    if (
                        attempts_used >= early_fallback_attempts
                        and fallback_candidate is not None
                        and fallback_semantic is not None
                        and fallback_difficulty is not None
                        and fallback_signature is not None
                        and fallback_action_pattern is not None
                        and fallback_env is not None
                        and difficulty_fallback_used < max_difficulty_fallback
                        and _difficulty_range_feasible(
                            difficulty_counts,
                            fallback_difficulty,
                            len(unfilled) - 1,
                            n,
                        )
                    ):
                        ex = fallback_candidate
                        chosen_semantic = fallback_semantic
                        ex["metadata"]["difficulty"] = fallback_difficulty
                        ex["audit"]["slot_attempts"] = attempts_used
                        seen_prompts.add(ex["prompt"])
                        event_signature_usage[fallback_signature] += 1
                        action_pattern_usage[fallback_action_pattern] += 1
                        env_counts[fallback_env] += 1
                        difficulty_counts[fallback_difficulty] += 1
                        difficulty_env_counts[(fallback_difficulty, fallback_env)] += 1
                        if fallback_configured_class is not None:
                            violation_class_usage[fallback_configured_class] += 1
                        difficulty_fallback_used += 1
                        RUNTIME_COUNTS["early_fallback_accepts"] += 1
                        _log(
                            f"[task3] warning: early difficulty fallback at slot {idx}, "
                            f"attempts={attempts_used}, assigned {fallback_difficulty} "
                            f"instead of {desired_difficulty}",
                        )
                        break

                    semantic_pool = [
                        s for s, c in semantic_remaining.items()
                        if c > 0
                    ]
                    if desired_difficulty == "medium":
                        semantic_pool = [s for s in semantic_pool if s != SEMANTIC_UNDETERMINED]
                    if not semantic_pool:
                        batch_rejection_reasons["semantic_pool_empty"] += 1
                        continue
                    weights = [float(max(1, semantic_remaining[s])) for s in semantic_pool]
                    if desired_difficulty == "easy" and SEMANTIC_UNDETERMINED in semantic_pool:
                        und_idx = semantic_pool.index(SEMANTIC_UNDETERMINED)
                        weights[und_idx] *= 1.8
                    semantic = random.choices(semantic_pool, weights=weights, k=1)[0]

                    desired_env = _pick_env_for_example(
                        semantic,
                        desired_difficulty,
                        env_counts,
                        env_quota,
                        difficulty_env_counts,
                        difficulty_env_targets,
                        allow_overquota=False,
                    )
                    blocked_action_patterns = _blocked_action_patterns(action_pattern_usage)
                    difficulty_hint = desired_difficulty
                    if _attempt_i >= int(0.90 * slot_attempt_budget):
                        difficulty_hint = None
                    if semantic == SEMANTIC_UNDETERMINED:
                        env_hint = Environment(desired_env)
                        gen_t0 = _perf_start()
                        candidate = generate_example(
                            idx,
                            correct_key,
                            semantic,
                            seed,
                            violation_class=None,
                            no_violation_env=env_hint,
                            desired_difficulty=difficulty_hint,
                            forbidden_event_signatures=set(event_signature_usage.keys()),
                            forbidden_action_patterns=blocked_action_patterns,
                        )
                        _perf_end("generate_examples", gen_t0)
                        RUNTIME_COUNTS["generate_example_calls"] += 1
                    else:
                        violation_class = _pick_violation_class_for_env(
                            desired_env,
                            violation_class_usage,
                        )
                        gen_t0 = _perf_start()
                        candidate = generate_example(
                            idx,
                            correct_key,
                            semantic,
                            seed,
                            violation_class=violation_class,
                            no_violation_env=None,
                            desired_difficulty=difficulty_hint,
                            forbidden_event_signatures=set(event_signature_usage.keys()),
                            forbidden_action_patterns=blocked_action_patterns,
                        )
                        _perf_end("generate_examples", gen_t0)
                        RUNTIME_COUNTS["generate_example_calls"] += 1

                    if candidate is None:
                        batch_rejection_reasons["candidate_none"] += 1
                        continue
                    candidate_difficulty = _assign_difficulty(candidate)

                    uniq_t0 = _perf_start()
                    if candidate["prompt"] in seen_prompts:
                        batch_rejection_reasons["prompt_seen"] += 1
                        _perf_end("uniqueness_checks", uniq_t0)
                        continue
                    signature = _event_signature_from_plan(candidate["event_plan"])
                    if event_signature_usage[signature] >= EVENT_SIGNATURE_REUSE_CAP:
                        batch_rejection_reasons["event_signature_cap"] += 1
                        _perf_end("uniqueness_checks", uniq_t0)
                        continue
                    action_pattern = "|".join(step["action"] for step in candidate["event_plan"])
                    if _action_pattern_would_exceed_caps(action_pattern_usage, action_pattern):
                        batch_rejection_reasons["action_pattern_cap"] += 1
                        _perf_end("uniqueness_checks", uniq_t0)
                        continue

                    candidate_env = candidate["scenario"]["environment"]
                    env_upper = env_quota.get(candidate_env, n)
                    if n == 100:
                        env_upper = 37
                    if env_counts[candidate_env] >= env_upper:
                        batch_rejection_reasons["env_quota_cap"] += 1
                        _perf_end("uniqueness_checks", uniq_t0)
                        continue
                    if n == 100 and difficulty_env_counts[(candidate_difficulty, candidate_env)] >= 14:
                        batch_rejection_reasons["difficulty_env_cell_cap"] += 1
                        _perf_end("uniqueness_checks", uniq_t0)
                        continue
                    _perf_end("uniqueness_checks", uniq_t0)

                    configured_class_raw = candidate["metadata"].get("configured_violation_class")
                    configured_class = configured_class_raw if isinstance(configured_class_raw, str) else None

                    if candidate_difficulty != desired_difficulty:
                        batch_rejection_reasons["difficulty_mismatch"] += 1
                        candidate_score = (
                            difficulty_counts[candidate_difficulty],
                            attempts_used,
                        )
                        if (
                            fallback_candidate is None
                            or fallback_score is None
                            or candidate_score < fallback_score
                        ):
                            fallback_candidate = candidate
                            fallback_semantic = semantic
                            fallback_difficulty = candidate_difficulty
                            fallback_signature = signature
                            fallback_action_pattern = action_pattern
                            fallback_env = candidate_env
                            fallback_configured_class = configured_class
                            fallback_score = candidate_score
                        continue

                    ex = candidate
                    chosen_semantic = semantic
                    ex["metadata"]["difficulty"] = candidate_difficulty
                    ex["audit"]["slot_attempts"] = attempts_used
                    seen_prompts.add(candidate["prompt"])
                    event_signature_usage[signature] += 1
                    action_pattern_usage[action_pattern] += 1
                    env_counts[candidate_env] += 1
                    difficulty_counts[candidate_difficulty] += 1
                    difficulty_env_counts[(candidate_difficulty, candidate_env)] += 1
                    if configured_class is not None:
                        violation_class_usage[configured_class] += 1
                    break

                if ex is not None:
                    examples.append(ex)
                    if chosen_semantic is not None:
                        semantic_remaining[chosen_semantic] -= 1
                    unfilled.remove(idx)
                    RUNTIME_COUNTS["accepted_examples"] += 1
                    progressed = True
                    continue

                swap_idx = _nearest_unfilled_same_tier(
                    idx,
                    unfilled,
                    difficulty_schedule,
                    swap_history,
                )
                if swap_idx is not None and slot_swap_counts[idx] >= 3:
                    swap_idx = None
                if swap_idx is None:
                    if (
                        fallback_candidate is not None
                        and fallback_semantic is not None
                        and fallback_difficulty is not None
                        and fallback_signature is not None
                        and fallback_action_pattern is not None
                        and fallback_env is not None
                    ):
                        if difficulty_fallback_used >= max_difficulty_fallback:
                            batch_rejection_reasons["fallback_budget_exceeded"] += 1
                            failed_idx = idx
                            last_error = (
                                f"difficulty fallback budget exceeded ({difficulty_fallback_used}/{max_difficulty_fallback}) "
                                f"at slot {idx}"
                            )
                            break
                        remaining_after = len(unfilled) - 1
                        if not _difficulty_range_feasible(
                            difficulty_counts,
                            fallback_difficulty,
                            remaining_after,
                            n,
                        ):
                            batch_rejection_reasons["fallback_range_blocked"] += 1
                            failed_idx = idx
                            last_error = (
                                f"difficulty fallback would break target range at slot {idx}: "
                                f"assigned={fallback_difficulty}, counts={dict(difficulty_counts)}, "
                                f"remaining={remaining_after}"
                            )
                            break
                        ex = fallback_candidate
                        chosen_semantic = fallback_semantic
                        ex["metadata"]["difficulty"] = fallback_difficulty
                        ex["audit"]["slot_attempts"] = attempts_used
                        seen_prompts.add(ex["prompt"])
                        event_signature_usage[fallback_signature] += 1
                        action_pattern_usage[fallback_action_pattern] += 1
                        env_counts[fallback_env] += 1
                        difficulty_counts[fallback_difficulty] += 1
                        difficulty_env_counts[(fallback_difficulty, fallback_env)] += 1
                        if fallback_configured_class is not None:
                            violation_class_usage[fallback_configured_class] += 1
                        examples.append(ex)
                        semantic_remaining[chosen_semantic] -= 1
                        unfilled.remove(idx)
                        difficulty_fallback_used += 1
                        RUNTIME_COUNTS["accepted_examples"] += 1
                        _log(
                            f"[task3] warning: difficulty fallback at slot {idx}, "
                            f"assigned {fallback_difficulty} instead of {desired_difficulty}",
                        )
                        progressed = True
                        continue

                    failed_idx = idx
                    batch_rejection_reasons["no_swap_available"] += 1
                    last_error = (
                        f"Unable to satisfy strict difficulty tier {desired_difficulty!r} at slot {idx} "
                        "and no same-tier unfilled slot is available for swap."
                    )
                    break
                swap_history.add((min(idx, swap_idx), max(idx, swap_idx)))
                slot_swap_counts[idx] += 1
                slot_swap_counts[swap_idx] += 1
                semantic_schedule[idx], semantic_schedule[swap_idx] = semantic_schedule[swap_idx], semantic_schedule[idx]
                key_schedule[idx], key_schedule[swap_idx] = key_schedule[swap_idx], key_schedule[idx]
                _log(
                    f"[task3] warning: slot-swap fallback idx={idx} <-> {swap_idx} "
                    f"tier={desired_difficulty}",
                )
                progressed = True

            if failed_idx is not None:
                break
            if not progressed:
                failed_idx = min(unfilled)
                batch_rejection_reasons["stalled_no_progress"] += 1
                last_error = (
                    f"stalled with unfilled slots at batch attempt "
                    f"{batch_attempt + 1}/{max_batch_attempts}: {sorted(unfilled)}"
                )
                break

        if failed_idx is not None or unfilled:
            _log(
                f"[task3] batch {batch_attempt + 1} rejected: "
                f"failed_idx={failed_idx}, unfilled={len(unfilled)}"
            )
            continue

        if n == 100:
            bad_tiers = [
                (d, difficulty_counts.get(d, 0))
                for d in DIFFICULTIES
                if not (30 <= difficulty_counts.get(d, 0) <= 37)
            ]
            if bad_tiers:
                last_error = (
                    f"difficulty distribution out of range "
                    f"(batch attempt {batch_attempt + 1}/{max_batch_attempts}): {bad_tiers}"
                )
                _log(f"[task3] {last_error}")
                continue

        if n == 100:
            bad_env = [
                (env, env_counts.get(env, 0))
                for env in VIOLATION_ENVS
                if not (30 <= env_counts.get(env, 0) <= 37)
            ]
            if bad_env:
                last_error = (
                    f"environment distribution out of range "
                    f"(batch attempt {batch_attempt + 1}/{max_batch_attempts}): {bad_env}"
                )
                _log(f"[task3] {last_error}")
                continue

        if n == 100:
            bad_cells = [
                (d, env, difficulty_env_counts[(d, env)])
                for d in DIFFICULTIES
                for env in VIOLATION_ENVS
                if not (8 <= difficulty_env_counts[(d, env)] <= 14)
            ]
            if bad_cells:
                last_error = (
                    f"difficulty x environment out of range "
                    f"(batch attempt {batch_attempt + 1}/{max_batch_attempts}): {bad_cells}"
                )
                _log(f"[task3] {last_error}")
                continue

        try:
            _assert_dataset_quality(examples)
        except RuntimeError as exc:
            last_error = (
                f"dataset quality gate failed "
                f"(batch attempt {batch_attempt + 1}/{max_batch_attempts}): {exc}"
            )
            _log(f"[task3] {last_error}")
            continue

        profile_rejection_reasons = batch_rejection_reasons
        break
    else:
        raise RuntimeError(last_error)

    generated_at_utc = (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    generation_config = {
        "num_vehicles": NUM_VEHICLES,
        "events_min": 1,
        "events_max": 3,
        "no_violation_target_ratio": 0.20,
        "event_signature_reuse_cap": EVENT_SIGNATURE_REUSE_CAP,
        "action_pattern_reuse_cap": ACTION_PATTERN_REUSE_CAP,
        "max_action_pattern_reuse_total": MAX_ACTION_PATTERN_REUSE_TOTAL,
        "max_slot_attempts_soft_cap": slot_attempt_budget,
        "slot_early_fallback_attempts": early_fallback_attempts,
        "high_attempt_slot_threshold": HIGH_ATTEMPT_SLOT_THRESHOLD,
        "max_difficulty_fallback_slots": max_difficulty_fallback,
        "used_difficulty_fallback_slots": difficulty_fallback_used,
        "env_quota": env_quota,
        "difficulty_quota": difficulty_quota,
        "difficulty_env_targets": {
            f"{d}@{env}": difficulty_env_targets[(d, env)]
            for d in DIFFICULTIES
            for env in VIOLATION_ENVS
        },
    }
    examples = sorted(examples, key=lambda ex: ex["id"])

    for ex in examples:
        ex["metadata"]["reproducibility"] = {
            "seed": seed,
            "generator_version": GENERATOR_VERSION,
            "n_examples": n,
            "generated_at_utc": generated_at_utc,
            "generation_config": generation_config,
        }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=f"{out_path.name}.tmp.",
        dir=str(out_path.parent),
    )
    write_t0 = _perf_start()
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    _perf_end("write_jsonl", write_t0)

    _log(
        f"Saved {len(examples)} examples to {out_path.resolve()} "
        f"({out_path.stat().st_size} bytes)",
        verbose_only=False,
    )

    if VERBOSE:
        answer_counts = Counter(ex["answer"] for ex in examples)
        _log("\nAnswer distribution:", verbose_only=False)
        for k in LETTERS:
            _log(f"  {k}: {answer_counts[k]}", verbose_only=False)

        gt_counts = Counter(ex["metadata"]["ground_truth"] for ex in examples)
        _log("\nGround truth distribution:", verbose_only=False)
        for k in ["A", "B", "C", "no_violation"]:
            _log(f"  {k}: {gt_counts[k]}", verbose_only=False)

        env_counts = Counter(ex["metadata"]["environment"] for ex in examples)
        _log("\nEnvironment distribution:", verbose_only=False)
        for k in sorted(env_counts):
            _log(f"  {k}: {env_counts[k]}", verbose_only=False)

        vtype_counts = Counter(ex["metadata"]["violation_type"] for ex in examples)
        _log("\nViolation-type distribution:", verbose_only=False)
        for k, c in sorted(vtype_counts.items()):
            _log(f"  {k}: {c}", verbose_only=False)

    total_runtime = time.perf_counter() - total_t0
    if PROFILE_RUNTIME:
        avg_time = total_runtime / max(1, len(examples))
        calls = RUNTIME_COUNTS["generate_example_calls"]
        accepted = max(1, RUNTIME_COUNTS["accepted_examples"])
        slot_attempts = [
            (
                ex["id"],
                int(ex.get("audit", {}).get("slot_attempts", int(ex.get("audit", {}).get("attempt", 0)) + 1)),
            )
            for ex in examples
        ]
        max_slot_attempts = max((attempt for _, attempt in slot_attempts), default=0)
        high_attempt_examples = [
            (eid, attempt)
            for eid, attempt in slot_attempts
            if attempt >= HIGH_ATTEMPT_SLOT_THRESHOLD
        ]
        _log("\nRuntime profile (seconds):", verbose_only=False)
        _log(f"  total_runtime: {total_runtime:.3f}", verbose_only=False)
        _log(f"  generate_examples: {RUNTIME_SECONDS['generate_examples']:.3f}", verbose_only=False)
        _log(f"  plan_generation: {RUNTIME_SECONDS['plan_generation']:.3f}", verbose_only=False)
        _log(f"  build_prompts_choices: {RUNTIME_SECONDS['build_prompts_choices']:.3f}", verbose_only=False)
        _log(f"  replay_audit: {RUNTIME_SECONDS['replay_audit']:.3f}", verbose_only=False)
        _log(f"  uniqueness_checks: {RUNTIME_SECONDS['uniqueness_checks']:.3f}", verbose_only=False)
        _log(f"  write_jsonl: {RUNTIME_SECONDS['write_jsonl']:.3f}", verbose_only=False)
        _log(
            f"  printing_logging_reporting: {RUNTIME_SECONDS['printing_logging_reporting']:.3f}",
            verbose_only=False,
        )
        _log(f"  generate_example_calls: {calls}", verbose_only=False)
        _log(f"  calls_per_accepted_example: {calls/accepted:.3f}", verbose_only=False)
        _log(f"  max_slot_attempts: {max_slot_attempts}", verbose_only=False)
        _log(
            f"  high_attempt_examples(>={HIGH_ATTEMPT_SLOT_THRESHOLD}): {len(high_attempt_examples)}",
            verbose_only=False,
        )
        if high_attempt_examples:
            preview = ", ".join(
                f"{eid}:{attempt}"
                for eid, attempt in sorted(high_attempt_examples, key=lambda t: t[1], reverse=True)[:10]
            )
            _log(f"  high_attempt_ids: {preview}", verbose_only=False)
        if profile_rejection_reasons:
            _log("  rejection_reasons:", verbose_only=False)
            for reason, count in sorted(profile_rejection_reasons.items(), key=lambda kv: kv[1], reverse=True):
                _log(f"    {reason}: {count}", verbose_only=False)
        _log(
            f"  early_fallback_accepts: {int(RUNTIME_COUNTS.get('early_fallback_accepts', 0))}",
            verbose_only=False,
        )
        _log(f"  avg_time_per_accepted_example: {avg_time:.3f}", verbose_only=False)


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3 violation detection generator")
    p.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N_EXAMPLES,
        help="number of examples (must be multiple of 5)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for deterministic generation",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "core" / "task3_violation.jsonl"),
        help="output JSONL path",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="enable detailed generation logs and distribution prints",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="print lightweight runtime timing breakdown",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    VERBOSE = bool(VERBOSE or args.verbose)
    PROFILE_RUNTIME = bool(args.profile)
    generate_task3(args.n, args.out, args.seed)
