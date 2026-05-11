"""Task 4 certainty-under-spatial-ambiguity generator."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from domain.entities import Action, Direction, Environment, ScenarioState, Vehicle
from domain.render import describe_scenario, render_prompt
from domain.rules import is_overlap_possible, vehicles_overlap
from domain.scenario import apply_action
from domain.vocabulary import label_of, labels_for_env, positions_for_env



N_EXAMPLES = 100
GENERATOR_VERSION = "task4_certainty_ambiguity_v1"
MAX_RETRIES = 150
NUM_VEHICLES = 3
LETTERS = ["A", "B", "C", "D", "E"]
QUESTION = "Which of the following statements is certainly true at the end of the sequence?"
TASK_NAME = "certainty_under_spatial_ambiguity"
STATEMENT_SIGNATURE_REUSE_CAP = 6
EVENT_SIG_CAP = 20
CORRECT_TEXT_CAP = 20
GENERATED_AT_UTC = "2026-04-27T00:00:00Z"  # deterministic stamp

DIFFICULTIES = ["easy", "medium", "hard"]
MULTI_LANE_POSITIONS_TASK4 = ("left_lane", "center_lane", "right_lane")
MULTI_LANE_LABELS_TASK4 = {
    "left_lane": "the left lane",
    "center_lane": "the center lane",
    "right_lane": "the right lane",
}

ALL_LABELS_BY_ENV = {
    Environment.INTERSECTION: set(labels_for_env(Environment.INTERSECTION)),
    Environment.MULTI_LANE: set(MULTI_LANE_LABELS_TASK4.values()),
    Environment.ROUNDABOUT: set(labels_for_env(Environment.ROUNDABOUT)),
}
ALL_LABELS = set().union(*ALL_LABELS_BY_ENV.values())


@dataclass(frozen=True)
class SlotSpec:
    environment: Environment
    scenario_type: str
    certainly_true_category: str



RE_BOTH_INSIDE_INTER = re.compile(r"^Vehicles ([ABC]) and ([ABC]) are both inside the intersection\.$")
RE_BOTH_ROUND = re.compile(r"^Vehicles ([ABC]) and ([ABC]) are both in the roundabout lane\.$")
RE_SINGLE_INSIDE_INTER = re.compile(r"^Vehicle ([ABC]) is inside the intersection\.$")
RE_SINGLE_ROUND = re.compile(r"^Vehicle ([ABC]) is in the roundabout lane\.$")
RE_BOTH_INSIDE_INTER_ALT = re.compile(r"^Both Vehicle ([ABC]) and Vehicle ([ABC]) are inside the intersection\.$")
RE_BOTH_ROUND_ALT = re.compile(r"^Both Vehicle ([ABC]) and Vehicle ([ABC]) are in the roundabout lane\.$")
RE_SINGLE_INSIDE_INTER_ALT = re.compile(r"^Vehicle ([ABC]) remains inside the intersection\.$")
RE_SINGLE_ROUND_ALT = re.compile(r"^Vehicle ([ABC]) remains in the roundabout lane\.$")
RE_NOT_ENTERED_INTER = re.compile(r"^Vehicle ([ABC]) has not entered the intersection\.$")
RE_NOT_ENTERED_ROUND = re.compile(r"^Vehicle ([ABC]) has not entered the roundabout\.$")
RE_EXITED_INTER = re.compile(r"^Vehicle ([ABC]) has exited the intersection\.$")
RE_EXITED_ROUND = re.compile(r"^Vehicle ([ABC]) has exited the roundabout\.$")
RE_AT_LABEL = re.compile(r"^Vehicle ([ABC]) is at (the [a-z ]+)\.$")
RE_IN_LANE = re.compile(r"^Vehicle ([ABC]) is in (the left lane|the center lane|the right lane)\.$")

RE_AHEAD = re.compile(r"^Vehicle ([ABC]) is ahead of Vehicle ([ABC])(?: in the roundabout lane)?\.$")
RE_LEFT_OF = re.compile(r"^Vehicle ([ABC]) is to the left of Vehicle ([ABC])(?: in the roundabout lane)?\.$")
RE_PAST = re.compile(r"^Vehicle ([ABC]) has already moved past Vehicle ([ABC])\.$")
RE_WILL_EXIT_BEFORE_ENTER_INTER = re.compile(
    r"^Vehicle ([ABC]) will exit before Vehicle ([ABC]) enters the intersection\.$"
)
RE_WILL_ENTER_BEFORE_EXIT_INTER = re.compile(
    r"^Vehicle ([ABC]) will enter the intersection before Vehicle ([ABC]) exits\.$"
)
RE_WILL_EXIT_BEFORE_ENTER_ROUND = re.compile(
    r"^Vehicle ([ABC]) will exit the roundabout before Vehicle ([ABC]) enters it\.$"
)
RE_WILL_ENTER_BEFORE_EXIT_ROUND = re.compile(
    r"^Vehicle ([ABC]) will enter the roundabout before Vehicle ([ABC]) exits\.$"
)
# Past-overlap uncertainty patterns are uncertain only if the referenced pair
# actually overlapped during replay.
RE_WAS_AHEAD_INTER = re.compile(
    r"^Vehicle ([ABC]) was ahead of Vehicle ([ABC]) inside the intersection\.$"
)
RE_WAS_LEFT_INTER = re.compile(
    r"^Vehicle ([ABC]) was to the left of Vehicle ([ABC]) inside the intersection\.$"
)
RE_WAS_AHEAD_ROUND = re.compile(
    r"^Vehicle ([ABC]) was ahead of Vehicle ([ABC]) in the roundabout lane\.$"
)
RE_WAS_LEFT_ROUND = re.compile(
    r"^Vehicle ([ABC]) was to the left of Vehicle ([ABC]) in the roundabout lane\.$"
)
RE_BEHIND_ROUND = re.compile(r"^Vehicle ([ABC]) is behind Vehicle ([ABC]) in the roundabout lane\.$")
RE_WAS_BEHIND_ROUND = re.compile(r"^Vehicle ([ABC]) was behind Vehicle ([ABC]) in the roundabout lane\.$")
RE_AHEAD_ROAD = re.compile(r"^Vehicle ([ABC]) is ahead of Vehicle ([ABC]) on the road\.$")
RE_PAST_ROAD = re.compile(r"^Vehicle ([ABC]) has already moved past Vehicle ([ABC]) on the road\.$")
RE_WILL_CHANGE_BEFORE = re.compile(r"^Vehicle ([ABC]) will change lanes before Vehicle ([ABC])\.$")
RE_BEHIND_ROAD = re.compile(r"^Vehicle ([ABC]) is behind Vehicle ([ABC]) on the road\.$")
RE_DIRECTLY_BEHIND_ROAD = re.compile(r"^Vehicle ([ABC]) is directly behind Vehicle ([ABC]) on the road\.$")



def _vehicle_ids() -> list[str]:
    return ["A", "B", "C"]


def _single_vehicle_statement_id(statement: str) -> str | None:
    m = re.match(r"^Vehicle ([ABC]) ", statement)
    if not m:
        return None
    return m.group(1)


def _correct_answer_vehicle_ids(statement: str) -> tuple[str, ...]:
    return tuple(sorted(_id_set_in_statement(statement)))


def _correct_text_cap_for(n: int) -> int:
    return max(CORRECT_TEXT_CAP, math.ceil(n / 15))


def _event_sig_cap_for(n: int) -> int:
    return max(EVENT_SIG_CAP, math.ceil(n / 10))


def _statement_signature_reuse_cap_for(n: int) -> int:
    return max(STATEMENT_SIGNATURE_REUSE_CAP, math.ceil(n / 25))


def _single_correct_gap_cap_for(n: int) -> int:
    return 10


def _correct_answer_vehicle_gap_cap() -> int:
    return 10


def _build_key_schedule(n: int, rng: random.Random) -> list[str]:
    if n % 5 != 0:
        raise ValueError("N must be a multiple of 5 for balanced answer letters.")
    per = n // 5
    schedule: list[str] = []
    for letter in LETTERS:
        schedule.extend([letter] * per)
    rng.shuffle(schedule)
    return schedule


def _build_difficulty_schedule(n: int, rng: random.Random) -> list[str]:
    if n == 100:
        schedule = (["easy"] * 33) + (["medium"] * 33) + (["hard"] * 34)
        rng.shuffle(schedule)
        return schedule
    base = n // 3
    rem = n % 3
    schedule = (["easy"] * base) + (["medium"] * base) + (["hard"] * base)
    for i in range(rem):
        schedule.append(DIFFICULTIES[i])
    rng.shuffle(schedule)
    return schedule


def _label_for_position(position: str) -> str:
    if position in MULTI_LANE_LABELS_TASK4:
        return MULTI_LANE_LABELS_TASK4[position]
    return label_of(position)


def _positions_for_env_task4(env: Environment) -> tuple[str, ...]:
    if env == Environment.MULTI_LANE:
        return MULTI_LANE_POSITIONS_TASK4
    return positions_for_env(env)


def _labels_for_env_task4(env: Environment) -> tuple[str, ...]:
    if env == Environment.MULTI_LANE:
        return tuple(MULTI_LANE_LABELS_TASK4[p] for p in MULTI_LANE_POSITIONS_TASK4)
    return labels_for_env(env)


def _scaled_counts(n: int, base: list[tuple[SlotSpec, int]]) -> list[tuple[SlotSpec, int]]:
    total = sum(c for _, c in base)
    raw = [(spec, (c * n) / total) for spec, c in base]
    floored = [(spec, int(v)) for spec, v in raw]
    used = sum(v for _, v in floored)
    rem = n - used
    order = sorted(
        ((spec, frac - int(frac)) for spec, frac in raw),
        key=lambda kv: kv[1],
        reverse=True,
    )
    counts = dict(floored)
    for i in range(rem):
        counts[order[i % len(order)][0]] += 1
    return [(spec, counts[spec]) for spec, _ in base]


def _build_slot_schedule(n: int, rng: random.Random) -> list[SlotSpec]:
    # Target mix (n=100):
    # intersection 60, roundabout 20, multi-lane 20
    base: list[tuple[SlotSpec, int]] = [
        (
            SlotSpec(
                environment=Environment.INTERSECTION,
                scenario_type="two_overlap_one_outside",
                certainly_true_category="containment_overlap",
            ),
            10,
        ),
        (
            SlotSpec(
                environment=Environment.INTERSECTION,
                scenario_type="two_overlap_third_exited",
                certainly_true_category="containment_overlap",
            ),
            10,
        ),
        (
            SlotSpec(
                environment=Environment.INTERSECTION,
                scenario_type="two_overlap_one_outside",
                certainly_true_category="containment_non_entry",
            ),
            10,
        ),
        (
            SlotSpec(
                environment=Environment.INTERSECTION,
                scenario_type="one_inside_one_exited_one_approach",
                certainly_true_category="exit_reached",
            ),
            20,
        ),
        (
            SlotSpec(
                environment=Environment.ROUNDABOUT,
                scenario_type="roundabout_overlap",
                certainly_true_category="roundabout_overlap",
            ),
            20,
        ),
        (
            SlotSpec(
                environment=Environment.ROUNDABOUT,
                scenario_type="roundabout_non_entry",
                certainly_true_category="containment_non_entry",
            ),
            10,
        ),
        (
            SlotSpec(
                environment=Environment.MULTI_LANE,
                scenario_type="multi_lane_positioning",
                certainly_true_category="lane_position",
            ),
            20,
        ),
    ]
    counts = base if n == 100 else _scaled_counts(n, base)
    schedule: list[SlotSpec] = []
    for spec, count in counts:
        schedule.extend([spec] * count)
    if len(schedule) != n:
        raise RuntimeError(f"internal slot schedule length mismatch: {len(schedule)} vs {n}")
    rng.shuffle(schedule)
    return schedule


def _state_from_roles(
    env: Environment,
    role_pos: dict[str, str],
    role_inside: dict[str, bool],
    role_stopped: dict[str, bool],
    role_direction: dict[str, Direction],
) -> ScenarioState:
    vehicles = []
    for vid in _vehicle_ids():
        vehicles.append(
            Vehicle(
                id=vid,
                position=role_pos[vid],
                direction=role_direction[vid],
                inside_intersection=role_inside[vid],
                stopped=role_stopped[vid],
            )
        )
    return ScenarioState(vehicles=vehicles, environment=env)


def _contains_abab(ids: list[str]) -> bool:
    if len(ids) < 4:
        return False
    for i in range(len(ids) - 3):
        a, b, c, d = ids[i : i + 4]
        if a == c and b == d and a != b:
            return True
    return False


def _has_action_streak_len3(actions: list[Action]) -> bool:
    if len(actions) < 3:
        return False
    for i in range(len(actions) - 2):
        if actions[i] == actions[i + 1] == actions[i + 2]:
            return True
    return False


def _find_overlaps(state: ScenarioState) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for v1, v2 in combinations(state.vehicles, 2):
        if vehicles_overlap(v1, v2):
            pairs.append((v1.id, v2.id))
    return pairs


def _pair_key(a: str, b: str) -> tuple[str, str]:
    if a <= b:
        return (a, b)
    return (b, a)


def _vehicle_entered(v: Vehicle, env: Environment) -> bool:
    if env == Environment.INTERSECTION:
        return v.inside_intersection or v.position == "inside_intersection" or v.position.endswith("_exit")
    if env == Environment.ROUNDABOUT:
        return v.inside_intersection or v.position == "roundabout_lane" or v.position.endswith("_exit")
    return False


def _vehicle_exited(v: Vehicle, env: Environment) -> bool:
    if env == Environment.INTERSECTION:
        return (not v.inside_intersection) and v.position.endswith("_exit")
    if env == Environment.ROUNDABOUT:
        return (not v.inside_intersection) and v.position.endswith("_exit")
    return False


def _vehicle_at_approach(v: Vehicle) -> bool:
    return (not v.inside_intersection) and v.position.endswith("_approach")


def classify_statement(
    statement: str,
    env: Environment,
    by_id: dict[str, Vehicle],
    overlap_pairs: set[tuple[str, str]] | None = None,
) -> str:
    m = RE_BOTH_INSIDE_INTER.match(statement)
    if m:
        if env != Environment.INTERSECTION:
            return "invalid"
        a, b = m.groups()
        return "true" if by_id[a].inside_intersection and by_id[b].inside_intersection else "false"

    m = RE_BOTH_INSIDE_INTER_ALT.match(statement)
    if m:
        if env != Environment.INTERSECTION:
            return "invalid"
        a, b = m.groups()
        return "true" if by_id[a].inside_intersection and by_id[b].inside_intersection else "false"

    m = RE_BOTH_ROUND.match(statement)
    if m:
        if env != Environment.ROUNDABOUT:
            return "invalid"
        a, b = m.groups()
        cond = (
            by_id[a].inside_intersection
            and by_id[b].inside_intersection
            and by_id[a].position == "roundabout_lane"
            and by_id[b].position == "roundabout_lane"
        )
        return "true" if cond else "false"

    m = RE_BOTH_ROUND_ALT.match(statement)
    if m:
        if env != Environment.ROUNDABOUT:
            return "invalid"
        a, b = m.groups()
        cond = (
            by_id[a].inside_intersection
            and by_id[b].inside_intersection
            and by_id[a].position == "roundabout_lane"
            and by_id[b].position == "roundabout_lane"
        )
        return "true" if cond else "false"

    m = RE_SINGLE_INSIDE_INTER.match(statement)
    if m:
        if env != Environment.INTERSECTION:
            return "invalid"
        a = m.group(1)
        cond = by_id[a].inside_intersection and by_id[a].position == "inside_intersection"
        return "true" if cond else "false"

    m = RE_SINGLE_INSIDE_INTER_ALT.match(statement)
    if m:
        if env != Environment.INTERSECTION:
            return "invalid"
        a = m.group(1)
        cond = by_id[a].inside_intersection and by_id[a].position == "inside_intersection"
        return "true" if cond else "false"

    m = RE_SINGLE_ROUND.match(statement)
    if m:
        if env != Environment.ROUNDABOUT:
            return "invalid"
        a = m.group(1)
        cond = by_id[a].inside_intersection and by_id[a].position == "roundabout_lane"
        return "true" if cond else "false"

    m = RE_SINGLE_ROUND_ALT.match(statement)
    if m:
        if env != Environment.ROUNDABOUT:
            return "invalid"
        a = m.group(1)
        cond = by_id[a].inside_intersection and by_id[a].position == "roundabout_lane"
        return "true" if cond else "false"

    m = RE_NOT_ENTERED_INTER.match(statement)
    if m:
        if env != Environment.INTERSECTION:
            return "invalid"
        a = m.group(1)
        return "true" if (not _vehicle_entered(by_id[a], env)) else "false"

    m = RE_NOT_ENTERED_ROUND.match(statement)
    if m:
        if env != Environment.ROUNDABOUT:
            return "invalid"
        a = m.group(1)
        return "true" if (not _vehicle_entered(by_id[a], env)) else "false"

    m = RE_EXITED_INTER.match(statement)
    if m:
        if env != Environment.INTERSECTION:
            return "invalid"
        a = m.group(1)
        return "true" if _vehicle_exited(by_id[a], env) else "false"

    m = RE_EXITED_ROUND.match(statement)
    if m:
        if env != Environment.ROUNDABOUT:
            return "invalid"
        a = m.group(1)
        return "true" if _vehicle_exited(by_id[a], env) else "false"

    m = RE_AT_LABEL.match(statement)
    if m:
        a, label = m.groups()
        label_to_position = {_label_for_position(pos): pos for pos in _positions_for_env_task4(env)}
        pos = label_to_position.get(label)
        if pos is None:
            return "invalid"
        return "true" if by_id[a].position == pos else "false"

    m = RE_IN_LANE.match(statement)
    if m:
        if env != Environment.MULTI_LANE:
            return "invalid"
        a, label = m.groups()
        label_to_position = {MULTI_LANE_LABELS_TASK4[pos]: pos for pos in MULTI_LANE_POSITIONS_TASK4}
        pos = label_to_position.get(label)
        if pos is None:
            return "invalid"
        return "true" if by_id[a].position == pos else "false"

    m = RE_AHEAD.match(statement)
    if m:
        a, b = m.groups()
        if a == b:
            return "false"
        return "uncertain" if (by_id[a].inside_intersection and by_id[b].inside_intersection) else "false"

    m = RE_LEFT_OF.match(statement)
    if m:
        a, b = m.groups()
        if a == b:
            return "false"
        return "uncertain" if (by_id[a].inside_intersection and by_id[b].inside_intersection) else "false"

    m = RE_PAST.match(statement)
    if m:
        a, b = m.groups()
        if a == b:
            return "false"
        return "uncertain" if (by_id[a].inside_intersection and by_id[b].inside_intersection) else "false"

    m = RE_WILL_EXIT_BEFORE_ENTER_INTER.match(statement)
    if m:
        if env != Environment.INTERSECTION:
            return "invalid"
        a, b = m.groups()
        a_inside = by_id[a].inside_intersection and by_id[a].position == "inside_intersection"
        b_approach = _vehicle_at_approach(by_id[b])
        return "uncertain" if (a_inside and b_approach) else "false"

    m = RE_WILL_ENTER_BEFORE_EXIT_INTER.match(statement)
    if m:
        if env != Environment.INTERSECTION:
            return "invalid"
        a, b = m.groups()
        a_approach = _vehicle_at_approach(by_id[a])
        b_inside = by_id[b].inside_intersection and by_id[b].position == "inside_intersection"
        return "uncertain" if (a_approach and b_inside) else "false"

    m = RE_WILL_EXIT_BEFORE_ENTER_ROUND.match(statement)
    if m:
        if env != Environment.ROUNDABOUT:
            return "invalid"
        a, b = m.groups()
        a_inside = by_id[a].inside_intersection and by_id[a].position == "roundabout_lane"
        b_approach = _vehicle_at_approach(by_id[b])
        return "uncertain" if (a_inside and b_approach) else "false"

    m = RE_WILL_ENTER_BEFORE_EXIT_ROUND.match(statement)
    if m:
        if env != Environment.ROUNDABOUT:
            return "invalid"
        a, b = m.groups()
        a_approach = _vehicle_at_approach(by_id[a])
        b_inside = by_id[b].inside_intersection and by_id[b].position == "roundabout_lane"
        return "uncertain" if (a_approach and b_inside) else "false"

    # Past-overlap patterns: internal relative state when both were inside is unknowable
    # from the final state alone → always "uncertain".
    m = RE_WAS_AHEAD_INTER.match(statement)
    if m:
        if env != Environment.INTERSECTION:
            return "invalid"
        a, b = m.groups()
        if a == b:
            return "false"
        return "uncertain" if _pair_key(a, b) in (overlap_pairs or set()) else "false"

    m = RE_WAS_LEFT_INTER.match(statement)
    if m:
        if env != Environment.INTERSECTION:
            return "invalid"
        a, b = m.groups()
        if a == b:
            return "false"
        return "uncertain" if _pair_key(a, b) in (overlap_pairs or set()) else "false"

    m = RE_WAS_AHEAD_ROUND.match(statement)
    if m:
        if env != Environment.ROUNDABOUT:
            return "invalid"
        a, b = m.groups()
        if a == b:
            return "false"
        return "uncertain" if _pair_key(a, b) in (overlap_pairs or set()) else "false"

    m = RE_WAS_LEFT_ROUND.match(statement)
    if m:
        if env != Environment.ROUNDABOUT:
            return "invalid"
        a, b = m.groups()
        if a == b:
            return "false"
        return "uncertain" if _pair_key(a, b) in (overlap_pairs or set()) else "false"

    m = RE_BEHIND_ROUND.match(statement)
    if m:
        if env != Environment.ROUNDABOUT:
            return "invalid"
        a, b = m.groups()
        if a == b:
            return "false"
        return "uncertain" if (by_id[a].inside_intersection and by_id[b].inside_intersection) else "false"

    m = RE_WAS_BEHIND_ROUND.match(statement)
    if m:
        if env != Environment.ROUNDABOUT:
            return "invalid"
        a, b = m.groups()
        if a == b:
            return "false"
        return "uncertain" if _pair_key(a, b) in (overlap_pairs or set()) else "false"

    m = RE_AHEAD_ROAD.match(statement)
    if m:
        if env != Environment.MULTI_LANE:
            return "invalid"
        a, b = m.groups()
        if a == b:
            return "false"
        return "uncertain"

    m = RE_BEHIND_ROAD.match(statement)
    if m:
        if env != Environment.MULTI_LANE:
            return "invalid"
        a, b = m.groups()
        if a == b:
            return "false"
        return "uncertain"

    m = RE_DIRECTLY_BEHIND_ROAD.match(statement)
    if m:
        if env != Environment.MULTI_LANE:
            return "invalid"
        a, b = m.groups()
        if a == b:
            return "false"
        return "uncertain"

    m = RE_PAST_ROAD.match(statement)
    if m:
        if env != Environment.MULTI_LANE:
            return "invalid"
        a, b = m.groups()
        if a == b:
            return "false"
        return "uncertain"

    m = RE_WILL_CHANGE_BEFORE.match(statement)
    if m:
        if env != Environment.MULTI_LANE:
            return "invalid"
        a, b = m.groups()
        if a == b:
            return "false"
        return "uncertain"

    return "invalid"


def _extract_position_labels(statement: str) -> set[str]:
    return {label for label in ALL_LABELS if label in statement}


def _assert_env_label_consistency(statements: list[str], env: Environment) -> bool:
    allowed = ALL_LABELS_BY_ENV[env]
    for s in statements:
        labels = _extract_position_labels(s)
        if any(label not in allowed for label in labels):
            return False
    return True


def _normalize_statement_structure(statement: str) -> str:
    out = statement
    out = re.sub(r"Vehicles [ABC] and [ABC]", "Vehicles V and V", out)
    out = re.sub(r"Vehicle [ABC]", "Vehicle V", out)
    for label in sorted(ALL_LABELS, key=len, reverse=True):
        out = out.replace(label, "<POS>")
    return out


def _serialize_state(state: ScenarioState) -> dict:
    return {
        "vehicles": [
            {
                "id": v.id,
                "position": v.position,
                "direction": v.direction.value,
                "inside_intersection": bool(v.inside_intersection),
                "stopped": bool(v.stopped),
            }
            for v in state.vehicles
        ],
        "environment": state.environment.value,
    }


def _final_state_snapshot(state: ScenarioState) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for v in state.vehicles:
        out[v.id] = {
            "position": v.position,
            "inside_intersection": bool(v.inside_intersection),
            "stopped": bool(v.stopped),
        }
    return out


def _id_set_in_statement(statement: str) -> set[str]:
    ids = set(re.findall(r"Vehicle ([A-Z])", statement))
    for a, b in re.findall(r"Vehicles ([A-Z]) and ([A-Z])", statement):
        ids.add(a)
        ids.add(b)
    return ids


def _make_initial_state_and_plan(
    slot: SlotSpec,
    difficulty: str,
    rng: random.Random,
) -> tuple[ScenarioState, list[tuple[str, Action]], dict[str, object]]:
    ids = _vehicle_ids()
    rng.shuffle(ids)
    a, b, c = ids

    dirs = rng.sample(list(Direction), NUM_VEHICLES)
    dir_map = {"A": dirs[0], "B": dirs[1], "C": dirs[2]}

    env = slot.environment
    if slot.scenario_type == "two_overlap_one_outside":
        overlap_pair = [a, b]
        outside = c
        pos = {vid: f"{dir_map[vid].value}_approach" for vid in _vehicle_ids()}
        inside = {vid: False for vid in _vehicle_ids()}
        stopped = {vid: False for vid in _vehicle_ids()}
        state = _state_from_roles(env, pos, inside, stopped, dir_map)
        # Randomise entry order (a first or b first) to avoid repetitive event sequences
        first_in, second_in = (a, b) if rng.random() < 0.5 else (b, a)
        if env == Environment.INTERSECTION:
            plan: list[tuple[str, Action]] = [(first_in, Action.MOVE_FORWARD), (second_in, Action.MOVE_FORWARD)]
        else:
            plan = [(first_in, Action.ENTER_ROUNDABOUT), (second_in, Action.ENTER_ROUNDABOUT)]
        # For easy: sometimes include the outside vehicle's STOP to add sequence variety
        if difficulty == "easy" and rng.random() < 0.5:
            plan.append((outside, Action.STOP))
        if difficulty in ("medium", "hard"):
            plan.append((outside, Action.STOP))
        if difficulty == "hard":
            plan.append((rng.choice(overlap_pair), Action.STOP))
        roles = {
            "overlap_pair": overlap_pair,
            "outside": outside,
            "inside_actor": overlap_pair[0],
            "exiter": None,
        }
        return state, plan, roles

    if slot.scenario_type == "two_overlap_third_exited":
        if env != Environment.INTERSECTION:
            raise ValueError("two_overlap_third_exited is intersection-only.")
        overlap_pair = [a, b]
        exiter = c
        pos = {vid: f"{dir_map[vid].value}_approach" for vid in _vehicle_ids()}
        inside = {vid: False for vid in _vehicle_ids()}
        stopped = {vid: False for vid in _vehicle_ids()}

        # Make one overlap vehicle and the exiter already inside to allow 2-event easy traces.
        pos[a] = "inside_intersection"
        inside[a] = True
        pos[exiter] = "inside_intersection"
        inside[exiter] = True

        state = _state_from_roles(env, pos, inside, stopped, dir_map)
        turn = rng.choice([Action.TURN_LEFT, Action.TURN_RIGHT])
        plan = [(exiter, turn), (b, Action.MOVE_FORWARD)]
        if difficulty in ("medium", "hard"):
            plan.append((a, Action.STOP))
        if difficulty == "hard":
            plan.append((b, Action.STOP))
        roles = {
            "overlap_pair": overlap_pair,
            "outside": None,
            "inside_actor": overlap_pair[0],
            "exiter": exiter,
        }
        return state, plan, roles

    if slot.scenario_type == "one_inside_one_exited_one_approach":
        exiter = a
        insider = b
        outside = c

        pos = {vid: f"{dir_map[vid].value}_approach" for vid in _vehicle_ids()}
        inside = {vid: False for vid in _vehicle_ids()}
        stopped = {vid: False for vid in _vehicle_ids()}

        if difficulty == "easy":
            pos[exiter] = "inside_intersection" if env == Environment.INTERSECTION else "roundabout_lane"
            inside[exiter] = True
            pos[insider] = "inside_intersection" if env == Environment.INTERSECTION else "roundabout_lane"
            inside[insider] = True
            state = _state_from_roles(env, pos, inside, stopped, dir_map)
            exiter_first = rng.random() < 0.5
            if env == Environment.INTERSECTION:
                turn = rng.choice([Action.TURN_LEFT, Action.TURN_RIGHT])
                if exiter_first:
                    plan = [(exiter, turn), (insider, Action.STOP)]
                else:
                    plan = [(insider, Action.STOP), (exiter, turn)]
            else:
                if exiter_first:
                    plan = [(exiter, Action.EXIT_ROUNDABOUT), (insider, Action.STOP)]
                else:
                    plan = [(insider, Action.STOP), (exiter, Action.EXIT_ROUNDABOUT)]
        else:
            state = _state_from_roles(env, pos, inside, stopped, dir_map)
            # Randomise who enters first (exiter or insider) to break first-actor bias:
            # the correct answer always references the exiter, so if exiter is always
            # the first actor the model can exploit this pattern.
            exiter_first = rng.random() < 0.5
            if env == Environment.INTERSECTION:
                turn = rng.choice([Action.TURN_LEFT, Action.TURN_RIGHT])
                if exiter_first:
                    plan = [(exiter, Action.MOVE_FORWARD), (insider, Action.MOVE_FORWARD), (exiter, turn)]
                else:
                    plan = [(insider, Action.MOVE_FORWARD), (exiter, Action.MOVE_FORWARD), (exiter, turn)]
            else:
                if exiter_first:
                    plan = [(exiter, Action.ENTER_ROUNDABOUT), (insider, Action.ENTER_ROUNDABOUT), (exiter, Action.EXIT_ROUNDABOUT)]
                else:
                    plan = [(insider, Action.ENTER_ROUNDABOUT), (exiter, Action.ENTER_ROUNDABOUT), (exiter, Action.EXIT_ROUNDABOUT)]
            if difficulty == "hard":
                plan.append((outside, Action.STOP))

        roles = {
            "overlap_pair": [exiter, insider],
            "outside": outside,
            "inside_actor": insider,
            "exiter": exiter,
        }
        return state, plan, roles

    if slot.scenario_type == "roundabout_overlap":
        if env != Environment.ROUNDABOUT:
            raise ValueError("roundabout_overlap is roundabout-only.")
        overlap_pair = [a, b]
        outside = c
        pos = {vid: f"{dir_map[vid].value}_approach" for vid in _vehicle_ids()}
        inside = {vid: False for vid in _vehicle_ids()}
        stopped = {vid: False for vid in _vehicle_ids()}
        state = _state_from_roles(env, pos, inside, stopped, dir_map)
        plan = [(a, Action.ENTER_ROUNDABOUT), (b, Action.ENTER_ROUNDABOUT)]
        if difficulty in ("medium", "hard"):
            plan.append((outside, Action.STOP))
        if difficulty == "hard":
            plan.append((rng.choice(overlap_pair), Action.STOP))
        roles = {
            "overlap_pair": overlap_pair,
            "outside": outside,
            "inside_actor": overlap_pair[0],
            "exiter": None,
        }
        return state, plan, roles

    if slot.scenario_type == "roundabout_non_entry":
        if env != Environment.ROUNDABOUT:
            raise ValueError("roundabout_non_entry is roundabout-only.")
        overlap_pair = [a, b]
        outside = c
        pos = {vid: f"{dir_map[vid].value}_approach" for vid in _vehicle_ids()}
        inside = {vid: False for vid in _vehicle_ids()}
        stopped = {vid: False for vid in _vehicle_ids()}
        state = _state_from_roles(env, pos, inside, stopped, dir_map)
        first_in, second_in = (a, b) if rng.random() < 0.5 else (b, a)
        plan = [(first_in, Action.ENTER_ROUNDABOUT), (second_in, Action.ENTER_ROUNDABOUT)]
        if difficulty == "easy":
            if rng.random() < 0.5:
                plan.append((outside, Action.STOP))
        elif difficulty == "medium":
            plan.append((outside, Action.STOP))
        else:
            plan.append((outside, Action.STOP))
            plan.append((rng.choice(overlap_pair), Action.STOP))
        roles = {
            "overlap_pair": overlap_pair,
            "outside": outside,
            "inside_actor": overlap_pair[0],
            "exiter": None,
        }
        return state, plan, roles

    if slot.scenario_type == "multi_lane_positioning":
        focus_a = a
        focus_b = b
        static = c
        dir_map = {vid: Direction.NORTH for vid in _vehicle_ids()}
        inside = {vid: False for vid in _vehicle_ids()}
        stopped = {vid: False for vid in _vehicle_ids()}
        layout = {
            focus_a: "left_lane",
            focus_b: "center_lane",
            static: "right_lane",
        }
        state = _state_from_roles(env, layout, inside, stopped, dir_map)

        templates: dict[str, list[list[tuple[str, Action]]]] = {
            "easy": [
                [(focus_a, Action.CHANGE_RIGHT), (focus_b, Action.CHANGE_RIGHT)],
                [(focus_b, Action.CHANGE_LEFT), (static, Action.CHANGE_LEFT)],
                [(focus_a, Action.CHANGE_RIGHT), (static, Action.CHANGE_LEFT)],
            ],
            "medium": [
                [(focus_a, Action.CHANGE_RIGHT), (focus_b, Action.CHANGE_RIGHT), (static, Action.STOP)],
                [(focus_b, Action.CHANGE_LEFT), (static, Action.CHANGE_LEFT), (focus_a, Action.STOP)],
                [(focus_a, Action.STOP), (focus_b, Action.CHANGE_LEFT), (static, Action.CHANGE_LEFT)],
            ],
            "hard": [
                [(focus_a, Action.CHANGE_RIGHT), (focus_b, Action.CHANGE_RIGHT), (static, Action.STOP), (focus_a, Action.STOP)],
                [(focus_b, Action.CHANGE_LEFT), (static, Action.CHANGE_LEFT), (focus_a, Action.STOP), (focus_b, Action.STOP)],
                [(focus_a, Action.STOP), (focus_b, Action.CHANGE_LEFT), (static, Action.CHANGE_LEFT), (focus_a, Action.CHANGE_RIGHT)],
            ],
        }
        plan = copy.deepcopy(rng.choice(templates[difficulty]))
        roles = {
            "overlap_pair": [],
            "outside": static,
            "inside_actor": None,
            "exiter": None,
            "focus_pair": [focus_a, focus_b],
        }
        return state, plan, roles

    raise ValueError(f"Unsupported scenario_type: {slot.scenario_type!r}")


def _replay_plan(
    init_state: ScenarioState,
    plan: list[tuple[str, Action]],
) -> tuple[
    ScenarioState,
    list[str],
    bool,
    list[str],
    set[tuple[str, str]],
    list[list[tuple[str, str]]],
] | None:
    sim = copy.deepcopy(init_state)
    events: list[str] = []
    overlap_any = False
    overlap_ids: set[str] = set()
    overlap_pairs: set[tuple[str, str]] = set()
    timeline: list[list[tuple[str, str]]] = []

    if _find_overlaps(sim):
        overlap_any = True
        for a, b in _find_overlaps(sim):
            overlap_ids.update([a, b])
            overlap_pairs.add(_pair_key(a, b))

    for vid, action in plan:
        ev = apply_action(sim, vid, action)
        if not ev:
            return None
        events.append(ev)
        pairs = _find_overlaps(sim)
        timeline.append(pairs)
        if pairs:
            overlap_any = True
            for a, b in pairs:
                overlap_ids.update([a, b])
                overlap_pairs.add(_pair_key(a, b))
    return sim, events, overlap_any, sorted(overlap_ids), overlap_pairs, timeline


def _certainly_true_statement(
    slot: SlotSpec,
    roles: dict[str, object],
    final_by_id: dict[str, Vehicle],
    correct_text_usage: Counter[str],
    correct_vehicle_usage: Counter[str],
    env_correct_vehicle_usage: dict[str, Counter[str]],
    rng: random.Random,
) -> str:
    env = slot.environment
    category = slot.certainly_true_category
    overlap_pair = roles["overlap_pair"]
    assert isinstance(overlap_pair, list)

    def choose(pool: list[str]) -> str:
        min_vehicle = min(correct_vehicle_usage.get(vid, 0) for vid in _vehicle_ids())
        scored: list[tuple[tuple[float, ...], str]] = []
        for text in pool:
            ids = _correct_answer_vehicle_ids(text)
            vehicle_pressure = max((correct_vehicle_usage[vid] - min_vehicle) for vid in ids)
            env_pressure = sum(env_correct_vehicle_usage[env.value][vid] for vid in ids)
            pair_penalty = 0 if len(ids) > 1 else 1
            score = (
                vehicle_pressure,
                correct_text_usage[text],
                env_pressure,
                pair_penalty,
                rng.random(),
            )
            scored.append((score, text))
        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    if category == "containment_overlap":
        a, b = overlap_pair
        pool = [
            f"Vehicles {a} and {b} are both inside the intersection.",
            f"Vehicles {b} and {a} are both inside the intersection.",
            f"Both Vehicle {a} and Vehicle {b} are inside the intersection.",
            f"Both Vehicle {b} and Vehicle {a} are inside the intersection.",
            f"Vehicle {a} is inside the intersection.",
            f"Vehicle {a} remains inside the intersection.",
            f"Vehicle {b} is inside the intersection.",
            f"Vehicle {b} remains inside the intersection.",
        ]
        return choose(pool)

    if category == "containment_non_entry":
        outside = roles["outside"]
        assert isinstance(outside, str)
        if env == Environment.INTERSECTION:
            pool = [
                f"Vehicle {outside} has not entered the intersection.",
                f"Vehicle {outside} is at {_label_for_position(final_by_id[outside].position)}.",
            ]
        else:
            pool = [
                f"Vehicle {outside} has not entered the roundabout.",
                f"Vehicle {outside} is at {_label_for_position(final_by_id[outside].position)}.",
            ]
        return choose(pool)

    if category == "roundabout_overlap":
        a, b = overlap_pair
        pool = [
            f"Vehicles {a} and {b} are both in the roundabout lane.",
            f"Vehicles {b} and {a} are both in the roundabout lane.",
            f"Both Vehicle {a} and Vehicle {b} are in the roundabout lane.",
            f"Both Vehicle {b} and Vehicle {a} are in the roundabout lane.",
            f"Vehicle {a} is in the roundabout lane.",
            f"Vehicle {a} remains in the roundabout lane.",
            f"Vehicle {b} is in the roundabout lane.",
            f"Vehicle {b} remains in the roundabout lane.",
        ]
        return choose(pool)

    if category == "exit_reached":
        exiter = roles["exiter"]
        assert isinstance(exiter, str)
        if env == Environment.INTERSECTION:
            pool = [
                f"Vehicle {exiter} has exited the intersection.",
                f"Vehicle {exiter} is at {_label_for_position(final_by_id[exiter].position)}.",
            ]
        else:
            pool = [
                f"Vehicle {exiter} has exited the roundabout.",
                f"Vehicle {exiter} is at {_label_for_position(final_by_id[exiter].position)}.",
            ]
        return choose(pool)

    if category == "lane_position":
        pool = [
            f"Vehicle {vid} is in {_label_for_position(final_by_id[vid].position)}."
            for vid in _vehicle_ids()
        ]
        return choose(pool)

    raise ValueError(f"Unsupported certainly_true category: {category!r}")


def _near_true_statements(
    slot: SlotSpec,
    roles: dict[str, object],
    rng: random.Random,
) -> list[str]:
    env = slot.environment
    category = slot.certainly_true_category

    if env == Environment.MULTI_LANE:
        focus_pair = roles["focus_pair"]
        assert isinstance(focus_pair, list)
        x, y = focus_pair[0], focus_pair[1]
        spatial_pool = [
            f"Vehicle {x} is ahead of Vehicle {y} on the road.",
            f"Vehicle {x} is behind Vehicle {y} on the road.",
            f"Vehicle {x} is directly behind Vehicle {y} on the road.",
            f"Vehicle {y} is ahead of Vehicle {x} on the road.",
            f"Vehicle {y} is behind Vehicle {x} on the road.",
            f"Vehicle {y} is directly behind Vehicle {x} on the road.",
            f"Vehicle {x} has already moved past Vehicle {y} on the road.",
            f"Vehicle {y} has already moved past Vehicle {x} on the road.",
        ]
        secondary_pool = [
            f"Vehicle {x} will change lanes before Vehicle {y}.",
            f"Vehicle {y} will change lanes before Vehicle {x}.",
        ]
        rng.shuffle(spatial_pool)
        rng.shuffle(secondary_pool)
        return [spatial_pool[0], secondary_pool[0] if rng.random() < 0.6 else spatial_pool[1]]

    overlap_pair = roles["overlap_pair"]
    assert isinstance(overlap_pair, list)
    x, y = overlap_pair[0], overlap_pair[1]

    if category in {"containment_overlap", "roundabout_overlap"}:
        if env == Environment.INTERSECTION:
            pool = [
                f"Vehicle {x} is ahead of Vehicle {y}.",
                f"Vehicle {y} is ahead of Vehicle {x}.",
                f"Vehicle {x} is to the left of Vehicle {y}.",
                f"Vehicle {y} is to the left of Vehicle {x}.",
                f"Vehicle {x} has already moved past Vehicle {y}.",
                f"Vehicle {y} has already moved past Vehicle {x}.",
            ]
        else:
            pool = [
                f"Vehicle {x} is ahead of Vehicle {y} in the roundabout lane.",
                f"Vehicle {x} is behind Vehicle {y} in the roundabout lane.",
                f"Vehicle {y} is ahead of Vehicle {x} in the roundabout lane.",
                f"Vehicle {y} is behind Vehicle {x} in the roundabout lane.",
                f"Vehicle {x} is to the left of Vehicle {y} in the roundabout lane.",
                f"Vehicle {y} is to the left of Vehicle {x} in the roundabout lane.",
                f"Vehicle {x} has already moved past Vehicle {y}.",
                f"Vehicle {y} has already moved past Vehicle {x}.",
            ]
        rng.shuffle(pool)
        return pool[:2]

    inside_actor = roles["inside_actor"]
    outside = roles["outside"]
    exiter = roles.get("exiter")
    assert isinstance(inside_actor, str)
    assert isinstance(outside, str)

    # For containment_non_entry: overlap_pair (x,y) are both still inside.
    # Mix one spatial-overlap statement (about x,y both inside) with one future-tense statement.
    if category == "containment_non_entry":
        if env == Environment.INTERSECTION:
            pool_spatial = [
                f"Vehicle {x} is ahead of Vehicle {y}.",
                f"Vehicle {y} is ahead of Vehicle {x}.",
                f"Vehicle {x} is to the left of Vehicle {y}.",
                f"Vehicle {y} is to the left of Vehicle {x}.",
                f"Vehicle {x} has already moved past Vehicle {y}.",
                f"Vehicle {y} has already moved past Vehicle {x}.",
            ]
            pool_future = [
                f"Vehicle {inside_actor} will exit before Vehicle {outside} enters the intersection.",
                f"Vehicle {outside} will enter the intersection before Vehicle {inside_actor} exits.",
            ]
        else:
            pool_spatial = [
                f"Vehicle {x} is ahead of Vehicle {y} in the roundabout lane.",
                f"Vehicle {x} is behind Vehicle {y} in the roundabout lane.",
                f"Vehicle {y} is ahead of Vehicle {x} in the roundabout lane.",
                f"Vehicle {y} is behind Vehicle {x} in the roundabout lane.",
                f"Vehicle {x} is to the left of Vehicle {y} in the roundabout lane.",
                f"Vehicle {y} is to the left of Vehicle {x} in the roundabout lane.",
                f"Vehicle {x} has already moved past Vehicle {y}.",
                f"Vehicle {y} has already moved past Vehicle {x}.",
            ]
            pool_future = [
                f"Vehicle {inside_actor} will exit the roundabout before Vehicle {outside} enters it.",
                f"Vehicle {outside} will enter the roundabout before Vehicle {inside_actor} exits.",
            ]
        rng.shuffle(pool_spatial)
        rng.shuffle(pool_future)
        return [pool_spatial[0], pool_future[0]]

    # For exit_reached: exiter has left, insider is still inside, outside at approach.
    # Mix one past-overlap uncertainty statement with one future-tense statement.
    assert isinstance(exiter, str)
    if env == Environment.INTERSECTION:
        pool_past = [
            f"Vehicle {exiter} was ahead of Vehicle {inside_actor} inside the intersection.",
            f"Vehicle {inside_actor} was ahead of Vehicle {exiter} inside the intersection.",
            f"Vehicle {exiter} was to the left of Vehicle {inside_actor} inside the intersection.",
            f"Vehicle {inside_actor} was to the left of Vehicle {exiter} inside the intersection.",
        ]
        pool_future = [
            f"Vehicle {inside_actor} will exit before Vehicle {outside} enters the intersection.",
            f"Vehicle {outside} will enter the intersection before Vehicle {inside_actor} exits.",
        ]
    else:
        pool_past = [
            f"Vehicle {exiter} was ahead of Vehicle {inside_actor} in the roundabout lane.",
            f"Vehicle {exiter} was behind Vehicle {inside_actor} in the roundabout lane.",
            f"Vehicle {inside_actor} was ahead of Vehicle {exiter} in the roundabout lane.",
            f"Vehicle {inside_actor} was behind Vehicle {exiter} in the roundabout lane.",
            f"Vehicle {exiter} was to the left of Vehicle {inside_actor} in the roundabout lane.",
            f"Vehicle {inside_actor} was to the left of Vehicle {exiter} in the roundabout lane.",
        ]
        pool_future = [
            f"Vehicle {inside_actor} will exit the roundabout before Vehicle {outside} enters it.",
            f"Vehicle {outside} will enter the roundabout before Vehicle {inside_actor} exits.",
        ]
    rng.shuffle(pool_past)
    rng.shuffle(pool_future)
    return [pool_past[0], pool_future[0]]


def _highly_false_candidates(env: Environment, by_id: dict[str, Vehicle]) -> list[str]:
    candidates: list[str] = []
    for vid, v in by_id.items():
        if env == Environment.INTERSECTION:
            if not (v.inside_intersection and v.position == "inside_intersection"):
                candidates.append(f"Vehicle {vid} is inside the intersection.")
            if not _vehicle_exited(v, env):
                candidates.append(f"Vehicle {vid} has exited the intersection.")
            if _vehicle_entered(v, env):
                candidates.append(f"Vehicle {vid} has not entered the intersection.")
        elif env == Environment.ROUNDABOUT:
            if not (v.inside_intersection and v.position == "roundabout_lane"):
                candidates.append(f"Vehicle {vid} is in the roundabout lane.")
            if not _vehicle_exited(v, env):
                candidates.append(f"Vehicle {vid} has exited the roundabout.")
            if _vehicle_entered(v, env):
                candidates.append(f"Vehicle {vid} has not entered the roundabout.")
        else:
            for pos in MULTI_LANE_POSITIONS_TASK4:
                if v.position != pos:
                    candidates.append(f"Vehicle {vid} is in {MULTI_LANE_LABELS_TASK4[pos]}.")

        for pos in _positions_for_env_task4(env):
            label = _label_for_position(pos)
            if env == Environment.MULTI_LANE:
                continue
            if v.position != pos:
                candidates.append(f"Vehicle {vid} is at {label}.")

    # De-duplicate while preserving order
    out: list[str] = []
    seen: set[str] = set()
    for s in candidates:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _build_option_set(
    slot: SlotSpec,
    roles: dict[str, object],
    final_state: ScenarioState,
    overlap_pairs: set[tuple[str, str]],
    correct_text_usage: Counter[str],
    correct_vehicle_usage: Counter[str],
    env_correct_vehicle_usage: dict[str, Counter[str]],
    rng: random.Random,
) -> dict[str, dict[str, str]] | None:
    by_id = {v.id: v for v in final_state.vehicles}
    certainly_true = _certainly_true_statement(
        slot,
        roles,
        by_id,
        correct_text_usage,
        correct_vehicle_usage,
        env_correct_vehicle_usage,
        rng,
    )
    near_true = _near_true_statements(slot, roles, rng)

    if classify_statement(certainly_true, slot.environment, by_id, overlap_pairs=overlap_pairs) != "true":
        return None
    if any(
        classify_statement(s, slot.environment, by_id, overlap_pairs=overlap_pairs) != "uncertain"
        for s in near_true
    ):
        return None

    excluded = {certainly_true, *near_true}
    hf_pool = _highly_false_candidates(slot.environment, by_id)
    hf_pool = [s for s in hf_pool if s not in excluded]
    hf_pool = [
        s
        for s in hf_pool
        if classify_statement(s, slot.environment, by_id, overlap_pairs=overlap_pairs) == "false"
    ]
    if len(hf_pool) < 2:
        return None
    rng.shuffle(hf_pool)
    highly_false = hf_pool[:2]

    all_statements = [certainly_true, near_true[0], near_true[1], highly_false[0], highly_false[1]]
    if len(set(all_statements)) != 5:
        return None
    if not _assert_env_label_consistency(all_statements, slot.environment):
        return None

    return {
        "correct": {
            "text": certainly_true,
            "type": "correct",
            "rationale": "certainly_true: derivable from replayed final state",
        },
        "near_true_1": {
            "text": near_true[0],
            "type": "near_true",
            "rationale": "near_true: plausible but not provable from available state information",
        },
        "near_true_2": {
            "text": near_true[1],
            "type": "near_true",
            "rationale": "near_true: plausible but not provable from available state information",
        },
        "highly_false_1": {
            "text": highly_false[0],
            "type": "highly_false",
            "rationale": "highly_false: contradicts replayed final state",
        },
        "highly_false_2": {
            "text": highly_false[1],
            "type": "highly_false",
            "rationale": "highly_false: contradicts replayed final state",
        },
    }


def assign_letters(
    options: dict[str, dict[str, str]],
    correct_key: str,
    rng: random.Random,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], str]:
    items = list(options.values())
    rng.shuffle(items)

    target_idx = LETTERS.index(correct_key)
    correct_item = next(x for x in items if x["type"] == "correct")
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


def _validate_invariants(
    *,
    init_state: ScenarioState,
    final_state: ScenarioState,
    plan: list[tuple[str, Action]],
    slot: SlotSpec,
    roles: dict[str, object],
    choices: dict[str, str],
    distractor_type: dict[str, str],
    answer: str,
    correct_key: str,
    overlap_detected: bool,
    overlap_pairs: set[tuple[str, str]],
    audit_option_rationale: dict[str, str],
    recorded_final_state: dict[str, dict[str, object]],
) -> tuple[bool, dict[str, bool], str]:
    env = slot.environment
    by_id = {v.id: v for v in final_state.vehicles}
    truths = {
        letter: classify_statement(text, env, by_id, overlap_pairs=overlap_pairs)
        for letter, text in choices.items()
    }

    # 1
    inv1 = truths.get(answer) == "true"
    # 2
    inv2 = all(truths[k] == "uncertain" for k, t in distractor_type.items() if t == "near_true")
    # 3
    inv3 = all(truths[k] == "false" for k, t in distractor_type.items() if t == "highly_false")
    # 4
    inv4 = len(set(choices.values())) == 5
    # 5
    inv5 = _assert_env_label_consistency(list(choices.values()), env)
    # 6
    replay_final = _final_state_snapshot(final_state)
    inv6 = replay_final == recorded_final_state
    # 7
    inv7 = overlap_detected or (slot.certainly_true_category in {"containment_non_entry", "lane_position"})
    # 8
    inv8 = len({vid for vid, _ in plan}) >= 2
    # 9
    inv9 = not _has_action_streak_len3([act for _, act in plan])
    # 10
    inv10 = not _contains_abab([vid for vid, _ in plan])
    # 11
    all_ids_ok = True
    for s in choices.values():
        ids = _id_set_in_statement(s)
        if not ids:
            all_ids_ok = False
            break
        if any(vid not in {"A", "B", "C"} for vid in ids):
            all_ids_ok = False
            break
    inv11 = all_ids_ok
    # 12
    inv12 = answer == correct_key
    # 13
    inv13 = (set(audit_option_rationale.keys()) == set(LETTERS)) and all(
        str(audit_option_rationale[k]).strip() for k in LETTERS
    )

    inv = {
        "certainly_true_is_derivable": inv1,
        "near_true_is_not_certainly_true": inv2,
        "highly_false_is_demonstrably_false": inv3,
        "five_distinct_statements": inv4,
        "no_cross_env_positions": inv5,
        "replay_matches_audit_final_state": inv6,
        "spatial_ambiguity_condition_met": inv7,
        "at_least_two_vehicles_act": inv8,
        "no_action_streak_len3": inv9,
        "no_abab_pattern": inv10,
        "all_statements_use_canonical_vehicle_ids": inv11,
        "answer_letter_matches_key_schedule": inv12,
        "audit_rationale_present": inv13,
    }
    ok = all(inv.values())
    reason = "ok" if ok else ",".join([k for k, v in inv.items() if not v])
    return ok, inv, reason


def generate_example(
    example_id: int,
    correct_key: str,
    slot: SlotSpec,
    difficulty: str,
    *,
    seed: int | None,
    correct_text_usage: Counter[str],
    correct_vehicle_usage: Counter[str],
    env_correct_vehicle_usage: dict[str, Counter[str]],
    rng: random.Random,
) -> dict | None:
    for attempt in range(MAX_RETRIES):
        init_state, plan, roles = _make_initial_state_and_plan(slot, difficulty, rng)

        # Safety gate for overlap-capable envs.
        if slot.environment != Environment.MULTI_LANE and not is_overlap_possible(slot.environment):
            continue

        replay = _replay_plan(init_state, plan)
        if replay is None:
            continue
        final_state, events, overlap_detected, overlap_vehicles, overlap_pairs, overlap_timeline = replay
        if len(events) not in {2, 3, 4}:
            continue

        options = _build_option_set(
            slot,
            roles,
            final_state,
            overlap_pairs,
            correct_text_usage,
            correct_vehicle_usage,
            env_correct_vehicle_usage,
            rng,
        )
        if options is None:
            continue

        choices, distractor_type, rationale_by_letter, answer = assign_letters(options, correct_key, rng)
        scenario_text = describe_scenario(init_state)
        prompt = render_prompt(scenario_text, events, QUESTION, choices)

        final_snapshot = _final_state_snapshot(final_state)
        option_rationale = {}
        by_id = {v.id: v for v in final_state.vehicles}
        for letter in LETTERS:
            cls = classify_statement(choices[letter], slot.environment, by_id, overlap_pairs=overlap_pairs)
            option_rationale[letter] = f"{cls}: {rationale_by_letter[letter]}"

        ok, invariant_map, _ = _validate_invariants(
            init_state=init_state,
            final_state=final_state,
            plan=plan,
            slot=slot,
            roles=roles,
            choices=choices,
            distractor_type=distractor_type,
            answer=answer,
            correct_key=correct_key,
            overlap_detected=overlap_detected,
            overlap_pairs=overlap_pairs,
            audit_option_rationale=option_rationale,
            recorded_final_state=final_snapshot,
        )
        if not ok:
            continue

        overlap_pair = roles["overlap_pair"]
        assert isinstance(overlap_pair, list)
        example = {
            "id": f"task4_{example_id:04d}",
            "task": TASK_NAME,
            "prompt": prompt,
            "scenario": _serialize_state(init_state),
            "events": events,
            "question": QUESTION,
            "choices": choices,
            "answer": answer,
            "distractor_type": distractor_type,
            "metadata": {
                "num_vehicles": NUM_VEHICLES,
                "num_events": len(events),
                "environment": slot.environment.value,
                "scenario_type": slot.scenario_type,
                "overlap_pair": overlap_pair,
                "certainly_true_category": slot.certainly_true_category,
                "difficulty": difficulty,
                "seed": seed,
                "generator_version": GENERATOR_VERSION,
                "generated_at_utc": GENERATED_AT_UTC,
            },
            "audit": {
                "attempt": attempt,
                "plan": [[vid, act.name] for vid, act in plan],
                "final_state": final_snapshot,
                "overlap_detected": overlap_detected,
                "overlap_vehicles": overlap_vehicles,
                "overlap_pairs": [list(pair) for pair in sorted(overlap_pairs)],
                "overlap_timeline": overlap_timeline,
                "replay_verified": True,
                "option_rationale": option_rationale,
                "invariants": invariant_map,
            },
        }
        return example

    return None


def _assert_dataset_quality(examples: list[dict]) -> None:
    if not examples:
        raise RuntimeError("empty dataset")

    n = len(examples)
    issues: list[str] = []

    prompts = [ex["prompt"] for ex in examples]
    if len(set(prompts)) != len(prompts):
        issues.append("duplicate prompts found")

    ans = Counter(ex["answer"] for ex in examples)
    if n % 5 == 0:
        target = n // 5
        for k in LETTERS:
            if ans.get(k, 0) != target:
                issues.append(f"answer distribution mismatch for {k}: {ans.get(k, 0)} vs {target}")

    env_counts = Counter(ex["metadata"]["environment"] for ex in examples)
    diff_counts = Counter(ex["metadata"]["difficulty"] for ex in examples)
    cat_counts = Counter(ex["metadata"]["certainly_true_category"] for ex in examples)
    stype_counts = Counter(ex["metadata"]["scenario_type"] for ex in examples)

    inter = env_counts.get(Environment.INTERSECTION.value, 0)
    rnd = env_counts.get(Environment.ROUNDABOUT.value, 0)
    if n == 100:
        ml = env_counts.get(Environment.MULTI_LANE.value, 0)
        if inter != 50:
            issues.append(f"intersection count out of range: {inter}")
        if rnd != 30:
            issues.append(f"roundabout count out of range: {rnd}")
        if ml != 20:
            issues.append(f"multi_lane count out of range: {ml}")
        if diff_counts.get("easy", 0) != 33 or diff_counts.get("medium", 0) != 33 or diff_counts.get("hard", 0) != 34:
            issues.append(f"difficulty distribution mismatch: {dict(diff_counts)}")
        expected_categories = {
            "containment_overlap": 20,
            "containment_non_entry": 20,
            "exit_reached": 20,
            "roundabout_overlap": 20,
            "lane_position": 20,
        }
        for category, expected in expected_categories.items():
            if cat_counts.get(category, 0) != expected:
                issues.append(f"category {category} count mismatch: {cat_counts.get(category, 0)} vs {expected}")
        expected_types = {
            "two_overlap_one_outside": 20,
            "one_inside_one_exited_one_approach": 20,
            "two_overlap_third_exited": 10,
            "roundabout_overlap": 20,
            "roundabout_non_entry": 10,
            "multi_lane_positioning": 20,
        }
        for stype, expected in expected_types.items():
            if stype_counts.get(stype, 0) != expected:
                issues.append(f"scenario_type {stype} count mismatch: {stype_counts.get(stype, 0)} vs {expected}")

    proportional_cap = max(40, math.ceil(n * 0.4))
    if any(c > proportional_cap for c in cat_counts.values()):
        issues.append(f"a certainly_true category exceeds cap: {dict(cat_counts)}")
    if any(c > proportional_cap for c in stype_counts.values()):
        issues.append(f"a scenario_type exceeds cap: {dict(stype_counts)}")

    structure_counts: Counter[tuple[str, str, str, tuple[str, ...]]] = Counter()
    event_sig_counts: Counter[tuple[str, ...]] = Counter()
    correct_text_counts: Counter[str] = Counter()
    single_correct_counts: Counter[str] = Counter()
    for idx, ex in enumerate(examples):
        env = Environment(ex["metadata"]["environment"])
        choices = ex["choices"]
        dtypes = ex["distractor_type"]
        answer = ex["answer"]

        if set(choices.keys()) != set(LETTERS):
            issues.append(f"{ex['id']}: choices keys must be A..E")
            continue
        if answer not in LETTERS:
            issues.append(f"{ex['id']}: invalid answer letter")
            continue
        if len(set(choices.values())) != 5:
            issues.append(f"{ex['id']}: duplicate statements")
        nt = sum(1 for t in dtypes.values() if t == "near_true")
        hf = sum(1 for t in dtypes.values() if t == "highly_false")
        if nt != 2 or hf != 2:
            issues.append(f"{ex['id']}: distractor counts near_true={nt}, highly_false={hf}")

        if not _assert_env_label_consistency(list(choices.values()), env):
            issues.append(f"{ex['id']}: cross-environment label contamination")

        sim = ScenarioState(
            vehicles=[
                Vehicle(
                    id=v["id"],
                    position=v["position"],
                    direction=Direction(v["direction"]),
                    inside_intersection=bool(v.get("inside_intersection", False)),
                    stopped=bool(v.get("stopped", False)),
                )
                for v in ex["scenario"]["vehicles"]
            ],
            environment=env,
        )
        overlap_any = bool(_find_overlaps(sim))
        replay_overlap_pairs = {_pair_key(a, b) for a, b in _find_overlaps(sim)}
        for ev in ex["events"]:
            m = re.match(r"^Vehicle ([ABC]) (.+)\.$", ev)
            if not m:
                issues.append(f"{ex['id']}: invalid event format")
                break
            vid, action_text = m.groups()
            action = None
            for a in Action:
                if a.value == action_text:
                    action = a
                    break
            if action is None:
                issues.append(f"{ex['id']}: unknown event action text {action_text!r}")
                break
            if not apply_action(sim, vid, action):
                issues.append(f"{ex['id']}: replay apply_action failed")
                break
            pairs = _find_overlaps(sim)
            if pairs:
                overlap_any = True
                for a, b in pairs:
                    replay_overlap_pairs.add(_pair_key(a, b))

        by_id = {v.id: v for v in sim.vehicles}
        truths = {
            k: classify_statement(v, env, by_id, overlap_pairs=replay_overlap_pairs)
            for k, v in choices.items()
        }
        if truths[answer] != "true":
            issues.append(f"{ex['id']}: answer is not certainly true ({truths[answer]})")
        for k, t in dtypes.items():
            if t == "near_true" and truths[k] != "uncertain":
                issues.append(f"{ex['id']}: near_true {k} classified as {truths[k]}")
            if t == "highly_false" and truths[k] != "false":
                issues.append(f"{ex['id']}: highly_false {k} classified as {truths[k]}")
        if sum(1 for t in truths.values() if t == "true") != 1:
            issues.append(f"{ex['id']}: expected exactly one true option, got {truths}")

        if env == Environment.MULTI_LANE:
            lane_match = RE_IN_LANE.match(choices[answer])
            if lane_match:
                vid, lane_label = lane_match.groups()
                audit_final = ex.get("audit", {}).get("final_state", {})
                final_pos = audit_final.get(vid, {}).get("position")
                expected_label = MULTI_LANE_LABELS_TASK4.get(str(final_pos), "")
                if expected_label != lane_label:
                    issues.append(
                        f"{ex['id']}: multi-lane correct answer mismatch {vid} label={lane_label!r} final_pos={final_pos!r}"
                    )

        if (not overlap_any) and (ex["metadata"]["certainly_true_category"] not in {"containment_non_entry", "lane_position"}):
            issues.append(
                f"{ex['id']}: spatial ambiguity condition not met for this category"
            )

        plan = ex.get("audit", {}).get("plan", [])
        actor_ids = [step[0] for step in plan if isinstance(step, list) and len(step) == 2]
        action_names = []
        for step in plan:
            if not isinstance(step, list) or len(step) != 2:
                continue
            action_names.append(Action[step[1]])
        if len(set(actor_ids)) < 2:
            issues.append(f"{ex['id']}: fewer than 2 acting vehicles")
        if _has_action_streak_len3(action_names):
            issues.append(f"{ex['id']}: action streak len3 detected")
        if _contains_abab(actor_ids):
            issues.append(f"{ex['id']}: ABAB actor pattern detected")

        # Statement structure cap per (env, scenario_type, category)
        combo = (
            ex["metadata"]["environment"],
            ex["metadata"]["scenario_type"],
            ex["metadata"]["certainly_true_category"],
        )
        normalized = tuple(sorted(_normalize_statement_structure(s) for s in choices.values()))
        structure_counts[(combo[0], combo[1], combo[2], normalized)] += 1
        event_sig_counts[tuple(ex["events"])] += 1
        correct_text_counts[choices[answer]] += 1
        single_vid = _single_vehicle_statement_id(choices[answer])
        if single_vid is not None:
            single_correct_counts[single_vid] += 1

    for key, count in structure_counts.items():
        if count > _statement_signature_reuse_cap_for(n):
            issues.append(f"statement structure reuse cap exceeded ({count}) for {key[:3]}")
    event_sig_cap = _event_sig_cap_for(n)
    if event_sig_counts and max(event_sig_counts.values()) > event_sig_cap:
        issues.append(f"event signature reuse cap exceeded: max={max(event_sig_counts.values())}")
    correct_text_cap = _correct_text_cap_for(n)
    if correct_text_counts and max(correct_text_counts.values()) > correct_text_cap:
        issues.append(f"correct text reuse cap exceeded: max={max(correct_text_counts.values())}")
    if single_correct_counts:
        gap = max(single_correct_counts.values()) - min(single_correct_counts.get(vid, 0) for vid in _vehicle_ids())
        if gap > _single_correct_gap_cap_for(n):
            issues.append(f"single-vehicle correct-answer gap too large: {dict(single_correct_counts)}")

    if issues:
        raise RuntimeError("Task4 quality gate failed: " + " | ".join(issues[:20]))


def generate_task4(n: int, output_path: str, seed: int | None = None) -> None:
    if seed is None:
        seed = random.SystemRandom().randrange(0, 2**32)
    rng = random.Random(seed)

    key_schedule = _build_key_schedule(n, rng)
    difficulty_schedule = _build_difficulty_schedule(n, rng)
    slot_schedule = _build_slot_schedule(n, rng)

    examples: list[dict] = []
    seen_prompts: set[str] = set()
    event_sig_usage: Counter[tuple[str, ...]] = Counter()
    correct_text_usage: Counter[str] = Counter()
    correct_vehicle_usage: Counter[str] = Counter()
    env_correct_vehicle_usage: dict[str, Counter[str]] = defaultdict(Counter)
    single_correct_usage: Counter[str] = Counter()
    reject: Counter[str] = Counter()
    correct_text_cap = _correct_text_cap_for(n)
    statement_signature_cap = _statement_signature_reuse_cap_for(n)
    event_sig_cap = _event_sig_cap_for(n)

    structure_usage: Counter[tuple[str, str, str, tuple[str, ...]]] = Counter()

    for idx in range(n):
        ex = None
        for _ in range(MAX_RETRIES * 2):
            candidate = generate_example(
                idx,
                key_schedule[idx],
                slot_schedule[idx],
                difficulty_schedule[idx],
                seed=seed,
                correct_text_usage=correct_text_usage,
                correct_vehicle_usage=correct_vehicle_usage,
                env_correct_vehicle_usage=env_correct_vehicle_usage,
                rng=rng,
            )
            if candidate is None:
                reject["candidate_none"] += 1
                continue
            if candidate["prompt"] in seen_prompts:
                reject["duplicate_prompt"] += 1
                continue

            ev_sig = tuple(candidate["events"])
            if event_sig_usage[ev_sig] >= event_sig_cap:
                reject["duplicate_event_sequence"] += 1
                continue

            correct_text = candidate["choices"][candidate["answer"]]
            if correct_text_usage[correct_text] >= correct_text_cap:
                reject["correct_text_cap"] += 1
                continue

            correct_vids = _correct_answer_vehicle_ids(correct_text)
            projected_vehicle_usage = {
                vid: correct_vehicle_usage.get(vid, 0) + (1 if vid in correct_vids else 0)
                for vid in _vehicle_ids()
            }
            if max(projected_vehicle_usage.values()) - min(projected_vehicle_usage.values()) >= _correct_answer_vehicle_gap_cap():
                reject["correct_vehicle_gap"] += 1
                continue

            single_vid = _single_vehicle_statement_id(correct_text)
            if single_vid is not None:
                other_min = min(single_correct_usage.get(vid, 0) for vid in _vehicle_ids() if vid != single_vid)
                projected = single_correct_usage[single_vid] + 1
                if projected - other_min >= _single_correct_gap_cap_for(n):
                    reject["single_correct_gap"] += 1
                    continue

            combo = (
                candidate["metadata"]["environment"],
                candidate["metadata"]["scenario_type"],
                candidate["metadata"]["certainly_true_category"],
            )
            normalized = tuple(
                sorted(_normalize_statement_structure(s) for s in candidate["choices"].values())
            )
            key = (combo[0], combo[1], combo[2], normalized)
            if structure_usage[key] >= statement_signature_cap:
                reject["statement_structure_cap"] += 1
                continue

            structure_usage[key] += 1
            seen_prompts.add(candidate["prompt"])
            event_sig_usage[ev_sig] += 1
            correct_text_usage[correct_text] += 1
            for vid in correct_vids:
                correct_vehicle_usage[vid] += 1
                env_correct_vehicle_usage[candidate["metadata"]["environment"]][vid] += 1
            if single_vid is not None:
                single_correct_usage[single_vid] += 1
            ex = candidate
            break
        if ex is None:
            raise RuntimeError(
                f"slot {idx} exhausted: produced {len(examples)}/{n}; "
                f"top_rejects={dict(reject.most_common(8))}"
            )
        examples.append(ex)

    _assert_dataset_quality(examples)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=f"{out_path.name}.tmp.",
        dir=str(out_path.parent),
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(out_path))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    answer_counts = Counter(ex["answer"] for ex in examples)
    env_counts = Counter(ex["metadata"]["environment"] for ex in examples)
    diff_counts = Counter(ex["metadata"]["difficulty"] for ex in examples)
    stype_counts = Counter(ex["metadata"]["scenario_type"] for ex in examples)
    cat_counts = Counter(ex["metadata"]["certainly_true_category"] for ex in examples)

    print(f"Seed used: {seed}")
    print(f"Saved {len(examples)} examples to {output_path}")
    print()
    print("Answer distribution:")
    for letter in LETTERS:
        print(f"  {letter}: {answer_counts.get(letter, 0)}")
    print("\nEnvironment distribution:")
    for env_name, count in sorted(env_counts.items()):
        print(f"  {env_name}: {count}")
    print("\nDifficulty distribution:")
    for d in DIFFICULTIES:
        print(f"  {d}: {diff_counts.get(d, 0)}")
    print("\nScenario type distribution:")
    for name, count in sorted(stype_counts.items()):
        print(f"  {name}: {count}")
    print("\nCertainly-true category distribution:")
    for name, count in sorted(cat_counts.items()):
        print(f"  {name}: {count}")
    if reject:
        print("\nTop rejection reasons:")
        for k, v in reject.most_common(10):
            print(f"  {k}: {v}")


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 4 certainty-under-spatial-ambiguity generator")
    p.add_argument("--n", type=int, default=N_EXAMPLES, help=f"Number of examples (default: {N_EXAMPLES})")
    p.add_argument("--seed", type=int, default=None, help="Random seed (omit for a fresh dataset each run)")
    p.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "core" / "task4_overlap.jsonl"),
        help="Output JSONL path",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    generate_task4(args.n, args.out, args.seed)
