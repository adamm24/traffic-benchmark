"""Independent Task 4 validator and quality auditor."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from domain.entities import Action, Direction, Environment, ScenarioState, Vehicle
from domain.rules import vehicles_overlap
from domain.scenario import apply_action
from domain.vocabulary import label_of, labels_for_env, positions_for_env


LETTERS = ("A", "B", "C", "D", "E")
TASK_NAME = "certainty_under_spatial_ambiguity"
EVENT_SIG_CAP = 20
CORRECT_TEXT_CAP = 20
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

# Past-tense uncertainty patterns used by Task 4 v6.
RE_WAS_AHEAD_INTER = re.compile(r"^Vehicle ([ABC]) was ahead of Vehicle ([ABC]) inside the intersection\.$")
RE_WAS_LEFT_INTER = re.compile(r"^Vehicle ([ABC]) was to the left of Vehicle ([ABC]) inside the intersection\.$")
RE_WAS_AHEAD_ROUND = re.compile(r"^Vehicle ([ABC]) was ahead of Vehicle ([ABC]) in the roundabout lane\.$")
RE_WAS_LEFT_ROUND = re.compile(r"^Vehicle ([ABC]) was to the left of Vehicle ([ABC]) in the roundabout lane\.$")
RE_BEHIND_ROUND = re.compile(r"^Vehicle ([ABC]) is behind Vehicle ([ABC]) in the roundabout lane\.$")
RE_WAS_BEHIND_ROUND = re.compile(r"^Vehicle ([ABC]) was behind Vehicle ([ABC]) in the roundabout lane\.$")
RE_AHEAD_ROAD = re.compile(r"^Vehicle ([ABC]) is ahead of Vehicle ([ABC]) on the road\.$")
RE_PAST_ROAD = re.compile(r"^Vehicle ([ABC]) has already moved past Vehicle ([ABC]) on the road\.$")
RE_WILL_CHANGE_BEFORE = re.compile(r"^Vehicle ([ABC]) will change lanes before Vehicle ([ABC])\.$")
RE_BEHIND_ROAD = re.compile(r"^Vehicle ([ABC]) is behind Vehicle ([ABC]) on the road\.$")
RE_DIRECTLY_BEHIND_ROAD = re.compile(r"^Vehicle ([ABC]) is directly behind Vehicle ([ABC]) on the road\.$")


@dataclass(frozen=True)
class ReplayResult:
    final_state: ScenarioState | None
    invalid: bool
    reason: str
    overlap_detected: bool
    overlap_pairs: set[tuple[str, str]]


@dataclass
class Totals:
    total: int = 0
    wrong: int = 0
    invalid: int = 0
    duplicate_prompts: int = 0
    near_true_not_uncertain: int = 0
    highly_false_not_false: int = 0
    cross_env_contamination: int = 0


def correct_text_cap_for(n: int) -> int:
    return max(CORRECT_TEXT_CAP, math.ceil(n / 15))


def event_sig_cap_for(n: int) -> int:
    return max(EVENT_SIG_CAP, math.ceil(n / 10))


def _vehicle_entered(v: Vehicle, env: Environment) -> bool:
    if env == Environment.INTERSECTION:
        return v.inside_intersection or v.position == "inside_intersection" or v.position.endswith("_exit")
    if env == Environment.ROUNDABOUT:
        return v.inside_intersection or v.position == "roundabout_lane" or v.position.endswith("_exit")
    return False


def _vehicle_exited(v: Vehicle, env: Environment) -> bool:
    return (not v.inside_intersection) and v.position.endswith("_exit")


def _vehicle_at_approach(v: Vehicle) -> bool:
    return (not v.inside_intersection) and v.position.endswith("_approach")


def _pair_key(a: str, b: str) -> tuple[str, str]:
    if a <= b:
        return (a, b)
    return (b, a)


def _label_for_position(position: str) -> str:
    if position in MULTI_LANE_LABELS_TASK4:
        return MULTI_LANE_LABELS_TASK4[position]
    return label_of(position)


def _positions_for_env_task4(env: Environment) -> tuple[str, ...]:
    if env == Environment.MULTI_LANE:
        return MULTI_LANE_POSITIONS_TASK4
    return positions_for_env(env)


def classify_statement(
    statement: str,
    env: Environment,
    by_id: dict[str, Vehicle],
    overlap_detected: bool = False,
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
        return "true" if not _vehicle_entered(by_id[a], env) else "false"

    m = RE_NOT_ENTERED_ROUND.match(statement)
    if m:
        if env != Environment.ROUNDABOUT:
            return "invalid"
        a = m.group(1)
        return "true" if not _vehicle_entered(by_id[a], env) else "false"

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

    # Past-tense uncertainty patterns: "Vehicle X was ahead/left of Vehicle Y inside the intersection"
    # These are uncertain iff both vehicles were simultaneously inside at some point during replay.
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


def _parse_action_from_event(event: str) -> tuple[str, Action] | None:
    m = re.match(r"^Vehicle ([ABC]) (.+)\.$", event)
    if not m:
        return None
    vid, action_text = m.groups()
    for action in Action:
        if action.value == action_text:
            return vid, action
    return None


def _has_overlap(state: ScenarioState) -> bool:
    for a, b in combinations(state.vehicles, 2):
        if vehicles_overlap(a, b):
            return True
    return False


def _reconstruct_state(ex: dict) -> ScenarioState:
    env = Environment(ex["scenario"]["environment"])
    vehicles = []
    for v in ex["scenario"]["vehicles"]:
        vehicles.append(
            Vehicle(
                id=v["id"],
                position=v["position"],
                direction=Direction(v["direction"]),
                inside_intersection=bool(v.get("inside_intersection", False)),
                stopped=bool(v.get("stopped", False)),
            )
        )
    return ScenarioState(vehicles=vehicles, environment=env)


def replay_example(ex: dict) -> ReplayResult:
    try:
        state = _reconstruct_state(ex)
    except Exception as exc:  # noqa: BLE001
        return ReplayResult(
            final_state=None,
            invalid=True,
            reason=f"reconstruct_error: {exc}",
            overlap_detected=False,
            overlap_pairs=set(),
        )

    overlap_detected = _has_overlap(state)
    overlap_pairs: set[tuple[str, str]] = set()
    if overlap_detected:
        for a, b in combinations(state.vehicles, 2):
            if vehicles_overlap(a, b):
                overlap_pairs.add(_pair_key(a.id, b.id))
    for i, event in enumerate(ex.get("events", []), start=1):
        parsed = _parse_action_from_event(event)
        if parsed is None:
            return ReplayResult(
                final_state=None,
                invalid=True,
                reason=f"event_parse_error at step {i}: {event!r}",
                overlap_detected=overlap_detected,
                overlap_pairs=overlap_pairs,
            )
        vid, action = parsed
        applied = apply_action(state, vid, action)
        if not applied:
            return ReplayResult(
                final_state=None,
                invalid=True,
                reason=f"apply_action failed at step {i}: ({vid}, {action.name})",
                overlap_detected=overlap_detected,
                overlap_pairs=overlap_pairs,
            )
        for a, b in combinations(state.vehicles, 2):
            if vehicles_overlap(a, b):
                overlap_detected = True
                overlap_pairs.add(_pair_key(a.id, b.id))

    return ReplayResult(
        final_state=state,
        invalid=False,
        reason="ok",
        overlap_detected=overlap_detected,
        overlap_pairs=overlap_pairs,
    )


def validate_example(ex: dict) -> tuple[bool, bool, str]:
    # (ok, invalid, reason)
    if ex.get("task") != TASK_NAME:
        return False, True, f"task label must be {TASK_NAME!r}"

    choices = ex.get("choices", {})
    answer = ex.get("answer")
    dtypes = ex.get("distractor_type", {})

    if set(choices.keys()) != set(LETTERS):
        return False, True, "choices keys must be exactly A..E"
    if answer not in LETTERS:
        return False, True, "invalid answer letter"
    if len(set(choices.values())) != 5:
        return False, True, "duplicate choice statements"
    nt = sum(1 for t in dtypes.values() if t == "near_true")
    hf = sum(1 for t in dtypes.values() if t == "highly_false")
    if nt != 2 or hf != 2:
        return False, True, f"distractor type counts invalid near_true={nt}, highly_false={hf}"

    replay = replay_example(ex)
    if replay.invalid or replay.final_state is None:
        return False, True, replay.reason

    env = replay.final_state.environment
    by_id = {v.id: v for v in replay.final_state.vehicles}
    truths = {
        k: classify_statement(
            v,
            env,
            by_id,
            overlap_detected=replay.overlap_detected,
            overlap_pairs=replay.overlap_pairs,
        )
        for k, v in choices.items()
    }

    if any(t == "invalid" for t in truths.values()):
        return False, True, f"invalid statement parser result: {truths}"

    true_letters = [k for k, t in truths.items() if t == "true"]
    if len(true_letters) != 1:
        return False, True, f"expected exactly one true statement, got {truths}"

    if answer != true_letters[0]:
        return False, False, f"wrong answer: declared={answer}, recomputed={true_letters[0]}"

    for k, t in dtypes.items():
        if t == "near_true" and truths[k] != "uncertain":
            return False, False, f"near_true not uncertain for {k}: {truths[k]}"
        if t == "highly_false" and truths[k] != "false":
            return False, False, f"highly_false not false for {k}: {truths[k]}"

    # Label consistency check.
    allowed_labels = ALL_LABELS_BY_ENV[env]
    for s in choices.values():
        labels = _extract_position_labels(s)
        if any(label not in allowed_labels for label in labels):
            return False, False, "cross-environment label contamination"

    cat = ex.get("metadata", {}).get("certainly_true_category")
    if (not replay.overlap_detected) and (cat not in {"containment_non_entry", "lane_position"}):
        return False, False, "spatial ambiguity condition not met"

    return True, False, "ok"


def validate_file(path: Path) -> int:
    if not path.exists():
        print(f"ERROR: input file not found: {path}")
        return 2

    totals = Totals()
    answer_counts = Counter()
    env_counts = Counter()
    difficulty_counts = Counter()
    category_counts = Counter()
    scenario_type_counts = Counter()
    num_events_counts = Counter()
    event_sig_counts = Counter()
    correct_text_counts = Counter()
    prompts_seen: set[str] = set()
    dataset_issues: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            totals.total += 1
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as exc:
                totals.invalid += 1
                print(f"[invalid] line {line_no}: JSON decode error: {exc}")
                continue

            prompt = ex.get("prompt", "")
            choices = ex.get("choices", {})
            if prompt in prompts_seen:
                totals.duplicate_prompts += 1
            prompts_seen.add(prompt)

            answer = ex.get("answer")
            if answer in LETTERS:
                answer_counts[answer] += 1
            metadata = ex.get("metadata", {})
            env_counts[metadata.get("environment")] += 1
            difficulty_counts[metadata.get("difficulty")] += 1
            category_counts[metadata.get("certainly_true_category")] += 1
            scenario_type_counts[metadata.get("scenario_type")] += 1
            num_events_counts[metadata.get("num_events")] += 1
            event_sig_counts[tuple(ex.get("events", []))] += 1
            if answer in LETTERS:
                correct_text_counts[choices[answer]] += 1

            ok, is_invalid, reason = validate_example(ex)
            if not ok:
                if is_invalid:
                    totals.invalid += 1
                    print(f"[invalid] line {line_no} {ex.get('id', '?')}: {reason}")
                else:
                    totals.wrong += 1
                    print(f"[wrong] line {line_no} {ex.get('id', '?')}: {reason}")

                if "near_true not uncertain" in reason:
                    totals.near_true_not_uncertain += 1
                if "highly_false not false" in reason:
                    totals.highly_false_not_false += 1
                if "cross-environment label contamination" in reason:
                    totals.cross_env_contamination += 1

    print(f"\nValidated file: {path}")
    print(f"total={totals.total} wrong={totals.wrong} invalid={totals.invalid}")
    print()
    print("Answer distribution:")
    for letter in LETTERS:
        print(f"  {letter}: {answer_counts.get(letter, 0)}")
    print("\nEnvironment distribution:")
    for env, c in sorted(env_counts.items()):
        print(f"  {env}: {c}")
    print("\nDifficulty distribution:")
    for d, c in sorted(difficulty_counts.items()):
        print(f"  {d}: {c}")
    print("\nCertainly-true category distribution:")
    for d, c in sorted(category_counts.items()):
        print(f"  {d}: {c}")
    print("\nScenario type distribution:")
    for d, c in sorted(scenario_type_counts.items()):
        print(f"  {d}: {c}")
    print("\nNum events distribution:")
    for d, c in sorted(num_events_counts.items()):
        print(f"  {d}: {c}")
    print("\nAuxiliary checks:")
    print(f"  duplicate_prompts: {totals.duplicate_prompts}")
    print(f"  near_true_not_uncertain: {totals.near_true_not_uncertain}")
    print(f"  highly_false_not_false: {totals.highly_false_not_false}")
    print(f"  cross_env_contamination: {totals.cross_env_contamination}")
    print(f"  event_sig_max: {max(event_sig_counts.values(), default=0)}")
    print(f"  correct_text_max: {max(correct_text_counts.values(), default=0)}")

    event_sig_cap = event_sig_cap_for(totals.total)
    if event_sig_counts and max(event_sig_counts.values()) > event_sig_cap:
        dataset_issues.append(
            f"event signature reuse cap exceeded: max={max(event_sig_counts.values())} cap={event_sig_cap}"
        )
    correct_text_cap = correct_text_cap_for(totals.total)
    if correct_text_counts and max(correct_text_counts.values()) > correct_text_cap:
        dataset_issues.append(
            f"correct text reuse cap exceeded: max={max(correct_text_counts.values())} cap={correct_text_cap}"
        )
    if dataset_issues:
        for issue in dataset_issues:
            print(f"[invalid] dataset-level: {issue}")

    return 0 if (totals.wrong == 0 and totals.invalid == 0 and not dataset_issues) else 1


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Independent validator for Task 4 certainty under spatial ambiguity")
    p.add_argument(
        "--input",
        type=str,
        default="dataset/core/task4_overlap.jsonl",
        help="Input JSONL file path",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    raise SystemExit(validate_file(Path(args.input)))
