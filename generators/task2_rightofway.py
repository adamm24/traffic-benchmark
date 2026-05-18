"""Task 2 right-of-way generator."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from domain.entities import (
    Direction,
    Environment,
    IntentDirection,
    ScenarioState,
    UnsupportedScenarioError,
    Vehicle,
)
from domain.render import describe_scenario
from domain.rules import right_of_way
from domain.rules import right_of_way_intersection
from domain.scenario import (
    build_intersection_scenario,
    build_roundabout_scenario,
)
from domain.trajectory import trajectories_conflict



GENERATOR_VERSION = "task2_rightofway_v4_1"
DEFAULT_N_EXAMPLES = 100
DEFAULT_SEED: int | None = None

NUM_VEHICLES = 3
MAX_RETRIES = 200

# Keep intersections dominant: this task should mostly test intent-sensitive
# right-of-way reasoning, with a smaller roundabout slice.
INTERSECTION_TARGET_SHARE = 0.60
# Sub-slice of all examples dedicated to 4-way cyclic/no-clear intersection.
INTERSECTION_NO_CLEAR_SHARE = 0.08
# Additional no-clear intersection slice with explicit right-vs-left conflicts.
INTERSECTION_NO_CLEAR_TURNMIX_SHARE = 0.08
# Additional no-clear 3-vehicle straight slice to avoid correlating
# "No unambiguous right of way" only with 4-vehicle scenarios.
INTERSECTION_NO_CLEAR_3WAY_SHARE = 0.06

LETTERS = ["A", "B", "C", "D", "E"]
PRIORITY_LABELS = ["A", "B", "C"]
ROUNDABOUT_PRIORITY_LABEL = "A"
PAIR_KEYS = ["A-B", "A-C", "B-C"]
BOTH_OPTION_TEXT = "Both can pass at the same time"
NO_CLEAR_OPTION_TEXT = "No vehicle has an unambiguous right of way"
NO_CLEAR_EXTRA_HF_TEXT = "One vehicle always has absolute priority regardless of intent"

INTERSECTION_QUESTIONS = [
    "Which vehicle has the right of way before the others?",
    "Which vehicle should be allowed to pass first?",
    "According to traffic rules, which vehicle has priority now?",
    "Which vehicle has priority in this situation?",
    "Which vehicle should proceed first under right-of-way rules?",
]

INTERSECTION_NO_CLEAR_QUESTIONS = [
    "Which option best describes right of way in this intersection?",
    "Is there one vehicle with clear right of way before the others?",
    "Which statement about priority is correct in this situation?",
]

ROUNDABOUT_QUESTIONS = [
    "Which vehicle has the right of way before the others?",
    "Which vehicle should be allowed to proceed first?",
    "According to roundabout rules, which vehicle has priority now?",
    "Which vehicle has priority in this roundabout situation?",
    "Which vehicle should proceed first under roundabout right-of-way rules?",
]

INTERSECTION_CONTEXTS = [
    "There are no traffic lights or signs.",
    "The intersection has no traffic signals.",
    "No stop signs or traffic lights are present.",
    "This is an unsignalized intersection.",
    "Standard unsignalized-intersection priority rules apply.",
]

ROUNDABOUT_CONTEXTS = [
    "Standard roundabout rules apply.",
    "Vehicles already inside the roundabout have priority.",
    "This roundabout follows standard right-of-way rules.",
    "No temporary traffic control is active at the roundabout.",
    "Use standard roundabout yielding rules.",
]

INTERSECTION_NO_CLEAR_STRAIGHT_CONTEXTS = [
    "All four approaches are active and no signal controls are present.",
    "All vehicles are proceeding straight through the unsignalized intersection.",
]

INTERSECTION_NO_CLEAR_TURNMIX_CONTEXTS = [
    "All four approaches are active and no signal controls are present.",
    "No temporary traffic control or police direction is present.",
    "Rule note: when trajectories conflict, left turns yield to oncoming straight/right traffic; otherwise priority-to-the-right applies.",
]

INTERSECTION_NO_CLEAR_3WAY_CONTEXTS = [
    "Three approaches are active and no signal controls are present.",
    "All vehicles are proceeding straight through the unsignalized intersection.",
    "No temporary traffic control or police direction is present.",
]

# Fallback highly-false statements for rare blocked distractor pools.
_GENERIC_HF_FALLBACK = {
    Environment.INTERSECTION: [
        "No vehicle has priority in this scenario",
        "All three vehicles may proceed simultaneously",
        "Priority follows alphabetical vehicle order",
        "The vehicle turning left always goes first",
        "The vehicle on the left always has priority",
        "All vehicles must wait until the intersection is empty",
    ],
    Environment.ROUNDABOUT: [
        "No vehicle has priority in this scenario",
        "Entering vehicles always go before circulating vehicles",
        "All three vehicles may proceed simultaneously",
        "Priority follows alphabetical vehicle order",
        "Vehicles outside the roundabout always go first",
        "All vehicles must stop until the roundabout is empty",
    ],
}

# Cardinal direction labels used in scenario-grounded distractor text.
_DIR_LABEL = {
    "north": "northern",
    "south": "southern",
    "east": "eastern",
    "west": "western",
}

# Right-of neighbor relative to an approach direction (priority-to-the-right).
# Vehicle approaching from X has the vehicle from Y to its right.
_RIGHT_OF = {
    "north": "west",   # heading south → vehicle on the right comes from the west arm
    "south": "east",
    "east":  "north",
    "west":  "south",
}

_OPPOSITE_DIR = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}

def _build_intersection_distractors(
    state: ScenarioState,
    priority_vid: str,
    yielding_vid: str,
    third_vid: str,
    pair_assessments: list[PairAssessment],
) -> list[str]:
    """
    Builds wrong-but-plausible policy strings for an intersection distractor pool.
    Uses the actual directions and intents in the scenario so they can't be
    rejected without checking the rules. Callers pick 2 from the returned list.
    """
    vmap = {v.id: v for v in state.vehicles}
    v_winner = vmap[priority_vid]
    winner_dir = v_winner.direction.value
    winner_intent = v_winner.intent.value if v_winner.intent else None
    winner_dir_label = _DIR_LABEL.get(winner_dir, winner_dir)

    distractors: list[str] = []

    # Collect intent distribution in the scenario
    intents = [v.intent.value if v.intent else None for v in state.vehicles]
    has_left_turner = "turn left" in intents
    has_straight = "go straight" in intents
    has_right_turner = "turn right" in intents
    winner_turns_left = winner_intent == "turn left"


    # D1: Inverted left-turn-yield rule.
    # When the winner is NOT turning left: claim that the left-turning vehicle
    # should proceed first because it "needs more space to manoeuvre".
    if has_left_turner and not winner_turns_left:
        distractors.append(
            "The vehicle intending to turn left should proceed first, as it "
            "requires more space and time to complete its manoeuvre safely"
        )

    # D2: When the winner IS going straight: claim straight-going vehicles
    # always yield to turning vehicles at unsignalized intersections.
    if winner_intent == "go straight" and (has_left_turner or has_right_turner):
        distractors.append(
            "Vehicles going straight must yield to vehicles performing a turn, "
            "since turning requires more space at an unsignalized intersection"
        )

    # D3: Inverted priority-to-the-right: the winner's direction is given as
    # the one that should yield (using the winner's actual direction label).
    distractors.append(
        f"The vehicle from the {winner_dir_label} approach must yield, because "
        f"it is approaching from a direction that does not have right of way "
        f"over perpendicular traffic at this intersection"
    )

    # D4: Wrong directional axis rule using the winner's approach axis.
    if winner_dir in ("north", "south"):
        distractors.append(
            "East-west traffic has right of way over north-south traffic "
            "at an unsignalized four-way intersection"
        )
    else:
        distractors.append(
            "North-south traffic has right of way over east-west traffic "
            "at an unsignalized four-way intersection"
        )

    # D5: Claim the vehicle going straight always wins over turning vehicles,
    # when the winner is actually a turning vehicle (turn right or a valid left).
    if winner_turns_left and has_straight:
        distractors.append(
            "A vehicle going straight always has priority over any vehicle "
            "intending to turn at an unsignalized intersection"
        )
    elif winner_intent == "turn right" and has_straight:
        distractors.append(
            "Vehicles going straight always yield last at an unsignalized "
            "intersection, regardless of their approach direction"
        )

    # D6: Claim opposing approaches share priority (wrong for this case).
    opposite_of_winner = _OPPOSITE_DIR.get(winner_dir, "")
    opposite_label = _DIR_LABEL.get(opposite_of_winner, opposite_of_winner)
    if opposite_label:
        distractors.append(
            f"Vehicles approaching from opposite directions — {winner_dir_label} "
            f"and {opposite_label} — share equal priority and must negotiate "
            f"passage before yielding to crossing traffic"
        )

    # D7: Claim the first vehicle to approach the stop line proceeds first.
    distractors.append(
        "The vehicle that reaches the intersection stop line first is entitled "
        "to proceed before others, regardless of approach direction"
    )

    # D8: Claim a right-turning vehicle always goes first (wrong generally).
    if has_right_turner and winner_intent != "turn right":
        distractors.append(
            "A vehicle turning right has priority because it takes the shortest "
            "path through the intersection and creates the least conflict"
        )

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for d in distractors:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def _build_roundabout_distractors(
    state: ScenarioState,
    priority_vid: str,
    yielding_vid: str,
    third_vid: str,
) -> list[str]:
    """
    Builds wrong-but-plausible policy strings for a roundabout distractor pool.
    Each statement misapplies the yield rule using the actual approach
    directions in the scenario. Callers pick 2 from the returned list.
    """
    vmap = {v.id: v for v in state.vehicles}
    entering_vehicles = [v for v in state.vehicles if not v.inside_intersection]
    entering_dirs = [_DIR_LABEL.get(v.direction.value, v.direction.value)
                     for v in entering_vehicles]

    distractors: list[str] = []

    # D1: Inverted roundabout rule — entering vehicles yield to nobody.
    distractors.append(
        "A vehicle entering the roundabout does not need to yield when the "
        "circulating vehicle has not yet reached the entry point"
    )

    # D2: Claim the circulating vehicle should yield to avoid blocking entry.
    distractors.append(
        "The vehicle already in the roundabout should yield to entering "
        "vehicles, since it has more space to slow down or loop around"
    )

    # D3: Claim priority goes to the entering vehicle waiting longest.
    distractors.append(
        "The vehicle that has been waiting at the roundabout entry the "
        "longest acquires priority and may enter before circulating traffic"
    )

    # D4: Directional priority claim using the actual approach direction.
    if entering_dirs:
        first_dir = entering_dirs[0]
        distractors.append(
            f"Vehicles approaching from the {first_dir} entry have priority "
            f"at this roundabout because they face the shortest merge distance"
        )

    # D5: Claim two-vehicle entry is allowed simultaneously.
    if len(entering_vehicles) >= 2 and len(entering_dirs) >= 2:
        d1, d2 = entering_dirs[0], entering_dirs[1]
        distractors.append(
            f"Both the {d1} and {d2} approaches may enter the roundabout "
            f"simultaneously since they use different sections of the ring"
        )

    # D6: Claim yield obligation transfers to circulating once entry is started.
    distractors.append(
        "Once a vehicle has begun its entry manoeuvre into the roundabout, "
        "the circulating vehicle is required to yield to avoid a collision"
    )

    # De-duplicate
    seen: set[str] = set()
    unique: list[str] = []
    for d in distractors:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique



def _pick_least_used(
    pool: list[str],
    usage: Counter[str],
    commit: bool = True,
) -> str:
    """Picks one element from the least-used bucket; ties broken randomly."""
    min_used = min(usage[item] for item in pool)
    candidates = [item for item in pool if usage[item] == min_used]
    chosen = random.choice(candidates)
    if commit:
        usage[chosen] += 1
    return chosen


def _pick_two_least_used(
    pool: list[str],
    usage: Counter[str],
    commit: bool = True,
) -> tuple[str, str]:
    """Picks two distinct elements with balancing pressure."""
    first = _pick_least_used(pool, usage, commit=commit)
    second_pool = [item for item in pool if item != first]
    second = _pick_least_used(second_pool, usage, commit=commit)
    return first, second


def _alphabetical_non_left_heuristic(state: ScenarioState) -> str:
    """
    Shallow baseline heuristic for intersections:
    pick alphabetically-first vehicle that is NOT turning left.
    """
    ordered = sorted(state.vehicles, key=lambda v: v.id)
    non_left = [
        v.id for v in ordered
        if v.intent != IntentDirection.TURN_LEFT
    ]
    return non_left[0] if non_left else ordered[0].id


def _relabel_intersection_for_anti_shortcut(
    state: ScenarioState,
    priority_label_usage: Counter[str],
    desired_priority_label: str,
    desired_conflict_pair: str | None = None,
) -> tuple[ScenarioState, PriorityDerivation] | None:
    """
    Finds an ID relabeling (A/B/C permutation) that:
      1) keeps the scenario valid under strict derive logic
      2) enforces the requested priority label
      3) prefers (but does not force) alphabetical-non-left shortcut failure
      4) balances residual usage inside the allowed label bucket
    """
    original_ids = [v.id for v in state.vehicles]
    best: tuple[tuple[int, str], ScenarioState, PriorityDerivation] | None = None

    for perm in itertools.permutations(["A", "B", "C"], len(original_ids)):
        old_to_new = {old: new for old, new in zip(original_ids, perm)}
        trial = copy.deepcopy(state)
        for v in trial.vehicles:
            v.id = old_to_new[v.id]

        derived = _derive_priority_structure(trial)
        if derived is None:
            continue

        priority_vid = derived[0]
        yielding_vid = derived[1]
        if priority_vid != desired_priority_label:
            continue
        if desired_conflict_pair is not None:
            if "-".join(sorted([priority_vid, yielding_vid])) != desired_conflict_pair:
                continue
        alpha_hit = int(_alphabetical_non_left_heuristic(trial) == priority_vid)

        score = (
            alpha_hit,
            priority_label_usage[priority_vid],
            priority_vid,
        )
        if best is None or score < best[0]:
            best = (score, trial, derived)

    if best is None:
        return None
    return best[1], best[2]


def _relabel_roundabout_for_priority_balance(
    state: ScenarioState,
    desired_priority_label: str,
) -> ScenarioState:
    """
    Relabel roundabout vehicles so the inside-priority vehicle uses the
    requested priority label.
    """
    vehicles = list(state.vehicles)
    inside = next(v for v in vehicles if v.inside_intersection or v.position == "roundabout_lane")
    outside = sorted([v for v in vehicles if v.id != inside.id], key=lambda v: v.id)

    winner_label = desired_priority_label
    remaining_labels = [l for l in PRIORITY_LABELS if l != winner_label]

    # Keep deterministic ordering for non-priority vehicles.
    old_to_new = {
        inside.id: winner_label,
        outside[0].id: remaining_labels[0],
        outside[1].id: remaining_labels[1],
    }
    relabeled = copy.deepcopy(state)
    for v in relabeled.vehicles:
        v.id = old_to_new[v.id]
    return relabeled


def _roundabout_conflict_pair_from_priority_label(priority_label: str) -> str:
    # With one inside winner + two entering vehicles, yielding tie-break is alphabetical.
    # Therefore pair is deterministic given winner label.
    if priority_label in {"A", "B"}:
        return "A-B"
    return "A-C"


def _build_env_schedule(n: int) -> list[Environment]:
    """
    Deterministic-sized environment schedule (shuffled):
    exact count for intersection/roundabout for the given n.
    """
    if n <= 0:
        return []
    n_intersection = int(round(n * INTERSECTION_TARGET_SHARE))
    if n > 1:
        n_intersection = max(1, min(n - 1, n_intersection))
    n_roundabout = n - n_intersection
    schedule = (
        [Environment.INTERSECTION] * n_intersection
        + [Environment.ROUNDABOUT] * n_roundabout
    )
    random.shuffle(schedule)
    return schedule


def _build_mode_schedule(n: int, env_schedule: list[Environment]) -> list[str]:
    """
    Build per-example mode schedule:
      - intersection_priority: unique global winner exists
      - intersection_no_clear_straight: 4-way straight cyclic case (no unique winner)
      - intersection_no_clear_turnmix: 4-way mixed-intent no-clear case
      - intersection_no_clear_threeway: 3-way straight no-clear case
      - roundabout_priority: standard inside-priority roundabout
    """
    if n != len(env_schedule):
        raise ValueError("env_schedule length mismatch")
    if n == 0:
        return []

    intersection_indices = [
        i for i, env in enumerate(env_schedule) if env == Environment.INTERSECTION
    ]
    n_intersection = len(intersection_indices)
    if n_intersection == 0:
        return ["roundabout_priority"] * n

    target_no_clear_straight = int(round(n * INTERSECTION_NO_CLEAR_SHARE))
    target_no_clear_turnmix = int(round(n * INTERSECTION_NO_CLEAR_TURNMIX_SHARE))
    target_no_clear_3way = int(round(n * INTERSECTION_NO_CLEAR_3WAY_SHARE))

    n_no_clear_straight = max(1, min(n_intersection - 3, target_no_clear_straight))
    remaining_after_straight = max(2, n_intersection - n_no_clear_straight - 1)
    n_no_clear_turnmix = max(1, min(remaining_after_straight - 1, target_no_clear_turnmix))
    remaining_after_turnmix = max(1, n_intersection - n_no_clear_straight - n_no_clear_turnmix - 1)
    n_no_clear_3way = max(1, min(remaining_after_turnmix, target_no_clear_3way))

    selected_straight = set(random.sample(intersection_indices, n_no_clear_straight))
    remaining_pool = [i for i in intersection_indices if i not in selected_straight]
    selected_turnmix = set(random.sample(remaining_pool, n_no_clear_turnmix))
    remaining_pool = [i for i in remaining_pool if i not in selected_turnmix]
    selected_3way = set(random.sample(remaining_pool, n_no_clear_3way))

    modes: list[str] = []
    for i, env in enumerate(env_schedule):
        if env == Environment.ROUNDABOUT:
            modes.append("roundabout_priority")
        elif i in selected_straight:
            modes.append("intersection_no_clear_straight")
        elif i in selected_turnmix:
            modes.append("intersection_no_clear_turnmix")
        elif i in selected_3way:
            modes.append("intersection_no_clear_threeway")
        else:
            modes.append("intersection_priority")
    return modes


def _build_priority_schedule(mode_schedule: list[str]) -> list[str]:
    """
    Build desired priority labels for priority-bearing modes.
    Empty string means "not applicable" (intersection_no_clear mode).
    """
    labels = [""] * len(mode_schedule)
    usage: Counter[str] = Counter()
    for i, mode in enumerate(mode_schedule):
        if mode.startswith("intersection_no_clear"):
            continue
        # Rebalance globally across A/B/C to avoid literal-answer dominance.
        label = min(PRIORITY_LABELS, key=lambda candidate: (usage[candidate], candidate))
        labels[i] = label
        usage[label] += 1
    return labels


def _build_pair_targets(total: int, pair_keys: list[str]) -> dict[str, int]:
    if total <= 0:
        return {k: 0 for k in pair_keys}
    base = total // len(pair_keys)
    rem = total % len(pair_keys)
    out = {k: base for k in pair_keys}
    for i, k in enumerate(pair_keys):
        if i < rem:
            out[k] += 1
    return out


def _pick_underused_pair(
    usage: Counter[str],
    targets: dict[str, int],
    candidates: list[str],
) -> str:
    # Prefer pairs farthest below target; ties broken by absolute usage.
    ranked = sorted(
        candidates,
        key=lambda p: (
            usage[p] - targets.get(p, 0),
            usage[p],
            p,
        ),
    )
    return ranked[0]



@dataclass(frozen=True)
class PairAssessment:
    v1: str
    v2: str
    conflict: bool
    winner: str | None
    direction_only_winner: str | None


PriorityDerivation = tuple[
    str,
    str,
    str,
    list[PairAssessment],
    dict[str, set[str]],
    dict[str, set[str]],
    bool,
    str | None,
]



def _pair_assessment(v1: Vehicle, v2: Vehicle, env: Environment) -> PairAssessment | None:
    """
    Returns pairwise conflict + winner info.

    Returns None when the pair is unsupported / inconsistent for Task 2.
    """
    if env == Environment.INTERSECTION:
        if v1.intent is None or v2.intent is None:
            return None
        direction_only_winner = right_of_way_intersection(v1, v2)
        has_conflict = trajectories_conflict(v1, v2)
        if not has_conflict:
            return PairAssessment(
                v1=v1.id,
                v2=v2.id,
                conflict=False,
                winner=None,
                direction_only_winner=direction_only_winner,
            )
        try:
            winner = right_of_way(v1, v2, env)
        except UnsupportedScenarioError:
            return None
        if winner not in (v1.id, v2.id):
            return None
        return PairAssessment(
            v1=v1.id,
            v2=v2.id,
            conflict=True,
            winner=winner,
            direction_only_winner=direction_only_winner,
        )

    if env == Environment.ROUNDABOUT:
        v1_inside = bool(v1.inside_intersection) or v1.position == "roundabout_lane"
        v2_inside = bool(v2.inside_intersection) or v2.position == "roundabout_lane"

        # Two entering vehicles are not a right-of-way conflict pair.
        if not v1_inside and not v2_inside:
            return PairAssessment(
                v1=v1.id,
                v2=v2.id,
                conflict=False,
                winner=None,
                direction_only_winner=None,
            )

        if v1_inside and v2_inside:
            return None

        winner = v1.id if v1_inside else v2.id
        return PairAssessment(
            v1=v1.id,
            v2=v2.id,
            conflict=True,
            winner=winner,
            direction_only_winner=None,
        )

    return None


def _derive_priority_structure(
    state: ScenarioState,
) -> PriorityDerivation | None:
    """
    Derives a unique global priority vehicle.

    Contract for Task 2 acceptance:
      • exactly one vehicle dominates all of its conflict pairs
      • that vehicle conflicts with all other vehicles (for 3 vehicles: degree 2)
    """
    vehicles = state.vehicles
    vids = [v.id for v in vehicles]

    wins: dict[str, set[str]] = {vid: set() for vid in vids}
    conflicts: dict[str, set[str]] = {vid: set() for vid in vids}
    pair_assessments: list[PairAssessment] = []

    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            v1, v2 = vehicles[i], vehicles[j]
            pair_info = _pair_assessment(v1, v2, state.environment)
            if pair_info is None:
                return None
            pair_assessments.append(pair_info)
            if not pair_info.conflict or pair_info.winner is None:
                continue

            winner = pair_info.winner
            a, b = pair_info.v1, pair_info.v2
            conflicts[a].add(b)
            conflicts[b].add(a)

            loser = b if winner == a else a
            wins[winner].add(loser)

    dominant = [
        vid
        for vid in vids
        if conflicts[vid] and wins[vid] == conflicts[vid]
    ]
    if len(dominant) != 1:
        return None

    priority_vid = dominant[0]
    if len(conflicts[priority_vid]) != len(vids) - 1:
        # Avoid partial-conflict ambiguity: "priority" must be global.
        return None

    direction_only_wins: dict[str, set[str]] = {vid: set() for vid in vids}
    direction_only_conflicts: dict[str, set[str]] = {vid: set() for vid in vids}
    for p in pair_assessments:
        if p.direction_only_winner is None:
            continue
        a, b = p.v1, p.v2
        direction_only_conflicts[a].add(b)
        direction_only_conflicts[b].add(a)
        loser = b if p.direction_only_winner == a else a
        direction_only_wins[p.direction_only_winner].add(loser)
    direction_only_dominant = [
        vid
        for vid in vids
        if direction_only_conflicts[vid]
        and direction_only_wins[vid] == direction_only_conflicts[vid]
        and len(direction_only_conflicts[vid]) == len(vids) - 1
    ]
    direction_only_priority = (
        direction_only_dominant[0] if len(direction_only_dominant) == 1 else None
    )

    intent_sensitive_with_priority = False
    if state.environment == Environment.INTERSECTION:
        # Require at least one pair where direction-only logic disagrees with
        # the intent-aware winner for the priority vehicle.
        intent_sensitive_with_priority = any(
            p.conflict
            and p.winner == priority_vid
            and p.direction_only_winner != priority_vid
            and priority_vid in (p.v1, p.v2)
            for p in pair_assessments
        )
        if not intent_sensitive_with_priority:
            return None
        if direction_only_priority == priority_vid:
            # Avoid datasets solvable by direction-only shortcut heuristics.
            return None
    else:
        # Roundabout priority is positional; intent sensitivity is not applicable.
        intent_sensitive_with_priority = True

    losers = sorted(
        list(conflicts[priority_vid]),
        key=lambda vid: (-len(wins[vid]), vid),
    )
    yielding_vid = losers[0]
    third_vid = next(vid for vid in vids if vid not in (priority_vid, yielding_vid))

    return (
        priority_vid,
        yielding_vid,
        third_vid,
        pair_assessments,
        wins,
        conflicts,
        intent_sensitive_with_priority,
        direction_only_priority,
    )



def _build_intersection_priority_scenario() -> tuple[
    ScenarioState,
    str,
    str,
    str,
    list[PairAssessment],
    dict[str, set[str]],
    dict[str, set[str]],
    bool,
    str | None,
] | None:
    for _ in range(MAX_RETRIES):
        state = build_intersection_scenario(NUM_VEHICLES, with_intent=True)
        derived = _derive_priority_structure(state)
        if derived is None:
            continue

        # Anti-shortcut gate 1: reject cases with exactly one non-left vehicle.
        # In those cases, a trivial "pick non-left" heuristic is unbeatable.
        left_count = sum(
            1 for v in state.vehicles if v.intent == IntentDirection.TURN_LEFT
        )
        if left_count >= 2:
            continue

        return (state, *derived)
    return None


def _build_roundabout_priority_scenario() -> tuple[
    ScenarioState,
    str,
    str,
    str,
    list[PairAssessment],
    dict[str, set[str]],
    dict[str, set[str]],
    bool,
    str | None,
] | None:
    for _ in range(MAX_RETRIES):
        state = build_roundabout_scenario(NUM_VEHICLES)

        # Randomize which vehicle is circulating.
        inside_idx = random.randrange(NUM_VEHICLES)
        for i, v in enumerate(state.vehicles):
            if i == inside_idx:
                v.inside_intersection = True
                v.position = "roundabout_lane"
            else:
                v.inside_intersection = False
                v.position = f"{v.direction.value}_approach"
                v.intent = None

        derived = _derive_priority_structure(state)
        if derived is None:
            continue
        return (state, *derived)
    return None


def _has_right_turn_vs_left_turn_conflict(
    state: ScenarioState,
    pair_assessments: list[PairAssessment],
) -> bool:
    """True when at least one conflicting pair is right-turn vs left-turn."""
    by_id = {v.id: v for v in state.vehicles}
    for p in pair_assessments:
        if not p.conflict:
            continue
        i1 = by_id[p.v1].intent
        i2 = by_id[p.v2].intent
        if {i1, i2} == {IntentDirection.TURN_LEFT, IntentDirection.TURN_RIGHT}:
            return True
    return False


def _has_opposite_straight_pair(state: ScenarioState) -> bool:
    by_dir: dict[Direction, Vehicle] = {v.direction: v for v in state.vehicles}
    for v in state.vehicles:
        if v.intent != IntentDirection.GO_STRAIGHT:
            continue
        opp_dir = Direction(_OPPOSITE_DIR[v.direction.value])
        ov = by_dir.get(opp_dir)
        if ov is None:
            continue
        if ov.intent == IntentDirection.GO_STRAIGHT:
            return True
    return False


def _derive_no_clear_structure(
    state: ScenarioState,
    require_all_straight: bool = False,
    require_right_left_conflict: bool = False,
    forbid_opposite_straight_pair: bool = False,
) -> tuple[list[PairAssessment], dict[str, set[str]], dict[str, set[str]]] | None:
    """
    Validate and derive the 4-way cyclic intersection structure where no unique
    global priority exists.
    """
    if state.environment != Environment.INTERSECTION:
        return None
    if len(state.vehicles) not in (3, 4):
        return None

    # Fixed no-clear scaffold: all sampled approaches are distinct.
    dirs = {v.direction for v in state.vehicles}
    if len(dirs) != len(state.vehicles):
        return None
    if require_all_straight and any(v.intent != IntentDirection.GO_STRAIGHT for v in state.vehicles):
        return None

    vids = [v.id for v in state.vehicles]
    wins: dict[str, set[str]] = {vid: set() for vid in vids}
    conflicts: dict[str, set[str]] = {vid: set() for vid in vids}
    pair_assessments: list[PairAssessment] = []

    for i in range(len(state.vehicles)):
        for j in range(i + 1, len(state.vehicles)):
            p = _pair_assessment(state.vehicles[i], state.vehicles[j], state.environment)
            if p is None:
                return None
            pair_assessments.append(p)
            if not p.conflict or p.winner is None:
                continue
            a, b = p.v1, p.v2
            conflicts[a].add(b)
            conflicts[b].add(a)
            loser = b if p.winner == a else a
            wins[p.winner].add(loser)

    dominant = [
        vid for vid in vids
        if conflicts[vid] and wins[vid] == conflicts[vid] and len(conflicts[vid]) == len(vids) - 1
    ]
    if dominant:
        return None

    if require_all_straight:
        # In the 4-way straight cyclic case every vehicle should conflict with
        # two adjacent vehicles, creating no global dominant winner.
        if len(state.vehicles) == 4 and any(len(conflicts[vid]) != 2 for vid in vids):
            return None
    if require_right_left_conflict and not _has_right_turn_vs_left_turn_conflict(
        state, pair_assessments
    ):
        return None
    if forbid_opposite_straight_pair and _has_opposite_straight_pair(state):
        return None

    return pair_assessments, wins, conflicts


def _build_intersection_no_clear_scenario() -> tuple[
    ScenarioState,
    list[PairAssessment],
    dict[str, set[str]],
    dict[str, set[str]],
] | None:
    return _build_intersection_no_clear_scenario_for_subtype("straight")


def _build_intersection_no_clear_scenario_for_subtype(
    subtype: str,
) -> tuple[
    ScenarioState,
    list[PairAssessment],
    dict[str, set[str]],
    dict[str, set[str]],
] | None:
    for _ in range(MAX_RETRIES):
        if subtype == "threeway":
            state = build_intersection_scenario(3, with_intent=True)
        else:
            state = build_intersection_scenario(4, with_intent=True)

        if subtype in {"straight", "threeway"}:
            for v in state.vehicles:
                v.intent = IntentDirection.GO_STRAIGHT

        # Shuffle label-role mapping to avoid fixed id->direction shortcuts.
        labels = ["A", "B", "C", "D"] if len(state.vehicles) == 4 else ["A", "B", "C"]
        random.shuffle(labels)
        old_to_new = {v.id: labels[i] for i, v in enumerate(state.vehicles)}
        for v in state.vehicles:
            v.id = old_to_new[v.id]

        if subtype == "straight":
            derived = _derive_no_clear_structure(
                state,
                require_all_straight=True,
                require_right_left_conflict=False,
            )
        elif subtype == "turnmix":
            derived = _derive_no_clear_structure(
                state,
                require_all_straight=False,
                require_right_left_conflict=True,
                forbid_opposite_straight_pair=True,
            )
        elif subtype == "threeway":
            derived = _derive_no_clear_structure(
                state,
                require_all_straight=True,
                require_right_left_conflict=False,
            )
        else:
            return None
        if derived is None:
            continue
        return state, *derived
    return None


def _enhance_scenario_text(
    base_text: str,
    env: Environment,
    context_usage: Counter[str],
    mode: str | None = None,
    commit: bool = True,
) -> tuple[str, str]:
    if mode == "intersection_no_clear_straight":
        pool = INTERSECTION_NO_CLEAR_STRAIGHT_CONTEXTS
    elif mode == "intersection_no_clear_turnmix":
        pool = INTERSECTION_NO_CLEAR_TURNMIX_CONTEXTS
    elif mode == "intersection_no_clear_threeway":
        pool = INTERSECTION_NO_CLEAR_3WAY_CONTEXTS
    else:
        pool = (
            INTERSECTION_CONTEXTS
            if env == Environment.INTERSECTION
            else ROUNDABOUT_CONTEXTS
        )
    context = _pick_least_used(pool, context_usage, commit=commit)
    return base_text + "\n" + context, context



def build_choices(
    priority_vid: str,
    yielding_vid: str,
    third_vid: str,
    env: Environment,
    hf_usage: Counter[str],
    mode: str = "priority",
    commit: bool = True,
    state: ScenarioState | None = None,
    pair_assessments: list[PairAssessment] | None = None,
) -> dict[str, dict[str, str]]:
    """
    Returns exactly 5 typed options.

    Mode "priority":
      - 1 correct vehicle + 2 near_true + 2 highly_false
      - always includes "Both can pass at the same time" (typed by conflict count)

    Mode "intersection_no_clear":
      - correct = NO_CLEAR_OPTION_TEXT
      - all four vehicle labels are distractors
    """
    if mode.startswith("intersection_no_clear"):
        if state is None or len(state.vehicles) not in (3, 4):
            raise ValueError("intersection_no_clear mode requires a 3- or 4-vehicle state")
        vids = sorted(v.id for v in state.vehicles)
        random.shuffle(vids)
        if len(vids) == 4:
            return {
                "correct": {"text": NO_CLEAR_OPTION_TEXT, "type": "correct"},
                "near_true_1": {"text": f"Vehicle {vids[0]}", "type": "near_true"},
                "near_true_2": {"text": f"Vehicle {vids[1]}", "type": "near_true"},
                "highly_false_1": {"text": f"Vehicle {vids[2]}", "type": "highly_false"},
                "highly_false_2": {"text": f"Vehicle {vids[3]}", "type": "highly_false"},
            }
        return {
            "correct": {"text": NO_CLEAR_OPTION_TEXT, "type": "correct"},
            "near_true_1": {"text": f"Vehicle {vids[0]}", "type": "near_true"},
            "near_true_2": {"text": f"Vehicle {vids[1]}", "type": "near_true"},
            "highly_false_1": {"text": f"Vehicle {vids[2]}", "type": "highly_false"},
            "highly_false_2": {"text": NO_CLEAR_EXTRA_HF_TEXT, "type": "highly_false"},
        }

    scenario_hf_pool: list[str] = []
    active_conflicts = (
        sum(1 for p in pair_assessments if p.conflict)
        if pair_assessments is not None
        else 0
    )
    both_type = "near_true" if active_conflicts == 1 else "highly_false"

    if state is not None and pair_assessments is not None:
        if env == Environment.INTERSECTION:
            scenario_hf_pool = _build_intersection_distractors(
                state, priority_vid, yielding_vid, third_vid, pair_assessments
            )
        elif env == Environment.ROUNDABOUT:
            scenario_hf_pool = _build_roundabout_distractors(
                state, priority_vid, yielding_vid, third_vid
            )

    policy_pool = [t for t in scenario_hf_pool if t != BOTH_OPTION_TEXT]
    if not policy_pool:
        policy_pool = [t for t in _GENERIC_HF_FALLBACK[env] if t != BOTH_OPTION_TEXT]

    policy_text = _pick_least_used(policy_pool, hf_usage, commit=commit)

    if both_type == "highly_false":
        return {
            "correct": {"text": f"Vehicle {priority_vid}", "type": "correct"},
            "near_true_1": {"text": f"Vehicle {yielding_vid}", "type": "near_true"},
            "near_true_2": {"text": f"Vehicle {third_vid}", "type": "near_true"},
            "highly_false_1": {"text": BOTH_OPTION_TEXT, "type": "highly_false"},
            "highly_false_2": {"text": policy_text, "type": "highly_false"},
        }

    # Rare branch for single-conflict scenarios: keep 2/2 type balance.
    return {
        "correct": {"text": f"Vehicle {priority_vid}", "type": "correct"},
        "near_true_1": {"text": f"Vehicle {yielding_vid}", "type": "near_true"},
        "near_true_2": {"text": BOTH_OPTION_TEXT, "type": "near_true"},
        "highly_false_1": {"text": f"Vehicle {third_vid}", "type": "highly_false"},
        "highly_false_2": {"text": policy_text, "type": "highly_false"},
    }



def assign_letters(
    choices_dict: dict[str, dict[str, str]],
    correct_key: str,
) -> tuple[dict[str, str], dict[str, str], str]:
    """
    Shuffles options and places the correct one at correct_key.
    """
    items = list(choices_dict.values())
    random.shuffle(items)

    target_idx = LETTERS.index(correct_key)
    correct_item = next(it for it in items if it["type"] == "correct")
    items.remove(correct_item)
    items.insert(target_idx, correct_item)

    choices: dict[str, str] = {}
    distractor_type: dict[str, str] = {}
    for letter, item in zip(LETTERS, items):
        choices[letter] = item["text"]
        if item["type"] != "correct":
            distractor_type[letter] = item["type"]

    return choices, distractor_type, correct_key


def _place_both_option_letter(
    choices: dict[str, str],
    distractor_type: dict[str, str],
    *,
    correct_key: str,
    preferred_letter: str | None,
) -> None:
    if preferred_letter is None or preferred_letter == correct_key:
        return
    current_letter = None
    for letter, text in choices.items():
        if text == BOTH_OPTION_TEXT:
            current_letter = letter
            break
    if current_letter is None or current_letter == preferred_letter:
        return
    if preferred_letter == correct_key:
        return
    choices[current_letter], choices[preferred_letter] = (
        choices[preferred_letter],
        choices[current_letter],
    )
    distractor_type[current_letter], distractor_type[preferred_letter] = (
        distractor_type[preferred_letter],
        distractor_type[current_letter],
    )



def _parse_intent(value: str | None) -> IntentDirection | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    for it in IntentDirection:
        if it.value == normalized or it.name.lower() == normalized:
            return it
    raise ValueError(f"Unknown intent string: {value!r}")


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


def validate_example_contract(example: dict) -> tuple[bool, str]:
    choices = example["choices"]
    answer = example["answer"]
    answer_text = choices[answer]
    dtypes = example["distractor_type"]
    resolution = example.get("metadata", {}).get("resolution", "unique_priority")

    # V1: 5 unique options
    if len(set(choices.values())) != 5:
        return False, "duplicate choice texts"

    # V2: distractor type balance
    nt = sum(1 for t in dtypes.values() if t == "near_true")
    hf = sum(1 for t in dtypes.values() if t == "highly_false")
    if nt != 2 or hf != 2:
        return False, f"distractor type counts invalid: near_true={nt}, highly_false={hf}"

    # V3: exactly one vehicle option per vehicle id
    vehicle_ids = [v["id"] for v in example["scenario"]["vehicles"]]
    for vid in vehicle_ids:
        if sum(1 for t in choices.values() if t == f"Vehicle {vid}") != 1:
            return False, f"vehicle option count invalid for Vehicle {vid}"

    state = _reconstruct_state(example["scenario"])
    metadata = example.get("metadata", {})
    if resolution == "intersection_no_clear":
        if example["scenario"]["environment"] != Environment.INTERSECTION.value:
            return False, "no-clear examples must be intersections"
        if answer_text != NO_CLEAR_OPTION_TEXT:
            return False, "no-clear answer text mismatch"
        if sum(1 for t in choices.values() if t == NO_CLEAR_OPTION_TEXT) != 1:
            return False, "no-clear answer option missing"
        if sum(1 for t in choices.values() if t == BOTH_OPTION_TEXT) > 0:
            return False, "'Both can pass' must not appear in no-clear examples"
        subtype = metadata.get("no_clear_subtype")
        if subtype not in {"straight", "turnmix", "threeway"}:
            return False, "invalid no-clear subtype"
        no_clear = _derive_no_clear_structure(
            state,
            require_all_straight=(subtype == "straight"),
            require_right_left_conflict=(subtype == "turnmix"),
            forbid_opposite_straight_pair=(subtype == "turnmix"),
        )
        if no_clear is None:
            return False, "scenario is not a valid no-clear case"
        pair_assessments, _, _ = no_clear
        conflict_count = sum(1 for p in pair_assessments if p.conflict)
        min_conflicts = 2 if subtype == "threeway" else 4
        if conflict_count < min_conflicts:
            return False, "no-clear case has too few conflicts"
        if metadata.get("priority_vehicle") is not None:
            return False, "no-clear metadata.priority_vehicle must be null"
        if metadata.get("yielding_vehicle") is not None:
            return False, "no-clear metadata.yielding_vehicle must be null"
        if metadata.get("conflict_pair", []):
            return False, "no-clear metadata.conflict_pair must be empty"
        return True, "ok"

    # Unique-priority mode
    if sum(1 for t in choices.values() if t == BOTH_OPTION_TEXT) != 1:
        return False, "missing fixed 'Both can pass at the same time' option"

    # V4: independent recomputation from scenario JSON
    derived = _derive_priority_structure(state)
    if derived is None:
        return False, "scenario has no unique global priority vehicle"

    (
        priority_vid,
        yielding_vid,
        _,
        pair_assessments,
        _,
        conflicts,
        intent_sensitive_with_priority,
        direction_only_priority,
    ) = derived
    if answer_text != f"Vehicle {priority_vid}":
        return False, (
            f"answer text {answer_text!r} inconsistent with recomputed "
            f"priority Vehicle {priority_vid}"
        )

    if metadata.get("priority_vehicle") != priority_vid:
        return False, "metadata.priority_vehicle mismatch"
    if metadata.get("yielding_vehicle") != yielding_vid:
        return False, "metadata.yielding_vehicle mismatch"
    if sorted(metadata.get("conflict_pair", [])) != sorted([priority_vid, yielding_vid]):
        return False, "metadata.conflict_pair mismatch"

    # V5: priority must conflict with all others (global, not local)
    if len(conflicts[priority_vid]) != len(vehicle_ids) - 1:
        return False, "priority vehicle does not conflict with all other vehicles"

    # V6: at least 2 conflict pairs for non-triviality
    conflict_count = sum(1 for p in pair_assessments if p.conflict)
    if conflict_count < 2:
        return False, "too few conflict pairs"

    # V7: intersection examples must require intent-aware reasoning.
    if example["scenario"]["environment"] == Environment.INTERSECTION.value:
        if not intent_sensitive_with_priority:
            return False, "priority vehicle has no intent-sensitive pair"
        if direction_only_priority == priority_vid:
            return False, "direction-only heuristic still recovers priority"

    return True, "ok"



def generate_example(
    example_id: int,
    correct_key: str,
    seed: int | None,
    mode: str,
    desired_priority_label: str,
    desired_conflict_pair: str | None,
    preferred_both_letter: str | None,
    question_usage: Counter[str],
    context_usage: Counter[str],
    hf_usage: Counter[str],
    priority_label_usage: Counter[str],
) -> dict | None:
    for attempt in range(MAX_RETRIES):
        no_clear_subtype: str | None = None
        if mode in {
            "intersection_no_clear_straight",
            "intersection_no_clear_turnmix",
            "intersection_no_clear_threeway",
        }:
            env = Environment.INTERSECTION
            if mode == "intersection_no_clear_straight":
                subtype = "straight"
            elif mode == "intersection_no_clear_turnmix":
                subtype = "turnmix"
            else:
                subtype = "threeway"
            no_clear_subtype = subtype
            result_no_clear = _build_intersection_no_clear_scenario_for_subtype(subtype)
            question = _pick_least_used(
                INTERSECTION_NO_CLEAR_QUESTIONS,
                question_usage,
                commit=False,
            )
            if result_no_clear is None:
                continue
            state, pair_assessments, wins, conflicts = result_no_clear
            priority_vid = ""
            yielding_vid = ""
            third_vid = ""
            intent_sensitive_with_priority = True
            direction_only_priority = None
            resolution = "intersection_no_clear"
        elif mode == "intersection_priority":
            env = Environment.INTERSECTION
            result = _build_intersection_priority_scenario()
            question = _pick_least_used(
                INTERSECTION_QUESTIONS,
                question_usage,
                commit=False,
            )
            if result is None:
                continue
            (
                state,
                priority_vid,
                yielding_vid,
                third_vid,
                pair_assessments,
                wins,
                conflicts,
                intent_sensitive_with_priority,
                direction_only_priority,
            ) = result
            relabeled = _relabel_intersection_for_anti_shortcut(
                state,
                priority_label_usage,
                desired_priority_label=desired_priority_label,
                desired_conflict_pair=desired_conflict_pair,
            )
            if relabeled is None:
                continue
            state, relabeled_derived = relabeled
            (
                priority_vid,
                yielding_vid,
                third_vid,
                pair_assessments,
                wins,
                conflicts,
                intent_sensitive_with_priority,
                direction_only_priority,
            ) = relabeled_derived
            resolution = "unique_priority"
        elif mode == "roundabout_priority":
            env = Environment.ROUNDABOUT
            result = _build_roundabout_priority_scenario()
            question = _pick_least_used(
                ROUNDABOUT_QUESTIONS,
                question_usage,
                commit=False,
            )
            if result is None:
                continue
            (
                state,
                priority_vid,
                yielding_vid,
                third_vid,
                pair_assessments,
                wins,
                conflicts,
                intent_sensitive_with_priority,
                direction_only_priority,
            ) = result
            state = _relabel_roundabout_for_priority_balance(
                state,
                desired_priority_label=desired_priority_label,
            )
            relabeled_derived = _derive_priority_structure(state)
            if relabeled_derived is None:
                continue
            (
                priority_vid,
                yielding_vid,
                third_vid,
                pair_assessments,
                wins,
                conflicts,
                intent_sensitive_with_priority,
                direction_only_priority,
            ) = relabeled_derived
            resolution = "unique_priority"
        else:
            raise RuntimeError(f"unknown generation mode: {mode!r}")

        scenario_text, context_line = _enhance_scenario_text(
            describe_scenario(state),
            env,
            context_usage,
            mode=mode,
            commit=False,
        )
        raw_choices = build_choices(
            priority_vid,
            yielding_vid,
            third_vid,
            env,
            hf_usage,
            mode=mode,
            commit=False,
            state=state,
            pair_assessments=pair_assessments,
        )
        choices, distractor_type, answer = assign_letters(raw_choices, correct_key)
        _place_both_option_letter(
            choices,
            distractor_type,
            correct_key=correct_key,
            preferred_letter=preferred_both_letter,
        )

        parts = [scenario_text, "", f"Question: {question}"]
        for key in sorted(choices):
            parts.append(f"{key}) {choices[key]}")
        prompt = "\n".join(parts)

        pairwise = [
            {
                "pair": sorted([p.v1, p.v2]),
                "conflict": p.conflict,
                "winner": p.winner,
                "direction_only_winner": p.direction_only_winner,
            }
            for p in pair_assessments
        ]

        conflict_count = sum(1 for p in pair_assessments if p.conflict)

        example = {
            "id": f"task2_{example_id:04d}",
            "task": "right_of_way",
            "prompt": prompt,
            "scenario": {
                "vehicles": [
                    {
                        "id": v.id,
                        "position": v.position,
                        "direction": v.direction.value,
                        "intent": v.intent.value if v.intent else None,
                        "inside_intersection": bool(v.inside_intersection),
                    }
                    for v in state.vehicles
                ],
                "environment": env.value,
            },
            "question": question,
            "choices": choices,
            "answer": answer,
            "distractor_type": distractor_type,
            "metadata": {
                "num_vehicles": len(state.vehicles),
                "environment": env.value,
                "priority_vehicle": priority_vid if resolution == "unique_priority" else None,
                "yielding_vehicle": yielding_vid if resolution == "unique_priority" else None,
                "third_vehicle": third_vid if resolution == "unique_priority" else None,
                "conflict_pair": (
                    sorted([priority_vid, yielding_vid])
                    if resolution == "unique_priority"
                    else []
                ),
                "pair_conflict_count": conflict_count,
                "intent_sensitive_priority_pair": intent_sensitive_with_priority,
                "direction_only_priority": direction_only_priority,
                "difficulty": "strict" if resolution == "unique_priority" else "cyclic_no_clear",
                "resolution": resolution,
                "mode": mode,
                "no_clear_subtype": no_clear_subtype,
            },
            "audit": {
                "generator_version": GENERATOR_VERSION,
                "seed": seed,
                "attempt": attempt,
                "pairwise": pairwise,
                "wins": {vid: sorted(list(wins[vid])) for vid in sorted(wins)},
                "conflicts": {
                    vid: sorted(list(conflicts[vid]))
                    for vid in sorted(conflicts)
                },
                "invariants": {
                    "five_distinct_options": len({choices[l] for l in LETTERS}) == 5,
                    "priority_conflicts_with_all_others": (
                        resolution != "unique_priority"
                        or len(conflicts[priority_vid]) == len(state.vehicles) - 1
                    ),
                    "pair_conflict_count_at_least_2": (
                        conflict_count >= (
                            2
                            if no_clear_subtype == "threeway"
                            else (4 if resolution == "intersection_no_clear" else 2)
                        )
                    ),
                    "intent_sensitive_priority_pair": intent_sensitive_with_priority,
                    "direction_only_does_not_match_priority": (
                        resolution != "unique_priority"
                        or direction_only_priority != priority_vid
                    ),
                    "answer_text_matches_priority": (
                        choices[answer] == f"Vehicle {priority_vid}"
                        if resolution == "unique_priority"
                        else choices[answer] == NO_CLEAR_OPTION_TEXT
                    ),
                    "no_clear_is_ambiguous": (
                        resolution != "intersection_no_clear"
                        or _derive_no_clear_structure(
                            state,
                            require_all_straight=(no_clear_subtype == "straight"),
                            require_right_left_conflict=(no_clear_subtype == "turnmix"),
                            forbid_opposite_straight_pair=(no_clear_subtype == "turnmix"),
                        ) is not None
                    ),
                },
                "context_line": context_line,
            },
        }

        ok, _ = validate_example_contract(example)
        if not ok:
            continue
        if resolution == "unique_priority" and desired_conflict_pair:
            pair_key = "-".join(example["metadata"]["conflict_pair"])
            if pair_key != desired_conflict_pair:
                continue

        # Commit balanced-template counters only on accepted example.
        question_usage[question] += 1
        context_usage[context_line] += 1
        hf_usage[raw_choices["highly_false_1"]["text"]] += 1
        hf_usage[raw_choices["highly_false_2"]["text"]] += 1
        if resolution == "unique_priority":
            priority_label_usage[priority_vid] += 1

        return example

    return None



def _max_answer_run(examples: list[dict]) -> int:
    prev: str | None = None
    run = 0
    best = 0
    for ex in examples:
        answer = ex["answer"]
        if answer == prev:
            run += 1
        else:
            prev = answer
            run = 1
        if run > best:
            best = run
    return best


def _reorder_examples_by_answer_run(examples: list[dict], max_run: int = 4) -> list[dict]:
    if not examples:
        return []

    by_letter: dict[str, list[dict]] = {letter: [] for letter in LETTERS}
    for ex in examples:
        by_letter[ex["answer"]].append(ex)

    remaining = {letter: len(by_letter[letter]) for letter in LETTERS}
    ordered: list[dict] = []
    prev: str | None = None
    current_run = 0

    def feasible_after_pick(counts: dict[str, int]) -> bool:
        total = sum(counts.values())
        if total == 0:
            return True
        max_count = max(counts.values())
        others = total - max_count
        return max_count <= max_run * (others + 1)

    while len(ordered) < len(examples):
        candidates: list[str] = []
        for letter in LETTERS:
            if remaining[letter] <= 0:
                continue
            if letter == prev and current_run >= max_run:
                continue
            trial = dict(remaining)
            trial[letter] -= 1
            if feasible_after_pick(trial):
                candidates.append(letter)

        if not candidates:
            raise RuntimeError(f"could not reorder Task 2 answers with max run <= {max_run}")

        candidates.sort(key=lambda letter: (-remaining[letter], letter == prev, letter))
        chosen = candidates[0]

        ordered.append(by_letter[chosen].pop(0))
        remaining[chosen] -= 1

        if chosen == prev:
            current_run += 1
        else:
            prev = chosen
            current_run = 1

    return ordered


def generate_task2(n: int, output_path: str, seed: int | None = DEFAULT_SEED) -> None:
    if seed is not None:
        random.seed(seed)

    if n % 5 != 0:
        raise ValueError("N must be a multiple of 5 for balanced key schedule.")

    key_schedule: list[str] = []
    per_key = n // 5
    for letter in LETTERS:
        key_schedule.extend([letter] * per_key)
    random.shuffle(key_schedule)

    env_schedule = _build_env_schedule(n)
    mode_schedule = _build_mode_schedule(n, env_schedule)
    priority_schedule = _build_priority_schedule(mode_schedule)
    n_intersection_priority = sum(1 for m in mode_schedule if m == "intersection_priority")
    n_roundabout_priority = sum(1 for m in mode_schedule if m == "roundabout_priority")
    intersection_pair_targets = _build_pair_targets(n_intersection_priority, PAIR_KEYS)
    # Under current roundabout tie-break semantics, only these two pairs are reachable.
    roundabout_pair_keys = ["A-B", "A-C"]
    roundabout_pair_targets = _build_pair_targets(n_roundabout_priority, roundabout_pair_keys)
    pair_usage_by_env: dict[str, Counter[str]] = {
        Environment.INTERSECTION.value: Counter(),
        Environment.ROUNDABOUT.value: Counter(),
    }
    question_usage: Counter[str] = Counter()
    context_usage: Counter[str] = Counter()
    hf_usage: Counter[str] = Counter()
    priority_label_usage: Counter[str] = Counter()
    both_option_letter_usage: Counter[str] = Counter()

    examples: list[dict] = []
    seen_prompts: set[str] = set()
    for idx in range(n):
        mode = mode_schedule[idx]
        desired_priority_label = priority_schedule[idx]
        if not mode.startswith("intersection_no_clear") and desired_priority_label not in PRIORITY_LABELS:
            raise RuntimeError(
                f"invalid desired priority label at idx={idx}: {desired_priority_label!r}"
            )
        ex = None
        if mode == "roundabout_priority":
            priority_candidates = sorted(
                PRIORITY_LABELS,
                key=lambda lbl: (
                    pair_usage_by_env[Environment.ROUNDABOUT.value][
                        _roundabout_conflict_pair_from_priority_label(lbl)
                    ] - roundabout_pair_targets.get(
                        _roundabout_conflict_pair_from_priority_label(lbl), 0
                    ),
                    pair_usage_by_env[Environment.ROUNDABOUT.value][
                        _roundabout_conflict_pair_from_priority_label(lbl)
                    ],
                    lbl != desired_priority_label,
                    lbl,
                ),
            )
        else:
            priority_candidates = [desired_priority_label]
            if not mode.startswith("intersection_no_clear"):
                for alt in PRIORITY_LABELS:
                    if alt not in priority_candidates:
                        priority_candidates.append(alt)
        for _ in range(MAX_RETRIES):
            for wanted_priority in priority_candidates:
                desired_conflict_pair: str | None = None
                if mode == "intersection_priority":
                    compatible_pairs = [
                        pair for pair in PAIR_KEYS
                        if wanted_priority in pair.split("-")
                    ]
                    desired_conflict_pair = _pick_underused_pair(
                        pair_usage_by_env[Environment.INTERSECTION.value],
                        intersection_pair_targets,
                        compatible_pairs,
                    )
                elif mode == "roundabout_priority":
                    desired_conflict_pair = _roundabout_conflict_pair_from_priority_label(
                        wanted_priority
                    )
                eligible_both_letters = [
                    letter for letter in LETTERS
                    if letter != key_schedule[idx]
                ]
                preferred_both_letter = min(
                    eligible_both_letters,
                    key=lambda letter: (both_option_letter_usage[letter], letter),
                )
                candidate = generate_example(
                    idx,
                    key_schedule[idx],
                    seed,
                    mode,
                    wanted_priority,
                    desired_conflict_pair,
                    preferred_both_letter,
                    question_usage,
                    context_usage,
                    hf_usage,
                    priority_label_usage,
                )
                if candidate is None:
                    continue
                if candidate["prompt"] in seen_prompts:
                    continue
                if candidate["metadata"]["resolution"] == "unique_priority":
                    env_name = candidate["metadata"]["environment"]
                    pair_list = candidate["metadata"].get("conflict_pair", [])
                    if isinstance(pair_list, list) and len(pair_list) == 2:
                        pair_key = "-".join(sorted(pair_list))
                        if env_name == Environment.INTERSECTION.value:
                            over_cap = (
                                pair_usage_by_env[env_name][pair_key]
                                >= intersection_pair_targets.get(pair_key, 0) + 2
                            )
                            if over_cap:
                                continue
                        elif env_name == Environment.ROUNDABOUT.value:
                            target = roundabout_pair_targets.get(pair_key, 0)
                            over_cap = pair_usage_by_env[env_name][pair_key] >= target + 2
                            if over_cap:
                                continue
                ex = candidate
                seen_prompts.add(candidate["prompt"])
                both_letter = next(
                    (
                        letter
                        for letter, text in ex["choices"].items()
                        if text == BOTH_OPTION_TEXT
                    ),
                    None,
                )
                if both_letter is not None:
                    both_option_letter_usage[both_letter] += 1
                if ex["metadata"]["resolution"] == "unique_priority":
                    env_name = ex["metadata"]["environment"]
                    pair_list = ex["metadata"].get("conflict_pair", [])
                    if isinstance(pair_list, list) and len(pair_list) == 2:
                        pair_key = "-".join(sorted(pair_list))
                        pair_usage_by_env[env_name][pair_key] += 1
                break
            if ex is not None:
                break
        if ex is None:
            raise RuntimeError(
                f"could not generate strict Task 2 example {idx} "
                f"(mode={mode}, priority={desired_priority_label}) "
                f"after {MAX_RETRIES} retries"
            )
        examples.append(ex)

    examples = _reorder_examples_by_answer_run(examples, max_run=4)

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
        os.replace(tmp_path, out_path)
    finally:
        # If os.replace succeeded tmp_path no longer exists; if it failed, clean up.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(
        f"Saved {len(examples)} examples to {out_path.resolve()} "
        f"({out_path.stat().st_size} bytes)"
    )
    if seed is None:
        print("Generation mode: non-deterministic (no seed provided).")
    else:
        print(f"Generation mode: deterministic (seed={seed}).")

    answer_counts = {letter: 0 for letter in LETTERS}
    for ex in examples:
        answer_counts[ex["answer"]] += 1

    print("\nAnswer distribution:")
    for letter in LETTERS:
        print(f"  {letter}: {answer_counts[letter]}")
    print(f"  max_run: {_max_answer_run(examples)}")

    env_counts: dict[str, int] = {}
    for ex in examples:
        env = ex["metadata"]["environment"]
        env_counts[env] = env_counts.get(env, 0) + 1
    print("\nEnvironment distribution:")
    for env, c in sorted(env_counts.items()):
        print(f"  {env}: {c}")

    print("\nTemplate usage:")
    print(f"  Unique questions used: {len(question_usage)}")
    print(f"  Unique contexts used: {len(context_usage)}")

    pv_counts: dict[str, int] = {}
    for ex in examples:
        pv = ex["metadata"].get("priority_vehicle")
        if pv is None:
            continue
        pv_counts[pv] = pv_counts.get(pv, 0) + 1
    print("\nPriority vehicle distribution:")
    for vid, c in sorted(pv_counts.items()):
        print(f"  Vehicle {vid}: {c}")

    conflict_counts: dict[int, int] = {}
    for ex in examples:
        c = ex["metadata"].get("pair_conflict_count", 0)
        conflict_counts[c] = conflict_counts.get(c, 0) + 1
    print("\nPair-conflict-count distribution:")
    for c, n_examples in sorted(conflict_counts.items()):
        print(f"  {c}: {n_examples}")

    mode_counts: dict[str, int] = {}
    for ex in examples:
        mode = ex.get("metadata", {}).get("mode", "<missing-mode>")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    print("\nMode distribution:")
    for mode, c in sorted(mode_counts.items()):
        print(f"  {mode}: {c}")

    pair_counts: dict[str, int] = {}
    for ex in examples:
        pair_list = ex["metadata"].get("conflict_pair", [])
        if isinstance(pair_list, list) and len(pair_list) == 2:
            k = "-".join(sorted(pair_list))
            pair_counts[k] = pair_counts.get(k, 0) + 1
    print("\nConflict-pair distribution:")
    for pair in PAIR_KEYS:
        print(f"  {pair}: {pair_counts.get(pair, 0)}")

    correct_text_counts: dict[str, int] = {}
    for ex in examples:
        t = ex["choices"][ex["answer"]]
        correct_text_counts[t] = correct_text_counts.get(t, 0) + 1
    top_text = max(correct_text_counts.values()) if correct_text_counts else 0
    if top_text > int(0.55 * n):
        raise RuntimeError("overuse of one literal correct answer detected")


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 2 right-of-way generator")
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
        help="RNG seed for deterministic generation (default: random run each time)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "core" / "task2_rightofway.jsonl"),
        help="output JSONL path",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    generate_task2(args.n, args.out, args.seed)
