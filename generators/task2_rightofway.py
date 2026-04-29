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



GENERATOR_VERSION = "task2_rightofway_v3"
DEFAULT_N_EXAMPLES = 100
DEFAULT_SEED: int | None = None

NUM_VEHICLES = 3
MAX_RETRIES = 200

# Keep enough intersections for intent-sensitive reasoning while avoiding a
# correct-vehicle shortcut: strict intersection cases mostly force Vehicle C.
INTERSECTION_TARGET_SHARE = 0.34

LETTERS = ["A", "B", "C", "D", "E"]
PRIORITY_LABELS = ["A", "B", "C"]
ROUNDABOUT_PRIORITY_LABEL = "A"
PAIR_KEYS = ["A-B", "A-C", "B-C"]
BOTH_OPTION_TEXT = "Both can pass at the same time"

INTERSECTION_QUESTIONS = [
    "Which vehicle has the right of way before the others?",
    "Which vehicle should be allowed to pass first?",
    "According to traffic rules, which vehicle has priority now?",
    "Which vehicle has priority in this situation?",
    "Which vehicle should proceed first under right-of-way rules?",
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
) -> tuple[ScenarioState, PriorityDerivation] | None:
    """
    Finds an ID relabeling (A/B/C permutation) that:
      1) keeps the scenario valid under strict derive logic
      2) enforces the requested priority label
      3) rejects alphabetical-non-left shortcut wins
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
        if priority_vid != desired_priority_label:
            continue
        if _alphabetical_non_left_heuristic(trial) == priority_vid:
            continue

        score = (
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


def _build_pair_schedule(env_schedule: list[Environment]) -> list[str]:
    """Build the target conflict-pair schedule."""
    n = len(env_schedule)
    if n == 0:
        return []

    intersection_indices = [
        i for i, env in enumerate(env_schedule) if env == Environment.INTERSECTION
    ]
    roundabout_indices = [
        i for i, env in enumerate(env_schedule) if env == Environment.ROUNDABOUT
    ]
    n_int = len(intersection_indices)
    n_rnd = len(roundabout_indices)

    # Balanced global thirds, with deterministic remainder assignment.
    targets = {
        "A-B": n // 3,
        "A-C": n // 3,
        "B-C": n // 3,
    }
    remainder = n - 3 * (n // 3)
    for key in ("B-C", "A-B", "A-C")[:remainder]:
        targets[key] += 1

    # Roundabout slots always materialize as A-B in this strict generator.
    if targets["A-B"] < n_rnd:
        deficit = n_rnd - targets["A-B"]
        targets["A-B"] += deficit
        # Pull quota from A-C first, then B-C while keeping a safety floor.
        min_bc_goal = 15 if n >= 45 and n_int >= 15 else 0
        take_ac = min(deficit, targets["A-C"])
        targets["A-C"] -= take_ac
        deficit -= take_ac
        if deficit > 0:
            take_bc = max(0, targets["B-C"] - min_bc_goal)
            take_bc = min(deficit, take_bc)
            targets["B-C"] -= take_bc
            deficit -= take_bc
        if deficit > 0:
            raise RuntimeError(
                "pair schedule infeasible after enforcing roundabout A-B floor"
            )

    # Enforce B-C feasibility and floor for larger datasets.
    if targets["B-C"] > n_int:
        overflow = targets["B-C"] - n_int
        targets["B-C"] = n_int
        targets["A-C"] += overflow
    if n >= 45 and n_int >= 15 and targets["B-C"] < 15:
        deficit = 15 - targets["B-C"]
        if targets["A-C"] < deficit:
            raise RuntimeError(
                "cannot enforce minimum B-C quota without violating constraints"
            )
        targets["A-C"] -= deficit
        targets["B-C"] += deficit

    rnd_ab = n_rnd
    rnd_ac = 0

    int_ab = targets["A-B"] - rnd_ab
    int_ac = targets["A-C"] - rnd_ac
    int_bc = targets["B-C"]

    if min(int_ab, int_ac, int_bc) < 0:
        raise RuntimeError("negative pair quota generated")
    if int_ab + int_ac + int_bc != n_int:
        raise RuntimeError("intersection pair quota does not match intersection slots")

    pair_schedule = [""] * n

    int_pairs = ["A-B"] * int_ab + ["A-C"] * int_ac + ["B-C"] * int_bc
    rnd_pairs = ["A-B"] * rnd_ab + ["A-C"] * rnd_ac
    random.shuffle(int_pairs)
    random.shuffle(rnd_pairs)

    for idx, pair in zip(intersection_indices, int_pairs):
        pair_schedule[idx] = pair
    for idx, pair in zip(roundabout_indices, rnd_pairs):
        pair_schedule[idx] = pair

    if any(pair not in PAIR_KEYS for pair in pair_schedule):
        raise RuntimeError("pair schedule contains invalid entries")
    return pair_schedule


def _build_priority_schedule(
    env_schedule: list[Environment],
    pair_schedule: list[str],
) -> list[str]:
    """
    Build desired priority labels compatible with (env, expected pair).

    The priority label must be one of the two labels in the scheduled conflict
    pair. Pick the least-used compatible label per slot so the final correct
    vehicle text is not dominated by one vehicle.
    """
    if len(env_schedule) != len(pair_schedule):
        raise ValueError("env_schedule and pair_schedule length mismatch")

    labels = [""] * len(env_schedule)
    usage: Counter[str] = Counter()

    for i, (env, pair) in enumerate(zip(env_schedule, pair_schedule)):
        if pair not in PAIR_KEYS:
            raise RuntimeError(f"invalid pair key at {i}: {pair!r}")
        if env == Environment.ROUNDABOUT and pair != "A-B":
            raise RuntimeError(
                f"roundabout slot {i} has unsupported pair {pair!r}"
            )
        possible = pair.split("-")
        label = min(possible, key=lambda candidate: (usage[candidate], candidate))
        labels[i] = label
        usage[label] += 1

    if any(label not in PRIORITY_LABELS for label in labels):
        raise RuntimeError("priority schedule contains invalid entries")
    return labels



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


def _enhance_scenario_text(
    base_text: str,
    env: Environment,
    context_usage: Counter[str],
    commit: bool = True,
) -> tuple[str, str]:
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
    commit: bool = True,
    state: ScenarioState | None = None,
    pair_assessments: list[PairAssessment] | None = None,
) -> dict[str, dict[str, str]]:
    """
    Returns exactly 5 typed options: 1 correct + 2 near_true + 2 highly_false.

    Always includes "Both can pass at the same time":
      - near_true when exactly one active conflict pair exists
      - highly_false otherwise

    When *state* and *pair_assessments* are provided, the two highly_false
    policy distractors are built from actual scenario directions/intents
    rather than falling back to generic pool statements.
    """
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

    # V1: 5 unique options
    if len(set(choices.values())) != 5:
        return False, "duplicate choice texts"
    if sum(1 for t in choices.values() if t == BOTH_OPTION_TEXT) != 1:
        return False, "missing fixed 'Both can pass at the same time' option"

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

    # V4: independent recomputation from scenario JSON
    state = _reconstruct_state(example["scenario"])
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

    metadata = example.get("metadata", {})
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
        if _alphabetical_non_left_heuristic(state) == priority_vid:
            return False, "alphabetical non-left heuristic still recovers priority"

    return True, "ok"



def generate_example(
    example_id: int,
    correct_key: str,
    seed: int | None,
    env_hint: Environment,
    desired_priority_label: str,
    question_usage: Counter[str],
    context_usage: Counter[str],
    hf_usage: Counter[str],
    priority_label_usage: Counter[str],
) -> dict | None:
    for attempt in range(MAX_RETRIES):
        env = env_hint

        if env == Environment.INTERSECTION:
            result = _build_intersection_priority_scenario()
            question = _pick_least_used(
                INTERSECTION_QUESTIONS,
                question_usage,
                commit=False,
            )
        else:
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

        if env == Environment.INTERSECTION:
            relabeled = _relabel_intersection_for_anti_shortcut(
                state,
                priority_label_usage,
                desired_priority_label=desired_priority_label,
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
        else:
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

        scenario_text, context_line = _enhance_scenario_text(
            describe_scenario(state),
            env,
            context_usage,
            commit=False,
        )
        raw_choices = build_choices(
            priority_vid,
            yielding_vid,
            third_vid,
            env,
            hf_usage,
            commit=False,
            state=state,
            pair_assessments=pair_assessments,
        )
        choices, distractor_type, answer = assign_letters(raw_choices, correct_key)

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
                "num_vehicles": NUM_VEHICLES,
                "environment": env.value,
                "priority_vehicle": priority_vid,
                "yielding_vehicle": yielding_vid,
                "third_vehicle": third_vid,
                "conflict_pair": sorted([priority_vid, yielding_vid]),
                "pair_conflict_count": conflict_count,
                "intent_sensitive_priority_pair": intent_sensitive_with_priority,
                "direction_only_priority": direction_only_priority,
                "difficulty": "strict",
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
                        len(conflicts[priority_vid]) == NUM_VEHICLES - 1
                    ),
                    "pair_conflict_count_at_least_2": conflict_count >= 2,
                    "intent_sensitive_priority_pair": intent_sensitive_with_priority,
                    "direction_only_does_not_match_priority": (
                        direction_only_priority != priority_vid
                    ),
                    "answer_text_matches_priority": (
                        choices[answer] == f"Vehicle {priority_vid}"
                    ),
                    "alphabetical_non_left_heuristic_fails": (
                        env != Environment.INTERSECTION
                        or _alphabetical_non_left_heuristic(state) != priority_vid
                    ),
                },
                "context_line": context_line,
            },
        }

        ok, _ = validate_example_contract(example)
        if not ok:
            continue

        # Commit balanced-template counters only on accepted example.
        question_usage[question] += 1
        context_usage[context_line] += 1
        hf_usage[raw_choices["highly_false_1"]["text"]] += 1
        hf_usage[raw_choices["highly_false_2"]["text"]] += 1
        priority_label_usage[priority_vid] += 1

        return example

    return None



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
    pair_schedule = _build_pair_schedule(env_schedule)
    priority_schedule = _build_priority_schedule(env_schedule, pair_schedule)
    question_usage: Counter[str] = Counter()
    context_usage: Counter[str] = Counter()
    hf_usage: Counter[str] = Counter()
    priority_label_usage: Counter[str] = Counter()

    examples: list[dict] = []
    seen_prompts: set[str] = set()
    for idx in range(n):
        env_hint = env_schedule[idx]
        expected_pair = pair_schedule[idx]
        desired_priority_label = priority_schedule[idx]
        if desired_priority_label not in PRIORITY_LABELS:
            raise RuntimeError(
                f"invalid desired priority label at idx={idx}: {desired_priority_label!r}"
            )
        ex = None
        priority_candidates = [desired_priority_label]
        for alt in expected_pair.split("-"):
            if alt != desired_priority_label:
                priority_candidates.append(alt)

        for _ in range(MAX_RETRIES):
            for wanted_priority in priority_candidates:
                candidate = generate_example(
                    idx,
                    key_schedule[idx],
                    seed,
                    env_hint,
                    wanted_priority,
                    question_usage,
                    context_usage,
                    hf_usage,
                    priority_label_usage,
                )
                if candidate is None:
                    continue
                cand_pair = candidate["metadata"].get("conflict_pair")
                if not isinstance(cand_pair, list) or len(cand_pair) != 2:
                    continue
                candidate_pair = "-".join(sorted(cand_pair))
                if candidate_pair != expected_pair:
                    continue
                if candidate["prompt"] in seen_prompts:
                    continue
                ex = candidate
                seen_prompts.add(candidate["prompt"])
                break
            if ex is not None:
                break
        if ex is None:
            raise RuntimeError(
                f"could not generate strict Task 2 example {idx} "
                f"(env={env_hint.value}, pair={expected_pair}, "
                f"priority={desired_priority_label}) after {MAX_RETRIES} retries"
            )
        examples.append(ex)

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
        pv = ex["metadata"]["priority_vehicle"]
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

    pair_counts: dict[str, int] = {}
    for ex in examples:
        pair_list = ex["metadata"].get("conflict_pair", [])
        if isinstance(pair_list, list) and len(pair_list) == 2:
            k = "-".join(sorted(pair_list))
            pair_counts[k] = pair_counts.get(k, 0) + 1
    print("\nConflict-pair distribution:")
    for pair in PAIR_KEYS:
        print(f"  {pair}: {pair_counts.get(pair, 0)}")


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
