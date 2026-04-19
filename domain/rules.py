"""
Right-of-way and violation rules.

This module hosts the canonical rule functions used by Task 2 generators
and by the independent Task 2 validator. It has two public entry points:

  • right_of_way(v1, v2, env)
        Dispatcher. Returns the id of the vehicle with priority, or None
        for "no conflict". Raises UnsupportedScenarioError for scenarios
        the ruleset cannot answer (e.g. a roundabout with no vehicle
        inside — previously silently misrouted to intersection logic,
        closes T2-B10).

  • right_of_way_intersection_with_intent(v1, v2)
        Intent-aware intersection rule. Uses the 4-cell trajectory model
        from domain.trajectory to detect actual path conflicts and
        applies:
          1. left-turn-yields-to-oncoming, then
          2. priority-to-the-right
        Callers that have declared intent for both vehicles should prefer
        this over right_of_way_intersection.

The old direction-only helpers (right_of_way_intersection,
right_of_way_roundabout) are preserved for backward compatibility with
callers that do not use intent (Task 1 and legacy code).
"""
from __future__ import annotations

from typing import Optional

from .entities import (
    Direction,
    Environment,
    IntentDirection,
    UnsupportedScenarioError,
    Vehicle,
)
from .trajectory import (
    both_have_intent,
    trajectories_conflict,
    trajectory_of,
)


# ── Right-of-way at intersections (priority-to-the-right rule) ──────────────

APPROACH_PRIORITY = {
    (Direction.NORTH, Direction.EAST):  "B",
    (Direction.NORTH, Direction.WEST):  "A",
    (Direction.SOUTH, Direction.WEST):  "B",
    (Direction.SOUTH, Direction.EAST):  "A",
    (Direction.EAST,  Direction.SOUTH): "B",
    (Direction.EAST,  Direction.NORTH): "A",
    (Direction.WEST,  Direction.NORTH): "B",
    (Direction.WEST,  Direction.SOUTH): "A",
}


def right_of_way_intersection(v1: Vehicle, v2: Vehicle) -> Optional[str]:
    """
    Direction-only right-of-way at a 4-way intersection (priority-to-the-right).

    Backward-compat helper for callers that have no intent information.
    Returns None when directions are opposite — this is the correct answer
    only when BOTH vehicles go straight. Callers with intent should use
    `right_of_way_intersection_with_intent`.
    """
    key = (v1.direction, v2.direction)
    result = APPROACH_PRIORITY.get(key)
    if result == "A":
        return v1.id
    elif result == "B":
        return v2.id
    return None


def right_of_way_intersection_with_intent(
    v1: Vehicle, v2: Vehicle
) -> Optional[str]:
    """
    Intent-aware intersection right-of-way.

    Precondition: both vehicles have declared intent (v.intent is not None).
    Raises UnsupportedScenarioError otherwise.

    Resolution order:
      1. If the two trajectories do NOT share any intersection cell, there
         is no conflict → returns None.
      2. Otherwise, apply yield-to-oncoming-when-turning-left:
         if exactly one vehicle is turning left and the vehicles approach
         from opposite directions, the left-turner yields (closes T2-B01).
      3. Otherwise, apply priority-to-the-right via APPROACH_PRIORITY.
         If that lookup is also undefined (same direction, identical
         approach — not a realistic 2-vehicle scenario), raise
         UnsupportedScenarioError rather than return None, since we know
         from step 1 that there IS a trajectory conflict.
    """
    if not both_have_intent(v1, v2):
        raise UnsupportedScenarioError(
            "right_of_way_intersection_with_intent requires both vehicles "
            "to have a declared intent."
        )

    if not trajectories_conflict(v1, v2):
        return None

    # Step 2: yield-to-oncoming-when-turning-left
    left_yielder = _resolve_left_turn_yield(v1, v2)
    if left_yielder is not None:
        return left_yielder

    # Step 3: priority-to-the-right
    key = (v1.direction, v2.direction)
    result = APPROACH_PRIORITY.get(key)
    if result == "A":
        return v1.id
    if result == "B":
        return v2.id

    raise UnsupportedScenarioError(
        f"Trajectories conflict but no rule applies: "
        f"v1=({v1.direction!r},{v1.intent!r}), "
        f"v2=({v2.direction!r},{v2.intent!r})."
    )


_OPPOSITE: dict[Direction, Direction] = {
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.EAST:  Direction.WEST,
    Direction.WEST:  Direction.EAST,
}


def _resolve_left_turn_yield(v1: Vehicle, v2: Vehicle) -> Optional[str]:
    """
    If exactly one vehicle is turning left AND the other approaches from
    the opposite direction, the left-turner yields to the other one.
    Returns the winner's id, or None if this rule does not apply.
    """
    v1_left = v1.intent == IntentDirection.TURN_LEFT
    v2_left = v2.intent == IntentDirection.TURN_LEFT
    if v1_left == v2_left:
        return None  # both or neither turning left → rule does not apply
    if _OPPOSITE.get(v1.direction) != v2.direction:
        return None  # not opposing directions → rule does not apply
    return v2.id if v1_left else v1.id


# ── Right-of-way at roundabouts ──────────────────────────────────────────────

def right_of_way_roundabout(v_inside: Vehicle, v_entering: Vehicle) -> str:
    """
    At a roundabout, vehicles already inside always have priority
    over vehicles attempting to enter.

    Precondition: v_inside must be the vehicle currently circulating inside
    the roundabout (inside_intersection == True). The caller is responsible
    for passing the correct vehicle as v_inside.
    """
    return v_inside.id


def roundabout_can_enter(entering: Vehicle, vehicles_inside: list[Vehicle]) -> bool:
    """
    Returns True if the entering vehicle may enter the roundabout,
    i.e. no vehicle is currently circulating inside.
    """
    return not any(v.inside_intersection for v in vehicles_inside)


# ── Generic dispatcher ───────────────────────────────────────────────────────

def right_of_way(v1: Vehicle, v2: Vehicle, env: Environment) -> Optional[str]:
    """
    Environment-aware dispatcher.

    Returns the id of the vehicle with priority, or None if there is no
    conflict between the two vehicles under the rules for `env`.

    Raises:
        UnsupportedScenarioError: if the scenario falls outside the
        supported input space for its environment. In particular, a
        roundabout scenario where NEITHER vehicle is inside is now
        rejected rather than silently falling back to intersection
        priority-to-the-right (closes T2-B10).
    """
    if env == Environment.INTERSECTION:
        if both_have_intent(v1, v2):
            return right_of_way_intersection_with_intent(v1, v2)
        return right_of_way_intersection(v1, v2)

    if env == Environment.ROUNDABOUT:
        v1_inside = v1.inside_intersection
        v2_inside = v2.inside_intersection
        if v1_inside and not v2_inside:
            return right_of_way_roundabout(v1, v2)
        if v2_inside and not v1_inside:
            return right_of_way_roundabout(v2, v1)
        if v1_inside and v2_inside:
            # Two vehicles simultaneously circulating is not a priority
            # question the Task 2 ruleset answers. Reject loudly.
            raise UnsupportedScenarioError(
                "Roundabout with both vehicles inside the ring is not a "
                "supported right-of-way scenario."
            )
        # Neither inside: previously fell back to intersection logic.
        # That was wrong — roundabout approach vehicles do not have
        # priority-to-the-right among themselves, they have nobody yet.
        raise UnsupportedScenarioError(
            "Roundabout with no vehicle inside the ring is not a supported "
            "right-of-way scenario (Task 2 requires one circulating vehicle)."
        )

    if env == Environment.MULTI_LANE:
        # Multi-lane right-of-way between two arbitrary vehicles is not a
        # Task 2 concept. Return None (no conflict) to preserve legacy
        # behavior rather than raise; generators should not call this
        # path for MULTI_LANE in the first place.
        return None

    raise UnsupportedScenarioError(f"Unknown environment: {env!r}")


# ── Violation rules ──────────────────────────────────────────────────────────

def is_violation_stop_sign(entered_without_stopping: bool) -> bool:
    """Vehicle must stop before entering intersection when stop sign is present."""
    return entered_without_stopping


def is_violation_right_of_way(vehicle_id: str, priority_vehicle_id: Optional[str]) -> bool:
    """Vehicle entered despite not having right of way."""
    if priority_vehicle_id is None:
        return False
    return vehicle_id != priority_vehicle_id


def is_violation_roundabout_entry(entering: Vehicle, vehicles_inside: list[Vehicle]) -> bool:
    """Vehicle entered roundabout without yielding to circulating vehicles."""
    return not roundabout_can_enter(entering, vehicles_inside)


def is_violation_no_overtake(zone: str) -> bool:
    """Overtaking is forbidden in marked no-overtake zones."""
    return zone == "no_overtake_zone"


# ── Valid actions per environment ────────────────────────────────────────────

VALID_ACTIONS = {
    Environment.INTERSECTION: [
        "moves forward", "stops", "turns left", "turns right"
    ],
    Environment.MULTI_LANE: [
        "moves forward", "stops",
        "changes to the left lane", "changes to the right lane"
    ],
    Environment.ROUNDABOUT: [
        "enters the roundabout", "exits the roundabout",
        "moves forward", "stops"
    ],
}


def get_valid_actions(env: Environment) -> list[str]:
    return VALID_ACTIONS.get(env, [])


# ── Overlap conditions ───────────────────────────────────────────────────────

def is_overlap_possible(env: Environment) -> bool:
    return env in (Environment.INTERSECTION, Environment.ROUNDABOUT)


def vehicles_overlap(v1: Vehicle, v2: Vehicle) -> bool:
    return v1.inside_intersection and v2.inside_intersection
