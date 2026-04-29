"""Right-of-way and violation rules."""
from __future__ import annotations

from typing import Optional

from .entities import (
    Action,
    Direction,
    Environment,
    IntentDirection,
    Lane,
    ScenarioState,
    UnsupportedScenarioError,
    Vehicle,
)
from .fsm import is_transition_applicable
from .trajectory import (
    both_have_intent,
    trajectories_conflict,
    trajectory_of,
)



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
    """Direction-only priority-to-the-right rule for a 4-way intersection."""
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
    """Intent-aware right-of-way for intersection conflicts."""
    if not both_have_intent(v1, v2):
        raise UnsupportedScenarioError(
            "right_of_way_intersection_with_intent requires both vehicles "
            "to have a declared intent."
        )

    if not trajectories_conflict(v1, v2):
        return None

    left_yielder = _resolve_left_turn_yield(v1, v2)
    if left_yielder is not None:
        return left_yielder

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
    """Return the winner when a left-turning vehicle yields to oncoming traffic."""
    v1_left = v1.intent == IntentDirection.TURN_LEFT
    v2_left = v2.intent == IntentDirection.TURN_LEFT
    if v1_left == v2_left:
        return None  # both or neither turning left → rule does not apply
    if _OPPOSITE.get(v1.direction) != v2.direction:
        return None  # not opposing directions → rule does not apply
    return v2.id if v1_left else v1.id



def right_of_way_roundabout(v_inside: Vehicle, v_entering: Vehicle) -> str:
    """Roundabout priority: circulating vehicles go first."""
    return v_inside.id


def roundabout_can_enter(entering: Vehicle, vehicles_inside: list[Vehicle]) -> bool:
    """True when no vehicle is circulating inside the roundabout."""
    return not any(v.inside_intersection for v in vehicles_inside)



def right_of_way(v1: Vehicle, v2: Vehicle, env: Environment) -> Optional[str]:
    """Priority vehicle id, or None if there is no conflict."""
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
            raise UnsupportedScenarioError(
                "Roundabout with both vehicles inside the ring is not a "
                "supported right-of-way scenario."
            )
        raise UnsupportedScenarioError(
            "Roundabout with no vehicle inside the ring is not a supported "
            "right-of-way scenario (Task 2 requires one circulating vehicle)."
        )

    if env == Environment.MULTI_LANE:
        return None

    raise UnsupportedScenarioError(f"Unknown environment: {env!r}")



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



def is_overlap_possible(env: Environment) -> bool:
    return env in (Environment.INTERSECTION, Environment.ROUNDABOUT)


def vehicles_overlap(v1: Vehicle, v2: Vehicle) -> bool:
    return v1.inside_intersection and v2.inside_intersection


def is_valid_transition(state: ScenarioState, vehicle_id: str, action: Action) -> bool:
    """True when the action is FSM-valid for the current state."""
    vehicle = state.get_vehicle(vehicle_id)
    if vehicle is None:
        return False
    return is_transition_applicable(vehicle, state.environment, action)


def _violation_result(vehicle_id: str, violation_type: str, reason: str) -> dict[str, object]:
    return {
        "is_violation": True,
        "vehicle": vehicle_id,
        "violation_type": violation_type,
        "reason": reason,
    }


def _ok_result(vehicle_id: str, reason: str = "No violation detected.") -> dict[str, object]:
    return {
        "is_violation": False,
        "vehicle": vehicle_id,
        "violation_type": None,
        "reason": reason,
    }


def detect_fsm_violation(
    state: ScenarioState,
    vehicle_id: str,
    action: Action,
) -> dict[str, object]:
    """Detect illegal FSM transitions."""
    vehicle = state.get_vehicle(vehicle_id)
    if vehicle is None:
        return _violation_result(
            vehicle_id,
            "invalid_vehicle",
            f"Vehicle {vehicle_id} is not present in the scenario state.",
        )

    if is_valid_transition(state, vehicle_id, action):
        return _ok_result(vehicle_id, "FSM transition is valid.")

    env = state.environment
    pos = vehicle.position
    inside = vehicle.inside_intersection

    if (
        env == Environment.INTERSECTION
        and action in (Action.TURN_LEFT, Action.TURN_RIGHT)
        and not inside
    ):
        return _violation_result(
            vehicle_id,
            "turn_without_entering",
            f"Vehicle {vehicle_id} tried to turn before entering the intersection.",
        )

    if action == Action.MOVE_FORWARD and pos.endswith("_exit"):
        return _violation_result(
            vehicle_id,
            "forward_from_exit",
            f"Vehicle {vehicle_id} tried to move forward from an exit position.",
        )

    if env == Environment.MULTI_LANE and action == Action.CHANGE_LEFT and pos == Lane.LEFT.value:
        return _violation_result(
            vehicle_id,
            "lane_change_out_of_bounds_left",
            f"Vehicle {vehicle_id} tried to change left from the left lane.",
        )

    if env == Environment.MULTI_LANE and action == Action.CHANGE_RIGHT and pos == Lane.RIGHT.value:
        return _violation_result(
            vehicle_id,
            "lane_change_out_of_bounds_right",
            f"Vehicle {vehicle_id} tried to change right from the right lane.",
        )

    return _violation_result(
        vehicle_id,
        "invalid_fsm_transition",
        (
            f"Vehicle {vehicle_id} attempted an invalid transition: "
            f"env={env.value}, position={pos}, action={action.value}."
        ),
    )


def detect_right_of_way_violation(
    state: ScenarioState,
    vehicle_id: str,
    action: Action,
) -> dict[str, object]:
    """Detect right-of-way violations."""
    vehicle = state.get_vehicle(vehicle_id)
    if vehicle is None:
        return _violation_result(
            vehicle_id,
            "invalid_vehicle",
            f"Vehicle {vehicle_id} is not present in the scenario state.",
        )

    env = state.environment

    if env == Environment.INTERSECTION and action == Action.MOVE_FORWARD:
        if vehicle.inside_intersection or not vehicle.position.endswith("_approach"):
            return _ok_result(vehicle_id, "Not an intersection entry action.")

        for other in state.vehicles:
            if other.id == vehicle_id:
                continue
            if other.inside_intersection and other.position == "inside_intersection":
                return _violation_result(
                    vehicle_id,
                    "intersection_right_of_way",
                    (
                        f"Vehicle {vehicle_id} entered while Vehicle {other.id} "
                        "was already inside the intersection."
                    ),
                )

        for other in state.vehicles:
            if other.id == vehicle_id:
                continue
            if other.inside_intersection or not other.position.endswith("_approach"):
                continue
            winner = right_of_way(vehicle, other, Environment.INTERSECTION)
            if winner is None:
                continue
            if winner != vehicle_id:
                return _violation_result(
                    vehicle_id,
                    "intersection_right_of_way",
                    (
                        f"Vehicle {vehicle_id} moved forward without right of way; "
                        f"Vehicle {winner} had priority."
                    ),
                )
        return _ok_result(vehicle_id, "Intersection right-of-way respected.")

    if env == Environment.ROUNDABOUT and action == Action.ENTER_ROUNDABOUT:
        inside_others = [
            v for v in state.vehicles
            if v.id != vehicle_id and v.inside_intersection
        ]
        if is_violation_roundabout_entry(vehicle, inside_others):
            inside_ids = ", ".join(v.id for v in inside_others)
            return _violation_result(
                vehicle_id,
                "roundabout_entry_no_yield",
                (
                    f"Vehicle {vehicle_id} entered the roundabout without yielding; "
                    f"circulating vehicle(s): {inside_ids}."
                ),
            )
        return _ok_result(vehicle_id, "Roundabout entry respected priority.")

    return _ok_result(vehicle_id, "No right-of-way rule applies to this action.")


def detect_violation(
    state: ScenarioState,
    vehicle_id: str,
    action: Action,
) -> dict[str, object]:
    """Run FSM and right-of-way checks for one action."""
    fsm_result = detect_fsm_violation(state, vehicle_id, action)
    if bool(fsm_result["is_violation"]):
        return fsm_result

    row_result = detect_right_of_way_violation(state, vehicle_id, action)
    if bool(row_result["is_violation"]):
        return row_result

    return _ok_result(vehicle_id)
