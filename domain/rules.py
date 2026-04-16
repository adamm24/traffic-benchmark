from .entities import Direction, Environment, Vehicle
from typing import Optional


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
    Returns the id of the vehicle with right of way at an intersection.
    Uses priority-to-the-right rule (no signs, no lights).
    Returns None if directions are opposite (no lateral conflict).
    """
    key = (v1.direction, v2.direction)
    result = APPROACH_PRIORITY.get(key)
    if result == "A":
        return v1.id
    elif result == "B":
        return v2.id
    return None


# ── Right-of-way at roundabouts ──────────────────────────────────────────────

def right_of_way_roundabout(v_inside: Vehicle, v_entering: Vehicle) -> str:
    """
    At a roundabout, vehicles already inside always have priority
    over vehicles attempting to enter.
    Returns the id of the vehicle with right of way.
    """
    if v_inside.inside_intersection:
        return v_inside.id
    return v_entering.id

def roundabout_can_enter(entering: Vehicle, vehicles_inside: list[Vehicle]) -> bool:
    """
    Returns True if the entering vehicle may enter the roundabout,
    i.e. no vehicle is currently circulating inside.
    """
    return not any(v.inside_intersection for v in vehicles_inside)


# ── Generic dispatcher ───────────────────────────────────────────────────────

def right_of_way(v1: Vehicle, v2: Vehicle, env: Environment) -> Optional[str]:
    """
    Dispatches right-of-way logic based on environment.
    Returns the id of the vehicle with priority, or None if no conflict.
    """
    if env == Environment.INTERSECTION:
        return right_of_way_intersection(v1, v2)
    elif env == Environment.ROUNDABOUT:
        if v1.inside_intersection:
            return right_of_way_roundabout(v1, v2)
        elif v2.inside_intersection:
            return right_of_way_roundabout(v2, v1)
        return right_of_way_intersection(v1, v2)
    return None


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