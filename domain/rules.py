from .entities import Direction, IntentDirection, Environment, Vehicle
from typing import Optional


# ── Right-of-way rules at intersections (no signs, no lights) ──────────────

APPROACH_PRIORITY = {
    # Italian/European rule: vehicle on the right has priority
    # Key: (vehicle_A_direction, vehicle_B_direction) → who has priority ("A" or "B")
    (Direction.NORTH, Direction.EAST):  "B",  # B comes from right of A
    (Direction.NORTH, Direction.WEST):  "A",
    (Direction.SOUTH, Direction.WEST):  "B",
    (Direction.SOUTH, Direction.EAST):  "A",
    (Direction.EAST,  Direction.SOUTH): "B",
    (Direction.EAST,  Direction.NORTH): "A",
    (Direction.WEST,  Direction.NORTH): "B",
    (Direction.WEST,  Direction.SOUTH): "A",
}

def right_of_way(v1: Vehicle, v2: Vehicle) -> Optional[str]:
    """
    Returns the id of the vehicle with right of way.
    Returns None if both can pass simultaneously (no conflict).
    """
    key = (v1.direction, v2.direction)
    result = APPROACH_PRIORITY.get(key)
    if result == "A":
        return v1.id
    elif result == "B":
        return v2.id
    return None


# ── Violation rules ─────────────────────────────────────────────────────────

def is_violation_stop_sign(vehicle: Vehicle, entered_without_stopping: bool) -> bool:
    """Vehicle must stop before entering intersection when stop sign present."""
    return entered_without_stopping

def is_violation_no_overtake(overtaking_vehicle: Vehicle, zone: str) -> bool:
    """Overtaking is forbidden in marked no-overtake zones."""
    return zone == "no_overtake_zone"

def is_violation_right_of_way(vehicle: Vehicle, had_priority: bool) -> bool:
    """Vehicle entered intersection despite not having right of way."""
    return not had_priority

def is_violation_roundabout_entry(vehicle: Vehicle, roundabout_occupied: bool) -> bool:
    """Vehicle entering roundabout must yield to vehicles already inside."""
    return roundabout_occupied


# ── Valid actions per environment ────────────────────────────────────────────

VALID_ACTIONS = {
    Environment.INTERSECTION: [
        "moves forward", "stops", "turns left", "turns right"
    ],
    Environment.MULTI_LANE: [
        "moves forward", "stops", "changes to the left lane",
        "changes to the right lane"
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
    """Overlap (multiple vehicles in same space) only occurs at intersections/roundabouts."""
    return env in (Environment.INTERSECTION, Environment.ROUNDABOUT)

def vehicles_overlap(v1: Vehicle, v2: Vehicle) -> bool:
    """Two vehicles overlap if both are inside the intersection simultaneously."""
    return v1.inside_intersection and v2.inside_intersection
