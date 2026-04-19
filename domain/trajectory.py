"""
Trajectory model for intersection conflict detection.

Closes T2-B01 / T2-B06: right-of-way logic for intersections needs to
distinguish between movements whose paths actually cross and movements
that can proceed simultaneously. The previous APPROACH_PRIORITY table
keyed on (direction_a, direction_b) alone returned None for
opposite-direction pairs even when one vehicle was turning left across
oncoming traffic — a real conflict the rule layer used to miss.

Model
-----
The intersection body is divided into four cells arranged on a 2×2 grid:

                  N arm
                 ┌─────┐
           W arm │NW│NE│ E arm
                 │──┼──│
                 │SW│SE│
                 └─────┘
                  S arm

Each (approach_direction, intent) pair maps to a frozenset of the cells
the trajectory occupies while crossing the intersection. Two trajectories
CONFLICT iff their cell sets intersect.

Conventions (right-hand drive)
------------------------------
A vehicle approaching from N is heading south; it enters the intersection
on the *west* half of the north arm (its right side) → entry cell = NW.
It exits on the right side of its target arm. TURN_RIGHT hugs the entry
corner (1 cell). GO_STRAIGHT cuts across the same side (2 cells).
TURN_LEFT sweeps through three cells to reach the diagonally opposite
corner.

Known conservative approximation
--------------------------------
Two opposite-direction left turns (e.g. N-left and S-left) share cells in
this model and therefore appear to conflict. In real right-hand-drive
traffic they can often proceed simultaneously. This is an accepted
over-approximation — it never grants priority wrongly; at worst it flags
a conflict that resolves to "both yield" under priority-to-the-right.
This does not affect any Task 2 scenario currently generated.

This module is pure data + total functions. It raises on malformed input
(e.g. missing intent) rather than silently returning empty sets.
"""
from __future__ import annotations

from typing import Optional

from .entities import Direction, IntentDirection, Vehicle


# ── Intersection cells ───────────────────────────────────────────────────────

NE = "NE"
NW = "NW"
SE = "SE"
SW = "SW"

ALL_CELLS: frozenset[str] = frozenset({NE, NW, SE, SW})


# ── Trajectory table ─────────────────────────────────────────────────────────

_TRAJECTORY_CELLS: dict[tuple[Direction, IntentDirection], frozenset[str]] = {
    # From NORTH (heading south); entry = NW
    (Direction.NORTH, IntentDirection.GO_STRAIGHT): frozenset({NW, SW}),
    (Direction.NORTH, IntentDirection.TURN_LEFT):   frozenset({NW, SW, SE}),
    (Direction.NORTH, IntentDirection.TURN_RIGHT):  frozenset({NW}),

    # From SOUTH (heading north); entry = SE
    (Direction.SOUTH, IntentDirection.GO_STRAIGHT): frozenset({SE, NE}),
    (Direction.SOUTH, IntentDirection.TURN_LEFT):   frozenset({SE, NE, NW}),
    (Direction.SOUTH, IntentDirection.TURN_RIGHT):  frozenset({SE}),

    # From EAST (heading west); entry = NE
    (Direction.EAST, IntentDirection.GO_STRAIGHT):  frozenset({NE, NW}),
    (Direction.EAST, IntentDirection.TURN_LEFT):    frozenset({NE, NW, SW}),
    (Direction.EAST, IntentDirection.TURN_RIGHT):   frozenset({NE}),

    # From WEST (heading east); entry = SW
    (Direction.WEST, IntentDirection.GO_STRAIGHT):  frozenset({SW, SE}),
    (Direction.WEST, IntentDirection.TURN_LEFT):    frozenset({SW, SE, NE}),
    (Direction.WEST, IntentDirection.TURN_RIGHT):   frozenset({SW}),
}


# ── Public API ───────────────────────────────────────────────────────────────

def trajectory_of(v: Vehicle) -> frozenset[str]:
    """
    Returns the set of intersection cells the vehicle occupies while
    traversing the intersection.

    Precondition: v.intent must be set. Scenarios without declared intent
    (e.g. Task 1) cannot use the trajectory model — callers should branch
    on intent availability and fall back to the direction-only
    `right_of_way_intersection` in that case.

    Raises:
        ValueError: if v.intent is None or (direction, intent) is not
        in the trajectory table (should be unreachable for valid enums).
    """
    if v.intent is None:
        raise ValueError(
            f"Vehicle {v.id} has no declared intent; trajectory is undefined."
        )
    key = (v.direction, v.intent)
    if key not in _TRAJECTORY_CELLS:
        raise ValueError(
            f"No trajectory defined for (direction={v.direction!r}, "
            f"intent={v.intent!r}). This indicates a corrupted enum."
        )
    return _TRAJECTORY_CELLS[key]


def trajectories_conflict(v1: Vehicle, v2: Vehicle) -> bool:
    """
    True iff the two vehicles' intersection trajectories share at least
    one cell. Both vehicles must have declared intent.
    """
    return bool(trajectory_of(v1) & trajectory_of(v2))


def trajectory_cells(direction: Direction, intent: IntentDirection) -> frozenset[str]:
    """
    Direct lookup variant for callers that only have (direction, intent)
    without a Vehicle instance (e.g. generator dry-runs, tests).
    """
    key = (direction, intent)
    if key not in _TRAJECTORY_CELLS:
        raise ValueError(
            f"No trajectory defined for (direction={direction!r}, intent={intent!r})."
        )
    return _TRAJECTORY_CELLS[key]


def has_intent(v: Vehicle) -> bool:
    """Convenience predicate: True iff the vehicle has declared intent."""
    return v.intent is not None


def both_have_intent(v1: Vehicle, v2: Vehicle) -> bool:
    """Convenience predicate for the intent-aware dispatcher path."""
    return v1.intent is not None and v2.intent is not None
