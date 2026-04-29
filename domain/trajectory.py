"""Trajectory model for intersection conflict detection.

The intersection body is divided into four cells arranged on a 2x2 grid:

                  N arm
                 ┌─────┐
           W arm │NW│NE│ E arm
                 │──┼──│
                 │SW│SE│
                 └─────┘
                  S arm

Each (approach_direction, intent) pair maps to a frozenset of the cells
the trajectory occupies while crossing. Two trajectories conflict when
their cell sets intersect.

Conventions: right-hand drive.
A vehicle approaching from N is heading south; it enters the intersection
on the *west* half of the north arm (its right side) → entry cell = NW.
It exits on the right side of its target arm. TURN_RIGHT hugs the entry
corner (1 cell). GO_STRAIGHT cuts across the same side (2 cells).
TURN_LEFT sweeps through three cells to reach the diagonally opposite
corner.

Two opposite-direction left turns (e.g. N-left and S-left) share cells in
this model and therefore appear to conflict. This conservative
approximation avoids granting priority where the model is unsure.
"""
from __future__ import annotations

from typing import Optional

from .entities import Direction, IntentDirection, Vehicle



NE = "NE"
NW = "NW"
SE = "SE"
SW = "SW"

ALL_CELLS: frozenset[str] = frozenset({NE, NW, SE, SW})



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



def trajectory_of(v: Vehicle) -> frozenset[str]:
    """Cells occupied by the vehicle while crossing the intersection."""
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
    """True when the two trajectories share at least one cell."""
    return bool(trajectory_of(v1) & trajectory_of(v2))


def trajectory_cells(direction: Direction, intent: IntentDirection) -> frozenset[str]:
    """Direct lookup for (direction, intent)."""
    key = (direction, intent)
    if key not in _TRAJECTORY_CELLS:
        raise ValueError(
            f"No trajectory defined for (direction={direction!r}, intent={intent!r})."
        )
    return _TRAJECTORY_CELLS[key]


def has_intent(v: Vehicle) -> bool:
    """True when the vehicle has declared intent."""
    return v.intent is not None


def both_have_intent(v1: Vehicle, v2: Vehicle) -> bool:
    """True when both vehicles have declared intent."""
    return v1.intent is not None and v2.intent is not None
