from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

_POSITION_PREFIXES = (
    "inside ",
    "outside ",
    "within ",
    "near ",
    "behind ",
    "beside ",
    "between ",
    "at ",
    "on ",
    "under ",
    "over ",
    "in front of ",
)


def _position_clause(position: str) -> str:
    label = position.replace("_", " ").strip()
    lower = label.lower()
    if any(lower.startswith(prefix) for prefix in _POSITION_PREFIXES):
        return f"is {label}"
    return f"is in {label}"


class Direction(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST  = "east"
    WEST  = "west"


class Action(str, Enum):
    MOVE_FORWARD     = "moves forward"
    STOP             = "stops"
    TURN_LEFT        = "turns left"
    TURN_RIGHT       = "turns right"
    CHANGE_LEFT      = "changes to the left lane"
    CHANGE_RIGHT     = "changes to the right lane"
    ENTER_ROUNDABOUT = "enters the roundabout"
    EXIT_ROUNDABOUT  = "exits the roundabout"


class Environment(str, Enum):
    INTERSECTION = "intersection"
    MULTI_LANE   = "multi_lane_road"
    ROUNDABOUT   = "roundabout"


class Lane(str, Enum):
    LEFT            = "left_lane"
    CENTER          = "center_lane"
    RIGHT           = "right_lane"
    ROUNDABOUT_LANE = "roundabout_lane"


class IntentDirection(str, Enum):
    """Declared intention of a vehicle at an intersection (Task 2 only)."""
    GO_STRAIGHT = "go straight"
    TURN_LEFT   = "turn left"
    TURN_RIGHT  = "turn right"


class VehicleState(str, Enum):
    """Vehicle lifecycle state derived from position and environment."""
    APPROACHING             = "approaching"
    INSIDE_INTERSECTION     = "inside_intersection"
    EXITED_INTERSECTION     = "exited_intersection"
    ON_LANE                 = "on_lane"
    ROUNDABOUT_APPROACHING  = "roundabout_approaching"
    IN_ROUNDABOUT           = "in_roundabout"
    EXITED_ROUNDABOUT       = "exited_roundabout"


class UnsupportedScenarioError(ValueError):
    """Scenario outside the supported rule space."""
    pass


@dataclass
class Vehicle:
    id: str
    position: str
    direction: Direction
    intent: Optional[IntentDirection] = None
    inside_intersection: bool = False
    stopped: bool = False

    def describe(self) -> str:
        base = f"Vehicle {self.id} {_position_clause(self.position)}"
        if self.intent:
            base += f", intending to {self.intent.value}"
        return base + "."


@dataclass
class ScenarioState:
    vehicles: list[Vehicle]
    environment: Environment
    step: int = 0
    event_log: list[str] = field(default_factory=list)

    def get_vehicle(self, vid: str) -> Optional[Vehicle]:
        for v in self.vehicles:
            if v.id == vid:
                return v
        return None
