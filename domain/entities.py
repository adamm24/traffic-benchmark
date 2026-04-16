from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Direction(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST  = "east"
    WEST  = "west"


class Action(str, Enum):
    MOVE_FORWARD  = "moves forward"
    STOP          = "stops"
    TURN_LEFT     = "turns left"
    TURN_RIGHT    = "turns right"
    CHANGE_LEFT   = "changes to the left lane"
    CHANGE_RIGHT  = "changes to the right lane"
    ENTER_ROUNDABOUT = "enters the roundabout"
    EXIT_ROUNDABOUT  = "exits the roundabout"


class Environment(str, Enum):
    INTERSECTION   = "intersection"
    MULTI_LANE     = "multi_lane_road"
    ROUNDABOUT     = "roundabout"


class Lane(str, Enum):
    LEFT   = "left_lane"
    CENTER = "center_lane"
    RIGHT  = "right_lane"
    ROUNDABOUT_LANE = "roundabout_lane"


class IntentDirection(str, Enum):
    """Used in Task 2: declared intention of vehicle at intersection."""
    GO_STRAIGHT  = "go straight"
    TURN_LEFT    = "turn left"
    TURN_RIGHT   = "turn right"
    EXIT_NORTH   = "exit north"
    EXIT_SOUTH   = "exit south"
    EXIT_EAST    = "exit east"
    EXIT_WEST    = "exit west"


@dataclass
class Vehicle:
    id: str                             # "A", "B", "C"
    position: str                       # Lane or named position
    direction: Direction                # approach direction
    intent: Optional[IntentDirection] = None   # declared intent (Task 2)
    inside_intersection: bool = False
    stopped: bool = False

    def describe(self) -> str:
        base = f"Vehicle {self.id} is in the {self.position}"
        if self.direction:
            base += f", approaching from the {self.direction.value}"
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
