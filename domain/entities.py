from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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
    """
    Explicit vehicle lifecycle state. Derived deterministically from
    (environment, position, inside_intersection) via domain.fsm.derive_state.

    Intersection states
      APPROACHING          — vehicle is on one of the arms (e.g. north_approach)
      INSIDE_INTERSECTION  — vehicle is inside the intersection body
      EXITED_INTERSECTION  — vehicle has reached an exit arm ({dir}_exit)

    Multi-lane states
      ON_LANE              — vehicle is on one of the three lanes

    Roundabout states
      ROUNDABOUT_APPROACHING  — vehicle is on one of the approach arms
      IN_ROUNDABOUT           — vehicle is circulating on the ring
      EXITED_ROUNDABOUT       — vehicle has reached an exit arm

    The enum is orthogonal to the `stopped` flag: a vehicle can be stopped
    in any state without changing state.
    """
    APPROACHING             = "approaching"
    INSIDE_INTERSECTION     = "inside_intersection"
    EXITED_INTERSECTION     = "exited_intersection"
    ON_LANE                 = "on_lane"
    ROUNDABOUT_APPROACHING  = "roundabout_approaching"
    IN_ROUNDABOUT           = "in_roundabout"
    EXITED_ROUNDABOUT       = "exited_roundabout"


class UnsupportedScenarioError(ValueError):
    """
    Raised by domain.rules dispatchers when a scenario falls outside the
    supported input space (e.g. a roundabout right-of-way query where no
    vehicle is inside the ring). Using a dedicated exception lets callers
    distinguish a genuine "no conflict" answer (None) from an ill-formed
    scenario.
    """
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
        # NOTE: prefer domain.render.describe_vehicle() for prompt generation —
        # it uses POSITION_LABELS for clean human-readable labels.
        # This method is kept only for quick debugging.
        pos_label = self.position.replace("_", " ")
        base = f"Vehicle {self.id} is in {pos_label}"
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
