import random
from .entities import (
    Vehicle, ScenarioState, Environment,
    Direction, Lane, Action, IntentDirection, VehicleState,
)
from .fsm import is_transition_applicable, derive_state


ALL_DIRECTIONS = list(Direction)
ALL_LANES      = [Lane.LEFT, Lane.CENTER, Lane.RIGHT]


# ── Scenario builders ────────────────────────────────────────────────────────

def build_intersection_scenario(num_vehicles: int = 3,
                                 with_intent: bool = False) -> ScenarioState:
    directions = random.sample(ALL_DIRECTIONS, num_vehicles)
    vehicles = []
    for i, d in enumerate(directions):
        vid = chr(65 + i)
        intent = random.choice([
            IntentDirection.GO_STRAIGHT,
            IntentDirection.TURN_LEFT,
            IntentDirection.TURN_RIGHT
        ]) if with_intent else None
        vehicles.append(Vehicle(
            id=vid,
            position=f"{d.value}_approach",
            direction=d,
            intent=intent
        ))
    return ScenarioState(vehicles=vehicles, environment=Environment.INTERSECTION)


def build_multi_lane_scenario(num_vehicles: int = 3) -> ScenarioState:
    lanes = random.sample(ALL_LANES, num_vehicles)
    vehicles = []
    for i, lane in enumerate(lanes):
        vid = chr(65 + i)
        vehicles.append(Vehicle(
            id=vid,
            position=lane.value,
            direction=Direction.NORTH
        ))
    return ScenarioState(vehicles=vehicles, environment=Environment.MULTI_LANE)


def build_roundabout_scenario(num_vehicles: int = 3) -> ScenarioState:
    directions = random.sample(ALL_DIRECTIONS, num_vehicles)
    vehicles = []
    for i, d in enumerate(directions):
        vid = chr(65 + i)
        inside = (i == 0)
        vehicles.append(Vehicle(
            id=vid,
            position="roundabout_lane" if inside else f"{d.value}_approach",
            direction=d,
            inside_intersection=inside
        ))
    return ScenarioState(vehicles=vehicles, environment=Environment.ROUNDABOUT)


SCENARIO_BUILDERS = {
    Environment.INTERSECTION: build_intersection_scenario,
    Environment.MULTI_LANE:   build_multi_lane_scenario,
    Environment.ROUNDABOUT:   build_roundabout_scenario,
}

def build_scenario(env: Environment, **kwargs) -> ScenarioState:
    return SCENARIO_BUILDERS[env](**kwargs)


# ── Action application (Task 1 core logic) ──────────────────────────────────

LANE_ORDER = [Lane.LEFT.value, Lane.CENTER.value, Lane.RIGHT.value]


def apply_action(state: ScenarioState, vehicle_id: str, action: Action) -> str:
    """
    Applies an action to a vehicle, mutates its state, and returns a
    natural-language description of the event.

    Contract (post FSM refactor):
      • If the transition (current_state, environment, action) is NOT
        defined in domain.fsm.TRANSITIONS, the function DOES NOT mutate
        state and returns the empty string "" (no event emitted, step not
        advanced).
      • If the transition is defined but the runtime precondition fails
        (e.g. CHANGE_LEFT from the leftmost lane), same behaviour: no
        mutation, empty string.
      • Otherwise state is mutated consistently, the event is appended to
        state.event_log, step is incremented, and the event string is
        returned.

    This closes T1-B02 / T1-B04 / T1-B07: apply_action() is now
    self-defensive. safe_apply_action() wrappers in generators remain
    useful for task-specific higher-level logic (retries, pool shuffling)
    but are no longer needed to enforce domain-level invariants.

    Returns:
        str: human-readable event string, or "" for no-op / invalid.
    """
    v = state.get_vehicle(vehicle_id)
    if v is None:
        raise ValueError(f"Vehicle {vehicle_id} not found in scenario.")

    # FSM guard — both state-level and runtime preconditions.
    if not is_transition_applicable(v, state.environment, action):
        return ""

    event = ""

    if action == Action.MOVE_FORWARD:
        # Only defined at INTERSECTION / APPROACHING (per FSM).
        v.inside_intersection = True
        v.position = "inside_intersection"
        event = f"Vehicle {v.id} moves forward."

    elif action == Action.STOP:
        v.stopped = True
        event = f"Vehicle {v.id} stops."

    elif action == Action.TURN_LEFT:
        v.direction = _rotate_direction(v.direction, "left")
        v.inside_intersection = False
        v.position = f"{v.direction.value}_exit"
        event = f"Vehicle {v.id} turns left."

    elif action == Action.TURN_RIGHT:
        v.direction = _rotate_direction(v.direction, "right")
        v.inside_intersection = False
        v.position = f"{v.direction.value}_exit"
        event = f"Vehicle {v.id} turns right."

    elif action == Action.CHANGE_LEFT:
        current_idx = _lane_index(v.position)
        # FSM guard already ensured idx > 0.
        v.position = LANE_ORDER[current_idx - 1]
        event = f"Vehicle {v.id} changes to the left lane."

    elif action == Action.CHANGE_RIGHT:
        current_idx = _lane_index(v.position)
        # FSM guard already ensured 0 <= idx < len-1.
        v.position = LANE_ORDER[current_idx + 1]
        event = f"Vehicle {v.id} changes to the right lane."

    elif action == Action.ENTER_ROUNDABOUT:
        v.inside_intersection = True
        v.position = "roundabout_lane"
        event = f"Vehicle {v.id} enters the roundabout."

    elif action == Action.EXIT_ROUNDABOUT:
        v.inside_intersection = False
        v.position = f"{v.direction.value}_exit"
        event = f"Vehicle {v.id} exits the roundabout."

    if event:
        state.event_log.append(event)
        state.step += 1
    return event


def _lane_index(position: str) -> int:
    """Returns index of lane in LEFT-CENTER-RIGHT order, or -1 if not a lane."""
    try:
        return LANE_ORDER.index(position)
    except ValueError:
        return -1


def _rotate_direction(direction: Direction, turn: str) -> Direction:
    """Returns new direction after turning left or right."""
    order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
    idx = order.index(direction)
    if turn == "right":
        return order[(idx + 1) % 4]
    else:
        return order[(idx - 1) % 4]


# ── Convenience re-exports for generators ────────────────────────────────────

def vehicle_state(v: Vehicle, env: Environment) -> VehicleState:
    """Re-export of derive_state() for callers that already import scenario."""
    return derive_state(v, env)
