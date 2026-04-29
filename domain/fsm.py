"""Finite-state transitions for vehicle movement."""
from __future__ import annotations

from .entities import (
    Action, Environment, Lane, Vehicle, VehicleState,
)



_LANE_POSITIONS = {Lane.LEFT.value, Lane.CENTER.value, Lane.RIGHT.value}


def derive_state(v: Vehicle, env: Environment) -> VehicleState:
    """Map current vehicle fields to a VehicleState."""
    pos = v.position
    inside = v.inside_intersection

    if env == Environment.INTERSECTION:
        if inside and pos == "inside_intersection":
            return VehicleState.INSIDE_INTERSECTION
        if not inside and pos.endswith("_approach"):
            return VehicleState.APPROACHING
        if not inside and pos.endswith("_exit"):
            return VehicleState.EXITED_INTERSECTION
        raise ValueError(
            f"Inconsistent intersection vehicle: position={pos!r}, "
            f"inside_intersection={inside!r}"
        )

    if env == Environment.MULTI_LANE:
        if not inside and pos in _LANE_POSITIONS:
            return VehicleState.ON_LANE
        raise ValueError(
            f"Inconsistent multi-lane vehicle: position={pos!r}, "
            f"inside_intersection={inside!r}"
        )

    if env == Environment.ROUNDABOUT:
        if inside and pos == "roundabout_lane":
            return VehicleState.IN_ROUNDABOUT
        if not inside and pos.endswith("_approach"):
            return VehicleState.ROUNDABOUT_APPROACHING
        if not inside and pos.endswith("_exit"):
            return VehicleState.EXITED_ROUNDABOUT
        raise ValueError(
            f"Inconsistent roundabout vehicle: position={pos!r}, "
            f"inside_intersection={inside!r}"
        )

    raise ValueError(f"Unknown environment: {env!r}")



# Keys: (VehicleState, Environment, Action)
# Values: VehicleState reached after the transition.
# STOP transitions do not change the VehicleState (they toggle the `stopped`
# flag, which is orthogonal to the FSM).

TRANSITIONS: dict[tuple[VehicleState, Environment, Action], VehicleState] = {
    (VehicleState.APPROACHING, Environment.INTERSECTION, Action.MOVE_FORWARD):
        VehicleState.INSIDE_INTERSECTION,
    (VehicleState.APPROACHING, Environment.INTERSECTION, Action.STOP):
        VehicleState.APPROACHING,

    (VehicleState.INSIDE_INTERSECTION, Environment.INTERSECTION, Action.TURN_LEFT):
        VehicleState.EXITED_INTERSECTION,
    (VehicleState.INSIDE_INTERSECTION, Environment.INTERSECTION, Action.TURN_RIGHT):
        VehicleState.EXITED_INTERSECTION,
    (VehicleState.INSIDE_INTERSECTION, Environment.INTERSECTION, Action.STOP):
        VehicleState.INSIDE_INTERSECTION,

    (VehicleState.EXITED_INTERSECTION, Environment.INTERSECTION, Action.STOP):
        VehicleState.EXITED_INTERSECTION,
    # MOVE_FORWARD from EXITED is intentionally absent to prevent
    # "exit -> inside" re-entry loops (T1-B04).

    (VehicleState.ON_LANE, Environment.MULTI_LANE, Action.CHANGE_LEFT):
        VehicleState.ON_LANE,
    (VehicleState.ON_LANE, Environment.MULTI_LANE, Action.CHANGE_RIGHT):
        VehicleState.ON_LANE,
    (VehicleState.ON_LANE, Environment.MULTI_LANE, Action.STOP):
        VehicleState.ON_LANE,
    # MOVE_FORWARD on multi-lane is intentionally absent.
    # In this discrete-lane model it would be a no-op.

    (VehicleState.ROUNDABOUT_APPROACHING, Environment.ROUNDABOUT, Action.ENTER_ROUNDABOUT):
        VehicleState.IN_ROUNDABOUT,
    (VehicleState.ROUNDABOUT_APPROACHING, Environment.ROUNDABOUT, Action.STOP):
        VehicleState.ROUNDABOUT_APPROACHING,

    (VehicleState.IN_ROUNDABOUT, Environment.ROUNDABOUT, Action.EXIT_ROUNDABOUT):
        VehicleState.EXITED_ROUNDABOUT,
    (VehicleState.IN_ROUNDABOUT, Environment.ROUNDABOUT, Action.STOP):
        VehicleState.IN_ROUNDABOUT,

    (VehicleState.EXITED_ROUNDABOUT, Environment.ROUNDABOUT, Action.STOP):
        VehicleState.EXITED_ROUNDABOUT,
}



def is_transition_defined(
    state: VehicleState,
    env: Environment,
    action: Action,
) -> bool:
    """True iff (state, env, action) is a declared transition in the FSM."""
    return (state, env, action) in TRANSITIONS


def next_state(
    state: VehicleState,
    env: Environment,
    action: Action,
) -> VehicleState | None:
    """Target VehicleState, or None for an invalid transition."""
    return TRANSITIONS.get((state, env, action))


_LANE_ORDER = (Lane.LEFT.value, Lane.CENTER.value, Lane.RIGHT.value)


def is_transition_applicable(v: Vehicle, env: Environment, action: Action) -> bool:
    """Check FSM validity plus runtime lane bounds."""
    try:
        state = derive_state(v, env)
    except ValueError:
        return False

    if not is_transition_defined(state, env, action):
        return False

    if action == Action.CHANGE_LEFT and env == Environment.MULTI_LANE:
        idx = _lane_index(v.position)
        return idx > 0
    if action == Action.CHANGE_RIGHT and env == Environment.MULTI_LANE:
        idx = _lane_index(v.position)
        return 0 <= idx < len(_LANE_ORDER) - 1

    return True


def valid_actions(v: Vehicle, env: Environment) -> list[Action]:
    """Actions applicable to `v` under `env`."""
    return [a for a in Action if is_transition_applicable(v, env, a)]


def _lane_index(position: str) -> int:
    """Lane index in left-center-right order, or -1 if not a lane."""
    try:
        return _LANE_ORDER.index(position)
    except ValueError:
        return -1
