"""
Explicit Finite State Machine for vehicles.

Closes T1-B03 / T1-B04 / T1-B07: transitions are now declarative data
instead of being scattered across if/elif branches in apply_action() and
duplicated in per-task safe_apply_action() wrappers.

Design
------
A vehicle's state is derived deterministically from
(environment, position, inside_intersection) by `derive_state()`. The
Vehicle.state attribute is not stored explicitly — the existing fields
(`position`, `inside_intersection`) remain the source of truth, and the
state is computed on demand. This keeps backward compatibility with the
frozen Task 1 dataset: the JSONL does not need to be re-serialized.

A transition is a triple (VehicleState, Environment, Action) → VehicleState.
Transitions not listed in TRANSITIONS are INVALID and must be rejected by
apply_action() without mutating state.

Some transitions require a runtime precondition beyond the state (e.g.
CHANGE_LEFT requires that the vehicle is not on the leftmost lane). Those
preconditions live in `is_transition_applicable()`.
"""
from __future__ import annotations

from .entities import (
    Action, Environment, Lane, Vehicle, VehicleState,
)


# ── State derivation ─────────────────────────────────────────────────────────

_LANE_POSITIONS = {Lane.LEFT.value, Lane.CENTER.value, Lane.RIGHT.value}


def derive_state(v: Vehicle, env: Environment) -> VehicleState:
    """
    Project (environment, position, inside_intersection) onto a VehicleState.

    This function is the single point where the legacy field representation
    is converted into the FSM representation. It is total: every combination
    of inputs maps to exactly one state. Unexpected combinations raise
    ValueError, making silent inconsistencies impossible.
    """
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


# ── Transitions table ────────────────────────────────────────────────────────

# Keys: (VehicleState, Environment, Action)
# Values: VehicleState reached after the transition.
# STOP transitions do not change the VehicleState (they toggle the `stopped`
# flag, which is orthogonal to the FSM).

TRANSITIONS: dict[tuple[VehicleState, Environment, Action], VehicleState] = {
    # ── Intersection ────────────────────────────────────────────────────
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
    # NOTE: MOVE_FORWARD from EXITED is intentionally absent. This rejects
    # the "exit → inside" re-entry loop (T1-B04).

    # ── Multi-lane ──────────────────────────────────────────────────────
    (VehicleState.ON_LANE, Environment.MULTI_LANE, Action.CHANGE_LEFT):
        VehicleState.ON_LANE,
    (VehicleState.ON_LANE, Environment.MULTI_LANE, Action.CHANGE_RIGHT):
        VehicleState.ON_LANE,
    (VehicleState.ON_LANE, Environment.MULTI_LANE, Action.STOP):
        VehicleState.ON_LANE,
    # NOTE: MOVE_FORWARD on a multi-lane is intentionally absent. In the
    # discrete-lane model it would be a no-op for position tracking
    # (documented design choice).

    # ── Roundabout ──────────────────────────────────────────────────────
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


# ── Public FSM API ───────────────────────────────────────────────────────────

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
    """Returns the target VehicleState, or None if the transition is invalid."""
    return TRANSITIONS.get((state, env, action))


_LANE_ORDER = (Lane.LEFT.value, Lane.CENTER.value, Lane.RIGHT.value)


def is_transition_applicable(v: Vehicle, env: Environment, action: Action) -> bool:
    """
    Checks both FSM validity AND runtime preconditions beyond the state.

    The only such precondition today is lane-bounds for CHANGE_LEFT /
    CHANGE_RIGHT: being on the leftmost lane makes CHANGE_LEFT inapplicable
    even though the FSM transition (ON_LANE, MULTI_LANE, CHANGE_LEFT) is
    defined.
    """
    try:
        state = derive_state(v, env)
    except ValueError:
        return False

    if not is_transition_defined(state, env, action):
        return False

    # Runtime precondition: lane edges
    if action == Action.CHANGE_LEFT and env == Environment.MULTI_LANE:
        idx = _lane_index(v.position)
        return idx > 0
    if action == Action.CHANGE_RIGHT and env == Environment.MULTI_LANE:
        idx = _lane_index(v.position)
        return 0 <= idx < len(_LANE_ORDER) - 1

    return True


def valid_actions(v: Vehicle, env: Environment) -> list[Action]:
    """
    Returns every Action that is applicable to `v` under `env` right now,
    according to the FSM + runtime preconditions.
    """
    return [a for a in Action if is_transition_applicable(v, env, a)]


def _lane_index(position: str) -> int:
    """Returns index of lane in LEFT-CENTER-RIGHT order, or -1 if not a lane."""
    try:
        return _LANE_ORDER.index(position)
    except ValueError:
        return -1
