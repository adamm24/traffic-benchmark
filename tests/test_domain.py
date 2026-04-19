"""
Smoke tests for the domain layer.

These are intentionally compact. They cover the three invariants that
changed in this refactor:

  1. FSM guard (domain.fsm + domain.scenario.apply_action):
     invalid transitions must return "" without mutating state.

  2. Trajectory conflict detection (domain.trajectory):
     perpendicular GO_STRAIGHT movements conflict, opposite
     GO_STRAIGHT movements do not, and a left turn across an oncoming
     straight movement IS flagged as a conflict.

  3. Roundabout dispatcher (domain.rules.right_of_way):
     roundabout-without-inside-vehicle is rejected as
     UnsupportedScenarioError instead of silently falling back to
     intersection logic.

Run with:
    python -m unittest tests.test_domain
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from domain.entities import (
    Action,
    Direction,
    Environment,
    IntentDirection,
    ScenarioState,
    UnsupportedScenarioError,
    Vehicle,
    VehicleState,
)
from domain.fsm import (
    derive_state,
    is_transition_applicable,
    next_state,
    valid_actions,
)
from domain.rules import right_of_way
from domain.scenario import apply_action
from domain.trajectory import trajectories_conflict, trajectory_of


# ── FSM tests ───────────────────────────────────────────────────────────────

class FSMTests(unittest.TestCase):
    def test_intersection_approaching_move_forward(self):
        v = Vehicle(id="A", position="north_approach", direction=Direction.NORTH)
        self.assertEqual(
            derive_state(v, Environment.INTERSECTION),
            VehicleState.APPROACHING,
        )
        self.assertIs(
            next_state(VehicleState.APPROACHING, Environment.INTERSECTION,
                       Action.MOVE_FORWARD),
            VehicleState.INSIDE_INTERSECTION,
        )

    def test_exited_cannot_re_enter(self):
        v = Vehicle(
            id="A", position="east_exit", direction=Direction.EAST,
            inside_intersection=False,
        )
        self.assertEqual(
            derive_state(v, Environment.INTERSECTION),
            VehicleState.EXITED_INTERSECTION,
        )
        self.assertFalse(
            is_transition_applicable(v, Environment.INTERSECTION, Action.MOVE_FORWARD)
        )
        state = ScenarioState(vehicles=[v], environment=Environment.INTERSECTION)
        event = apply_action(state, "A", Action.MOVE_FORWARD)
        self.assertEqual(event, "")
        self.assertEqual(state.step, 0)
        self.assertEqual(state.event_log, [])
        self.assertEqual(v.position, "east_exit")

    def test_leftmost_cannot_change_left(self):
        v = Vehicle(id="A", position="left_lane", direction=Direction.NORTH)
        self.assertFalse(
            is_transition_applicable(v, Environment.MULTI_LANE, Action.CHANGE_LEFT)
        )

    def test_valid_actions_on_lane(self):
        v = Vehicle(id="A", position="center_lane", direction=Direction.NORTH)
        actions = set(valid_actions(v, Environment.MULTI_LANE))
        self.assertIn(Action.CHANGE_LEFT, actions)
        self.assertIn(Action.CHANGE_RIGHT, actions)
        self.assertIn(Action.STOP, actions)
        self.assertNotIn(Action.MOVE_FORWARD, actions)


# ── Trajectory tests ────────────────────────────────────────────────────────

def _v(vid: str, direction: Direction, intent: IntentDirection) -> Vehicle:
    return Vehicle(
        id=vid,
        position=f"{direction.value}_approach",
        direction=direction,
        intent=intent,
    )


class TrajectoryTests(unittest.TestCase):
    def test_opposite_straight_no_conflict(self):
        a = _v("A", Direction.NORTH, IntentDirection.GO_STRAIGHT)
        b = _v("B", Direction.SOUTH, IntentDirection.GO_STRAIGHT)
        self.assertFalse(trajectories_conflict(a, b))

    def test_perpendicular_straight_conflict(self):
        a = _v("A", Direction.NORTH, IntentDirection.GO_STRAIGHT)
        b = _v("B", Direction.EAST, IntentDirection.GO_STRAIGHT)
        self.assertTrue(trajectories_conflict(a, b))

    def test_left_turn_vs_opposing_straight(self):
        # North turning left must yield to oncoming south-to-north traffic.
        a = _v("A", Direction.NORTH, IntentDirection.TURN_LEFT)
        b = _v("B", Direction.SOUTH, IntentDirection.GO_STRAIGHT)
        self.assertTrue(trajectories_conflict(a, b))

    def test_right_turn_is_smallest_trajectory(self):
        a = _v("A", Direction.NORTH, IntentDirection.TURN_RIGHT)
        self.assertEqual(len(trajectory_of(a)), 1)


# ── Dispatcher tests ────────────────────────────────────────────────────────

class DispatcherTests(unittest.TestCase):
    def test_roundabout_no_inside_vehicle_raises(self):
        a = Vehicle(id="A", position="north_approach", direction=Direction.NORTH)
        b = Vehicle(id="B", position="east_approach", direction=Direction.EAST)
        with self.assertRaises(UnsupportedScenarioError):
            right_of_way(a, b, Environment.ROUNDABOUT)

    def test_roundabout_inside_wins(self):
        a = Vehicle(
            id="A", position="roundabout_lane",
            direction=Direction.NORTH, inside_intersection=True,
        )
        b = Vehicle(id="B", position="east_approach", direction=Direction.EAST)
        self.assertEqual(right_of_way(a, b, Environment.ROUNDABOUT), "A")

    def test_intersection_no_intent_priority_to_right(self):
        # North vs East: east is to the north's right → east wins.
        a = Vehicle(id="A", position="north_approach", direction=Direction.NORTH)
        b = Vehicle(id="B", position="east_approach", direction=Direction.EAST)
        self.assertEqual(
            right_of_way(a, b, Environment.INTERSECTION),
            "B",
        )

    def test_intersection_with_intent_left_yields(self):
        # North turning left vs south going straight → south wins.
        a = _v("A", Direction.NORTH, IntentDirection.TURN_LEFT)
        b = _v("B", Direction.SOUTH, IntentDirection.GO_STRAIGHT)
        self.assertEqual(
            right_of_way(a, b, Environment.INTERSECTION),
            "B",
        )

    def test_intersection_with_intent_opposite_straight_no_conflict(self):
        a = _v("A", Direction.NORTH, IntentDirection.GO_STRAIGHT)
        b = _v("B", Direction.SOUTH, IntentDirection.GO_STRAIGHT)
        self.assertIsNone(right_of_way(a, b, Environment.INTERSECTION))


if __name__ == "__main__":
    unittest.main()
