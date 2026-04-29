"""Smoke tests for the domain layer."""
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
from domain.rules import detect_fsm_violation
from domain.rules import detect_right_of_way_violation
from domain.rules import detect_violation
from domain.rules import is_valid_transition
from domain.scenario import apply_action
from domain.trajectory import trajectories_conflict, trajectory_of



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


class ViolationDetectionTests(unittest.TestCase):
    def test_turn_without_entering_detected(self):
        v = Vehicle(id="A", position="north_approach", direction=Direction.NORTH)
        state = ScenarioState(vehicles=[v], environment=Environment.INTERSECTION)
        out = detect_fsm_violation(state, "A", Action.TURN_LEFT)
        self.assertTrue(out["is_violation"])
        self.assertEqual(out["violation_type"], "turn_without_entering")

    def test_forward_from_exit_detected(self):
        v = Vehicle(id="A", position="east_exit", direction=Direction.EAST)
        state = ScenarioState(vehicles=[v], environment=Environment.INTERSECTION)
        out = detect_fsm_violation(state, "A", Action.MOVE_FORWARD)
        self.assertTrue(out["is_violation"])
        self.assertEqual(out["violation_type"], "forward_from_exit")

    def test_lane_boundaries_detected(self):
        vl = Vehicle(id="A", position="left_lane", direction=Direction.NORTH)
        vr = Vehicle(id="B", position="right_lane", direction=Direction.NORTH)
        state = ScenarioState(
            vehicles=[vl, vr],
            environment=Environment.MULTI_LANE,
        )
        left = detect_fsm_violation(state, "A", Action.CHANGE_LEFT)
        right = detect_fsm_violation(state, "B", Action.CHANGE_RIGHT)
        self.assertTrue(left["is_violation"])
        self.assertEqual(left["violation_type"], "lane_change_out_of_bounds_left")
        self.assertTrue(right["is_violation"])
        self.assertEqual(right["violation_type"], "lane_change_out_of_bounds_right")

    def test_intersection_right_of_way_violation_detected(self):
        a = Vehicle(id="A", position="north_approach", direction=Direction.NORTH)
        b = Vehicle(id="B", position="east_approach", direction=Direction.EAST)
        c = Vehicle(id="C", position="south_approach", direction=Direction.SOUTH)
        state = ScenarioState(vehicles=[a, b, c], environment=Environment.INTERSECTION)
        out = detect_right_of_way_violation(state, "A", Action.MOVE_FORWARD)
        self.assertTrue(out["is_violation"])
        self.assertEqual(out["violation_type"], "intersection_right_of_way")

    def test_roundabout_entry_violation_detected(self):
        a = Vehicle(id="A", position="north_approach", direction=Direction.NORTH)
        b = Vehicle(
            id="B",
            position="roundabout_lane",
            direction=Direction.EAST,
            inside_intersection=True,
        )
        c = Vehicle(id="C", position="west_approach", direction=Direction.WEST)
        state = ScenarioState(vehicles=[a, b, c], environment=Environment.ROUNDABOUT)
        out = detect_right_of_way_violation(state, "A", Action.ENTER_ROUNDABOUT)
        self.assertTrue(out["is_violation"])
        self.assertEqual(out["violation_type"], "roundabout_entry_no_yield")

    def test_stop_is_legal_and_transition_valid(self):
        v = Vehicle(id="A", position="north_approach", direction=Direction.NORTH)
        state = ScenarioState(vehicles=[v], environment=Environment.INTERSECTION)
        self.assertTrue(is_valid_transition(state, "A", Action.STOP))
        out = detect_violation(state, "A", Action.STOP)
        self.assertFalse(out["is_violation"])


if __name__ == "__main__":
    unittest.main()
