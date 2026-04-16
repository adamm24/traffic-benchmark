import random
from .entities import (
    Vehicle, ScenarioState, Environment,
    Direction, Lane, IntentDirection
)


ALL_DIRECTIONS = list(Direction)
ALL_LANES      = [Lane.LEFT, Lane.CENTER, Lane.RIGHT]


def build_intersection_scenario(num_vehicles: int = 3,
                                 with_intent: bool = False) -> ScenarioState:
    """Builds a valid starting scenario at an intersection."""
    directions = random.sample(ALL_DIRECTIONS, num_vehicles)
    vehicles = []
    for i, d in enumerate(directions):
        vid = chr(65 + i)  # "A", "B", "C"
        intent = random.choice(list(IntentDirection)[:3]) if with_intent else None
        v = Vehicle(
            id=vid,
            position=f"{d.value}_approach",
            direction=d,
            intent=intent
        )
        vehicles.append(v)
    return ScenarioState(vehicles=vehicles, environment=Environment.INTERSECTION)


def build_multi_lane_scenario(num_vehicles: int = 3) -> ScenarioState:
    """Builds a valid starting scenario on a multi-lane road."""
    lanes = random.sample(ALL_LANES, num_vehicles)
    vehicles = []
    for i, lane in enumerate(lanes):
        vid = chr(65 + i)
        v = Vehicle(
            id=vid,
            position=lane.value,
            direction=Direction.NORTH,
            intent=None
        )
        vehicles.append(v)
    return ScenarioState(vehicles=vehicles, environment=Environment.MULTI_LANE)


def build_roundabout_scenario(num_vehicles: int = 3) -> ScenarioState:
    """Builds a valid starting scenario at a roundabout."""
    directions = random.sample(ALL_DIRECTIONS, num_vehicles)
    vehicles = []
    for i, d in enumerate(directions):
        vid = chr(65 + i)
        inside = (i == 0)  # first vehicle is already inside
        v = Vehicle(
            id=vid,
            position="roundabout_lane" if inside else f"{d.value}_approach",
            direction=d,
            inside_intersection=inside
        )
        vehicles.append(v)
    return ScenarioState(vehicles=vehicles, environment=Environment.ROUNDABOUT)


SCENARIO_BUILDERS = {
    Environment.INTERSECTION: build_intersection_scenario,
    Environment.MULTI_LANE:   build_multi_lane_scenario,
    Environment.ROUNDABOUT:   build_roundabout_scenario,
}

def build_scenario(env: Environment, **kwargs) -> ScenarioState:
    return SCENARIO_BUILDERS[env](**kwargs)
