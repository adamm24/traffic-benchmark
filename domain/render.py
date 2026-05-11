from .entities import ScenarioState, Vehicle, Environment



POSITION_LABELS = {
    "left_lane":            "the left lane",
    "center_lane":          "the center lane",
    "right_lane":           "the right lane",
    "roundabout_lane":      "the roundabout lane",
    "inside_intersection":  "inside the intersection",
    "north_approach":       "the northern approach",
    "south_approach":       "the southern approach",
    "east_approach":        "the eastern approach",
    "west_approach":        "the western approach",
    "north_exit":           "the northern exit",
    "south_exit":           "the southern exit",
    "east_exit":            "the eastern exit",
    "west_exit":            "the western exit",
}

def _pos(position: str) -> str:
    return POSITION_LABELS.get(position, position.replace("_", " "))

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


def _position_clause(position_label: str) -> str:
    label = position_label.strip()
    lower = label.lower()
    if any(lower.startswith(prefix) for prefix in _POSITION_PREFIXES):
        return f"is {label}"
    return f"is in {label}"


def describe_vehicle(v: Vehicle) -> str:
    """Natural-language description of a vehicle's initial state."""
    base = f"Vehicle {v.id} {_position_clause(_pos(v.position))}"
    if v.intent:
        base += f", intending to {v.intent.value}"
    return base + "."



ENV_LABELS = {
    "intersection":    "an intersection",
    "multi_lane_road": "a multi-lane road",
    "roundabout":      "a roundabout",
}

def describe_scenario(state: ScenarioState) -> str:
    """Natural-language description of the initial scenario."""
    count_word = {2: "Two", 3: "Three", 4: "Four", 5: "Five"}.get(
        len(state.vehicles), str(len(state.vehicles))
    )
    env_label = ENV_LABELS.get(state.environment.value, state.environment.value)
    header = f"{count_word} vehicles are at {env_label}."
    lines = [header] + [describe_vehicle(v) for v in state.vehicles]
    return "\n".join(lines)



def describe_events(events: list[str]) -> str:
    """Format events as a numbered sequence."""
    if not events:
        return "No events occurred."
    numbered = "\n".join(f"{i+1}. {e}" for i, e in enumerate(events))
    return f"Sequence of events:\n{numbered}"



def render_prompt(scenario_text: str,
                  events: list[str],
                  question: str,
                  choices: dict[str, str]) -> str:
    """Assemble the multiple-choice prompt."""
    parts = [
        scenario_text,
        "",
        describe_events(events),
        "",
        f"Question: {question}",
    ]
    for key in sorted(choices):
        parts.append(f"{key}) {choices[key]}")
    return "\n".join(parts)
