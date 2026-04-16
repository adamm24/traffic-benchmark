from .entities import ScenarioState, Vehicle, Environment


# ── Position descriptions ────────────────────────────────────────────────────

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


# ── Vehicle description ───────────────────────────────────────────────────────

def describe_vehicle(v: Vehicle) -> str:
    """
    Returns a natural language description of a vehicle's initial state.
    Examples:
      - "Vehicle A is in the left lane."
      - "Vehicle B is at the northern approach, intending to turn left."
    """
    base = f"Vehicle {v.id} is in {_pos(v.position)}"
    if v.intent:
        base += f", intending to {v.intent.value}"
    return base + "."


# ── Scenario description ──────────────────────────────────────────────────────

ENV_LABELS = {
    "intersection":    "an intersection",
    "multi_lane_road": "a multi-lane road",
    "roundabout":      "a roundabout",
}

def describe_scenario(state: ScenarioState) -> str:
    """
    Returns a full natural language description of the initial scenario.
    Example output:
        Three vehicles are at an intersection.
        Vehicle A is in the northern approach, intending to go straight.
        Vehicle B is in the eastern approach, intending to turn left.
        Vehicle C is in the southern approach.
    """
    count_word = {2: "Two", 3: "Three", 4: "Four", 5: "Five"}.get(
        len(state.vehicles), str(len(state.vehicles))
    )
    env_label = ENV_LABELS.get(state.environment.value, state.environment.value)
    header = f"{count_word} vehicles are at {env_label}."
    lines = [header] + [describe_vehicle(v) for v in state.vehicles]
    return "\n".join(lines)


# ── Event sequence description ────────────────────────────────────────────────

def describe_events(events: list[str]) -> str:
    """
    Formats a list of event strings as a numbered sequence.
    Example output:
        Sequence of events:
        1. Vehicle A moves forward.
        2. Vehicle B changes to the left lane.
        3. Vehicle C stops.
    """
    if not events:
        return "No events occurred."
    numbered = "\n".join(f"{i+1}. {e}" for i, e in enumerate(events))
    return f"Sequence of events:\n{numbered}"


# ── Full prompt renderer ──────────────────────────────────────────────────────

def render_prompt(scenario_text: str,
                  events: list[str],
                  question: str,
                  choices: dict[str, str]) -> str:
    """
    Assembles the full multiple-choice prompt ready for LLM evaluation.

    Args:
        scenario_text: output of describe_scenario()
        events:        list of event strings
        question:      the question string
        choices:       dict mapping "A".."E" to answer text

    Returns:
        A single formatted string.
    """
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
