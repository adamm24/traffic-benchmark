# Domain Documentation — Traffic Benchmark

**Author:** Adam Amrani  
**Last updated:** April 2026

---

## 1. Overview

The `domain/` module defines the shared contract for all benchmark generators. It specifies the state representation, traffic rules, simulation engine, and output vocabulary. Generators are not permitted to define positions, rules, or labels outside this module.

The domain models a discrete road traffic scenario across three environments:

| Environment | Description |
|---|---|
| `intersection` | Unsignalized four-way intersection |
| `multi_lane_road` | Three-lane road (left, center, right), all vehicles travelling north |
| `roundabout` | Rotary with approach arms and an internal circulation lane |

---

## 2. Module Structure

```
domain/
├── entities.py     # Core data classes and enums
├── rules.py        # Traffic rules: right-of-way, violation detection
├── fsm.py          # Finite-state machine: vehicle lifecycle states and transitions
├── scenario.py     # Scenario builders and the apply_action() simulation engine
├── render.py       # State-to-text conversion for LLM prompts
└── vocabulary.py   # Controlled vocabulary: position keys to human-readable labels
```

---

## 3. State Representation

### Enums (`entities.py`)

- **`Direction`** — four cardinal directions: `NORTH`, `SOUTH`, `EAST`, `WEST`
- **`Action`** — eight vehicle actions: `MOVE_FORWARD`, `STOP`, `TURN_LEFT`, `TURN_RIGHT`, `CHANGE_LEFT`, `CHANGE_RIGHT`, `ENTER_ROUNDABOUT`, `EXIT_ROUNDABOUT`
- **`Environment`** — three environments: `INTERSECTION`, `MULTI_LANE`, `ROUNDABOUT`
- **`Lane`** — four lane positions: `LEFT`, `CENTER`, `RIGHT`, `ROUNDABOUT_LANE`
- **`IntentDirection`** — declared movement intention, used only in Task 2: `GO_STRAIGHT`, `TURN_LEFT`, `TURN_RIGHT`
- **`VehicleState`** — derived lifecycle state used by the FSM: `APPROACHING`, `INSIDE_INTERSECTION`, `EXITED_INTERSECTION`, `ON_LANE`, `ROUNDABOUT_APPROACHING`, `IN_ROUNDABOUT`, `EXITED_ROUNDABOUT`

### `Vehicle`

A dataclass with the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Single-letter identifier: `A`, `B`, or `C` |
| `position` | `str` | Current position key (e.g. `north_approach`, `left_lane`) |
| `direction` | `Direction` | Compass direction the vehicle faces |
| `intent` | `IntentDirection` or `None` | Declared intention at an intersection (Task 2 only) |
| `inside_intersection` | `bool` | True when the vehicle is inside the intersection or roundabout ring |
| `stopped` | `bool` | True after a STOP action |

`position` and `inside_intersection` are kept in sync by `apply_action()`. When `inside_intersection` is `True`, `position` is always `"inside_intersection"` (for intersections) or `"roundabout_lane"` (for roundabouts).

### `ScenarioState`

Holds the full simulation state:

- `vehicles`: list of three `Vehicle` objects
- `environment`: the active `Environment`
- `step`: integer step counter, incremented with each applied action
- `event_log`: list of natural-language event strings produced by `apply_action()`

---

## 4. Traffic Rules (`rules.py`)

### Right-of-way at intersections

Two functions handle intersection priority:

- `right_of_way_intersection(v1, v2)` — direction-only priority using the `APPROACH_PRIORITY` lookup table. Returns the id of the vehicle with right-of-way, or `None` if no conflict exists.
- `right_of_way_intersection_with_intent(v1, v2)` — intent-aware priority. First applies the left-turn yield rule (a vehicle turning left yields to oncoming traffic), then falls back to direction-only priority.

The dispatcher `right_of_way(v1, v2, env)` selects the appropriate function based on the environment and whether both vehicles have a declared intent.

`APPROACH_PRIORITY` covers the eight lateral conflict pairs (e.g. north–east, south–west). Pairs of vehicles approaching from directly opposite directions return `None` — these are treated as having no direct conflict.

### Right-of-way in roundabouts

`right_of_way_roundabout(v_inside, v_entering)` always returns the id of the circulating vehicle. A vehicle already inside the roundabout has unconditional priority over one attempting to enter.

### Violation detection

- `detect_fsm_violation(state, vehicle_id, action)` — checks whether the action is a valid FSM transition. Returns a result dict with `is_violation`, `violation_type`, and a reason string.
- `detect_right_of_way_violation(state, vehicle_id, action)` — checks whether the action violates priority rules (entering an intersection without right-of-way, entering a roundabout without yielding).

`detect_violation(state, vehicle_id, action)` runs both checks in sequence and returns the first violation found, or a no-violation result.

### Valid actions per environment

`get_valid_actions(env)` returns the set of actions that are physically valid for each environment:

| Environment | Valid actions |
|---|---|
| `intersection` | MOVE_FORWARD, STOP, TURN_LEFT, TURN_RIGHT |
| `multi_lane_road` | MOVE_FORWARD, STOP, CHANGE_LEFT, CHANGE_RIGHT |
| `roundabout` | ENTER_ROUNDABOUT, EXIT_ROUNDABOUT, MOVE_FORWARD, STOP |

---

## 5. Finite-State Machine (`fsm.py`)

The FSM defines the vehicle lifecycle within each environment as explicit states and transitions.

### State derivation

`derive_state(v, env)` maps the current `position` and `inside_intersection` fields to a `VehicleState`. `is_transition_applicable()` uses it before each action to check validity.

### Transition table

The `TRANSITIONS` dictionary encodes valid `(VehicleState, Environment, Action) → VehicleState` mappings. Key entries:

| From state | Environment | Action | To state |
|---|---|---|---|
| `APPROACHING` | intersection | MOVE_FORWARD | `INSIDE_INTERSECTION` |
| `INSIDE_INTERSECTION` | intersection | TURN_LEFT / TURN_RIGHT | `EXITED_INTERSECTION` |
| `ON_LANE` | multi_lane_road | CHANGE_LEFT / CHANGE_RIGHT | `ON_LANE` |
| `ROUNDABOUT_APPROACHING` | roundabout | ENTER_ROUNDABOUT | `IN_ROUNDABOUT` |
| `IN_ROUNDABOUT` | roundabout | EXIT_ROUNDABOUT | `EXITED_ROUNDABOUT` |
| Any state | any | STOP | same state |

MOVE_FORWARD from an exit position is intentionally absent to prevent re-entry loops. MOVE_FORWARD on multi-lane is also absent because it is a no-op in the discrete lane model.

### `is_transition_applicable(v, env, action)`

Combines table lookup with runtime lane-boundary checks for CHANGE_LEFT and CHANGE_RIGHT: a vehicle must not already be at the leftmost or rightmost lane.

---

## 6. Simulation Engine (`scenario.py`)

### Scenario builders

Three functions initialize a valid starting state:

- `build_intersection_scenario(num_vehicles, with_intent)` — assigns distinct approach directions to each vehicle; optionally assigns a random `IntentDirection`
- `build_multi_lane_scenario(num_vehicles)` — assigns distinct lanes from {LEFT, CENTER, RIGHT}
- `build_roundabout_scenario(num_vehicles)` — places the first vehicle inside the roundabout ring, the rest at approach positions

All builders use `random.sample` to ensure no two vehicles start at the same position.

### `apply_action(state, vehicle_id, action)`

The central simulation function. Applies one action to the named vehicle, mutates `ScenarioState` in place, and returns a natural-language event string. If the action has no effect (e.g. CHANGE_LEFT from a non-lane position), no event is generated and `step` is not incremented.

Behavior per action type:

- `MOVE_FORWARD` at intersection: sets `inside_intersection = True`, `position = "inside_intersection"`
- `TURN_LEFT` / `TURN_RIGHT`: updates `direction`, sets `position` to the corresponding exit arm, sets `inside_intersection = False`
- `CHANGE_LEFT` / `CHANGE_RIGHT`: updates `position` only if the lane-boundary check passes
- `ENTER_ROUNDABOUT`: sets `inside_intersection = True`, `position = "roundabout_lane"`
- `EXIT_ROUNDABOUT`: sets `inside_intersection = False`, `position` to the directional exit arm
- `STOP`: sets `stopped = True`

---

## 7. Vocabulary (`vocabulary.py`)

All human-readable position labels used in prompts and answer choices are defined in `POSITION_LABELS`, a dictionary mapping internal position keys to English strings. Examples:

| Key | Label |
|---|---|
| `north_approach` | the northern approach |
| `inside_intersection` | inside the intersection |
| `east_exit` | the eastern exit |
| `roundabout_lane` | the roundabout lane |

The function `label_of(position)` raises a `ValueError` for any key not in the dictionary, which prevents generators from producing answer choices with arbitrary strings.

**Known gap:** The three active lane positions used in simulation (`left_lane`, `center_lane`, `right_lane`) are not registered in `POSITION_LABELS`. Task 4 handles this with a local mapping inside its generator. No other task requires human-readable labels for these positions.

---

## 8. Design Assumptions

The following design decisions are intentional constraints on the domain model. They are documented here for clarity when reading the generators or extending the benchmark.

**Lane occupancy.** In the discrete multi-lane model, two vehicles may occupy the same lane at the same time. They are understood to be at different longitudinal positions along the lane. The benchmark evaluates lateral position tracking (which lane a vehicle is in), not spatial separation within a lane. Enforcing exclusive lane occupancy would make most CHANGE_LEFT/CHANGE_RIGHT actions impossible with three vehicles on three lanes.

**MOVE_FORWARD semantics.** The action has different effects depending on the environment. In intersections, it moves a vehicle from an approach arm into the center. On multi-lane roads, it has no effect on position and is excluded from generator action pools in Task 1. In roundabouts, dedicated ENTER_ROUNDABOUT and EXIT_ROUNDABOUT actions are used instead.

**`intent` and right-of-way.** The `IntentDirection` field is visible in prompts, but the direction-only function `right_of_way_intersection()` does not use it. The intent-aware function `right_of_way_intersection_with_intent()` is used in Task 2 when both vehicles have a declared intent. For all other tasks, `intent` is `None`.

**Action pools in generators.** `get_valid_actions(env)` in `rules.py` lists all actions that are physically valid for an environment. Individual generators maintain their own filtered pools that exclude actions which are valid at the domain level but uninformative for a specific task (e.g. MOVE_FORWARD is excluded from Task 1 on multi-lane roads).

**Opposite-direction conflicts.** Pairs of vehicles approaching from opposite directions return `None` from `right_of_way_intersection()`, meaning no conflict is assigned. Scenarios requiring priority resolution between opposite-direction vehicles are excluded from the current generators.

**Render module.** `render.py` is a pure formatting layer — it converts `ScenarioState` objects to natural-language text for LLM prompts. It contains no domain logic.
