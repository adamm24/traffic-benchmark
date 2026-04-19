"""
Independent Task 2 validator.

Purpose
-------
Re-compute the expected answer for every Task 2 example by rebuilding
the ScenarioState from the JSON scenario block alone (no coupling to
generator internals) and calling `domain.rules.right_of_way`.
Compare against the recorded answer letter and report discrepancies.

Closes T2-B12 (no replay) and the "support a later adversarial critic
pass through better metadata and cleaner generation logic" requirement:
if the generator and the validator ever disagree, we find out at
dataset freeze time, not after the benchmark has shipped.

Usage
-----
    python scripts/validate_task2.py \
        --input dataset/core/task2_rightofway.jsonl

Exit code is 0 iff every example validates, 1 otherwise. Discrepancies
are printed to stderr as structured lines so they can be consumed by CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Make project root importable ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from domain.entities import (
    Direction,
    Environment,
    IntentDirection,
    ScenarioState,
    UnsupportedScenarioError,
    Vehicle,
)
from domain.rules import right_of_way


# ── Scenario reconstruction from JSON alone ─────────────────────────────────

def _parse_direction(value: str) -> Direction:
    return Direction(value)


def _parse_intent(value: str | None) -> IntentDirection | None:
    if value is None:
        return None
    # Intent may be stored as human label ("turn left") or enum name.
    normalized = value.strip().lower()
    for it in IntentDirection:
        if it.value == normalized or it.name.lower() == normalized:
            return it
    raise ValueError(f"Unknown intent string: {value!r}")


def reconstruct_scenario(scenario_json: dict) -> ScenarioState:
    """
    Rebuild a ScenarioState from the JSON `scenario` block of an example.
    Intentionally does NOT call any generator helper — this is the
    validator's independence guarantee.
    """
    env = Environment(scenario_json["environment"])
    vehicles: list[Vehicle] = []
    for v in scenario_json["vehicles"]:
        vehicles.append(
            Vehicle(
                id=v["id"],
                position=v["position"],
                direction=_parse_direction(v["direction"]),
                intent=_parse_intent(v.get("intent")),
                inside_intersection=bool(v.get("inside_intersection", False))
                or v["position"] == "roundabout_lane"
                or v["position"] == "inside_intersection",
                stopped=bool(v.get("stopped", False)),
            )
        )
    return ScenarioState(vehicles=vehicles, environment=env)


# ── Per-example validation ──────────────────────────────────────────────────

def validate_example(example: dict) -> tuple[bool, str]:
    """
    Returns (ok, message). When ok is False, message explains the
    discrepancy.

    Strategy
    --------
    The existing Task 2 JSONL stores `metadata.priority_vehicle`
    (the expected winner) and optionally `metadata.yielding_vehicle`.
    We reconstruct the scenario and verify that `right_of_way(priority,
    other, env) == priority.id` for every other vehicle in the
    scenario. This re-derives the generator's claim from scratch using
    only the JSON scenario block and the domain ruleset.
    """
    state = reconstruct_scenario(example["scenario"])
    metadata = example.get("metadata", {})
    priority_id = metadata.get("priority_vehicle")
    if priority_id is None:
        return False, "example has no metadata.priority_vehicle"

    priority_v = state.get_vehicle(priority_id)
    if priority_v is None:
        return False, f"priority_vehicle {priority_id!r} not in scenario"

    yielding_id = metadata.get("yielding_vehicle")
    candidates = (
        [state.get_vehicle(yielding_id)]
        if yielding_id and state.get_vehicle(yielding_id) is not None
        else [v for v in state.vehicles if v.id != priority_id]
    )

    for other in candidates:
        if other is None:
            continue
        try:
            winner = right_of_way(priority_v, other, state.environment)
        except UnsupportedScenarioError as e:
            return False, (
                f"scenario rejected by dispatcher for pair "
                f"({priority_v.id},{other.id}): {e}"
            )
        if winner is None:
            # No conflict: acceptable only if generator also says so.
            continue
        if winner != priority_id:
            return False, (
                f"pair ({priority_v.id},{other.id}): dispatcher says "
                f"{winner!r} wins but metadata.priority_vehicle={priority_id!r}"
            )

    # Cross-check the answer letter resolves to a text mentioning priority_id.
    answer_letter = example["answer"]
    answer_text = example["choices"][answer_letter].lower()
    if f"vehicle {priority_id.lower()}" not in answer_text and \
            f"vehicle {priority_id}" not in example["choices"][answer_letter]:
        return False, (
            f"answer letter {answer_letter} resolves to "
            f"{example['choices'][answer_letter]!r}, which does not name "
            f"priority_vehicle={priority_id!r}"
        )
    return True, "ok"


# ── Driver ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Independent Task 2 validator.")
    p.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "core" / "task2_rightofway.jsonl"),
        help="path to Task 2 JSONL dataset",
    )
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"input not found: {path}", file=sys.stderr)
        return 2

    total = 0
    failed = 0
    unsupported = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"[LINE {line_no}] invalid JSON: {e}", file=sys.stderr
                )
                failed += 1
                continue

            ok, msg = validate_example(ex)
            if not ok:
                failed += 1
                if "unsupported" in msg:
                    unsupported += 1
                print(
                    f"[FAIL {ex.get('id', f'line{line_no}')}] {msg}",
                    file=sys.stderr,
                )

    print(f"Validated {total} examples; {failed} failed "
          f"({unsupported} unsupported scenarios).")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
