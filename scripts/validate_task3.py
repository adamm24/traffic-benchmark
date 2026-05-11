"""Independent Task 3 validator and quality auditor."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from domain.entities import Action, Direction, Environment, IntentDirection, ScenarioState, Vehicle
from domain.rules import detect_violation
from domain.scenario import apply_action


LETTERS = ("A", "B", "C", "D", "E")
NO_VIOLATION_OPTION = "No vehicle performed an illegal action"
QUESTION_TEXT = "Which vehicle performs the first illegal action?"

FIXED_OPTIONS = {
    "Vehicle A",
    "Vehicle B",
    "Vehicle C",
    NO_VIOLATION_OPTION,
    "Another vehicle (not A, B, or C)",
}


@dataclass(frozen=True)
class ReplayResult:
    has_violation: bool
    violation_vehicle: str | None
    violation_type: str
    violation_step: int | None
    reason: str
    invalid: bool


@dataclass
class Totals:
    total: int = 0
    wrong: int = 0
    invalid: int = 0
    duplicate_option_texts: int = 0
    undetermined_correct: int = 0


def _parse_intent(value: str | None) -> IntentDirection | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    for it in IntentDirection:
        if it.value == normalized or it.name.lower() == normalized:
            return it
    raise ValueError(f"unknown intent string: {value!r}")


def _reconstruct_state(scenario_json: dict) -> ScenarioState:
    env = Environment(scenario_json["environment"])
    vehicles: list[Vehicle] = []
    for v in scenario_json["vehicles"]:
        vehicles.append(
            Vehicle(
                id=v["id"],
                position=v["position"],
                direction=Direction(v["direction"]),
                intent=_parse_intent(v.get("intent")),
                inside_intersection=bool(v.get("inside_intersection", False)),
                stopped=bool(v.get("stopped", False)),
            )
        )
    return ScenarioState(vehicles=vehicles, environment=env)


def replay_first_violation(example: dict) -> ReplayResult:
    try:
        state = _reconstruct_state(example["scenario"])
    except Exception as exc:  # noqa: BLE001
        return ReplayResult(
            has_violation=True,
            violation_vehicle=None,
            violation_type="reconstruction_error",
            violation_step=None,
            reason=str(exc),
            invalid=True,
        )

    plan = example.get("event_plan", [])
    events = example.get("events", [])

    if len(plan) != len(events):
        return ReplayResult(
            has_violation=True,
            violation_vehicle=None,
            violation_type="event_length_mismatch",
            violation_step=None,
            reason="event_plan length does not match events length",
            invalid=True,
        )

    for idx, step in enumerate(plan, start=1):
        try:
            vehicle_id = step["vehicle"]
            action = Action[step["action"]]
        except Exception as exc:  # noqa: BLE001
            return ReplayResult(
                has_violation=True,
                violation_vehicle=None,
                violation_type="event_parse_error",
                violation_step=idx,
                reason=str(exc),
                invalid=True,
            )

        detection = detect_violation(state, vehicle_id, action)
        if detection["is_violation"]:
            return ReplayResult(
                has_violation=True,
                violation_vehicle=vehicle_id,
                violation_type=str(detection["violation_type"]),
                violation_step=idx,
                reason=str(detection["reason"]),
                invalid=False,
            )

        emitted = apply_action(state, vehicle_id, action)
        if not emitted:
            return ReplayResult(
                has_violation=True,
                violation_vehicle=vehicle_id,
                violation_type="apply_failed_after_legal_check",
                violation_step=idx,
                reason="Action failed to apply after legal pre-check.",
                invalid=True,
            )

    return ReplayResult(
        has_violation=False,
        violation_vehicle=None,
        violation_type="no_violation",
        violation_step=None,
        reason="No violation detected.",
        invalid=False,
    )


def _semantic_from_replay(replay: ReplayResult) -> str:
    if not replay.has_violation:
        return NO_VIOLATION_OPTION
    return f"Vehicle {replay.violation_vehicle}"


def validate_example(example: dict) -> tuple[bool, bool, str, ReplayResult]:
    choices = example.get("choices", {})
    answer = example.get("answer")
    question = example.get("question")

    if question != QUESTION_TEXT:
        replay = replay_first_violation(example)
        return False, True, "question text mismatch", replay

    if sorted(choices.keys()) != list(LETTERS):
        replay = replay_first_violation(example)
        return False, True, "choices must contain keys A..E", replay

    if len(set(choices.values())) != 5:
        replay = replay_first_violation(example)
        return False, False, "duplicate option texts", replay

    if set(choices.values()) != FIXED_OPTIONS:
        replay = replay_first_violation(example)
        return False, True, "choice set is not fixed semantic set", replay

    if answer not in LETTERS:
        replay = replay_first_violation(example)
        return False, True, "invalid answer letter", replay

    option_rationale = example.get("audit", {}).get("option_rationale_by_letter", {})
    if sorted(option_rationale.keys()) != list(LETTERS):
        replay = replay_first_violation(example)
        return False, True, "missing option rationale coverage in audit", replay
    if any(not str(option_rationale[k]).strip() for k in LETTERS):
        replay = replay_first_violation(example)
        return False, True, "empty option rationale in audit", replay

    if "option_rationale_by_letter" in example.get("metadata", {}):
        replay = replay_first_violation(example)
        return False, True, "option rationale duplicated in metadata", replay

    replay = replay_first_violation(example)
    if replay.invalid:
        return False, True, f"invalid replay: {replay.reason}", replay

    answer_text = choices[answer]
    replay_answer_text = _semantic_from_replay(replay)
    if answer_text != replay_answer_text:
        return False, False, (
            f"wrong answer text: answer={answer_text!r}, replay={replay_answer_text!r}"
        ), replay

    metadata = example.get("metadata", {})
    expected_gt = replay.violation_vehicle if replay.has_violation else "no_violation"
    if metadata.get("ground_truth") != expected_gt:
        return False, False, "metadata.ground_truth mismatch", replay

    if replay.has_violation:
        if metadata.get("violation_vehicle") != replay.violation_vehicle:
            return False, False, "metadata.violation_vehicle mismatch", replay
        if metadata.get("violation_type") != replay.violation_type:
            return False, False, "metadata.violation_type mismatch", replay
        if metadata.get("violation_step") != replay.violation_step:
            return False, False, "metadata.violation_step mismatch", replay
    else:
        if metadata.get("violation_vehicle") is not None:
            return False, False, "metadata.violation_vehicle must be null", replay
        if metadata.get("violation_type") != "no_violation":
            return False, False, "metadata.violation_type must be no_violation", replay
        if metadata.get("violation_step") is not None:
            return False, False, "metadata.violation_step must be null", replay

    return True, False, "ok", replay


def validate_file(path: Path) -> int:
    if not path.exists():
        print(f"ERROR: input file not found: {path}")
        return 2

    totals = Totals()
    answer_counts = Counter()
    gt_counts = Counter()
    env_counts = Counter()
    vtype_counts = Counter()
    seq_len_counts = Counter()
    no_violation_seq_len_counts = Counter()
    violation_seq_len_counts = Counter()
    violation_step_counts = Counter()
    violator_counts = Counter()
    class_env_counts = Counter()
    undetermined_letter_counts = Counter()
    fifth_letter_counts = Counter()
    acted_vehicle_count_distribution = Counter()
    event_sequence_counts = Counter()
    legal_prefix_count = 0
    non_final_violation_count = 0

    violator_last_actor = 0
    violating_examples = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            totals.total += 1
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as exc:
                totals.invalid += 1
                print(f"[invalid] line {line_no}: JSON decode error: {exc}")
                continue

            ok, is_invalid, reason, replay = validate_example(ex)

            choices = ex.get("choices", {})
            if len(set(choices.values())) != len(choices.values()):
                totals.duplicate_option_texts += 1
            for letter, text in choices.items():
                if text == NO_VIOLATION_OPTION:
                    undetermined_letter_counts[letter] += 1
                if text == "Another vehicle (not A, B, or C)":
                    fifth_letter_counts[letter] += 1

            answer = ex.get("answer")
            if answer in LETTERS:
                answer_counts[answer] += 1

            metadata = ex.get("metadata", {})
            gt_counts[metadata.get("ground_truth")] += 1
            env_counts[metadata.get("environment")] += 1
            vtype_counts[metadata.get("violation_type")] += 1
            class_env_counts[(metadata.get("violation_type"), metadata.get("environment"))] += 1
            sig = "|".join(f"{s.get('vehicle')}:{s.get('action')}" for s in ex.get("event_plan", []))
            event_sequence_counts[sig] += 1

            seq_len = int(metadata.get("num_events", len(ex.get("event_plan", []))))
            seq_len_counts[seq_len] += 1
            acted_vehicle_count_distribution[len({s.get("vehicle") for s in ex.get("event_plan", [])})] += 1
            if metadata.get("ground_truth") == "no_violation":
                no_violation_seq_len_counts[seq_len] += 1
            else:
                violation_seq_len_counts[seq_len] += 1

            if choices.get(answer) == NO_VIOLATION_OPTION:
                totals.undetermined_correct += 1

            if replay.has_violation:
                violating_examples += 1
                if replay.violation_step is not None:
                    violation_step_counts[replay.violation_step] += 1
                    if replay.violation_step >= 2:
                        legal_prefix_count += 1
                    if replay.violation_step < seq_len:
                        non_final_violation_count += 1
                if replay.violation_vehicle is not None:
                    violator_counts[replay.violation_vehicle] += 1
                plan = ex.get("event_plan", [])
                if plan:
                    if plan[-1].get("vehicle") == replay.violation_vehicle:
                        violator_last_actor += 1

            if not ok:
                if is_invalid:
                    totals.invalid += 1
                    print(f"[invalid] line {line_no}: {reason}")
                else:
                    totals.wrong += 1
                    print(f"[wrong] line {line_no}: {reason}")

    if totals.undetermined_correct != gt_counts["no_violation"]:
        totals.wrong += 1
        print(
            "[wrong] undetermined-correct count mismatch: "
            f"{totals.undetermined_correct} vs no_violation {gt_counts['no_violation']}"
        )

    print("\n=== Task 3 Validation Summary ===")
    print(f"examples: {totals.total}")
    print(f"wrong: {totals.wrong}")
    print(f"invalid: {totals.invalid}")
    print(f"duplicate option texts: {totals.duplicate_option_texts}")
    print(f"undetermined correct: {totals.undetermined_correct}")

    if totals.total > 0:
        no_violation_ratio = gt_counts["no_violation"] / totals.total
        print(f"no_violation ratio: {no_violation_ratio:.3f} ({gt_counts['no_violation']}/{totals.total})")

    print("\nAnswer key distribution:")
    for k in LETTERS:
        print(f"  {k}: {answer_counts[k]}")

    print("\nGround truth distribution:")
    for k in ["A", "B", "C", "no_violation"]:
        print(f"  {k}: {gt_counts[k]}")

    print("\nEnvironment distribution:")
    for k in sorted(env_counts):
        print(f"  {k}: {env_counts[k]}")

    print("\nViolation type distribution:")
    for k in sorted(vtype_counts):
        print(f"  {k}: {vtype_counts[k]}")

    if violating_examples > 0:
        ratio = violator_last_actor / violating_examples
        print("\nViolator-is-last-actor:")
        print(f"  {violator_last_actor}/{violating_examples} ({ratio:.3f})")

    print("\nSequence length distribution:")
    for k in sorted(seq_len_counts):
        print(f"  {k}: {seq_len_counts[k]}")

    avg_len_all = (
        sum(k * v for k, v in seq_len_counts.items()) / totals.total
        if totals.total else 0.0
    )
    avg_len_no_violation = (
        sum(k * v for k, v in no_violation_seq_len_counts.items()) / gt_counts["no_violation"]
        if gt_counts["no_violation"] else 0.0
    )
    avg_len_violation = (
        sum(k * v for k, v in violation_seq_len_counts.items()) / violating_examples
        if violating_examples else 0.0
    )
    print(f"  avg_all: {avg_len_all:.3f}")
    print(f"  avg_no_violation: {avg_len_no_violation:.3f}")
    print(f"  avg_violation: {avg_len_violation:.3f}")

    print("\nNo-violation sequence lengths:")
    for k in sorted(no_violation_seq_len_counts):
        print(f"  {k}: {no_violation_seq_len_counts[k]}")

    print("\nViolation sequence lengths:")
    for k in sorted(violation_seq_len_counts):
        print(f"  {k}: {violation_seq_len_counts[k]}")

    print("\nViolation step position distribution:")
    for k in sorted(violation_step_counts):
        print(f"  {k}: {violation_step_counts[k]}")

    if violating_examples > 0:
        print("\nViolation prefix/non-final stats:")
        print(f"  legal-prefix (step>=2): {legal_prefix_count}/{violating_examples} ({legal_prefix_count/violating_examples:.3f})")
        print(f"  non-final violation step: {non_final_violation_count}/{violating_examples} ({non_final_violation_count/violating_examples:.3f})")

    print("\nViolator distribution:")
    for vid in ["A", "B", "C"]:
        print(f"  {vid}: {violator_counts[vid]}")

    print("\nActing-vehicle count distribution:")
    for k in sorted(acted_vehicle_count_distribution):
        print(f"  {k}: {acted_vehicle_count_distribution[k]}")

    print("\nViolation-class x environment:")
    for (vclass, env), c in sorted(class_env_counts.items()):
        print(f"  {vclass} @ {env}: {c}")

    print("\nOption-position distribution:")
    print(f"  {NO_VIOLATION_OPTION}:")
    for k in LETTERS:
        print(f"    {k}: {undetermined_letter_counts[k]}")
    print("  Another vehicle (not A, B, or C):")
    for k in LETTERS:
        print(f"    {k}: {fifth_letter_counts[k]}")

    repeated_event_sequence_count = sum(1 for c in event_sequence_counts.values() if c > 1)
    max_event_sequence_repeat = max(event_sequence_counts.values()) if event_sequence_counts else 0
    print("\nEvent-sequence repetition:")
    print(f"  repeated_event_sequence_count: {repeated_event_sequence_count}")
    print(f"  max_event_sequence_repeat: {max_event_sequence_repeat}")

    # Quality warnings (non-fatal): shortcut signals to inspect.
    warnings: list[str] = []
    if violating_examples > 0 and (violator_last_actor / violating_examples) > 0.75:
        warnings.append("violator-too-often-last-actor")
    if violating_examples > 0 and (legal_prefix_count / violating_examples) < 0.40:
        warnings.append("too-few-legal-prefix-violations")
    if violating_examples > 0 and (non_final_violation_count / violating_examples) < 0.15:
        warnings.append("too-few-non-final-violations")
    if no_violation_seq_len_counts[1] > 0:
        warnings.append("some-no-violation-examples-have-length-1")
    if max_event_sequence_repeat > 3:
        warnings.append("event-sequence-template-repetition-high")
    if warnings:
        print("\nQuality warnings:")
        for w in warnings:
            print(f"  - {w}")

    if totals.wrong == 0 and totals.invalid == 0 and totals.duplicate_option_texts == 0:
        return 0
    return 1


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Task 3 violation dataset")
    p.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "core" / "task3_violation.jsonl"),
        help="Task 3 JSONL file",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    raise SystemExit(validate_file(Path(args.input)))
