"""Lightweight Task 4 distribution checker."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from domain.entities import Environment
import validate_task4 as validator


LETTERS = ("A", "B", "C", "D", "E")
EXPECTED_ANSWER_COUNTS_100 = {letter: 20 for letter in LETTERS}
EXPECTED_ENV_COUNTS_100 = {
    Environment.INTERSECTION.value: 50,
    Environment.ROUNDABOUT.value: 30,
    Environment.MULTI_LANE.value: 20,
}
EXPECTED_DIFFICULTY_COUNTS_100 = {"easy": 33, "medium": 33, "hard": 34}
RE_SINGLE_VEHICLE = re.compile(r"^Vehicle ([ABC]) ")

STATE_FACT_PATTERNS = (
    validator.RE_BOTH_INSIDE_INTER,
    validator.RE_BOTH_ROUND,
    validator.RE_SINGLE_INSIDE_INTER,
    validator.RE_SINGLE_ROUND,
    validator.RE_BOTH_INSIDE_INTER_ALT,
    validator.RE_BOTH_ROUND_ALT,
    validator.RE_SINGLE_INSIDE_INTER_ALT,
    validator.RE_SINGLE_ROUND_ALT,
    validator.RE_NOT_ENTERED_INTER,
    validator.RE_NOT_ENTERED_ROUND,
    validator.RE_EXITED_INTER,
    validator.RE_EXITED_ROUND,
    validator.RE_AT_LABEL,
    validator.RE_IN_LANE,
)
UNCERTAIN_PATTERNS = (
    validator.RE_AHEAD,
    validator.RE_LEFT_OF,
    validator.RE_PAST,
    validator.RE_WILL_EXIT_BEFORE_ENTER_INTER,
    validator.RE_WILL_ENTER_BEFORE_EXIT_INTER,
    validator.RE_WILL_EXIT_BEFORE_ENTER_ROUND,
    validator.RE_WILL_ENTER_BEFORE_EXIT_ROUND,
    validator.RE_WAS_AHEAD_INTER,
    validator.RE_WAS_LEFT_INTER,
    validator.RE_WAS_AHEAD_ROUND,
    validator.RE_WAS_LEFT_ROUND,
    validator.RE_BEHIND_ROUND,
    validator.RE_WAS_BEHIND_ROUND,
    validator.RE_AHEAD_ROAD,
    validator.RE_PAST_ROAD,
    validator.RE_WILL_CHANGE_BEFORE,
    validator.RE_BEHIND_ROAD,
    validator.RE_DIRECTLY_BEHIND_ROAD,
)


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.match(text) for pattern in patterns)


def _single_vehicle_correct_id(text: str) -> str | None:
    m = RE_SINGLE_VEHICLE.match(text)
    if not m:
        return None
    return m.group(1)


def check_file(path: Path) -> int:
    if not path.exists():
        print(f"ERROR: input file not found: {path}")
        return 2

    total = 0
    semantic_wrong = 0
    semantic_invalid = 0
    duplicate_prompts = 0
    answer_counts = Counter()
    env_counts = Counter()
    difficulty_counts = Counter()
    event_sig_counts = Counter()
    correct_text_counts = Counter()
    correct_vehicle_counts = Counter()
    correct_vehicle_counts_by_env: dict[str, Counter[str]] = defaultdict(Counter)
    prompts_seen: set[str] = set()
    issues: list[str] = []
    warnings: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as exc:
                semantic_invalid += 1
                issues.append(f"line {line_no}: invalid json: {exc}")
                continue

            prompt = ex.get("prompt", "")
            if prompt in prompts_seen:
                duplicate_prompts += 1
            prompts_seen.add(prompt)

            metadata = ex.get("metadata", {})
            env = metadata.get("environment", ex.get("scenario", {}).get("environment"))
            difficulty = metadata.get("difficulty")
            answer = ex.get("answer")
            choices = ex.get("choices", {})
            distractor_type = ex.get("distractor_type", {})

            if answer in LETTERS:
                answer_counts[answer] += 1
                if answer in choices:
                    correct_text = choices[answer]
                    correct_text_counts[correct_text] += 1
                    single_vid = _single_vehicle_correct_id(correct_text)
                    if single_vid is not None:
                        correct_vehicle_counts[single_vid] += 1
                        correct_vehicle_counts_by_env[env][single_vid] += 1

            env_counts[env] += 1
            difficulty_counts[difficulty] += 1
            event_sig_counts[tuple(ex.get("events", []))] += 1

            ok, is_invalid, reason = validator.validate_example(ex)
            if not ok:
                if is_invalid:
                    semantic_invalid += 1
                else:
                    semantic_wrong += 1
                issues.append(f"line {line_no} {ex.get('id', '?')}: semantic check failed: {reason}")

            if answer not in LETTERS or answer not in choices:
                issues.append(f"line {line_no} {ex.get('id', '?')}: missing/invalid answer key")
                continue

            if not _matches_any(choices[answer], STATE_FACT_PATTERNS):
                issues.append(
                    f"line {line_no} {ex.get('id', '?')}: correct answer pattern unrecognized: {choices[answer]!r}"
                )

            for letter, text in choices.items():
                if letter == answer:
                    continue
                kind = distractor_type.get(letter)
                if kind == "near_true":
                    if not _matches_any(text, UNCERTAIN_PATTERNS):
                        issues.append(
                            f"line {line_no} {ex.get('id', '?')}: near_true pattern unrecognized for {letter}: {text!r}"
                        )
                elif kind == "highly_false":
                    if not _matches_any(text, STATE_FACT_PATTERNS):
                        issues.append(
                            f"line {line_no} {ex.get('id', '?')}: highly_false pattern unrecognized for {letter}: {text!r}"
                        )
                else:
                    issues.append(
                        f"line {line_no} {ex.get('id', '?')}: unexpected distractor type for {letter}: {kind!r}"
                    )

    if total == 100:
        if answer_counts != EXPECTED_ANSWER_COUNTS_100:
            issues.append(f"answer distribution mismatch: {dict(answer_counts)}")
        if env_counts != EXPECTED_ENV_COUNTS_100:
            issues.append(f"environment distribution mismatch: {dict(env_counts)}")
        if difficulty_counts != EXPECTED_DIFFICULTY_COUNTS_100:
            issues.append(f"difficulty distribution mismatch: {dict(difficulty_counts)}")

    event_sig_max = max(event_sig_counts.values(), default=0)
    correct_text_max = max(correct_text_counts.values(), default=0)
    if duplicate_prompts != 0:
        issues.append(f"duplicate prompts found: {duplicate_prompts}")
    event_sig_cap = validator.event_sig_cap_for(total)
    if event_sig_max > event_sig_cap:
        issues.append(f"event_sig_max exceeded: {event_sig_max} > {event_sig_cap}")
    correct_text_cap = validator.correct_text_cap_for(total)
    if correct_text_max > correct_text_cap:
        issues.append(f"correct_text_max exceeded: {correct_text_max} > {correct_text_cap}")

    a_count = correct_vehicle_counts.get("A", 0)
    b_count = correct_vehicle_counts.get("B", 0)
    c_count = correct_vehicle_counts.get("C", 0)
    if correct_text_max == correct_text_cap:
        warnings.append(
            f"correct_text_max is at the cap edge: {correct_text_max} (text={correct_text_counts.most_common(1)[0][0]!r})"
        )
    warnings.append(f"single-vehicle correct answers: A={a_count} B={b_count} C={c_count}")
    warnings.append(
        "single-vehicle correct answers by env: "
        + ", ".join(
            f"{env}:A={counts.get('A', 0)} B={counts.get('B', 0)} C={counts.get('C', 0)}"
            for env, counts in sorted(correct_vehicle_counts_by_env.items())
        )
    )
    if c_count - a_count >= 15:
        warnings.append(
            "Vehicle C vs A single-answer gap is large, but the new multi_lane_road slice is not the cause "
            f"(multi_lane_road: A={correct_vehicle_counts_by_env[Environment.MULTI_LANE.value].get('A', 0)} "
            f"B={correct_vehicle_counts_by_env[Environment.MULTI_LANE.value].get('B', 0)} "
            f"C={correct_vehicle_counts_by_env[Environment.MULTI_LANE.value].get('C', 0)})."
        )

    print(f"Checked file: {path}")
    print(f"total={total}")
    print(f"semantic_wrong={semantic_wrong}")
    print(f"semantic_invalid={semantic_invalid}")
    print(f"duplicate_prompts={duplicate_prompts}")
    print()
    print("Answer distribution:")
    for letter in LETTERS:
        print(f"  {letter}: {answer_counts.get(letter, 0)}")
    print()
    print("Environment distribution:")
    for env in (
        Environment.INTERSECTION.value,
        Environment.ROUNDABOUT.value,
        Environment.MULTI_LANE.value,
    ):
        print(f"  {env}: {env_counts.get(env, 0)}")
    print()
    print("Difficulty distribution:")
    for difficulty in ("easy", "medium", "hard"):
        print(f"  {difficulty}: {difficulty_counts.get(difficulty, 0)}")
    print()
    print("Reuse caps:")
    print(f"  event_sig_max: {event_sig_max}")
    print(f"  correct_text_max: {correct_text_max}")
    print()
    print("Observations:")
    for warning in warnings:
        print(f"  - {warning}")

    if issues:
        print()
        print("Failures:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print()
    print("Status: PASS")
    return 0


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 4 v5 dataset checker")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/core/task4_overlap.jsonl",
        help="Input JSONL file path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    raise SystemExit(check_file(Path(args.input)))
