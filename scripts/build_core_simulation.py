#!/usr/bin/env python3
"""Build quiz-only files under dataset/core_simulation and final variants."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from domain.vocabulary import label_of
from scripts.validate_task2 import NO_CLEAR_OPTION_TEXT, recompute_priority
from scripts.validate_task3 import validate_example as validate_task3_example
from scripts.validate_task4 import validate_example as validate_task4_example


LETTERS = ("A", "B", "C", "D", "E")
ALLOWED_CLEAN_KEYS = {"id", "task", "prompt", "choices", "answer"}
ALLOWED_NO_ANSWER_KEYS = ALLOWED_CLEAN_KEYS - {"answer"}
FORBIDDEN_PATTERNS = (
    "near_true",
    "highly_false",
    "entirely_true",
    "distractor_type",
    "rationale",
    "audit",
    "metadata",
    "ground_truth",
    "semantic_by_letter",
    "option_rationale",
    "generator_version",
    "invariants",
)

TASKS = (
    {
        "name": "task1",
        "source": "task1_position.jsonl",
        "out": "task1_position.jsonl",
        "generator": "generators/task1_position.py",
        "seed": 101,
    },
    {
        "name": "task2",
        "source": "task2_rightofway.jsonl",
        "out": "task2_rightofway.jsonl",
        "generator": "generators/task2_rightofway.py",
        "seed": 102,
    },
    {
        "name": "task3",
        "source": "task3_violation.jsonl",
        "out": "task3_violation.jsonl",
        "generator": "generators/task3_violation.py",
        "seed": 103,
    },
    {
        "name": "task4",
        "source": "task4_overlap.jsonl",
        "out": "task4_overlap.jsonl",
        "generator": "generators/task4_overlap.py",
        "seed": 104,
    },
)


class ValidationError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def strip_choice_lines(prompt: str, choices: dict[str, str]) -> str:
    formatted = {f"{letter}) {text}" for letter, text in choices.items()}
    kept: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped in formatted:
            continue
        if re.match(r"^[A-E]\)\s", stripped):
            continue
        kept.append(line.rstrip())
    return "\n".join(kept).strip()


def clean_example(ex: dict) -> dict:
    choices = ex.get("choices")
    if not isinstance(choices, dict):
        raise ValidationError(f"{ex.get('id', '?')}: choices missing or invalid")
    return {
        "id": ex["id"],
        "task": ex["task"],
        "prompt": strip_choice_lines(ex["prompt"], choices),
        "choices": {letter: choices[letter] for letter in LETTERS},
        "answer": ex["answer"],
    }


def remove_answer(ex: dict) -> dict:
    if "answer" not in ex:
        raise ValidationError(f"{ex.get('id', '?')}: answer missing before no-answer export")
    return {key: value for key, value in ex.items() if key != "answer"}


def final_paths(out_dir: Path, file_name: str) -> tuple[Path, Path]:
    stem = Path(file_name).stem
    final_dir = out_dir / "final"
    return (
        final_dir / f"{stem}_with_answers.jsonl",
        final_dir / f"{stem}_no_answers.jsonl",
    )


def check_choices_answer(ex: dict, *, where: str) -> None:
    choices = ex.get("choices")
    answer = ex.get("answer")
    if not isinstance(choices, dict) or set(choices) != set(LETTERS):
        raise ValidationError(f"{where}: choices must be exactly A-E")
    if answer not in LETTERS:
        raise ValidationError(f"{where}: invalid answer {answer!r}")
    if len(set(choices.values())) != len(LETTERS):
        raise ValidationError(f"{where}: duplicate choice text")


def check_no_false_invariants(ex: dict, *, where: str) -> None:
    invariants = ex.get("audit", {}).get("invariants", {})
    if isinstance(invariants, dict):
        bad = [k for k, v in invariants.items() if v is False]
        if bad:
            raise ValidationError(f"{where}: failing source invariants: {bad}")


def validate_source_task1(rows: list[dict]) -> None:
    for ex in rows:
        where = ex.get("id", "?")
        check_choices_answer(ex, where=where)
        check_no_false_invariants(ex, where=where)
        trace = ex.get("audit", {}).get("queried_trace", [])
        if not trace:
            raise ValidationError(f"{where}: missing queried trace")
        expected = label_of(trace[-1])
        if ex["choices"][ex["answer"]] != expected:
            raise ValidationError(
                f"{where}: wrong answer, expected {expected!r}, got {ex['choices'][ex['answer']]!r}"
            )


def validate_source_task2(rows: list[dict]) -> None:
    for ex in rows:
        where = ex.get("id", "?")
        check_choices_answer(ex, where=where)
        check_no_false_invariants(ex, where=where)
        recomputed = recompute_priority(ex["scenario"])
        answer_text = ex["choices"][ex["answer"]]
        if recomputed.priority is None:
            if answer_text != NO_CLEAR_OPTION_TEXT:
                raise ValidationError(
                    f"{where}: expected no-clear answer {NO_CLEAR_OPTION_TEXT!r}, got {answer_text!r}"
                )
            continue
        expected = f"Vehicle {recomputed.priority}"
        if answer_text != expected:
            raise ValidationError(
                f"{where}: wrong answer, expected {expected!r}, got {answer_text!r}"
            )


def validate_source_task3(rows: list[dict]) -> None:
    for ex in rows:
        where = ex.get("id", "?")
        check_choices_answer(ex, where=where)
        check_no_false_invariants(ex, where=where)
        ok, invalid, reason, _ = validate_task3_example(ex)
        if not ok:
            kind = "invalid" if invalid else "wrong"
            raise ValidationError(f"{where}: task3 {kind}: {reason}")


def validate_source_task4(rows: list[dict]) -> None:
    for ex in rows:
        where = ex.get("id", "?")
        check_choices_answer(ex, where=where)
        check_no_false_invariants(ex, where=where)
        ok, invalid, reason = validate_task4_example(ex)
        if not ok:
            kind = "invalid" if invalid else "wrong"
            raise ValidationError(f"{where}: task4 {kind}: {reason}")


SOURCE_VALIDATORS = {
    "task1": validate_source_task1,
    "task2": validate_source_task2,
    "task3": validate_source_task3,
    "task4": validate_source_task4,
}


def validate_clean(rows: list[dict], *, task_name: str, expected_n: int) -> dict:
    if len(rows) != expected_n:
        raise ValidationError(f"{task_name}: expected {expected_n} rows, got {len(rows)}")

    ids = set()
    quiz_signatures = set()
    prompt_counts = Counter()
    answers = Counter()
    correct_texts = Counter()
    answer_runs: list[int] = []
    prev_answer: str | None = None
    current_run = 0

    for idx, ex in enumerate(rows):
        where = f"{task_name}:{idx}:{ex.get('id', '?')}"
        if set(ex) != ALLOWED_CLEAN_KEYS:
            raise ValidationError(f"{where}: clean keys are {sorted(ex)}")
        check_choices_answer(ex, where=where)

        if ex["id"] in ids:
            raise ValidationError(f"{where}: duplicate id")
        ids.add(ex["id"])
        quiz_sig = (ex["prompt"], tuple(ex["choices"].items()), ex["answer"])
        if quiz_sig in quiz_signatures:
            raise ValidationError(f"{where}: duplicate full quiz")
        quiz_signatures.add(quiz_sig)
        prompt_counts[ex["prompt"]] += 1

        blob = json.dumps(ex, ensure_ascii=False).lower()
        leaked = [p for p in FORBIDDEN_PATTERNS if p in blob]
        if leaked:
            raise ValidationError(f"{where}: leakage tokens found: {leaked}")
        if re.search(r"\bfalse\b", blob):
            raise ValidationError(f"{where}: visible 'false' token found")

        if any(re.search(rf"^{letter}\)\s", line.strip()) for line in ex["prompt"].splitlines() for letter in LETTERS):
            raise ValidationError(f"{where}: prompt still contains inline options")

        answer = ex["answer"]
        answers[answer] += 1
        correct_texts[ex["choices"][answer]] += 1
        if answer == prev_answer:
            current_run += 1
        else:
            if current_run:
                answer_runs.append(current_run)
            prev_answer = answer
            current_run = 1
    if current_run:
        answer_runs.append(current_run)

    expected_each = expected_n // len(LETTERS)
    if any(answers[letter] != expected_each for letter in LETTERS):
        raise ValidationError(f"{task_name}: unbalanced answers {dict(answers)}")

    if max(answer_runs, default=0) > 4:
        raise ValidationError(f"{task_name}: answer key has a run longer than 4: {max(answer_runs)}")

    most_common_text, most_common_count = correct_texts.most_common(1)[0]
    correct_text_cap = max(20, ((expected_n + 2) // 3) + 5)
    if most_common_count > correct_text_cap:
        raise ValidationError(
            f"{task_name}: correct text overused {most_common_text!r} x{most_common_count}"
        )

    return {
        "rows": len(rows),
        "answers": dict(answers),
        "max_answer_run": max(answer_runs, default=0),
        "max_prompt_reuse": max(prompt_counts.values(), default=0),
        "reused_prompts": sum(1 for c in prompt_counts.values() if c > 1),
        "unique_correct_texts": len(correct_texts),
    }


def validate_no_answer(
    with_answer_rows: list[dict],
    no_answer_rows: list[dict],
    *,
    task_name: str,
) -> None:
    if len(with_answer_rows) != len(no_answer_rows):
        raise ValidationError(
            f"{task_name}: no-answer row count mismatch "
            f"{len(no_answer_rows)} vs {len(with_answer_rows)}"
        )
    for idx, (with_answer, no_answer) in enumerate(zip(with_answer_rows, no_answer_rows)):
        where = f"{task_name}:{idx}:{with_answer.get('id', '?')}"
        if set(no_answer) != ALLOWED_NO_ANSWER_KEYS:
            raise ValidationError(f"{where}: no-answer keys are {sorted(no_answer)}")
        expected = remove_answer(with_answer)
        if no_answer != expected:
            raise ValidationError(f"{where}: no-answer row differs beyond removed answer field")


def run_generator(task: dict, n: int, source_dir: Path, seed: int) -> str:
    out_path = source_dir / task["source"]
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / task["generator"]),
        "--n",
        str(n),
        "--seed",
        str(seed),
        "--out",
        str(out_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return (completed.stdout + "\n" + completed.stderr).strip()
    return ""


def build(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)

    summary: dict[str, dict] = {}
    for task in TASKS:
        source_path = source_dir / task["source"]
        out_path = out_dir / task["out"]
        final_with_answers_path, final_no_answers_path = final_paths(out_dir, task["out"])
        last_error = ""
        attempts = 1 if args.skip_generate else args.seed_attempts
        for offset in range(attempts):
            seed = int(task["seed"]) + offset
            if not args.skip_generate:
                last_error = run_generator(task, args.n, source_dir, seed)
                if last_error:
                    continue
            try:
                source_rows = read_jsonl(source_path)
                SOURCE_VALIDATORS[task["name"]](source_rows)
                clean_rows = [clean_example(ex) for ex in source_rows]
                clean_stats = validate_clean(
                    clean_rows,
                    task_name=task["name"],
                    expected_n=args.n,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if args.skip_generate:
                    raise
                continue

            write_jsonl(out_path, clean_rows)
            write_jsonl(final_with_answers_path, clean_rows)
            no_answer_rows = [remove_answer(ex) for ex in clean_rows]
            validate_no_answer(clean_rows, no_answer_rows, task_name=task["name"])
            write_jsonl(final_no_answers_path, no_answer_rows)

            reread = read_jsonl(out_path)
            reread_with_answers = read_jsonl(final_with_answers_path)
            reread_no_answers = read_jsonl(final_no_answers_path)
            validate_clean(reread, task_name=task["name"], expected_n=args.n)
            validate_clean(reread_with_answers, task_name=task["name"], expected_n=args.n)
            validate_no_answer(reread_with_answers, reread_no_answers, task_name=task["name"])

            clean_stats["final_with_answers"] = str(final_with_answers_path.relative_to(PROJECT_ROOT))
            clean_stats["final_no_answers"] = str(final_no_answers_path.relative_to(PROJECT_ROOT))
            summary[task["out"]] = clean_stats
            source_label = "existing source" if args.skip_generate else f"seed {seed}"
            print(f"{task['name']}: clean export passed with {source_label}", flush=True)
            break
        else:
            raise RuntimeError(
                f"{task['name']}: no clean dataset after {attempts} attempt(s).\n"
                f"Last error:\n{last_error}"
            )

    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clean core_simulation quiz datasets.")
    parser.add_argument("--n", type=int, default=100, help="examples per task; must be multiple of 5")
    parser.add_argument(
        "--source-dir",
        default=str(PROJECT_ROOT / "dataset" / "core"),
        help="internal source JSONL directory",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "dataset" / "core_simulation"),
        help="clean output JSONL directory",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="export from existing source files instead of running generators",
    )
    parser.add_argument(
        "--seed-attempts",
        type=int,
        default=25,
        help="number of consecutive seeds to try per generator",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
