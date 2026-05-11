from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LETTERS = ("A", "B", "C", "D", "E")


def compute_metrics(rows: list[dict[str, Any]], qualitative_limit: int = 12) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["is_correct"])
    parse_failed = sum(1 for row in rows if row["parse_failed"])
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)

    answer_counts = Counter(row["parsed_answer"] for row in rows if row["parsed_answer"] in LETTERS)

    return {
        "num_examples": total,
        "overall_accuracy": _ratio(correct, total),
        "overall_correct": correct,
        "per_task_accuracy": _per_task_accuracy(by_task),
        "answer_distribution": {
            letter: {
                "count": answer_counts[letter],
                "percentage": _percentage(answer_counts[letter], total),
            }
            for letter in LETTERS
        },
        "unparsed": {
            "count": parse_failed,
            "percentage": _percentage(parse_failed, total),
        },
        "format_errors": {
            "count": parse_failed,
            "percentage": _percentage(parse_failed, total),
        },
        "qualitative_errors": _qualitative_errors(rows, qualitative_limit),
    }


def write_report(output_root: Path | str, model_order: list[dict[str, str]]) -> Path:
    root = Path(output_root)
    summaries = _read_summaries(root)
    by_key = {summary["model_key"]: summary for summary in summaries}
    ordered = [by_key[spec["key"]] for spec in model_order if spec["key"] in by_key]

    lines: list[str] = [
        "# Pilot Evaluation Report",
        "",
        "Dataset: `dataset/core_simulation/`.",
        "Prompting: zero-shot; dataset prompt plus A-E choices plus the fixed instruction.",
        "Gold labels are read only after generation for scoring.",
        "Decoding: deterministic for Hugging Face runs (`do_sample=False`).",
        "",
        "## Models",
        "",
        "| Model | Profile | Status |",
        "|---|---|---|",
    ]
    for summary in ordered:
        lines.append(
            f"| `{summary['model_id']}` | {summary.get('profile', '')} | {summary['status']} |"
        )

    lines.extend(["", "## Overall Accuracy", "", "| Model | Examples | Accuracy | Correct | Limitation |", "|---|---:|---:|---:|---|"])
    for summary in ordered:
        lines.append(
            "| {model} | {n} | {acc} | {correct} | {limitation} |".format(
                model=f"`{summary['model_key']}`",
                n=summary.get("num_examples", 0) or 0,
                acc=_display_pct(summary.get("overall_accuracy")),
                correct=summary.get("overall_correct", ""),
                limitation=summary.get("limitation", ""),
            )
        )

    task_names = _task_names(ordered)
    if task_names:
        lines.extend(["", "## Per-task Accuracy", "", _task_header(task_names), _task_divider(task_names)])
        for summary in ordered:
            cells = [f"`{summary['model_key']}`"]
            per_task = summary.get("per_task_accuracy", {})
            for task in task_names:
                cells.append(_display_pct(per_task.get(task, {}).get("accuracy")))
            lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Answer Distribution", ""])
    for summary in ordered:
        dist = summary.get("answer_distribution", {})
        if not dist:
            lines.append(f"- `{summary['model_key']}`: not available ({summary['status']})")
            continue
        parts = [f"{letter}: {_display_pct(dist.get(letter, {}).get('percentage'), already_percentage=True)}" for letter in LETTERS]
        lines.append(f"- `{summary['model_key']}`: " + ", ".join(parts))

    lines.extend(["", "## Format Errors", "", "| Model | Count | Percentage |", "|---|---:|---:|"])
    for summary in ordered:
        errors = summary.get("format_errors", {})
        if not errors:
            lines.append(f"| `{summary['model_key']}` | n/a | n/a |")
            continue
        lines.append(
            f"| `{summary['model_key']}` | {errors.get('count', '')} | "
            f"{_display_pct(errors.get('percentage'), already_percentage=True)} |"
        )

    lines.extend(["", "## Qualitative Errors", ""])
    any_errors = False
    for summary in ordered:
        errors = summary.get("qualitative_errors", [])
        if not errors:
            continue
        any_errors = True
        lines.append(f"### {summary['model_key']}")
        for err in errors[:8]:
            parsed = err.get("parsed_answer") or "unparsed"
            raw = str(err.get("raw_response", "")).replace("\n", " ")[:140]
            lines.append(
                f"- `{err['id']}` ({err['task']}): predicted {parsed}, "
                f"gold {err['correct_answer']}. Raw: `{raw}`"
            )
        lines.append("")
    if not any_errors:
        lines.append("No wrong predictions recorded.")

    path = root / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _per_task_accuracy(by_task: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for task, rows in sorted(by_task.items()):
        total = len(rows)
        correct = sum(1 for row in rows if row["is_correct"])
        out[task] = {
            "accuracy": _ratio(correct, total),
            "correct": correct,
            "total": total,
        }
    return out


def _qualitative_errors(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["is_correct"]:
            by_task[row["task"]].append(row)

    while len(out) < limit and any(by_task.values()):
        for task in sorted(by_task):
            if by_task[task] and len(out) < limit:
                row = by_task[task].pop(0)
                out.append(
                    {
                        "id": row["id"],
                        "task": row["task"],
                        "correct_answer": row["correct_answer"],
                        "parsed_answer": row["parsed_answer"],
                        "raw_response": row["raw_response"],
                        "parse_failed": row["parse_failed"],
                    }
                )
    return out


def _read_summaries(root: Path) -> list[dict[str, Any]]:
    summary_dir = root / "summaries"
    if not summary_dir.exists():
        return []
    import json

    summaries = []
    for path in sorted(summary_dir.glob("*.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def _task_names(summaries: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for summary in summaries:
        names.update(summary.get("per_task_accuracy", {}).keys())
    return sorted(names)


def _task_header(task_names: list[str]) -> str:
    return "| Model | " + " | ".join(task_names) + " |"


def _task_divider(task_names: list[str]) -> str:
    return "|---|" + "|".join("---:" for _ in task_names) + "|"


def _ratio(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return round(num / den, 4)


def _percentage(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return round(100 * num / den, 2)


def _display_pct(value: Any, *, already_percentage: bool = False) -> str:
    if value is None or value == "":
        return ""
    numeric = float(value)
    if not already_percentage:
        numeric *= 100
    return f"{numeric:.2f}%"
