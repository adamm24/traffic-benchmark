from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.backends.base import ModelBackend
from evaluation.loader import Example
from evaluation.metrics import compute_metrics
from evaluation.parser import parse_answer
from evaluation.prompt import build_prompt


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    profile: str


DEFAULT_MODELS = (
    ModelSpec("qwen1_5_1_8b_chat", "Qwen/Qwen1.5-1.8B-Chat", "small open baseline"),
    ModelSpec("phi3_mini_4k_instruct", "microsoft/Phi-3-mini-4k-instruct", "compact instruction model"),
    ModelSpec("llama3_8b_instruct", "meta-llama/Meta-Llama-3-8B-Instruct", "stronger open model"),
)


def select_models(selector: str) -> list[ModelSpec]:
    if selector == "all":
        return list(DEFAULT_MODELS)

    selected = []
    for spec in DEFAULT_MODELS:
        if selector in {spec.key, spec.model_id}:
            selected.append(spec)
    if not selected:
        known = ", ".join(["all", *[spec.key for spec in DEFAULT_MODELS]])
        raise ValueError(f"Unknown model {selector!r}. Use one of: {known}")
    return selected


def model_order_dicts() -> list[dict[str, str]]:
    return [{"key": spec.key, "model_id": spec.model_id, "profile": spec.profile} for spec in DEFAULT_MODELS]


def run_model(
    spec: ModelSpec,
    backend: ModelBackend,
    dataset: dict[str, list[Example]],
    output_root: Path | str,
    *,
    backend_name: str,
    limit_per_task: int | None = None,
    resume: bool = True,
    qualitative_limit: int = 12,
) -> dict[str, Any]:
    root = Path(output_root)
    raw_dir = root / "raw" / spec.key
    summary_dir = root / "summaries"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for task_file, examples in sorted(dataset.items()):
        selected = examples[:limit_per_task] if limit_per_task is not None else examples
        out_path = raw_dir / f"{task_file}.jsonl"
        existing = _load_existing(out_path) if resume else {}
        if not resume:
            out_path.write_text("", encoding="utf-8")

        with out_path.open("a", encoding="utf-8") as f:
            for example in selected:
                if example.id in existing:
                    rows.append(existing[example.id])
                    continue

                raw_response = backend.generate(build_prompt(example))
                parsed = parse_answer(raw_response)
                row = {
                    "id": example.id,
                    "task": example.task,
                    "correct_answer": example.answer,
                    "raw_response": raw_response,
                    "parsed_answer": parsed.answer,
                    "is_correct": parsed.answer == example.answer,
                    "parse_failed": parsed.parse_failed,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                rows.append(row)

    summary = compute_metrics(rows, qualitative_limit=qualitative_limit)
    summary.update(
        {
            "status": "completed",
            "model_key": spec.key,
            "model_id": spec.model_id,
            "profile": spec.profile,
            "backend": backend_name,
            "limit_per_task": limit_per_task,
        }
    )
    _write_summary(summary_dir / f"{spec.key}.json", summary)
    return summary


def write_failure_summary(spec: ModelSpec, output_root: Path | str, backend_name: str, reason: str) -> dict[str, Any]:
    summary = {
        "status": "failed",
        "model_key": spec.key,
        "model_id": spec.model_id,
        "profile": spec.profile,
        "backend": backend_name,
        "limitation": reason,
        "num_examples": 0,
        "overall_accuracy": None,
        "overall_correct": "",
        "per_task_accuracy": {},
        "answer_distribution": {},
        "format_errors": {},
        "qualitative_errors": [],
    }
    summary_dir = Path(output_root) / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(summary_dir / f"{spec.key}.json", summary)
    return summary


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

