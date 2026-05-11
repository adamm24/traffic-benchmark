from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


LETTERS = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class Example:
    id: str
    task: str
    prompt: str
    choices: dict[str, str]
    answer: str
    source_file: str


def load_dataset(dataset_dir: Path | str) -> dict[str, list[Example]]:
    base = Path(dataset_dir)
    if not base.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base}")

    task_files = sorted(base.glob("*.jsonl"))
    if not task_files:
        raise FileNotFoundError(f"No JSONL task files found in {base}")

    return {path.stem: load_task_file(path) for path in task_files}


def load_task_file(path: Path | str) -> list[Example]:
    file_path = Path(path)
    examples: list[Example] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(_parse_example(row, file_path.stem, line_no))
    return examples


def _parse_example(row: dict, source_file: str, line_no: int) -> Example:
    missing = {"id", "task", "prompt", "choices", "answer"} - set(row)
    if missing:
        raise ValueError(f"{source_file}:{line_no}: missing keys {sorted(missing)}")

    choices = row["choices"]
    if not isinstance(choices, dict) or set(choices) != set(LETTERS):
        raise ValueError(f"{source_file}:{line_no}: choices must be A-E")

    answer = row["answer"]
    if answer not in LETTERS:
        raise ValueError(f"{source_file}:{line_no}: invalid answer {answer!r}")

    return Example(
        id=str(row["id"]),
        task=str(row["task"]),
        prompt=str(row["prompt"]),
        choices={letter: str(choices[letter]) for letter in LETTERS},
        answer=answer,
        source_file=source_file,
    )

