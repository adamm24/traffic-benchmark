# Traffic Benchmark

Traffic Benchmark is a stage project for the Master's degree in Data Science at the University of Milano-Bicocca. It evaluates structured reasoning in traffic scenarios.

## Project Status

Completed in this repository:
- Task 1: position tracking (`dataset/core/task1_position.jsonl`)
- Task 2: right-of-way reasoning (`dataset/core/task2_rightofway.jsonl`)
- Task 3: violation detection (`dataset/core/task3_violation.jsonl`)
- Task 4: overlap reasoning (`dataset/core/task4_overlap.jsonl`)
- Validators and quality checks for Tasks 2–4

## Repository Entry Points

- Project overview: `project_documentation.md`
- Task-level reports: `documentation/`
  - `task1_documentation.md`
  - `task2_documentation.md`
  - `task3_documentation.md`
  - `task4_documentation.md`

## Repository Structure

- `domain/`: shared simulation logic, FSM, rules, rendering, vocabulary
- `generators/`: dataset generators (`task1_position.py` ... `task4_overlap.py`)
- `dataset/core/`: full benchmark records (rich schema with scenario, metadata, audit)
- `dataset/core_simulation/`: evaluation-ready records (`id`, `task`, `prompt`, `choices`, `answer`)
- `scripts/`: validators and dataset utility scripts
- `tests/`: unit tests for domain behavior
- `documentation/`: task reports and implementation notes

## Dataset Organization

- Core benchmark data: `dataset/core/`
- Core simulation data for model evaluation: `dataset/core_simulation/`
- Results/outputs: not committed by default; create locally when running evaluations

## Notes

- `project_documentation.md` contains the full design rationale and methodology.
