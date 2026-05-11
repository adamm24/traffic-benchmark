# Traffic Benchmark

Traffic Benchmark is a stage project for the Master's degree in Data Science at the University of Milano-Bicocca. It evaluates structured reasoning in traffic scenarios.

## Project Status

Completed in this repository:
- Task 1: position tracking (`dataset/core/task1_position.jsonl`)
- Task 2: right-of-way reasoning (`dataset/core/task2_rightofway.jsonl`)
- Task 3: violation detection (`dataset/core/task3_violation.jsonl`)
- Task 4: certainty under spatial ambiguity (`dataset/core/task4_overlap.jsonl`)
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

## Scripts

- `scripts/validate_task2.py`: independent Task 2 validator and shortcut audit
- `scripts/validate_task3.py`: independent Task 3 replay validator and quality audit
- `scripts/validate_task4.py`: independent Task 4 semantic replay validator
- `scripts/validate_task4_distribution.py`: Task 4 distribution and statement-pattern validator
- `scripts/build_core_simulation.py`: exports quiz-only files to `dataset/core_simulation/`

## Dataset Organization

- Core benchmark data: `dataset/core/`
- Core simulation data for model evaluation: `dataset/core_simulation/`
- Results/outputs: not committed by default; create locally when running evaluations

## Notes

- `project_documentation.md` contains the full design rationale and methodology.
