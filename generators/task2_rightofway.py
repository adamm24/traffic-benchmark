"""
Task 2 — Right-of-Way Reasoning Generator
==========================================
Generates multiple-choice questions that test an LLM's ability to
determine which vehicle has the right of way at an intersection or
roundabout, based on traffic priority rules.

Environments: intersection, roundabout
Vehicles:     3
Choices:      1 correct, 2 near_true, 2 highly_false

Usage:
    python generators/task2_rightofway.py
"""

from __future__ import annotations

import copy
import json
import random
import sys
import os
from pathlib import Path
from typing import Optional

# ── Make project root importable ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from domain.entities import (
    Direction, Environment, IntentDirection, ScenarioState, Vehicle,
)
from domain.scenario import (
    build_intersection_scenario, build_roundabout_scenario,
)
from domain.rules import (
    right_of_way, right_of_way_intersection, right_of_way_roundabout,
)
from domain.render import describe_scenario
# render_prompt intentionally not imported: Task 2 has no event sequence,
# so the prompt is assembled inline without the events block.

# ── Constants ───────────────────────────────────────────────────────────────

N_EXAMPLES   = 100          # change to 300 for the full core dataset
NUM_VEHICLES = 3
MAX_RETRIES  = 50           # per-example retry budget

TASK_ENVS = [Environment.INTERSECTION, Environment.ROUNDABOUT]

# Weight: more intersection examples since it has more rule variety
ENV_WEIGHTS = [0.65, 0.35]

LETTERS = ["A", "B", "C", "D", "E"]

# ── Opposite direction pairs (no lateral conflict at intersection) ──────────

OPPOSITE_PAIRS = {
    (Direction.NORTH, Direction.SOUTH),
    (Direction.SOUTH, Direction.NORTH),
    (Direction.EAST, Direction.WEST),
    (Direction.WEST, Direction.EAST),
}


# ── Question templates ──────────────────────────────────────────────────────

INTERSECTION_QUESTIONS = [
    "Which vehicle has the right of way?",
    "Which vehicle should be allowed to pass first?",
    "According to traffic rules, which vehicle has priority?",
]

ROUNDABOUT_QUESTIONS = [
    "Which vehicle has the right of way?",
    "Which vehicle should be allowed to proceed first?",
    "According to traffic rules, which vehicle has priority?",
]


# ── Context phrases (added to scenario for realism) ─────────────────────────

INTERSECTION_CONTEXTS = [
    "There are no traffic lights or signs.",
    "The intersection has no traffic signals.",
    "No traffic signs or signals are present.",
    "There are no stop signs or traffic lights.",
]

ROUNDABOUT_CONTEXTS = [
    "Standard roundabout rules apply.",
    "Vehicles inside the roundabout have priority.",
    "The roundabout follows standard right-of-way rules.",
]


# ── Scenario generation helpers ─────────────────────────────────────────────

def _has_lateral_conflict(v1: Vehicle, v2: Vehicle) -> bool:
    """Returns True if two vehicles have a lateral conflict (not opposite)."""
    return (v1.direction, v2.direction) not in OPPOSITE_PAIRS


def _build_intersection_with_conflict(num_vehicles: int = 3) -> tuple[ScenarioState, str, str] | None:
    """
    Builds an intersection scenario where at least two vehicles have a
    lateral conflict (so the right-of-way rule applies).

    Returns (state, priority_vid, yielding_vid) or None if no valid
    conflict is found after retries.
    """
    for _ in range(MAX_RETRIES):
        state = build_intersection_scenario(num_vehicles, with_intent=True)

        # Collect ALL valid conflict pairs, then pick one randomly
        # This avoids always selecting the first pair (A,B)
        conflict_pairs = []
        vehicles = state.vehicles
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1, v2 = vehicles[i], vehicles[j]
                if _has_lateral_conflict(v1, v2):
                    priority_vid = right_of_way_intersection(v1, v2)
                    if priority_vid is not None:
                        yielding_vid = v2.id if priority_vid == v1.id else v1.id
                        conflict_pairs.append((priority_vid, yielding_vid))

        if conflict_pairs:
            priority_vid, yielding_vid = random.choice(conflict_pairs)
            return state, priority_vid, yielding_vid

    return None


def _build_roundabout_with_conflict(num_vehicles: int = 3) -> tuple[ScenarioState, str, str] | None:
    """
    Builds a roundabout scenario where one vehicle is inside and at least
    one is trying to enter.  The "inside" vehicle is chosen randomly among
    the three to avoid always assigning priority to Vehicle A.

    Returns (state, priority_vid, yielding_vid) or None.
    """
    for _ in range(MAX_RETRIES):
        state = build_roundabout_scenario(num_vehicles)

        # Randomise which vehicle is the one inside the roundabout
        # (build_roundabout_scenario always puts index 0 inside)
        inside_idx = random.randrange(num_vehicles)
        for i, v in enumerate(state.vehicles):
            if i == inside_idx:
                v.inside_intersection = True
                v.position = "roundabout_lane"
            else:
                v.inside_intersection = False
                v.position = f"{v.direction.value}_approach"

        inside_v = state.vehicles[inside_idx]
        # Pick a random entering vehicle from the others
        entering_candidates = [v for v in state.vehicles if not v.inside_intersection]
        if not entering_candidates:
            continue
        entering_v = random.choice(entering_candidates)

        priority_vid = right_of_way_roundabout(inside_v, entering_v)
        yielding_vid = entering_v.id if priority_vid == inside_v.id else inside_v.id
        return state, priority_vid, yielding_vid

    return None


# ── Scenario text enhancer ──────────────────────────────────────────────────

def _enhance_scenario_text(base_text: str, env: Environment) -> str:
    """Adds a context line about the traffic environment."""
    if env == Environment.INTERSECTION:
        context = random.choice(INTERSECTION_CONTEXTS)
    else:
        context = random.choice(ROUNDABOUT_CONTEXTS)
    return base_text + "\n" + context


# ── Choice builder ──────────────────────────────────────────────────────────

def build_choices(
    priority_vid: str,
    yielding_vid: str,
    third_vid: str,
    env: Environment,
) -> dict[str, dict]:
    """
    Returns a dict with exactly 5 entries:
        "correct":        { "text": ..., "type": "correct" }
        "near_true_1":    { "text": ..., "type": "near_true" }
        "near_true_2":    { "text": ..., "type": "near_true" }
        "highly_false_1": { "text": ..., "type": "highly_false" }
        "highly_false_2": { "text": ..., "type": "highly_false" }
    """
    correct_text = f"Vehicle {priority_vid}"

    # near_true 1: the vehicle that should yield (most plausible error)
    nt1_text = f"Vehicle {yielding_vid}"

    # near_true 2: "Both can pass" — plausible if someone doesn't know the rule
    nt2_text = "Both can pass at the same time"

    # highly_false 1: the third vehicle not involved in the conflict
    hf1_text = f"Vehicle {third_vid}"

    # highly_false 2: a clearly wrong statement
    if env == Environment.INTERSECTION:
        hf2_options = [
            "No vehicle can pass",
            "All vehicles must stop",
            "The intersection must be cleared first",
        ]
    else:  # ROUNDABOUT
        hf2_options = [
            "No vehicle can pass",
            "All vehicles must exit the roundabout",
            "All vehicles must stop and wait",
        ]
    hf2_text = random.choice(hf2_options)

    return {
        "correct":        {"text": correct_text,  "type": "correct"},
        "near_true_1":    {"text": nt1_text,       "type": "near_true"},
        "near_true_2":    {"text": nt2_text,       "type": "near_true"},
        "highly_false_1": {"text": hf1_text,       "type": "highly_false"},
        "highly_false_2": {"text": hf2_text,       "type": "highly_false"},
    }


# ── Letter assignment ───────────────────────────────────────────────────────

def assign_letters(
    choices_dict: dict[str, dict],
    correct_key: str,
) -> tuple[dict[str, str], dict[str, str], str]:
    """
    Shuffles the five options and places the correct answer at *correct_key*.

    Returns:
        choices        – dict  {A..E: text}
        distractor_type – dict  {A..E: type}  (only for distractors)
        answer         – the letter of the correct answer
    """
    items = list(choices_dict.values())
    random.shuffle(items)

    # Ensure the correct item lands at correct_key
    target_idx = LETTERS.index(correct_key)
    correct_item = next(it for it in items if it["type"] == "correct")
    items.remove(correct_item)
    items.insert(target_idx, correct_item)

    choices = {}
    distractor_type = {}
    for letter, item in zip(LETTERS, items):
        choices[letter] = item["text"]
        if item["type"] != "correct":
            distractor_type[letter] = item["type"]

    return choices, distractor_type, correct_key


# ── Single example generator ────────────────────────────────────────────────

def generate_example(example_id: int, correct_key: str) -> dict | None:
    """
    Generates a single Task 2 example.
    Returns None if no valid scenario is found after MAX_RETRIES attempts.
    """
    for _ in range(MAX_RETRIES):
        # 1. Choose environment
        env = random.choices(TASK_ENVS, weights=ENV_WEIGHTS, k=1)[0]

        # 2. Build scenario with a guaranteed conflict
        if env == Environment.INTERSECTION:
            result = _build_intersection_with_conflict(NUM_VEHICLES)
        else:
            result = _build_roundabout_with_conflict(NUM_VEHICLES)

        if result is None:
            continue

        state, priority_vid, yielding_vid = result

        # 3. Find the third vehicle (not in the main conflict)
        all_vids = [v.id for v in state.vehicles]
        third_vid = [v for v in all_vids if v != priority_vid and v != yielding_vid][0]

        # 4. Build scenario text
        scenario_text = describe_scenario(state)
        scenario_text = _enhance_scenario_text(scenario_text, env)

        # 5. Choose question
        if env == Environment.INTERSECTION:
            question = random.choice(INTERSECTION_QUESTIONS)
        else:
            question = random.choice(ROUNDABOUT_QUESTIONS)

        # 6. Build choices
        raw_choices = build_choices(priority_vid, yielding_vid, third_vid, env)

        # 7. Assign letters
        choices, distractor_type, answer = assign_letters(raw_choices, correct_key)

        # 8. Render prompt (no events for Task 2 — it's a static reasoning task)
        parts = [
            scenario_text,
            "",
            f"Question: {question}",
        ]
        for key in sorted(choices):
            parts.append(f"{key}) {choices[key]}")
        prompt = "\n".join(parts)

        # 9. Build the conflict pair info for metadata
        conflict_pair = sorted([priority_vid, yielding_vid])

        return {
            "id": f"task2_{example_id:04d}",
            "task": "right_of_way",
            "prompt": prompt,
            "scenario": {
                "vehicles": [
                    {
                        "id": v.id,
                        "position": v.position,
                        "direction": v.direction.value,
                        "intent": v.intent.value if v.intent else None,
                        "inside_intersection": v.inside_intersection,
                    }
                    for v in state.vehicles
                ],
                "environment": env.value,
            },
            "question": question,
            "choices": choices,
            "answer": answer,
            "distractor_type": distractor_type,
            "metadata": {
                "num_vehicles": NUM_VEHICLES,
                "environment": env.value,
                "priority_vehicle": priority_vid,
                "yielding_vehicle": yielding_vid,
                "conflict_pair": conflict_pair,
                "difficulty": "base",
            },
        }

    return None  # exhausted retries


# ── Main generation loop ────────────────────────────────────────────────────

def generate_task2(n: int, output_path: str) -> None:
    """Generates *n* examples, writes JSONL, prints distribution stats."""

    # Key schedule: exactly n/5 of each letter, shuffled
    assert n % 5 == 0, "N_EXAMPLES must be a multiple of 5 for balanced keys"
    key_schedule: list[str] = []
    per_key = n // 5
    for letter in LETTERS:
        key_schedule.extend([letter] * per_key)
    random.shuffle(key_schedule)

    examples: list[dict] = []
    for idx in range(n):
        ex = generate_example(idx, key_schedule[idx])
        if ex is None:
            print(f"WARNING: could not generate example {idx}, skipping.")
            continue
        examples.append(ex)

    # ── Write output ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(examples)} examples to {output_path}\n")

    # ── Answer distribution ─────────────────────────────────────────────
    answer_counts = {l: 0 for l in LETTERS}
    for ex in examples:
        answer_counts[ex["answer"]] += 1

    print("Answer distribution:")
    for letter in LETTERS:
        bar = "\u2588" * answer_counts[letter]
        print(f"  {letter}: {answer_counts[letter]:3d}  {bar}")

    # ── Environment distribution ────────────────────────────────────────
    env_counts: dict[str, int] = {}
    for ex in examples:
        e = ex["metadata"]["environment"]
        env_counts[e] = env_counts.get(e, 0) + 1

    print("\nEnvironment distribution:")
    for e, c in sorted(env_counts.items()):
        print(f"  {e}: {c}")

    # ── Priority vehicle distribution ───────────────────────────────────
    pv_counts: dict[str, int] = {}
    for ex in examples:
        pv = ex["metadata"]["priority_vehicle"]
        pv_counts[pv] = pv_counts.get(pv, 0) + 1

    print("\nPriority vehicle distribution:")
    for pv, c in sorted(pv_counts.items()):
        print(f"  Vehicle {pv}: {c}")


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_PATH = str(PROJECT_ROOT / "dataset" / "core" / "task2_rightofway.jsonl")
    generate_task2(N_EXAMPLES, OUTPUT_PATH)
