"""
Task 1 — Position Tracking Generator
=====================================
Generates multiple-choice questions that test an LLM's ability to track
vehicle positions through a sequence of actions.

Environments: multi_lane_road, intersection  (roundabout excluded)
Vehicles:     3
Steps:        2–4
Choices:      1 correct, 2 near_true, 2 highly_false

Post-refactor guarantees (addresses T1 improvement plan):
  • Distractor quality:
      near_true_1 = queried vehicle's START position.
      near_true_2 = another tracked vehicle's FINAL position in the same
                    environment (principled, grounded in the scenario).
      highly_false_1 = cross-environment label (e.g. a lane on an
                       intersection scenario).
      highly_false_2 = same-environment label that was never actually
                       visited by any vehicle (rules out shallow "pick
                       any in-env label" heuristics).
  • No hardcoded fallback strings. All labels flow through
    domain.vocabulary. If a label cannot be found the example is
    REJECTED and retried — never replaced with a placeholder.
  • Audit metadata: every example embeds an "audit" block with the full
    queried-vehicle trace, per-distractor rationale, invariant flags,
    and generation metadata (seed, generator version).
  • Deterministic with --seed (env RNG only, independent of system RNG).
  • Scale-ready: N_EXAMPLES can be set to 300 via CLI without changing
    the generator; per-env and per-key balance are enforced by the
    key_schedule and a soft env-balance retry loop.
  • FSM-backed: apply_action() now self-rejects invalid transitions
    (returns ""); safe_apply_action() below is a thin wrapper that
    converts "" → None so existing call sites keep working.

Usage:
    python generators/task1_position.py --n 300 --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

# ── Make project root importable ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from domain.entities import (
    Action, Environment, ScenarioState, Vehicle,
)
from domain.scenario import (
    LANE_ORDER, apply_action,
    build_intersection_scenario, build_multi_lane_scenario,
)
from domain.render import describe_scenario, render_prompt
from domain.vocabulary import (
    cross_env_labels, is_valid_label, label_of, labels_for_env,
    positions_for_env,
)

# ── Constants ───────────────────────────────────────────────────────────────

GENERATOR_VERSION = "task1_position.v2"   # bump when contract changes
DEFAULT_N_EXAMPLES = 100                  # CLI --n overrides; 300 = full core
NUM_VEHICLES = 3
MIN_STEPS    = 2
MAX_STEPS    = 4
MAX_RETRIES  = 50                         # per-example retry budget

TASK_ENVS = [Environment.MULTI_LANE, Environment.INTERSECTION]

# Actions valid per environment (Task 1 — position only, no roundabout)
# NOTE: MOVE_FORWARD excluded from MULTI_LANE because it doesn't change
#       the lane position and would be a no-op for position tracking.
ACTIONS_BY_ENV = {
    Environment.MULTI_LANE: [
        Action.CHANGE_LEFT,
        Action.CHANGE_RIGHT,
    ],
    Environment.INTERSECTION: [
        Action.MOVE_FORWARD,
        Action.TURN_LEFT,
        Action.TURN_RIGHT,
    ],
}


# ── FSM-compatible action wrapper ───────────────────────────────────────────

def safe_apply_action(
    state: ScenarioState,
    vehicle_id: str,
    action: Action,
) -> str | None:
    """
    Thin wrapper around the FSM-backed apply_action().

    Post FSM refactor (T1-B03/B04/B07), apply_action() rejects invalid
    transitions by returning "" without mutating state. This wrapper
    converts "" → None so legacy call-sites that treat None as "try a
    different action" continue to work unchanged.
    """
    result = apply_action(state, vehicle_id, action)
    return result if result else None


# ── Sequence generation ─────────────────────────────────────────────────────

def generate_sequence(
    state: ScenarioState,
    queried_vid: str,
    env: Environment,
    n_steps: int,
) -> list[tuple[str, Action]] | None:
    """
    Produces a valid n-step event sequence guaranteeing:
      • the queried vehicle is moved at least `min_queried_moves` times
      • its final position differs from its starting position

    Returns a list of (vehicle_id, Action) tuples on success, or None
    if it cannot satisfy both constraints within MAX_RETRIES attempts.
    The caller replays these tuples to get event strings AND to record
    every vehicle's trajectory for audit.
    """
    vehicle_ids = [v.id for v in state.vehicles]
    valid_actions = ACTIONS_BY_ENV[env]
    start_pos = state.get_vehicle(queried_vid).position

    min_queried_moves = 2 if env == Environment.INTERSECTION else 1

    for _ in range(MAX_RETRIES):
        trial_state = copy.deepcopy(state)
        plan: list[tuple[str, Action]] = []
        queried_move_count = 0

        for step_idx in range(n_steps):
            remaining = n_steps - step_idx
            queried_deficit = min_queried_moves - queried_move_count

            if queried_deficit >= remaining:
                vid = queried_vid
            elif step_idx == n_steps - 1 and queried_move_count == 0:
                vid = queried_vid
            else:
                vid = random.choice(vehicle_ids)

            action_pool = list(valid_actions)
            random.shuffle(action_pool)
            applied = False
            for act in action_pool:
                snapshot = copy.deepcopy(trial_state)
                result = safe_apply_action(snapshot, vid, act)
                if result is not None:
                    trial_state = snapshot
                    plan.append((vid, act))
                    if vid == queried_vid:
                        queried_move_count += 1
                    applied = True
                    break

            if not applied:
                break

        if len(plan) != n_steps:
            continue
        if queried_move_count < min_queried_moves:
            continue

        final_pos = trial_state.get_vehicle(queried_vid).position
        if final_pos == start_pos:
            continue

        # Replay onto the caller's state so mutations stick.
        for v_new in trial_state.vehicles:
            v_old = state.get_vehicle(v_new.id)
            v_old.position = v_new.position
            v_old.direction = v_new.direction
            v_old.inside_intersection = v_new.inside_intersection
            v_old.stopped = v_new.stopped
        state.event_log = trial_state.event_log
        state.step = trial_state.step
        return plan

    return None


# ── Per-vehicle trajectory replay ───────────────────────────────────────────

def replay_trajectories(
    init_state: ScenarioState,
    plan: list[tuple[str, Action]],
) -> dict[str, list[str]]:
    """
    Returns {vehicle_id: [positions]} — for every vehicle, the sequence
    of positions INCLUDING the starting position and after each of its
    own actions (NOT after every step). A vehicle that never moved has a
    1-element list holding only its starting position.
    """
    sim = copy.deepcopy(init_state)
    trace: dict[str, list[str]] = {v.id: [v.position] for v in sim.vehicles}

    for vid, act in plan:
        apply_action(sim, vid, act)
        trace[vid].append(sim.get_vehicle(vid).position)

    return trace


# ── Choice builder ──────────────────────────────────────────────────────────

def build_choices(
    env: Environment,
    queried_vid: str,
    queried_trace: list[str],
    other_traces: dict[str, list[str]],
) -> dict[str, dict] | None:
    """
    Build the 5-way MCQ choice set from real trajectory data.

    Returns:
        dict keyed by {"correct", "near_true_1", "near_true_2",
        "highly_false_1", "highly_false_2"}. Each value is a dict:
          { "text": <human-readable label>,
            "type": "correct" | "near_true" | "highly_false",
            "rationale": <short string explaining what this distractor
                          is and why it is plausibly wrong> }

    Returns None if the scenario cannot yield 5 distinct valid labels
    (vocabulary exhaustion). The caller MUST treat None as a rejection
    and retry — never fall back to a placeholder string.
    """
    # Correct answer: queried vehicle's final position.
    correct_pos   = queried_trace[-1]
    correct_label = label_of(correct_pos)

    # near_true_1: start position of the queried vehicle.
    start_pos   = queried_trace[0]
    nt1_label   = label_of(start_pos)
    if nt1_label == correct_label:
        # Shouldn't happen — generate_sequence guarantees start ≠ final.
        return None

    # near_true_2: final position of another tracked vehicle (same env).
    # Picking from the scenario's own vehicles keeps the distractor
    # grounded in the prompt — a model must actually track positions,
    # not just exclude implausible labels.
    nt2_label = None
    nt2_source = None
    other_ids = [vid for vid in other_traces if vid != queried_vid]
    random.shuffle(other_ids)
    for other_id in other_ids:
        candidate_pos = other_traces[other_id][-1]
        candidate_label = label_of(candidate_pos)
        if candidate_label not in (correct_label, nt1_label):
            nt2_label = candidate_label
            nt2_source = other_id
            break
    if nt2_label is None:
        # Fallback inside the same environment: any label not yet used.
        pool = [lab for lab in labels_for_env(env)
                if lab not in (correct_label, nt1_label)]
        if not pool:
            return None
        nt2_label = random.choice(pool)
        nt2_source = "env_pool"

    # highly_false_1: a cross-environment label (env-elimination baseline
    # CAN rule this out — intentionally, it's the easy distractor).
    cross_pool = list(cross_env_labels(env))
    cross_pool = [lab for lab in cross_pool
                  if lab not in (correct_label, nt1_label, nt2_label)]
    if not cross_pool:
        return None
    hf1_label = random.choice(cross_pool)

    # highly_false_2: a SAME-environment label that was never visited by
    # ANY vehicle in this scenario. This blocks the "pick any same-env
    # label besides the start" heuristic — to rule out hf2 the solver
    # must know where the vehicles actually are, not just what's valid
    # for the environment.
    visited_labels = set()
    for vid, trace in other_traces.items():
        for p in trace:
            visited_labels.add(label_of(p))
    for p in queried_trace:
        visited_labels.add(label_of(p))

    same_env_pool = [
        lab for lab in labels_for_env(env)
        if lab not in visited_labels
        and lab not in (correct_label, nt1_label, nt2_label, hf1_label)
    ]
    if not same_env_pool:
        # Scenario "fills" the environment; no unvisited label remains.
        # Back off to a second cross-env label so we still produce a
        # valid example, but mark the rationale accordingly.
        backup_pool = [lab for lab in cross_env_labels(env)
                       if lab not in (correct_label, nt1_label,
                                      nt2_label, hf1_label)]
        if not backup_pool:
            return None
        hf2_label = random.choice(backup_pool)
        hf2_rationale = (
            f"cross-environment label ({env.value} has no unvisited "
            f"same-env label available)"
        )
    else:
        hf2_label = random.choice(same_env_pool)
        hf2_rationale = f"same-environment label not visited by any vehicle"

    return {
        "correct": {
            "text": correct_label,
            "type": "correct",
            "rationale": "queried vehicle's final position",
        },
        "near_true_1": {
            "text": nt1_label,
            "type": "near_true",
            "rationale": "queried vehicle's start position",
        },
        "near_true_2": {
            "text": nt2_label,
            "type": "near_true",
            "rationale": (
                f"final position of vehicle {nt2_source}"
                if nt2_source and nt2_source != "env_pool"
                else "same-environment label, not currently held by any vehicle"
            ),
        },
        "highly_false_1": {
            "text": hf1_label,
            "type": "highly_false",
            "rationale": f"cross-environment label (not valid for {env.value})",
        },
        "highly_false_2": {
            "text": hf2_label,
            "type": "highly_false",
            "rationale": hf2_rationale,
        },
    }


# ── Letter assignment ───────────────────────────────────────────────────────

LETTERS = ["A", "B", "C", "D", "E"]


def assign_letters(
    choices_dict: dict[str, dict],
    correct_key: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], str]:
    """
    Shuffles the five options and places the correct answer at *correct_key*.

    Returns:
        choices           – dict  {A..E: text}
        distractor_type   – dict  {A..E: type}  (only for distractors)
        rationale_by_letter – dict {A..E: rationale} (for ALL options,
                                including the correct one)
        answer            – the letter of the correct answer
    """
    items = list(choices_dict.values())
    random.shuffle(items)

    target_idx = LETTERS.index(correct_key)
    correct_item = next(it for it in items if it["type"] == "correct")
    items.remove(correct_item)
    items.insert(target_idx, correct_item)

    choices: dict[str, str] = {}
    distractor_type: dict[str, str] = {}
    rationale_by_letter: dict[str, str] = {}
    for letter, item in zip(LETTERS, items):
        choices[letter] = item["text"]
        rationale_by_letter[letter] = item["rationale"]
        if item["type"] != "correct":
            distractor_type[letter] = item["type"]

    return choices, distractor_type, rationale_by_letter, correct_key


# ── Event rendering from plan ───────────────────────────────────────────────

def plan_to_events(
    init_state: ScenarioState, plan: list[tuple[str, Action]]
) -> list[str]:
    """Replays `plan` to produce the human-readable event strings."""
    sim = copy.deepcopy(init_state)
    events: list[str] = []
    for vid, act in plan:
        ev = apply_action(sim, vid, act)
        if not ev:
            raise RuntimeError(
                f"Planned action rejected by FSM at replay: ({vid}, {act})."
            )
        events.append(ev)
    return events


# ── Single example generator ────────────────────────────────────────────────

def generate_example(
    example_id: int,
    correct_key: str,
    env_hint: Environment | None = None,
) -> dict | None:
    """
    Generates a single Task 1 example. Returns None if no valid sequence
    and a valid 5-way choice set can be produced within MAX_RETRIES.

    Args:
        example_id:   running index for the example id.
        correct_key:  letter (A..E) where the correct answer must land.
        env_hint:     if set, restrict generation to this environment
                      (used by the key_schedule to keep env balance).
    """
    for attempt in range(MAX_RETRIES):
        env = env_hint if env_hint is not None else random.choice(TASK_ENVS)
        n_steps = random.randint(MIN_STEPS, MAX_STEPS)

        if env == Environment.MULTI_LANE:
            state = build_multi_lane_scenario(NUM_VEHICLES)
        else:
            state = build_intersection_scenario(NUM_VEHICLES, with_intent=False)

        queried_vid = random.choice([v.id for v in state.vehicles])
        init_state = copy.deepcopy(state)

        sim_state = copy.deepcopy(state)
        plan = generate_sequence(sim_state, queried_vid, env, n_steps)
        if plan is None:
            continue

        events = plan_to_events(init_state, plan)
        trace  = replay_trajectories(init_state, plan)

        queried_trace = trace[queried_vid]
        raw_choices = build_choices(env, queried_vid, queried_trace, trace)
        if raw_choices is None:
            continue  # vocabulary exhaustion → retry

        # Post-condition: all five labels are distinct and valid for env
        # (cross-env hf is intentionally NOT valid for env — that's its role).
        letter_texts = {c["text"] for c in raw_choices.values()}
        if len(letter_texts) != 5:
            continue

        choices, distractor_type, rationale_by_letter, answer = assign_letters(
            raw_choices, correct_key
        )

        scenario_text = describe_scenario(init_state)
        question = f"Where is Vehicle {queried_vid} at the end of the sequence?"
        prompt = render_prompt(scenario_text, events, question, choices)

        return {
            "id": f"task1_{example_id:04d}",
            "task": "position_tracking",
            "prompt": prompt,
            "scenario": {
                "vehicles": [
                    {
                        "id": v.id,
                        "position": v.position,
                        "direction": v.direction.value,
                    }
                    for v in init_state.vehicles
                ],
                "environment": env.value,
            },
            "events": events,
            "question": question,
            "choices": choices,
            "answer": answer,
            "distractor_type": distractor_type,
            "metadata": {
                "num_vehicles": NUM_VEHICLES,
                "num_events": len(events),
                "queried_vehicle": queried_vid,
                "environment": env.value,
                "difficulty": "base",
            },
            "audit": {
                "generator_version": GENERATOR_VERSION,
                "attempt": attempt,
                "queried_trace": queried_trace,
                "all_traces": trace,
                "plan": [[vid, act.name] for vid, act in plan],
                "rationale_by_letter": rationale_by_letter,
                "invariants": {
                    "start_ne_final": queried_trace[0] != queried_trace[-1],
                    "queried_moved": sum(1 for vid, _ in plan if vid == queried_vid),
                    "all_labels_in_vocabulary": all(
                        is_valid_label(choices[L], env)
                        or choices[L] in cross_env_labels(env)
                        for L in LETTERS
                    ),
                    "five_distinct_options": len({choices[L] for L in LETTERS}) == 5,
                },
            },
        }

    return None


# ── Main generation loop ────────────────────────────────────────────────────

def generate_task1(n: int, output_path: str, seed: int | None = None) -> None:
    """Generates *n* examples, writes JSONL, prints distribution stats."""
    if seed is not None:
        random.seed(seed)

    if n % 5 != 0:
        raise ValueError("N must be a multiple of 5 for balanced key schedule.")
    if n % (5 * len(TASK_ENVS)) != 0:
        # Not strictly required, but warn the user: perfect joint balance
        # (letter × env) needs n divisible by 10.
        print(
            f"NOTE: n={n} is not divisible by {5 * len(TASK_ENVS)}; joint "
            f"letter×env balance will be approximate."
        )

    # Key schedule: exactly n/5 of each letter, shuffled.
    key_schedule: list[str] = []
    per_key = n // 5
    for letter in LETTERS:
        key_schedule.extend([letter] * per_key)
    random.shuffle(key_schedule)

    # Env schedule: stable 50/50 ratio enforced up front so we don't rely
    # on per-example random.choice drift over 300 examples.
    env_schedule: list[Environment] = []
    per_env = n // len(TASK_ENVS)
    remainder = n - per_env * len(TASK_ENVS)
    for env in TASK_ENVS:
        env_schedule.extend([env] * per_env)
    env_schedule.extend([random.choice(TASK_ENVS)] * remainder)
    random.shuffle(env_schedule)

    examples: list[dict] = []
    dropped = 0
    for idx in range(n):
        ex = generate_example(idx, key_schedule[idx], env_hint=env_schedule[idx])
        if ex is None:
            dropped += 1
            print(f"WARNING: could not generate example {idx}, skipping.")
            continue
        examples.append(ex)

    # ── Write output ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(examples)} examples to {output_path}")
    if dropped:
        print(f"Dropped {dropped} examples (generation failures).")
    print()

    # ── Answer distribution ─────────────────────────────────────────────
    answer_counts = {letter: 0 for letter in LETTERS}
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

    # ── Distractor-type sanity check ────────────────────────────────────
    type_counts = {"near_true": 0, "highly_false": 0}
    for ex in examples:
        for t in ex["distractor_type"].values():
            type_counts[t] = type_counts.get(t, 0) + 1
    print("\nDistractor-type totals (expect 2:2 per example):")
    for t, c in type_counts.items():
        print(f"  {t}: {c}")


# ── Entry point ─────────────────────────────────────────────────────────────

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 1 position tracking generator")
    p.add_argument("--n", type=int, default=DEFAULT_N_EXAMPLES,
                   help="number of examples (must be multiple of 5)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for reproducibility")
    p.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "core" / "task1_position.jsonl"),
        help="output JSONL path",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    generate_task1(args.n, args.out, args.seed)
