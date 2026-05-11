"""Independent Task 2 validator and quality auditor."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from domain.entities import Direction, Environment, IntentDirection
from domain.trajectory import trajectory_cells


LETTERS = ("A", "B", "C", "D", "E")
NO_CLEAR_OPTION_TEXT = "No vehicle has an unambiguous right of way"
BOTH_OPTION_TEXT = "Both can pass at the same time"

RIGHT_OF = {
    Direction.NORTH: Direction.EAST,
    Direction.EAST: Direction.SOUTH,
    Direction.SOUTH: Direction.WEST,
    Direction.WEST: Direction.NORTH,
}

OPPOSITE = {
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.EAST: Direction.WEST,
    Direction.WEST: Direction.EAST,
}


@dataclass(frozen=True)
class PairResult:
    v1: str
    v2: str
    conflict: bool
    winner: str | None
    unsupported: bool
    reason: str


@dataclass(frozen=True)
class PriorityResult:
    priority: str | None
    yielding: str | None
    pairwise: tuple[PairResult, ...]
    wins: dict[str, tuple[str, ...]]
    conflicts: dict[str, tuple[str, ...]]
    reason: str


def _parse_direction(value: str) -> Direction:
    return Direction(value)


def _parse_intent(value: str | None) -> IntentDirection | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    for it in IntentDirection:
        if it.value == normalized or it.name.lower() == normalized:
            return it
    raise ValueError(f"unknown intent string: {value!r}")


def _pair_intersection(v1: dict, v2: dict) -> PairResult:
    i1 = _parse_intent(v1.get("intent"))
    i2 = _parse_intent(v2.get("intent"))
    if i1 is None or i2 is None:
        return PairResult(v1["id"], v2["id"], False, None, True, "missing intent")

    d1 = _parse_direction(v1["direction"])
    d2 = _parse_direction(v2["direction"])

    c1 = trajectory_cells(d1, i1)
    c2 = trajectory_cells(d2, i2)
    has_conflict = bool(c1 & c2)
    if not has_conflict:
        return PairResult(v1["id"], v2["id"], False, None, False, "no_conflict")

    v1_left = i1 == IntentDirection.TURN_LEFT
    v2_left = i2 == IntentDirection.TURN_LEFT

    # left-turn-yields-to-oncoming
    if v1_left != v2_left and OPPOSITE[d1] == d2:
        winner = v2["id"] if v1_left else v1["id"]
        return PairResult(v1["id"], v2["id"], True, winner, False, "left_turn_yields")

    # priority-to-the-right
    if RIGHT_OF[d1] == d2:
        return PairResult(v1["id"], v2["id"], True, v2["id"], False, "priority_right")
    if RIGHT_OF[d2] == d1:
        return PairResult(v1["id"], v2["id"], True, v1["id"], False, "priority_right")

    return PairResult(v1["id"], v2["id"], True, None, True, "conflict_without_rule")


def _pair_roundabout(v1: dict, v2: dict) -> PairResult:
    v1_inside = bool(v1.get("inside_intersection", False)) or v1.get("position") == "roundabout_lane"
    v2_inside = bool(v2.get("inside_intersection", False)) or v2.get("position") == "roundabout_lane"

    if v1_inside and v2_inside:
        return PairResult(v1["id"], v2["id"], True, None, True, "both_inside_roundabout")
    if not v1_inside and not v2_inside:
        return PairResult(v1["id"], v2["id"], False, None, False, "both_outside_roundabout")

    winner = v1["id"] if v1_inside else v2["id"]
    return PairResult(v1["id"], v2["id"], True, winner, False, "inside_priority")


def _direction_only_winner(v1: dict, v2: dict) -> str | None:
    d1 = _parse_direction(v1["direction"])
    d2 = _parse_direction(v2["direction"])
    if OPPOSITE[d1] == d2:
        return None
    if RIGHT_OF[d1] == d2:
        return v2["id"]
    if RIGHT_OF[d2] == d1:
        return v1["id"]
    return None


def _has_intent_sensitive_priority_pair(scenario_json: dict, priority: str) -> bool:
    if scenario_json["environment"] != Environment.INTERSECTION.value:
        return False
    vehicles = scenario_json["vehicles"]
    by_id = {v["id"]: v for v in vehicles}
    if priority not in by_id:
        return False
    pv = by_id[priority]
    for ov in vehicles:
        if ov["id"] == priority:
            continue
        pr = _pair_intersection(pv, ov)
        if not pr.conflict or pr.unsupported or pr.winner != priority:
            continue
        if _direction_only_winner(pv, ov) != priority:
            return True
    return False


def _is_no_clear_four_way_straight(scenario_json: dict) -> bool:
    if scenario_json.get("environment") != Environment.INTERSECTION.value:
        return False
    vehicles = scenario_json.get("vehicles", [])
    if len(vehicles) != 4:
        return False
    directions = {v.get("direction") for v in vehicles}
    if len(directions) != 4:
        return False
    for v in vehicles:
        intent = (v.get("intent") or "").strip().lower()
        if intent != IntentDirection.GO_STRAIGHT.value:
            return False
    return True


def _is_no_clear_threeway_straight(scenario_json: dict) -> bool:
    if scenario_json.get("environment") != Environment.INTERSECTION.value:
        return False
    vehicles = scenario_json.get("vehicles", [])
    if len(vehicles) != 3:
        return False
    directions = {v.get("direction") for v in vehicles}
    if len(directions) != 3:
        return False
    for v in vehicles:
        intent = (v.get("intent") or "").strip().lower()
        if intent != IntentDirection.GO_STRAIGHT.value:
            return False
    return True


def _has_right_left_conflict_pair(scenario_json: dict) -> bool:
    if scenario_json.get("environment") != Environment.INTERSECTION.value:
        return False
    vehicles = scenario_json.get("vehicles", [])
    by_id = {v["id"]: v for v in vehicles if "id" in v}
    ids = list(by_id.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            v1 = by_id[ids[i]]
            v2 = by_id[ids[j]]
            i1 = _parse_intent(v1.get("intent"))
            i2 = _parse_intent(v2.get("intent"))
            if {i1, i2} != {IntentDirection.TURN_LEFT, IntentDirection.TURN_RIGHT}:
                continue
            pr = _pair_intersection(v1, v2)
            if pr.conflict and not pr.unsupported:
                return True
    return False


def _dominant_direction_only(scenario_json: dict) -> str | None:
    if scenario_json["environment"] != Environment.INTERSECTION.value:
        return None
    vehicles = scenario_json["vehicles"]
    wins: dict[str, set[str]] = {v["id"]: set() for v in vehicles}
    conflicts: dict[str, set[str]] = {v["id"]: set() for v in vehicles}
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            v1, v2 = vehicles[i], vehicles[j]
            w = _direction_only_winner(v1, v2)
            if w is None:
                continue
            a, b = v1["id"], v2["id"]
            conflicts[a].add(b)
            conflicts[b].add(a)
            loser = b if w == a else a
            wins[w].add(loser)
    cands = [vid for vid in wins if conflicts[vid] and wins[vid] == conflicts[vid]]
    cands = [vid for vid in cands if len(conflicts[vid]) == len(vehicles) - 1]
    if len(cands) == 1:
        return cands[0]
    return None


def _alphabetical_non_left_heuristic(scenario_json: dict) -> str | None:
    if scenario_json["environment"] != Environment.INTERSECTION.value:
        return None
    vehicles = sorted(scenario_json["vehicles"], key=lambda v: v["id"])
    non_left = [
        v["id"]
        for v in vehicles
        if (v.get("intent") or "").strip().lower() != IntentDirection.TURN_LEFT.value
    ]
    if non_left:
        return non_left[0]
    return vehicles[0]["id"] if vehicles else None


def recompute_priority(scenario_json: dict) -> PriorityResult:
    env = Environment(scenario_json["environment"])
    vehicles = scenario_json["vehicles"]
    vids = [v["id"] for v in vehicles]

    wins: dict[str, set[str]] = {vid: set() for vid in vids}
    conflicts: dict[str, set[str]] = {vid: set() for vid in vids}
    pairwise: list[PairResult] = []

    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            if env == Environment.INTERSECTION:
                pr = _pair_intersection(vehicles[i], vehicles[j])
            elif env == Environment.ROUNDABOUT:
                pr = _pair_roundabout(vehicles[i], vehicles[j])
            else:
                return PriorityResult(
                    priority=None,
                    yielding=None,
                    pairwise=tuple(),
                    wins={k: tuple() for k in vids},
                    conflicts={k: tuple() for k in vids},
                    reason=f"unsupported environment {env.value}",
                )

            pairwise.append(pr)
            if pr.unsupported:
                return PriorityResult(
                    priority=None,
                    yielding=None,
                    pairwise=tuple(pairwise),
                    wins={k: tuple(sorted(v)) for k, v in wins.items()},
                    conflicts={k: tuple(sorted(v)) for k, v in conflicts.items()},
                    reason=f"unsupported pair ({pr.v1},{pr.v2}): {pr.reason}",
                )

            if not pr.conflict:
                continue

            a, b = pr.v1, pr.v2
            conflicts[a].add(b)
            conflicts[b].add(a)
            loser = b if pr.winner == a else a
            wins[pr.winner].add(loser)

    candidates = [
        vid
        for vid in vids
        if conflicts[vid] and wins[vid] == conflicts[vid]
    ]

    if len(candidates) != 1:
        return PriorityResult(
            priority=None,
            yielding=None,
            pairwise=tuple(pairwise),
            wins={k: tuple(sorted(v)) for k, v in wins.items()},
            conflicts={k: tuple(sorted(v)) for k, v in conflicts.items()},
            reason=f"expected one dominant vehicle, got {candidates}",
        )

    priority = candidates[0]
    if len(conflicts[priority]) != len(vids) - 1:
        return PriorityResult(
            priority=None,
            yielding=None,
            pairwise=tuple(pairwise),
            wins={k: tuple(sorted(v)) for k, v in wins.items()},
            conflicts={k: tuple(sorted(v)) for k, v in conflicts.items()},
            reason="dominant vehicle does not conflict with all others",
        )

    losers = sorted(
        list(conflicts[priority]),
        key=lambda vid: (-len(wins[vid]), vid),
    )

    return PriorityResult(
        priority=priority,
        yielding=losers[0] if losers else None,
        pairwise=tuple(pairwise),
        wins={k: tuple(sorted(v)) for k, v in wins.items()},
        conflicts={k: tuple(sorted(v)) for k, v in conflicts.items()},
        reason="ok",
    )


def _context_from_prompt(prompt: str) -> str:
    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    try:
        qidx = next(i for i, ln in enumerate(lines) if ln.startswith("Question:"))
    except StopIteration:
        return "<missing-question-line>"
    if qidx == 0:
        return "<missing-context-line>"
    return lines[qidx - 1]


def validate_example(example: dict) -> tuple[list[str], PriorityResult]:
    errs: list[str] = []

    # shape checks
    if set(example.get("choices", {}).keys()) != set(LETTERS):
        errs.append("choices keys must be exactly A..E")

    answer = example.get("answer")
    if answer not in LETTERS:
        errs.append(f"invalid answer letter: {answer!r}")

    if len(set(example.get("choices", {}).values())) != 5:
        errs.append("duplicate choice texts")

    dtypes = example.get("distractor_type", {})
    nt = sum(1 for t in dtypes.values() if t == "near_true")
    hf = sum(1 for t in dtypes.values() if t == "highly_false")
    if nt != 2 or hf != 2:
        errs.append(f"distractor type counts invalid: near_true={nt}, highly_false={hf}")

    # each vehicle option should appear exactly once (strict Task 2 contract)
    vehicles = example.get("scenario", {}).get("vehicles", [])
    vids = [v.get("id") for v in vehicles]
    for vid in vids:
        expected = f"Vehicle {vid}"
        c = sum(1 for txt in example.get("choices", {}).values() if txt == expected)
        if c != 1:
            errs.append(f"vehicle option occurrence for {expected!r} is {c}, expected 1")

    # independent recomputation
    recomputed = recompute_priority(example["scenario"])
    answer_text = example.get("choices", {}).get(answer, "")
    meta = example.get("metadata", {})
    resolution = meta.get("resolution")
    no_clear_mode = (
        resolution == "intersection_no_clear"
        or answer_text == NO_CLEAR_OPTION_TEXT
    )

    if no_clear_mode:
        subtype = meta.get("no_clear_subtype")
        if answer_text != NO_CLEAR_OPTION_TEXT:
            errs.append("no-clear example answer text mismatch")
        if BOTH_OPTION_TEXT in example.get("choices", {}).values():
            errs.append("'Both can pass at the same time' should not appear in no-clear mode")
        if subtype == "straight":
            if not _is_no_clear_four_way_straight(example.get("scenario", {})):
                errs.append("no-clear straight example does not match 4-way straight structure")
        elif subtype == "turnmix":
            sc = example.get("scenario", {})
            if sc.get("environment") != Environment.INTERSECTION.value or len(sc.get("vehicles", [])) != 4:
                errs.append("no-clear turnmix example must be 4-vehicle intersection")
            if not _has_right_left_conflict_pair(sc):
                errs.append("no-clear turnmix example lacks right-vs-left conflict pair")
        elif subtype == "threeway":
            if not _is_no_clear_threeway_straight(example.get("scenario", {})):
                errs.append("no-clear threeway example does not match 3-way straight structure")
        else:
            errs.append("no-clear example missing/invalid subtype")
        if recomputed.priority is not None:
            errs.append("no-clear example unexpectedly has a unique dominant priority")
        if meta.get("priority_vehicle") is not None:
            errs.append("metadata.priority_vehicle should be null in no-clear mode")
        if meta.get("yielding_vehicle") is not None:
            errs.append("metadata.yielding_vehicle should be null in no-clear mode")
        if meta.get("conflict_pair"):
            errs.append("metadata.conflict_pair should be empty in no-clear mode")
    else:
        if recomputed.priority is None:
            errs.append(f"recompute failed: {recomputed.reason}")
            return errs, recomputed
        if answer in LETTERS:
            expected_answer = f"Vehicle {recomputed.priority}"
            if answer_text != expected_answer:
                errs.append(
                    f"gold mismatch: answer text {answer_text!r}, expected {expected_answer!r}"
                )
        if meta.get("priority_vehicle") != recomputed.priority:
            errs.append(
                f"metadata.priority_vehicle={meta.get('priority_vehicle')!r} "
                f"!= recomputed {recomputed.priority!r}"
            )
        if meta.get("yielding_vehicle") != recomputed.yielding:
            errs.append(
                f"metadata.yielding_vehicle={meta.get('yielding_vehicle')!r} "
                f"!= recomputed {recomputed.yielding!r}"
            )
        cp = sorted(meta.get("conflict_pair", []))
        expected_cp = sorted([recomputed.priority, recomputed.yielding])
        if cp != expected_cp:
            errs.append(f"metadata.conflict_pair={cp!r} != expected {expected_cp!r}")

        if example.get("scenario", {}).get("environment") == Environment.INTERSECTION.value:
            intent_sensitive = _has_intent_sensitive_priority_pair(
                example["scenario"], recomputed.priority
            )
            if not intent_sensitive:
                errs.append("intersection example has no intent-sensitive priority pair")
            expected_meta_flag = meta.get("intent_sensitive_priority_pair")
            if expected_meta_flag is not None and bool(expected_meta_flag) != intent_sensitive:
                errs.append(
                    "metadata.intent_sensitive_priority_pair does not match recomputation"
                )
            direction_only_priority = _dominant_direction_only(example["scenario"])
            if direction_only_priority == recomputed.priority:
                errs.append("direction-only global heuristic matches priority")
            expected_donly_meta = meta.get("direction_only_priority")
            if expected_donly_meta is not None and expected_donly_meta != direction_only_priority:
                errs.append("metadata.direction_only_priority does not match recomputation")

    # non-triviality: require >=2 conflicting pairs
    conflict_count = sum(1 for p in recomputed.pairwise if p.conflict)
    if no_clear_mode:
        min_conflicts = 2 if meta.get("no_clear_subtype") == "threeway" else 4
        if conflict_count < min_conflicts:
            errs.append(f"too few conflict pairs for no-clear mode: {conflict_count}")
    else:
        if conflict_count < 2:
            errs.append(f"too few conflict pairs: {conflict_count}")

    return errs, recomputed


def main() -> int:
    p = argparse.ArgumentParser(description="Independent Task 2 validator + quality audit")
    p.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "core" / "task2_rightofway.jsonl"),
        help="path to Task 2 JSONL dataset",
    )
    p.add_argument(
        "--show-fails",
        type=int,
        default=20,
        help="how many failing examples to print",
    )
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"input not found: {path}", file=sys.stderr)
        return 2

    total = 0
    failed = 0
    json_errors = 0

    answer_counts: Counter[str] = Counter()
    env_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    question_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    choice_text_counts: Counter[str] = Counter()
    correct_text_counts: Counter[str] = Counter()
    conflict_count_dist: Counter[int] = Counter()
    no_clear_vehicle_counts: Counter[int] = Counter()

    with_both_phrase = 0
    both_by_letter: Counter[str] = Counter()
    intersection_total = 0
    intersection_unique_total = 0
    intersection_unique_priority_counts: Counter[str] = Counter()
    intersection_role_pattern_counts: Counter[tuple[tuple[str, str, str], ...]] = Counter()
    direction_only_correct = 0
    alpha_non_left_correct = 0
    distractor_letter_counts: Counter[str] = Counter()
    failed_records: list[tuple[str, list[str]]] = []

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                json_errors += 1
                failed += 1
                failed_records.append((f"line{line_no}", [f"invalid JSON: {e}"]))
                continue

            ex_id = ex.get("id", f"line{line_no}")

            # aggregate stats
            env = ex.get("scenario", {}).get("environment", "<missing-env>")
            env_counts[env] += 1
            answer_counts[ex.get("answer", "<missing-answer>")] += 1
            question_counts[ex.get("question", "<missing-question>")] += 1
            context_counts[_context_from_prompt(ex.get("prompt", ""))] += 1
            for txt in ex.get("choices", {}).values():
                choice_text_counts[txt] += 1
            if ex.get("answer") in LETTERS:
                correct_text_counts[ex["choices"][ex["answer"]]] += 1
            if BOTH_OPTION_TEXT in ex.get("choices", {}).values():
                with_both_phrase += 1
                for letter, text in ex.get("choices", {}).items():
                    if text == BOTH_OPTION_TEXT:
                        both_by_letter[letter] += 1
                        break
            for letter in LETTERS:
                if letter != ex.get("answer"):
                    distractor_letter_counts[letter] += 1
            resolution = ex.get("metadata", {}).get("resolution", "unique_priority")
            resolution_counts[resolution] += 1
            if resolution == "intersection_no_clear":
                no_clear_vehicle_counts[len(ex.get("scenario", {}).get("vehicles", []))] += 1
            if env == Environment.INTERSECTION.value:
                intersection_total += 1
                if resolution != "intersection_no_clear":
                    intersection_unique_total += 1
                    direction_only_guess = _dominant_direction_only(ex["scenario"])
                    if direction_only_guess is not None:
                        if ex["choices"].get(ex.get("answer")) == f"Vehicle {direction_only_guess}":
                            direction_only_correct += 1
                    alpha_guess = _alphabetical_non_left_heuristic(ex["scenario"])
                    if alpha_guess is not None:
                        if ex["choices"].get(ex.get("answer")) == f"Vehicle {alpha_guess}":
                            alpha_non_left_correct += 1
                    ans_text = ex["choices"].get(ex.get("answer"), "")
                    if ans_text.startswith("Vehicle "):
                        intersection_unique_priority_counts[ans_text.split(" ", 1)[1]] += 1
                    pattern = tuple(sorted(
                        (
                            str(v.get("id")),
                            str(v.get("direction")),
                            str(v.get("intent")),
                        )
                        for v in ex.get("scenario", {}).get("vehicles", [])
                    ))
                    intersection_role_pattern_counts[pattern] += 1

            errs, recomputed = validate_example(ex)
            if recomputed.priority is not None:
                priority_counts[recomputed.priority] += 1
                conflict_count_dist[sum(1 for p in recomputed.pairwise if p.conflict)] += 1

            if errs:
                failed += 1
                failed_records.append((ex_id, errs))

    # report
    print(f"Validated {total} examples; {failed} failed ({json_errors} JSON decode errors).")

    print("\nDistribution checks:")
    print("  Environment:", dict(sorted(env_counts.items())))
    print("  Resolution:", dict(sorted(resolution_counts.items())))
    print("  Answer letters:", dict(sorted(answer_counts.items())))
    print("  Recomputed priority vehicle:", dict(sorted(priority_counts.items())))
    print("  Pair-conflict-count:", dict(sorted(conflict_count_dist.items())))

    print("\nDiversity checks:")
    print(f"  Unique question templates observed: {len(question_counts)}")
    print(f"  Unique context templates observed: {len(context_counts)}")
    print(f"  Unique choice texts observed: {len(choice_text_counts)}")

    top_questions = question_counts.most_common(5)
    top_contexts = context_counts.most_common(5)
    top_choices = choice_text_counts.most_common(10)

    print("  Top questions:")
    for q, c in top_questions:
        print(f"    - {c:3d} | {q}")

    print("  Top contexts:")
    for ctext, c in top_contexts:
        print(f"    - {c:3d} | {ctext}")

    print("  Top choice texts:")
    for txt, c in top_choices:
        print(f"    - {c:3d} | {txt}")

    print("\nShortcut-leakage signals:")
    both_ratio = (with_both_phrase / total * 100.0) if total else 0.0
    print(
        f"  'Both can pass at the same time' appears in {with_both_phrase}/{total} "
        f"examples ({both_ratio:.1f}%)."
    )
    if intersection_unique_total:
        donly_acc = direction_only_correct / intersection_unique_total * 100.0
        print(
            f"  Direction-only intersection heuristic accuracy: "
            f"{direction_only_correct}/{intersection_unique_total} ({donly_acc:.1f}%)."
        )
        alpha_acc = alpha_non_left_correct / intersection_unique_total * 100.0
        print(
            f"  Alphabetical non-left heuristic accuracy: "
            f"{alpha_non_left_correct}/{intersection_unique_total} ({alpha_acc:.1f}%)."
        )
    roundabout_total = env_counts.get(Environment.ROUNDABOUT.value, 0)
    if total:
        inside_only_overall = roundabout_total / total * 100.0
        print(
            f"  Roundabout-inside shortcut ceiling (overall): "
            f"{roundabout_total}/{total} ({inside_only_overall:.1f}%)."
        )

    if failed_records:
        print("\nSample failures:")
        for ex_id, errs in failed_records[: max(0, args.show_fails)]:
            print(f"  [FAIL {ex_id}] {errs[0]}")

    quality_issues: list[str] = []
    if total and total % len(LETTERS) == 0:
        expected_per_letter = total // len(LETTERS)
        for letter in LETTERS:
            if answer_counts.get(letter, 0) != expected_per_letter:
                quality_issues.append(
                    f"answer letter imbalance: {letter}={answer_counts.get(letter, 0)} "
                    f"(expected {expected_per_letter})"
                )

    if intersection_unique_total:
        if len(intersection_unique_priority_counts) < 2:
            quality_issues.append("intersection unique-priority answers collapse to <2 vehicles")
        max_v = max(intersection_unique_priority_counts.values(), default=0)
        if max_v > int(0.70 * intersection_unique_total):
            quality_issues.append("intersection unique-priority vehicle imbalance too high")

    if intersection_role_pattern_counts:
        top_pattern = max(intersection_role_pattern_counts.values())
        if top_pattern > int(0.35 * max(1, intersection_unique_total)):
            quality_issues.append("repeated role-label pattern overuse detected")

    if total:
        max_correct = max(correct_text_counts.values(), default=0)
        if max_correct > int(0.55 * total):
            quality_issues.append("overuse of the same literal correct answer text")

    if resolution_counts.get("intersection_no_clear", 0) > 0:
        if sum(no_clear_vehicle_counts.values()) > 0 and len(no_clear_vehicle_counts) == 1 and 4 in no_clear_vehicle_counts:
            quality_issues.append("no-clear answer remains perfectly correlated with 4-vehicle cases")

    if both_by_letter:
        vals = [both_by_letter.get(letter, 0) for letter in LETTERS]
        if max(vals) - min(vals) > 12:
            quality_issues.append("'Both can pass...' shows distractor-letter bias")

    if quality_issues:
        print("\nQuality gate failures:")
        for issue in quality_issues:
            print(f"  - {issue}")

    return 0 if (failed == 0 and not quality_issues) else 1


if __name__ == "__main__":
    raise SystemExit(main())
