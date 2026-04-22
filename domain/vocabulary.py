"""
Controlled vocabulary for human-readable position labels.

This module is the single source of truth for the strings that may appear
as MCQ options across the benchmark. Generators MUST sample their option
texts from this vocabulary and MUST NOT fabricate placeholder strings like
"off the road" or "unknown location" — those are forbidden leakage channels
(a model can learn that such a label is never the correct answer).

If a generator cannot find a valid label from the pool, it MUST reject the
example and retry rather than fall back to an invented string.
"""
from __future__ import annotations

from .entities import Environment, Lane


# ── Raw position → human-readable label ──────────────────────────────────────
#
# These labels are the ONLY strings allowed to appear as MCQ option text for
# position-based tasks. The renderer (domain.render) uses the same table.

POSITION_LABELS: dict[str, str] = {
    # Multi-lane positions
    "left_lane":            "the left lane",
    "center_lane":          "the center lane",
    "right_lane":           "the right lane",
    # Roundabout internal position
    "roundabout_lane":      "the roundabout lane",
    # Intersection body (logical "inside" state)
    "inside_intersection":  "inside the intersection",
    # Intersection approach arms
    "north_approach":       "the northern approach",
    "south_approach":       "the southern approach",
    "east_approach":        "the eastern approach",
    "west_approach":        "the western approach",
    # Intersection / roundabout exits
    "north_exit":           "the northern exit",
    "south_exit":           "the southern exit",
    "east_exit":            "the eastern exit",
    "west_exit":            "the western exit",
}


# ── Per-environment vocabularies ─────────────────────────────────────────────
#
# For each environment, the set of position labels that may legitimately
# appear in an example set in that environment.

_INTERSECTION_POSITIONS = (
    "north_approach", "south_approach", "east_approach", "west_approach",
    "inside_intersection",
    "north_exit", "south_exit", "east_exit", "west_exit",
)

_MULTI_LANE_POSITIONS = (
    "left_lane", "center_lane", "right_lane",
)

_ROUNDABOUT_POSITIONS = (
    "north_approach", "south_approach", "east_approach", "west_approach",
    "roundabout_lane",
    "north_exit", "south_exit", "east_exit", "west_exit",
)

_POSITIONS_BY_ENV: dict[Environment, tuple[str, ...]] = {
    Environment.INTERSECTION: _INTERSECTION_POSITIONS,
    Environment.MULTI_LANE:   _MULTI_LANE_POSITIONS,
    Environment.ROUNDABOUT:   _ROUNDABOUT_POSITIONS,
}


# ── Dedicated highly_false pool (same-environment, never reachable) ──────────
#
# These are real road-context concepts that a driver would recognise as
# same-domain labels but that the simulator never produces as a vehicle
# position. They exist so the generator can fill the 5-slot MCQ for
# environments whose reachable position set is too small (multi_lane_road
# only has 3 lanes, so same-environment distractors would exhaust after
# the correct answer + near_true_1).
#
# Rules:
#   • A highly_false label is NEVER a correct answer.
#   • A highly_false label belongs to the same driving domain as the env
#     (lane-like for multi_lane, intersection-geometry-like for
#     intersection if ever needed) so it is not trivially eliminable by
#     "environment keyword matching".
#   • Intersection already has 9 reachable labels → empty pool.
#   • Roundabout is excluded from Task 1 → empty pool.
#
# If a future task requires a same-env highly_false for intersection or
# roundabout, extend this table — do NOT fall back to cross-environment
# labels in generator code.

HIGHLY_FALSE_LABELS_BY_ENV: dict[Environment, tuple[str, ...]] = {
    Environment.INTERSECTION: (),
    Environment.MULTI_LANE: (
        "the road shoulder",
        "the emergency lane",
        "the median strip",
        "the oncoming lane",
    ),
    Environment.ROUNDABOUT: (),
}


# ── Public API ───────────────────────────────────────────────────────────────

def label_of(position: str) -> str:
    """
    Returns the human-readable label for a raw position string.

    Raises ValueError if the position is not in the controlled vocabulary.
    Callers should not catch and fall back — a missing label is a domain
    contract violation and should fail loudly.
    """
    if position not in POSITION_LABELS:
        raise ValueError(
            f"Position {position!r} is not in the controlled vocabulary. "
            f"Known positions: {sorted(POSITION_LABELS)}"
        )
    return POSITION_LABELS[position]


def positions_for_env(env: Environment) -> tuple[str, ...]:
    """Returns the tuple of raw position strings valid for an environment."""
    return _POSITIONS_BY_ENV[env]


def labels_for_env(env: Environment) -> tuple[str, ...]:
    """Returns the tuple of human-readable labels valid for an environment."""
    return tuple(POSITION_LABELS[p] for p in _POSITIONS_BY_ENV[env])


def cross_env_labels(env: Environment) -> tuple[str, ...]:
    """
    Returns labels that are valid in other environments but NOT valid in the
    given environment. Used to build `highly_false` distractors that are
    unambiguously wrong by environment.
    """
    current = set(labels_for_env(env))
    foreign: list[str] = []
    for other_env, positions in _POSITIONS_BY_ENV.items():
        if other_env == env:
            continue
        for p in positions:
            label = POSITION_LABELS[p]
            if label not in current and label not in foreign:
                foreign.append(label)
    return tuple(foreign)


def is_valid_label(label: str, env: Environment) -> bool:
    """True iff `label` is a human-readable label legitimate for `env`."""
    return label in labels_for_env(env)


def highly_false_labels_for_env(env: Environment) -> tuple[str, ...]:
    """
    Returns the dedicated highly_false label pool for `env`. These are
    same-domain, same-environment distractor labels that the simulator
    never produces as a real vehicle position. They are part of the
    controlled vocabulary but never appear as correct answers.

    See the module-level comment on HIGHLY_FALSE_LABELS_BY_ENV for the
    rules governing this pool.
    """
    return HIGHLY_FALSE_LABELS_BY_ENV.get(env, ())


def is_env_consistent_label(label: str, env: Environment) -> bool:
    """
    True iff `label` is legitimate for `env` as an MCQ option: either a
    real reachable position label for `env`, or one of the dedicated
    highly_false pool labels for `env`. Cross-environment labels return
    False — the generator MUST NOT use them as distractors.
    """
    return label in labels_for_env(env) or label in highly_false_labels_for_env(env)
