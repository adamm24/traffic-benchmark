"""Controlled vocabulary for human-readable position labels.

Generators sample option text from this file instead of inventing fallback
labels such as "off the road" or "unknown location".
"""
from __future__ import annotations

from .entities import Environment


POSITION_LABELS: dict[str, str] = {
    # Multi-lane positions
    "left_shoulder":       "the left shoulder",
    "far_left_lane":        "the far-left lane",
    "left_center_lane":     "the left-center lane",
    "right_center_lane":    "the right-center lane",
    "far_right_lane":       "the far-right lane",
    "right_shoulder":       "the right shoulder",
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


_INTERSECTION_POSITIONS = (
    "north_approach", "south_approach", "east_approach", "west_approach",
    "inside_intersection",
    "north_exit", "south_exit", "east_exit", "west_exit",
)

_MULTI_LANE_POSITIONS = (
    "left_shoulder",
    "far_left_lane",
    "left_center_lane",
    "right_center_lane",
    "far_right_lane",
    "right_shoulder",
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


HIGHLY_FALSE_LABELS_BY_ENV: dict[Environment, tuple[str, ...]] = {
    Environment.INTERSECTION: (),
    Environment.MULTI_LANE: (),
    Environment.ROUNDABOUT: (),
}



def label_of(position: str) -> str:
    """Human-readable label for a raw position string."""
    if position not in POSITION_LABELS:
        raise ValueError(
            f"Position {position!r} is not in the controlled vocabulary. "
            f"Known positions: {sorted(POSITION_LABELS)}"
        )
    return POSITION_LABELS[position]


def positions_for_env(env: Environment) -> tuple[str, ...]:
    """Raw position strings valid for an environment."""
    return _POSITIONS_BY_ENV[env]


def labels_for_env(env: Environment) -> tuple[str, ...]:
    """Human-readable labels valid for an environment."""
    return tuple(POSITION_LABELS[p] for p in _POSITIONS_BY_ENV[env])


def cross_env_labels(env: Environment) -> tuple[str, ...]:
    """Labels valid in other environments, kept for audit checks."""
    import warnings
    warnings.warn(
        "cross_env_labels() is deprecated for use in generators. "
        "Use labels_for_env() or highly_false_labels_for_env() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    """True when `label` belongs to `env`."""
    return label in labels_for_env(env)


def highly_false_labels_for_env(env: Environment) -> tuple[str, ...]:
    """Dedicated highly_false label pool for `env`."""
    return HIGHLY_FALSE_LABELS_BY_ENV.get(env, ())


def is_env_consistent_label(label: str, env: Environment) -> bool:
    """True when `label` is reachable in `env`."""
    return label in labels_for_env(env)
