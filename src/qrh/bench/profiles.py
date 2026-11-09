"""Benchmark profiles. ``quick`` is for tests and smoke checks; ``full`` is for real comparisons.

Both are deterministic for a given seed. Replications per suite can be overridden with ``--reps``.
"""

from __future__ import annotations

import copy
from typing import Any

SUITES = ("planted-bugs", "noise-fpr", "known-edge-power")
OPTIONAL_SUITES = ("history", "llm-gate", "onboarding", "param-stability")
WALK_FORWARD = {"train": 0.3, "test": 0.1}  # fractions of the development window, whatever its length
DEFAULT_SEED = 20240601

PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "quick": {
        "planted-bugs": {"reps": 3},
        "noise-fpr": {"grids": [12, 36], "periods": [500], "reps": 30},
        "known-edge-power": {"grid": 12, "periods": 800, "strengths": [0.002, 0.004, 0.008], "reps": 25},
        "llm-gate": {"reps": 3},
        "onboarding": {"reps": 2},
        "param-stability": {"reps": 3},
    },
    "full": {
        "planted-bugs": {"reps": 30},
        "noise-fpr": {"grids": [12, 36, 120], "periods": [500, 1000, 2000], "reps": 150},
        "known-edge-power": {"grid": 36, "periods": 1000, "strengths": [0.002, 0.003, 0.004, 0.006, 0.008], "reps": 200},
        "llm-gate": {"reps": 30},
        "onboarding": {"reps": 10},
        "param-stability": {"reps": 30},
    },
}


def resolve_profile(name: str, reps: int | None = None) -> dict[str, dict[str, Any]]:
    """A copy of a profile's suite configuration, with the replications overridden if ``reps`` is given."""
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; use one of {sorted(PROFILES)}")
    cfg = copy.deepcopy(PROFILES[name])
    if reps is not None:
        if isinstance(reps, bool) or reps < 1:
            raise ValueError("reps must be an integer >= 1")
        for suite in cfg.values():
            suite["reps"] = int(reps)
    return cfg
