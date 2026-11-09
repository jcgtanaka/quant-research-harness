"""Counts, exact binomial intervals and Fisher comparisons for benchmark outcomes."""

from __future__ import annotations

from typing import Any

from scipy import stats

ALPHA = 0.05
CI_LEVEL = 0.95
DIRECTIONS = ("higher", "lower")  # which way is better for a metric


def exact_ci(k: int, n: int, level: float = CI_LEVEL) -> tuple[float, float]:
    """Clopper-Pearson interval for k successes in n trials."""
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=level, method="exact")
    return float(ci.low), float(ci.high)


def metric(k: int, n: int, *, direction: str, gate: bool = False, label: str = "") -> dict[str, Any]:
    """A benchmark outcome: k of n, its rate and exact 95% interval, and which direction is better.

    ``gate`` marks the metrics a promotion is judged on.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"inconsistent counts: {k} of {n}")
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci": None, "direction": direction, "gate": gate, "label": label}
    lo, hi = exact_ci(k, n)
    return {"k": int(k), "n": int(n), "rate": k / n, "ci": [lo, hi], "direction": direction, "gate": gate, "label": label}


def compare_counts(k_a: int, n_a: int, k_b: int, n_b: int, alpha: float = ALPHA) -> dict[str, Any]:
    """Fisher exact test (two-sided) of two proportions; ``delta`` is rate_b - rate_a.

    The two samples are treated as independent, which is conservative when both come from the same seeds.
    """
    if n_a == 0 or n_b == 0:
        return {"delta": None, "p_value": None, "distinguishable": False}
    _, p = stats.fisher_exact([[k_a, n_a - k_a], [k_b, n_b - k_b]])
    return {"delta": k_b / n_b - k_a / n_a, "p_value": float(p), "distinguishable": bool(p < alpha)}
