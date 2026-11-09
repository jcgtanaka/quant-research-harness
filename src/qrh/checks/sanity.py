"""Sanity bounds on a run: returns, leverage and position size must be plausible."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qrh.adapters.base import RunResult

PERCENT_MEDIAN_ABS = 0.5
PERCENT_HINT = "returns may be in percent, expected fractions"


@dataclass
class SanityResult:
    verdict: str  # PASS | FAIL | BLOCKED
    reasons: list[str]
    metrics: dict[str, float] = field(default_factory=dict)


def check_sanity(result: RunResult, bounds: dict[str, float]) -> SanityResult:
    """Compare run metrics with the bounds fixed at registration. Unknowns are BLOCKED."""
    if not bounds:
        return SanityResult("BLOCKED", ["no sanity bounds declared"])
    rets = result.returns.astype(float) if len(result.returns) else result.returns
    if len(rets) == 0:
        return SanityResult("BLOCKED", ["run produced no returns"])
    if np.isinf(rets.to_numpy()).any():
        return SanityResult("FAIL", ["returns contain infinite values"])
    values = rets.dropna().abs()
    if values.empty:
        return SanityResult("BLOCKED", ["all returns are missing"])
    metrics = {"max_abs_period_return": float(values.max()), "median_abs_period_return": float(values.median())}
    weights = result.positions
    wants_weights = any(k in bounds for k in ("max_gross_leverage", "max_position_weight"))
    if weights is not None and weights.shape[0] and weights.shape[1]:
        w = weights.astype(float).abs().fillna(0.0)
        metrics["max_gross_leverage"] = float(w.sum(axis=1).max())
        metrics["max_position_weight"] = float(w.to_numpy().max())
    elif wants_weights:
        return SanityResult("BLOCKED", ["no positions to check leverage and position weight against"], metrics)
    reasons = [
        f"{k} {metrics[k]:.6g} exceeds bound {limit:.6g}"
        for k, limit in sorted(bounds.items())
        if k in metrics and metrics[k] > limit
    ]
    if metrics["median_abs_period_return"] > PERCENT_MEDIAN_ABS:
        reasons.append(f"median absolute period return {metrics['median_abs_period_return']:.6g}: {PERCENT_HINT}")
    return SanityResult("FAIL" if reasons else "PASS", reasons, metrics)


def sanity_rows(metrics: dict[str, float], bounds: dict[str, float]) -> list[dict[str, object]]:
    """One row per measured quantity: its value, the bound it is held to (or None) and whether it is within it."""
    limits = {**bounds, "median_abs_period_return": PERCENT_MEDIAN_ABS}
    return [
        {
            "name": k, "value": metrics[k], "bound": limits.get(k),
            "ok": limits.get(k) is None or metrics[k] <= limits[k],
        }
        for k in sorted(metrics)
    ]
