"""Minimum track record length (Bailey and Lopez de Prado, 2012, "The Sharpe Ratio Efficient Frontier").

All Sharpe ratios are per-period (not annualized).

PSR = Phi((SR - SR*) * sqrt(T - 1) / sqrt(1 - g3 * SR + (g4 - 1) / 4 * SR^2))
with sample skewness g3 and non-excess kurtosis g4 (normal: 0 and 3).

MinTRL is the smallest integer T >= 2 with PSR >= confidence. Solving PSR = confidence for T gives
T* = 1 + (1 - g3 * SR + (g4 - 1) / 4 * SR^2) * (z_confidence / (SR - SR*))^2, and the answer is ceil(T*).
"""

from __future__ import annotations

import math
from typing import Any

from scipy.stats import norm


def _finite(**values: Any) -> None:
    for name, v in values.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name} must be a finite number, got {v!r}")


def _variance_term(sr: float, skew: float, kurt: float) -> float:
    term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if not term > 0:
        raise ValueError(f"PSR variance term is not positive ({term:.4g}); skew, kurtosis and Sharpe are inconsistent")
    return term


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_periods: int, skew: float, kurt: float) -> float:
    """Probability that the true Sharpe exceeds ``sr_benchmark`` (all per-period), in [0, 1]."""
    _finite(sr=sr, sr_benchmark=sr_benchmark, skew=skew, kurt=kurt)
    if isinstance(n_periods, bool) or not isinstance(n_periods, int) or n_periods < 2:
        raise ValueError(f"n_periods must be an integer >= 2, got {n_periods!r}")
    term = _variance_term(sr, skew, kurt)
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(n_periods - 1) / math.sqrt(term)))


def min_track_record_length(sr: float, sr_benchmark: float, skew: float, kurt: float, confidence: float) -> int:
    """Smallest number of periods T (>= 2) with PSR >= ``confidence``."""
    _finite(sr=sr, sr_benchmark=sr_benchmark, skew=skew, kurt=kurt, confidence=confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence!r}")
    if sr <= sr_benchmark:
        raise ValueError("sr must be greater than sr_benchmark; no finite track record length exists")
    term = _variance_term(sr, skew, kurt)
    z = float(norm.ppf(confidence))
    t = max(2, math.ceil(1.0 + term * (z / (sr - sr_benchmark)) ** 2))
    # Guard against floating-point rounding at the boundary.
    while t > 2 and probabilistic_sharpe(sr, sr_benchmark, t - 1, skew, kurt) >= confidence:
        t -= 1
    while probabilistic_sharpe(sr, sr_benchmark, t, skew, kurt) < confidence:
        t += 1
    return int(t)
