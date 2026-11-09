"""Performance metrics on simple per-period returns.

Undefined quantities are ``None``, never ``inf`` or ``NaN``. Non-finite input
is an error (the caller turns it into BLOCKED).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365.25 * 86400.0
_REL_ZERO = 1e-12  # std this small relative to |mean| counts as zero (float noise)


def as_array(returns: Any) -> np.ndarray:
    """1-D float array; raises ValueError on NaN or inf."""
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not np.isfinite(arr).all():
        raise ValueError("returns contain NaN or inf")
    return arr


def sharpe(returns: Any) -> float | None:
    """Per-period Sharpe ratio mean/std (ddof=1, zero risk-free). None if undefined."""
    r = as_array(returns)
    if r.size < 2:
        return None
    mean, std = float(r.mean()), float(r.std(ddof=1))
    if not math.isfinite(std) or std <= _REL_ZERO * abs(mean):
        return None
    return mean / std


def annualized_sharpe(returns: Any, periods_per_year: float) -> float | None:
    if not periods_per_year > 0:
        raise ValueError("periods_per_year must be positive")
    s = sharpe(returns)
    return None if s is None else s * math.sqrt(periods_per_year)


def cagr(returns: Any, periods_per_year: float) -> float | None:
    """Compound annual growth rate from simple returns; -1.0 on total loss."""
    if not periods_per_year > 0:
        raise ValueError("periods_per_year must be positive")
    r = as_array(returns)
    if r.size == 0:
        return None
    if (r <= -1.0).any():
        return -1.0
    years = r.size / periods_per_year
    return float(math.expm1(np.log1p(r).sum() / years))


def max_drawdown(returns: Any) -> float | None:
    """Largest peak-to-trough loss as a negative fraction (0.0 if none), from equity 1."""
    r = as_array(returns)
    if r.size == 0:
        return None
    equity = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def infer_periods_per_year(index: Any) -> float | None:
    """Observed bars per year of a sorted timestamp index, else None.

    The rate is the observed density ``(n - 1) / (span_seconds / SECONDS_PER_YEAR)``
    with a 365.25-day year, so business-day and session calendars give their real
    rate (about 261 for weekday-only daily bars) instead of the spacing of the
    typical bar. Holes in the data lower the rate. Needs at least 3 timestamps,
    sorted, without duplicates.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return None
    if not index.is_monotonic_increasing or index.has_duplicates:
        return None
    span = float((index[-1] - index[0]).total_seconds())
    if not span > 0:
        return None
    return (len(index) - 1) / (span / SECONDS_PER_YEAR)
