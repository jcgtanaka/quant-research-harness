"""Bounds on the unexplained remainder of a PnL reconstruction.

Pure functions. Every cap, lag and alpha is a required parameter with no
default. Nothing here decides PASS or FAIL: each function returns a small
typed result so the caller can build the verdict.

Status values:
    EVALUATED      the bound was computed and compared.
    NOT_EVALUATED  the cap is None (pending); the measurement is still returned.
    UNDEFINED      the comparison has no meaning (zero net explained PnL);
                   the caller must treat this as inconclusive, not a pass.

A cap parameter is one of: None (pending), a number (explicit override), or ``NOISE_SCALED``
(the limit is derived from the noise measured in the remainder itself, see ``NOISE_SCALE_K``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

EVALUATED = "evaluated"
NOT_EVALUATED = "not_evaluated"
UNDEFINED = "undefined"

NOISE_SCALED = "noise_scaled"  # cap sentinel: derive the limit from the remainder's own noise
# Number of standard deviations of honest noise allowed before the net remainder is called unexplained profit.
# A design constant, not a tuned value.
NOISE_SCALE_K = 3


@dataclass(frozen=True)
class AbsoluteCapResult:
    status: str
    annualised_bp: float
    cap_bp_per_year: Optional[float]
    breached: Optional[bool]


@dataclass(frozen=True)
class ShareResult:
    status: str
    share: Optional[float]
    max_unexplained_share: Optional[float]
    breached: Optional[bool]


@dataclass(frozen=True)
class DriftResult:
    status: str
    mean: float
    std_error: float
    t_stat: float
    p_value: float
    alpha: float
    significant: bool
    lag: int


@dataclass(frozen=True)
class SlopeResult:
    status: str
    slope: Optional[float]
    intercept: Optional[float]
    std_error: Optional[float]
    limit: Optional[float]  # K x SE(b): the largest |b - 1| allowed
    breached: Optional[bool]
    lag: int


def _as_series(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numbers")
    return arr


def _pair(remainder, explained) -> tuple[np.ndarray, np.ndarray]:
    r = _as_series(remainder, "remainder")
    e = _as_series(explained, "explained_net_pnl")
    if r.size != e.size:
        raise ValueError("remainder and explained_net_pnl must have equal length")
    return r, e


def _positive_finite(value, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return v


def _optional_cap(value, name: str) -> Optional[float]:
    if value is None:
        return None
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} cap must be a finite number >= 0 or None")
    return v


def _noise_sigma(r: np.ndarray) -> float:
    """Per-bar standard deviation of the demeaned remainder."""
    return float(np.std(r))


def noise_scaled_limits(remainder, capital_base: float, bars_per_year: float) -> tuple[float, float]:
    """(absolute limit in bp per year, net-remainder limit in currency) at K standard deviations of honest noise.

    sigma_year_bp = sigma_bar / capital * 1e4 * sqrt(bars_per_year); absolute limit = K * sigma_year_bp.
    The currency limit is K * sigma_bar * sqrt(n_bars); the share limit is that divided by |net explained PnL|."""
    r = _as_series(remainder, "remainder")
    capital = _positive_finite(capital_base, "capital_base")
    bpy = _positive_finite(bars_per_year, "bars_per_year")
    sigma = _noise_sigma(r)
    return (NOISE_SCALE_K * sigma / capital * 1e4 * math.sqrt(bpy),
            NOISE_SCALE_K * sigma * math.sqrt(r.size))


def absolute_cap_bound(
    remainder,
    capital_base: float,
    bars_per_year: float,
    max_unexplained_bp_per_year,
) -> AbsoluteCapResult:
    """Annualised NET remainder (sum over all bars, up and down) in bp of capital.

    Zero-mean rounding and timing noise cancels; a steady hidden profit does not."""
    r = _as_series(remainder, "remainder")
    capital = _positive_finite(capital_base, "capital_base")
    bpy = _positive_finite(bars_per_year, "bars_per_year")
    if isinstance(max_unexplained_bp_per_year, str) and max_unexplained_bp_per_year == NOISE_SCALED:
        cap = noise_scaled_limits(r, capital, bpy)[0]
    else:
        cap = _optional_cap(max_unexplained_bp_per_year, "max_unexplained_bp_per_year")
    net = float(np.sum(r))
    annualised_bp = net / capital * 1e4 * bpy / r.size
    if cap is None:
        return AbsoluteCapResult(NOT_EVALUATED, annualised_bp, None, None)
    return AbsoluteCapResult(EVALUATED, annualised_bp, cap, annualised_bp > cap)


def share_bound(
    remainder,
    explained_net_pnl,
    max_unexplained_share,
) -> ShareResult:
    """|sum remainder| relative to |sum explained net PnL|."""
    r, e = _pair(remainder, explained_net_pnl)
    denom = abs(float(np.sum(e)))
    scaled = isinstance(max_unexplained_share, str) and max_unexplained_share == NOISE_SCALED
    if scaled:
        cap = NOISE_SCALE_K * _noise_sigma(r) * math.sqrt(r.size) / denom if denom != 0.0 else 0.0
    else:
        cap = _optional_cap(max_unexplained_share, "max_unexplained_share")
    if denom == 0.0:
        if not np.any(r):  # flat, all-zero remainder: nothing unexplained, clean
            return ShareResult(EVALUATED if cap is not None else NOT_EVALUATED, 0.0, cap,
                               False if cap is not None else None)
        return ShareResult(UNDEFINED, None, cap, None)
    share = abs(float(np.sum(r))) / denom
    if cap is None:
        return ShareResult(NOT_EVALUATED, share, None, None)
    return ShareResult(EVALUATED, share, cap, share > cap)


def _newey_west_variance_of_mean(x: np.ndarray, lag: int) -> float:
    n = x.size
    d = x - x.mean()
    lrv = float(np.dot(d, d)) / n
    for k in range(1, lag + 1):
        gamma = float(np.dot(d[k:], d[:-k])) / n
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    return max(lrv, 0.0) / n


def drift_bound(remainder, newey_west_lag: int, alpha: float) -> DriftResult:
    """One-sided test of mean remainder > 0 with a Newey-West standard error."""
    r = _as_series(remainder, "remainder")
    if isinstance(newey_west_lag, bool) or int(newey_west_lag) != newey_west_lag:
        raise ValueError("newey_west_lag must be an integer")
    lag = int(newey_west_lag)
    if lag < 0 or lag >= r.size:
        raise ValueError("newey_west_lag must satisfy 0 <= lag < number of bars")
    a = float(alpha)
    if not math.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    mean = float(r.mean())
    se = math.sqrt(_newey_west_variance_of_mean(r, lag))
    if se == 0.0:
        # Degenerate (e.g. constant) series: the sign of the mean is certain.
        t = math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
    else:
        t = mean / se
    p = float(stats.norm.sf(t))
    return DriftResult(EVALUATED, mean, se, t, p, a, p < a, lag)


def slope_bound(reported, explained, newey_west_lag: int) -> SlopeResult:
    """Scale check: reported = a + b x explained + e (OLS with intercept); honest runs have b near 1.

    SE(b) is Newey-West on the score x_c[t] * u[t] (x_c demeaned explained, u the OLS residual, whose score has
    zero mean), divided by mean(x_c^2). Breach when |b - 1| > NOISE_SCALE_K x SE(b). Zero-variance explained PnL
    gives UNDEFINED: the caller must treat that as inconclusive, not a pass."""
    rep, exp = _pair(reported, explained)
    if isinstance(newey_west_lag, bool) or int(newey_west_lag) != newey_west_lag:
        raise ValueError("newey_west_lag must be an integer")
    lag = int(newey_west_lag)
    if lag < 0 or lag >= rep.size:
        raise ValueError("newey_west_lag must satisfy 0 <= lag < number of bars")
    xc = exp - exp.mean()
    sxx = float(np.mean(xc * xc))
    if sxx == 0.0:
        return SlopeResult(UNDEFINED, None, None, None, None, None, lag)
    slope = float(np.mean(xc * (rep - rep.mean())) / sxx)
    intercept = float(rep.mean() - slope * exp.mean())
    score = xc * (rep - intercept - slope * exp)
    se = math.sqrt(_newey_west_variance_of_mean(score, lag)) / sxx
    limit = NOISE_SCALE_K * se
    deviation = abs(slope - 1.0)
    return SlopeResult(EVALUATED, slope, intercept, se, limit, deviation > limit and deviation > 1e-9, lag)
