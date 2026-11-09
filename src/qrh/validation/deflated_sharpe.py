"""Probabilistic and Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014).

All Sharpe ratios here are per-period (not annualized).

PSR(SR*) = Phi((SR - SR*) * sqrt(T - 1) / sqrt(1 - g3 * SR + (g4 - 1) / 4 * SR^2))
with sample skewness g3 and non-excess kurtosis g4 (normal: 0 and 3).

Expected maximum Sharpe of N independent trials under the null:
SR0 = sqrt(V) * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N * e)))
with gamma the Euler-Mascheroni constant and V the variance of the trial Sharpe
ratios. DSR = PSR(SR0); N = 1 gives SR0 = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from qrh.validation.metrics import as_array, sharpe

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


@dataclass(frozen=True)
class DSRResult:
    dsr: float
    psr: float  # PSR against a zero benchmark
    sharpe: float  # per-period
    sr0: float
    n_trials: int
    var_sharpe: float
    n_obs: int
    skew: float
    kurtosis: float


def sample_moments(returns: Any) -> tuple[float, float]:
    """Sample skewness and non-excess kurtosis (central moments, divisor T)."""
    r = as_array(returns)
    d = r - r.mean()
    m2 = float((d**2).mean())
    if not m2 > 0:
        raise ValueError("returns have zero variance; skewness and kurtosis are undefined")
    return float((d**3).mean() / m2**1.5), float((d**4).mean() / m2**2)


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int, skew: float, kurtosis: float) -> float:
    """PSR: probability that the true Sharpe exceeds ``sr_benchmark`` (all per-period)."""
    if n_obs < 2:
        raise ValueError("PSR needs at least 2 observations")
    if not all(math.isfinite(x) for x in (sr, sr_benchmark, skew, kurtosis)):
        raise ValueError("PSR inputs must be finite")
    radicand = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2
    if not radicand > 0:
        raise ValueError(f"PSR variance term is not positive ({radicand:.4g}); moments are inconsistent")
    return _NORMAL.cdf((sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(radicand))


def _check_trials(n_trials: Any, var_sharpe: Any) -> tuple[int, float]:
    if isinstance(n_trials, bool) or not isinstance(n_trials, (int, np.integer)) or n_trials < 1:
        raise ValueError(f"n_trials must be an integer >= 1, got {n_trials!r}")
    v = float(var_sharpe)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"var_sharpe must be finite and >= 0, got {var_sharpe!r}")
    return int(n_trials), v


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """SR0, the expected maximum per-period Sharpe among ``n_trials`` null trials."""
    n, v = _check_trials(n_trials, var_sharpe)
    if n == 1 or v == 0.0:
        return 0.0
    g = EULER_GAMMA
    return math.sqrt(v) * ((1 - g) * _NORMAL.inv_cdf(1 - 1 / n) + g * _NORMAL.inv_cdf(1 - 1 / (n * math.e)))


def deflated_sharpe(returns: Any, n_trials: int, var_sharpe: float) -> DSRResult:
    """DSR of a return series given the number of trials and the variance of their Sharpe ratios."""
    n, v = _check_trials(n_trials, var_sharpe)
    r = as_array(returns)
    sr = sharpe(r)
    if sr is None:
        raise ValueError("Sharpe ratio is undefined for these returns")
    skew, kurt = sample_moments(r)
    sr0 = expected_max_sharpe(n, v)
    return DSRResult(
        dsr=probabilistic_sharpe(sr, sr0, r.size, skew, kurt),
        psr=probabilistic_sharpe(sr, 0.0, r.size, skew, kurt),
        sharpe=sr, sr0=sr0, n_trials=n, var_sharpe=v, n_obs=int(r.size), skew=skew, kurtosis=kurt,
    )
