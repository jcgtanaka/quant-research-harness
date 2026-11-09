"""Stationary bootstrap (Politis and Romano, 1994) with percentile intervals.

Resampled series are built from blocks of consecutive observations (circular)
whose lengths are geometric with a given mean, which keeps the resampled series
stationary and preserves short-range dependence. A mean block length of 1 is
the ordinary i.i.d. bootstrap. All randomness comes from a seeded
``numpy.random.Generator``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qrh.validation.metrics import as_array

MIN_OBSERVATIONS = 5


def default_block_length(n: int) -> int:
    """Mean block length ceil(n ** (1/3))."""
    return max(1, math.ceil(round(n ** (1.0 / 3.0), 12)))


def stationary_bootstrap_indices(n: int, reps: int, mean_block_length: float, rng: np.random.Generator) -> np.ndarray:
    """A (reps, n) array of indices in [0, n). Each step starts a new block with prob 1/L."""
    if n < 1 or reps < 1:
        raise ValueError("n and reps must be >= 1")
    if not mean_block_length >= 1:
        raise ValueError("mean block length must be >= 1")
    p = 1.0 / mean_block_length
    starts = rng.integers(0, n, size=(reps, n))
    fresh = rng.random((reps, n)) < p
    fresh[:, 0] = True
    pos = np.arange(n)
    last_fresh = np.maximum.accumulate(np.where(fresh, pos, 0), axis=1)
    base = np.take_along_axis(starts, last_fresh, axis=1)
    return (base + pos - last_fresh) % n


@dataclass(frozen=True)
class BootstrapResult:
    point: dict[str, float | None]
    intervals: dict[str, tuple[float, float] | None]
    level: float
    reps: int
    seed: int
    mean_block_length: float
    periods_per_year: float
    samples: dict[str, np.ndarray] = field(default_factory=dict, compare=False, repr=False)  # per-replication metrics


def _stats(samples: np.ndarray, ppy: float) -> dict[str, np.ndarray]:
    """Vectorized annualized Sharpe (NaN if no variation), CAGR and max drawdown per row."""
    n = samples.shape[1]
    mean, std = samples.mean(axis=1), samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 1e-12 * np.abs(mean), mean / std * math.sqrt(ppy), np.nan)
        ruined = (samples <= -1.0).any(axis=1)
        log_growth = np.log1p(np.where(samples <= -1.0, 0.0, samples)).sum(axis=1)
        cagr = np.where(ruined, -1.0, np.expm1(log_growth / (n / ppy)))
    equity = np.concatenate([np.ones((samples.shape[0], 1)), np.cumprod(1.0 + samples, axis=1)], axis=1)
    mdd = (equity / np.maximum.accumulate(equity, axis=1) - 1.0).min(axis=1)
    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": mdd}


def bootstrap_ci(
    returns: Any,
    periods_per_year: float,
    reps: int = 2000,
    level: float = 0.95,
    mean_block_length: float | None = None,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile intervals for annualized Sharpe, CAGR and max drawdown."""
    r = as_array(returns)
    if r.size < MIN_OBSERVATIONS:
        raise ValueError(f"need at least {MIN_OBSERVATIONS} observations, got {r.size}")
    if not periods_per_year > 0:
        raise ValueError("periods_per_year must be positive")
    if isinstance(reps, bool) or int(reps) != reps or reps < 1:
        raise ValueError("reps must be an integer >= 1")
    if not 0 < level < 1:
        raise ValueError("level must be strictly between 0 and 1")
    length = float(default_block_length(r.size) if mean_block_length is None else mean_block_length)
    if not length >= 1:
        raise ValueError("mean block length must be >= 1")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    idx = stationary_bootstrap_indices(r.size, int(reps), length, rng)
    boot = _stats(r[idx], periods_per_year)
    point = _stats(r[None, :], periods_per_year)
    alpha = (1.0 - level) / 2.0
    intervals: dict[str, tuple[float, float] | None] = {}
    points: dict[str, float | None] = {}
    for name, sample in boot.items():
        p = float(point[name][0])
        points[name] = None if math.isnan(p) else p
        ok = sample[~np.isnan(sample)]
        if ok.size == 0 or ok.size < 0.5 * sample.size:  # mostly undefined: no honest interval
            intervals[name] = None
            continue
        lo, hi = np.percentile(ok, [100 * alpha, 100 * (1 - alpha)])
        intervals[name] = (float(lo), float(hi))
    return BootstrapResult(
        points, intervals, level, int(reps), int(seed), length, float(periods_per_year), samples=boot
    )
