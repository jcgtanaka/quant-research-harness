"""The S5 computation shared by the pipeline stage and the benchmarks: PBO and DSR on a returns matrix.

The method callables (``s5.pbo``, ``s5.dsr``) are passed in, so a stage and a benchmark that select the
same versions get the same numbers. Raises ValueError (with a plain reason) when the inputs cannot support the statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from qrh.validation.deflated_sharpe import sample_moments
from qrh.validation.metrics import sharpe


PBO_NOT_APPLICABLE = "not applicable (one combination, no selection among variants)"


@dataclass(frozen=True)
class OverfitResult:
    pbo: Any  # PBOResult of the s5.pbo method; None when there is one combination (nothing was selected)
    dsr: Any  # DSRResult of the s5.dsr method
    var_sharpe: float  # the V the deflated Sharpe used: max(var_sharpe_grid, var_sharpe_estimator)
    n_trials: int
    pbo_note: str = ""  # why PBO is missing (PBO_NOT_APPLICABLE), else empty
    var_sharpe_source: str = "grid"  # which of the two was larger and so used: "grid" or "estimator"
    var_sharpe_grid: float | None = None  # variance of the in-sample Sharpe ratios across the combinations; None for one
    var_sharpe_estimator: float = 0.0  # sampling variance of the selected variant's out-of-sample Sharpe estimator


def estimator_sharpe_variance(returns: Any) -> float:
    """Sampling variance of the per-period Sharpe ratio of one return series (Mertens 2002, as used in the PSR).

    var(SR) = (1 - skew * SR + (kurtosis - 1) / 4 * SR^2) / (T - 1), with the sample skewness, non-excess kurtosis and
    Sharpe ratio of the series itself. It is the floor of V in the deflated Sharpe for every grid size, and V itself
    when only one combination was run, so the number of trials charged always raises the bar for the winner."""
    r = np.asarray(returns, dtype=float)
    sr = sharpe(r)
    if sr is None or r.size < 3:
        raise ValueError("the out-of-sample Sharpe ratio is undefined, so its sampling variance cannot be estimated")
    skew, kurt = sample_moments(r)
    var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (r.size - 1)
    if not var > 0:
        raise ValueError("the sampling variance of the Sharpe ratio is not positive; the moments are inconsistent")
    return float(var)


def overfitting_statistics(
    matrix: pd.DataFrame,
    oos_returns: Any,
    family_count: int,
    n_blocks: int,
    pbo_fn: Callable[..., Any],
    dsr_fn: Callable[..., Any],
) -> OverfitResult:
    """PBO of the in-sample matrix and the DSR of the out-of-sample series.

    ``n_trials`` is at least the grid size (the matrix width). V, the variance of the Sharpe ratios of the null
    trials, is ``max(grid, estimator)`` for every grid size: ``grid`` is the variance of the in-sample per-period
    Sharpe ratios across the combinations, ``estimator`` the sampling variance of the selected variant's
    out-of-sample Sharpe estimator (``estimator_sharpe_variance``). The floor stops near-duplicate variants (the same
    rule with a hair more cost) from collapsing the cross-variant variance and so the luck bar. Both values and the
    one used are returned. With exactly one combination nothing was selected, so PBO is not applicable (``pbo`` is
    None, ``pbo_note`` says so) and there is no grid variance; V is the estimator variance.
    """
    if matrix.shape[1] < 1:
        raise ValueError("the in-sample returns have no combination: at least one combination is needed")
    n_trials = max(int(family_count), int(matrix.shape[1]))
    oos = np.asarray(oos_returns, dtype=float)
    estimator = estimator_sharpe_variance(oos)
    if matrix.shape[1] == 1:
        dsr = dsr_fn(oos, n_trials=n_trials, var_sharpe=estimator)
        return OverfitResult(None, dsr, estimator, n_trials, PBO_NOT_APPLICABLE, "estimator", None, estimator)
    pbo = pbo_fn(matrix, n_blocks=n_blocks)
    per_period = [s for s in (sharpe(matrix[c].to_numpy()) for c in matrix.columns) if s is not None]
    if len(per_period) < 2:
        raise ValueError("fewer than two strategies have a defined in-sample Sharpe ratio")
    grid = float(np.var(per_period, ddof=1))
    var_sharpe, source = (grid, "grid") if grid >= estimator else (estimator, "estimator")
    dsr = dsr_fn(oos, n_trials=n_trials, var_sharpe=var_sharpe)
    return OverfitResult(pbo, dsr, var_sharpe, n_trials, "", source, grid, estimator)
