"""Probability of Backtest Overfitting by Combinatorially Symmetric Cross-Validation.

Bailey, Borwein, Lopez de Prado and Zhu (2017). The T x N return matrix (one
column per strategy) is cut into S contiguous blocks. For every way of choosing
S/2 blocks as in-sample (the other S/2 are out-of-sample) the best in-sample
strategy n* is found and its relative out-of-sample rank
omega = rank / (N + 1) gives the logit lambda = ln(omega / (1 - omega)).
PBO is the share of combinations with lambda <= 0, i.e. where the in-sample
winner lands in the lower half out-of-sample.

Sharpe ratios are per-period (mean/std, ddof=1). A (sub)sample with no
variation has Sharpe 0 by convention. Ties in the argmax go to the lowest column
index, ties in ranks are averaged. The computation is exact and deterministic.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_BLOCKS = 16
_CHUNK = 1024
_FLAT = 1e-12  # variance below this fraction of the column's mean square counts as zero


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    logits: np.ndarray
    n_combinations: int
    n_blocks: int
    n_strategies: int
    rows_used: int
    rows_dropped: int
    mean_oos_sharpe: float  # per-period, of the in-sample winner, averaged over combinations
    prob_oos_loss: float  # share of combinations where the winner's OOS Sharpe is negative


def _prepare(matrix: Any, n_blocks: Any) -> tuple[np.ndarray, int]:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("returns matrix must be 2-D (rows = periods, columns = strategies)")
    if m.shape[1] < 2:
        raise ValueError("PBO needs at least 2 strategies (columns)")
    if isinstance(n_blocks, bool) or not isinstance(n_blocks, (int, np.integer)) or n_blocks < 2 or n_blocks % 2:
        raise ValueError(f"n_blocks must be an even integer >= 2, got {n_blocks!r}")
    if not np.isfinite(m).all():
        raise ValueError("returns matrix contains NaN or inf")
    s = int(n_blocks)
    if m.shape[0] // s < 2:
        raise ValueError(f"too few rows: {m.shape[0]} rows cannot form {s} blocks of at least 2 rows")
    return m, s


def _sharpe_from_stats(sum_c: np.ndarray, ssq_c: np.ndarray, n: int, centre: np.ndarray, ref2: np.ndarray) -> np.ndarray:
    """Per-column Sharpe from sums of centred values (centre = column mean of all rows)."""
    mean_c = sum_c / n
    var = (ssq_c - n * mean_c**2) / (n - 1)
    flat = var <= _FLAT * ref2
    std = np.sqrt(np.where(flat, 1.0, var))
    return np.where(flat, 0.0, (mean_c + centre) / std)


def pbo_cscv(matrix: Any, n_blocks: int = DEFAULT_BLOCKS) -> PBOResult:
    """Run CSCV on a T x N returns matrix (DataFrame or array). Raises ValueError on bad input."""
    m, s = _prepare(matrix, n_blocks)
    t, n_strat = m.shape
    length = t // s
    dropped = t - length * s
    m = m[dropped:]  # leading remainder rows are dropped

    centre = m.mean(axis=0)
    ref2 = (m**2).mean(axis=0)
    std_all = m.std(axis=0, ddof=1)
    const = (std_all <= 1e-12 * np.abs(centre)) & (centre != 0)
    if const.any():
        raise ValueError(f"column {int(np.argmax(const))} is constant and non-zero; its Sharpe ratio is not defined")
    ref2 = np.where(ref2 > 0, ref2, 1.0)

    centred = (m - centre).reshape(s, length, n_strat)
    block_sum = centred.sum(axis=1)  # (S, N)
    block_ssq = (centred**2).sum(axis=1)
    total_sum, total_ssq = block_sum.sum(axis=0), block_ssq.sum(axis=0)

    half = s // 2
    n_is, n_oos = half * length, (s - half) * length
    combos = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(s), half)), dtype=np.int64)
    combos = combos.reshape(-1, half)

    logits = np.empty(len(combos))
    oos_best = np.empty(len(combos))
    for lo in range(0, len(combos), _CHUNK):
        idx = combos[lo : lo + _CHUNK]
        is_sum, is_ssq = block_sum[idx].sum(axis=1), block_ssq[idx].sum(axis=1)  # (c, N)
        sr_is = _sharpe_from_stats(is_sum, is_ssq, n_is, centre, ref2)
        sr_oos = _sharpe_from_stats(total_sum - is_sum, total_ssq - is_ssq, n_oos, centre, ref2)
        best = np.argmax(sr_is, axis=1)
        rows = np.arange(len(idx))
        chosen = sr_oos[rows, best]
        less = (sr_oos < chosen[:, None]).sum(axis=1)
        equal = (sr_oos == chosen[:, None]).sum(axis=1)  # includes the winner itself
        omega = (less + (equal + 1) / 2.0) / (n_strat + 1)
        logits[lo : lo + len(idx)] = np.log(omega / (1.0 - omega))
        oos_best[lo : lo + len(idx)] = chosen

    return PBOResult(
        pbo=float((logits <= 0).mean()),
        logits=logits,
        n_combinations=math.comb(s, half),
        n_blocks=s,
        n_strategies=n_strat,
        rows_used=length * s,
        rows_dropped=dropped,
        mean_oos_sharpe=float(oos_best.mean()),
        prob_oos_loss=float((oos_best < 0).mean()),
    )
