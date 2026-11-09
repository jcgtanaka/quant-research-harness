"""Walk-forward analysis over the development window.

The window is cut into folds on its actual bars. For each fold the best
parameter combination is chosen from the in-sample (S3) returns restricted to
the fold's train span only, then the strategy is run on the fold's test span and
the test returns are stitched into one out-of-sample series.

Spec: ``{train, test, step, anchored}``. ``train``, ``test`` and ``step`` are a
number of bars (integer >= 1) or a fraction of the window's bars (float in
(0, 1)); ``step`` defaults to ``test``; ``anchored`` keeps the train start fixed.
A trailing remainder shorter than ``test`` is not used. ``step`` must be at
least ``test`` so that test spans never overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from qrh.adapters.base import DataView
from qrh.validation.metrics import annualized_sharpe

_KEYS = {"train", "test", "step", "anchored"}


@dataclass(frozen=True)
class WalkForwardSpec:
    train: int | float
    test: int | float
    step: int | float | None = None
    anchored: bool = False


@dataclass(frozen=True)
class Fold:
    """Bar positions in the window; ends are exclusive."""

    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class FoldResult:
    fold: Fold
    combo_hash: str
    params: dict[str, Any]
    train_sharpe: float
    test_returns: pd.Series


@dataclass(frozen=True)
class WalkForwardResult:
    folds: list[FoldResult]
    oos_returns: pd.Series


def _size(name: str, v: Any) -> int | float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"walk_forward.{name}: must be a number of bars or a fraction, got {v!r}")
    if isinstance(v, int):
        if v < 1:
            raise ValueError(f"walk_forward.{name}: bars must be >= 1")
    elif not 0 < v < 1:
        raise ValueError(f"walk_forward.{name}: a float must be a fraction strictly between 0 and 1")
    return v


def parse_walk_forward(raw: Any) -> WalkForwardSpec:
    """Validate a walk-forward spec mapping; raises ValueError with the reason."""
    if not isinstance(raw, dict):
        raise ValueError("walk_forward must be a mapping with train and test")
    unknown = sorted(set(raw) - _KEYS)
    if unknown:
        raise ValueError(f"walk_forward: unknown key(s) {unknown}")
    for key in ("train", "test"):
        if key not in raw:
            raise ValueError(f"walk_forward.{key}: required")
    anchored = raw.get("anchored", False)
    if not isinstance(anchored, bool):
        raise ValueError("walk_forward.anchored: must be true or false")
    step = raw.get("step")
    return WalkForwardSpec(
        train=_size("train", raw["train"]),
        test=_size("test", raw["test"]),
        step=None if step is None else _size("step", step),
        anchored=anchored,
    )


def _bars(v: int | float, n: int) -> int:
    return v if isinstance(v, int) else max(1, round(v * n))


def make_folds(n_bars: int, spec: WalkForwardSpec) -> list[Fold]:
    train, test = _bars(spec.train, n_bars), _bars(spec.test, n_bars)
    step = test if spec.step is None else _bars(spec.step, n_bars)
    if step < test:
        raise ValueError(f"walk_forward.step ({step} bars) must be >= test ({test} bars) so test spans do not overlap")
    folds: list[Fold] = []
    k = 0
    while True:
        train_start = 0 if spec.anchored else k * step
        train_end = train + k * step
        test_end = train_end + test
        if test_end > n_bars:
            break
        folds.append(Fold(k, train_start, train_end, train_end, test_end))
        k += 1
    if not folds:
        raise ValueError(f"window too short: {n_bars} bars cannot fit train {train} + test {test}")
    return folds


def select_best(is_returns: pd.DataFrame, fold: Fold, periods_per_year: float) -> tuple[str, float]:
    """Best column by annualized Sharpe on the fold's train span only (ties: lowest label)."""
    train = is_returns.iloc[fold.train_start : fold.train_end]
    best: tuple[str, float] | None = None
    for label in sorted(is_returns.columns):  # sorted: the first of equal Sharpes wins
        sr = annualized_sharpe(train[label].to_numpy(), periods_per_year)
        if sr is not None and (best is None or sr > best[1]):
            best = (str(label), sr)
    if best is None:
        raise ValueError(f"fold {fold.index}: no strategy has a defined in-sample Sharpe ratio")
    return best


def run_walk_forward(
    adapter: Any,
    view: DataView,
    bars: pd.DatetimeIndex,
    is_returns: pd.DataFrame,
    params_by_label: dict[str, dict[str, Any]],
    spec: WalkForwardSpec,
    periods_per_year: float,
    run_kwargs: dict[str, Any] | None = None,
    folds: list[Fold] | None = None,
) -> WalkForwardResult:
    """Select per fold from ``is_returns`` (index = ``bars``), run the adapter on each test span.

    ``folds`` overrides the folds derived from ``spec`` (used by variants that shape the folds differently).
    """
    if not is_returns.index.equals(bars):
        raise ValueError("in-sample returns do not align with the window bars")
    results: list[FoldResult] = []
    for fold in folds if folds is not None else make_folds(len(bars), spec):
        label, train_sr = select_best(is_returns, fold, periods_per_year)
        expected = bars[fold.test_start : fold.test_end]
        run = adapter.run(params_by_label[label], view, expected[0], expected[-1], **(run_kwargs or {}))
        returns = run.returns
        if not isinstance(returns, pd.Series) or not returns.index.equals(expected):
            raise ValueError(
                f"fold {fold.index}: adapter returns do not cover exactly the {len(expected)} bars of the test span"
            )
        results.append(FoldResult(fold, label, dict(params_by_label[label]), train_sr, returns.astype(float)))
    return WalkForwardResult(results, pd.concat([r.test_returns for r in results]))


# Registry entries for method s4.walkforward=rolling_anchored@1 -------------------------------------
# Entry contract (shared by every s4.walkforward version): entry(adapter, view, bars, is_returns,
# params_by_label, raw_spec, periods_per_year, run_kwargs=None) -> WalkForwardResult, and
# validate(raw_spec, n_bars) raises ValueError when the spec cannot be used on a window of n_bars.


def validate_rolling_anchored(raw: Any, n_bars: int) -> None:
    make_folds(n_bars, parse_walk_forward(raw))


def run_rolling_anchored(
    adapter: Any,
    view: DataView,
    bars: pd.DatetimeIndex,
    is_returns: pd.DataFrame,
    params_by_label: dict[str, dict[str, Any]],
    raw_spec: Any,
    periods_per_year: float,
    run_kwargs: dict[str, Any] | None = None,
) -> WalkForwardResult:
    return run_walk_forward(
        adapter, view, bars, is_returns, params_by_label, parse_walk_forward(raw_spec), periods_per_year, run_kwargs
    )
