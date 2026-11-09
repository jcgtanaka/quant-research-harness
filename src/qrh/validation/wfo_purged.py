"""Walk-forward with a purge gap and an embargo (``s4.walkforward=purged_embargo@1``, experimental).

Same fold layout and selection rule as the default walk-forward, plus two gaps between the train and
test windows. Lopez de Prado (2018, Advances in Financial Machine Learning, ch. 7) purges training
observations whose labels overlap the test period and embargoes observations right after it. A forward
walk-forward has no training data after the test span, so the two ideas are adapted as follows:

- ``purge``: the last ``purge`` bars of every train span are not used to select the parameters
  (their returns may still carry information about the first test bars);
- ``embargo``: the first ``embargo`` bars of every test span are neither run nor scored.

``purge`` and ``embargo`` are a number of bars (integer >= 0) or a fraction of the window's bars (float
strictly between 0 and 1); the defaults are ``DEFAULT_PURGE`` and ``DEFAULT_EMBARGO`` bars. All other keys
are those of ``qrh.validation.wfo``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from qrh.adapters.base import DataView
from qrh.validation.wfo import (
    Fold,
    WalkForwardResult,
    WalkForwardSpec,
    _bars,
    make_folds,
    parse_walk_forward,
    run_walk_forward,
)

DEFAULT_PURGE = 5
DEFAULT_EMBARGO = 5


@dataclass(frozen=True)
class PurgedSpec:
    base: WalkForwardSpec
    purge: int | float
    embargo: int | float


def _gap(name: str, v: Any) -> int | float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"walk_forward.{name}: must be a number of bars or a fraction, got {v!r}")
    if isinstance(v, int):
        if v < 0:
            raise ValueError(f"walk_forward.{name}: bars must be >= 0")
    elif not 0 < v < 1:
        raise ValueError(f"walk_forward.{name}: a float must be a fraction strictly between 0 and 1")
    return v


def parse_purged(raw: Any) -> PurgedSpec:
    """Validate a purged walk-forward spec mapping; raises ValueError with the reason."""
    if not isinstance(raw, dict):
        raise ValueError("walk_forward must be a mapping with train and test")
    rest = {k: v for k, v in raw.items() if k not in ("purge", "embargo")}
    return PurgedSpec(
        base=parse_walk_forward(rest),
        purge=_gap("purge", raw.get("purge", DEFAULT_PURGE)),
        embargo=_gap("embargo", raw.get("embargo", DEFAULT_EMBARGO)),
    )


def make_folds_purged(n_bars: int, spec: PurgedSpec) -> list[Fold]:
    purge = 0 if spec.purge == 0 else _bars(spec.purge, n_bars)
    embargo = 0 if spec.embargo == 0 else _bars(spec.embargo, n_bars)
    out: list[Fold] = []
    for f in make_folds(n_bars, spec.base):
        train_end, test_start = f.train_end - purge, f.test_start + embargo
        if train_end - f.train_start < 2:
            raise ValueError(f"walk_forward.purge ({purge} bars) leaves fewer than 2 train bars in fold {f.index}")
        if f.test_end - test_start < 2:
            raise ValueError(f"walk_forward.embargo ({embargo} bars) leaves fewer than 2 test bars in fold {f.index}")
        out.append(Fold(f.index, f.train_start, train_end, test_start, f.test_end))
    return out


def validate_purged_embargo(raw: Any, n_bars: int) -> None:
    make_folds_purged(n_bars, parse_purged(raw))


def run_purged_embargo(
    adapter: Any,
    view: DataView,
    bars: pd.DatetimeIndex,
    is_returns: pd.DataFrame,
    params_by_label: dict[str, dict[str, Any]],
    raw_spec: Any,
    periods_per_year: float,
    run_kwargs: dict[str, Any] | None = None,
) -> WalkForwardResult:
    spec = parse_purged(raw_spec)
    return run_walk_forward(
        adapter, view, bars, is_returns, params_by_label, spec.base, periods_per_year, run_kwargs,
        folds=make_folds_purged(len(bars), spec),
    )
