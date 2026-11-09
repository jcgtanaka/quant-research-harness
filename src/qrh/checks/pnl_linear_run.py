"""Runs the pnl.linear verdict on a strategy run (``s2.pnl_linear=pnl_linear@1``, experimental, opt-in).

``evaluate_pnl_linear`` is a pure verdict over series; this module builds those series from what S2 has: the adapter's
reported returns and positions, the dataset frame, the hypothesis ``execution`` block (lag) and its ``reconstruction``
block (settings). It runs only where a hypothesis declares ``reconstruction``.

Scope of this first integration, each limit answered with BLOCKED and a reason, never with a silent approximation:

- positions must be weights (``position_units: weight_start_of_bar``); a quantity needs a capital base the hypothesis
  does not declare, except ``asset_class: futures`` with ``position_units: quantity``, which is handed to
  ``pnl_linear_futures.check_pnl_linear_futures``. PnL is then a fraction of NAV, so the capital base is 1.0 and caps in bp are bp of NAV;
- the lag applied to positions is ``execution.lag_bars``, as in ``s2.pnl_reconstruction`` (a position decided at bar t
  earns from bar t + lag);
- the explained cost is ``max(min_cost_per_turnover, cost_series_column)`` per unit of lagged turnover; the free-form
  ``cost_model`` mapping is validated but not interpreted;
- income: ``dividend`` entries are added to the explained return (one price column only); other kinds are not yet
  supported;
- the Newey-West lag is the usual ``floor(4 (n/100)^(2/9))`` and alpha is 0.05; both are recorded in the receipt.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from qrh.adapters.base import RunResult
from qrh.checks.pnl_linear import PnlLinearVerdict, evaluate_pnl_linear
from qrh.checks.pnl_linear_config import (
    HARNESS_ADJUSTMENT,
    ReconstructionConfigError,
    check_frame_columns,
    fill_built_adjustment,
    validate_reconstruction,
)
from qrh.checks.pnl_reconstruction import _align_positions, validate_execution

CAPITAL_BASE = 1.0
ALPHA = 0.05
MIN_BARS = 20


def plain(obj: Any) -> Any:
    """JSON-safe copy: dataclasses and numpy values become builtins, non-finite numbers become None."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return plain(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    return obj


def adjustment_record(reconstruction: Any, built: bool, answer: str | None) -> tuple[dict[str, str], list[str]]:
    """What the receipt records about the price adjustment, and the warnings it causes.

    Built backtest: the harness knows its data, so the type is its own (or the declared one) and nothing is asked.
    Brought backtest: the type is the user's Track 2 answer; a missing or unknown answer is a warning."""
    if built:
        declared = reconstruction.get("adjustment") if isinstance(reconstruction, dict) else None
        types = sorted({str(t) for t in declared.values()}) if isinstance(declared, dict) and declared else []
        return {"type": types[0] if len(types) == 1 else (",".join(types) or HARNESS_ADJUSTMENT),
                "source": "harness"}, []
    if answer is None or answer == "unknown":
        return ({"type": "unknown", "source": "unanswered"},
                ["warning: price adjustment is unknown (Track 2 question price_adjustment was not answered)"])
    return {"type": answer, "source": "track2_answer"}, []


def _blocked(reason: str) -> PnlLinearVerdict:
    return PnlLinearVerdict("BLOCKED", [f"blocked: {reason}"], {})


def _newey_west_lag(n: int) -> int:
    return max(0, min(int(4 * (n / 100.0) ** (2.0 / 9.0)), n - 1))


def _is_futures_quantity(reconstruction: Any) -> bool:
    """The futures branch is selected by the declaration alone: asset_class futures with position_units quantity."""
    return (isinstance(reconstruction, dict) and reconstruction.get("asset_class") == "futures"
            and reconstruction.get("position_units") == "quantity")


def _is_option_quantity(reconstruction: Any) -> bool:
    """The options branch is selected by the declaration alone: asset_class option with position_units quantity."""
    return (isinstance(reconstruction, dict) and reconstruction.get("asset_class") == "option"
            and reconstruction.get("position_units") == "quantity")


def check_pnl_linear(
    result: RunResult,
    frame: pd.DataFrame,
    execution: dict[str, Any] | None,
    reconstruction: Any,
    periods_per_year: float | None,
    commission_records: Sequence[Any] | None = None,
    as_of: date | None = None,
    built: bool = True,
) -> PnlLinearVerdict:
    """``built`` is True for a backtest the harness built (the default) and False for one the user brought. For a
    built one the adjustment need not be declared and any unexplained difference is a bug warning (see
    ``evaluate_pnl_linear``)."""
    # forwarded only when given, so a call without them is identical to the pre-wiring call
    extra: dict[str, Any] = {}
    if commission_records is not None:
        extra["commission_records"] = commission_records
    if as_of is not None:
        extra["as_of"] = as_of
    if _is_option_quantity(reconstruction):
        from qrh.checks import pnl_linear_options  # imported here: that module imports this one

        return pnl_linear_options.check_pnl_linear_options(
            result, frame, execution, reconstruction, periods_per_year, **extra)
    if _is_futures_quantity(reconstruction):
        from qrh.checks import pnl_linear_futures  # imported here: that module imports this one

        return pnl_linear_futures.check_pnl_linear_futures(
            result, frame, execution, reconstruction, periods_per_year, **extra)
    if built and isinstance(execution, dict):
        exec_cols = [execution["price_column"]] if "price_column" in execution else list(execution.get("price_columns") or [])
        reconstruction = fill_built_adjustment(reconstruction, exec_cols)
    try:
        config = validate_reconstruction(reconstruction)
        check_frame_columns(config, set(frame.columns))
    except ReconstructionConfigError as exc:
        return _blocked(f"configuration rejected: {exc}")
    if not execution:
        return _blocked("execution is required (execution: {lag_bars: >= 1, price_column: ...}); "
                        "lag_bars has no default")
    try:
        validate_execution(execution)
    except ValueError as exc:
        return _blocked(f"execution rejected: {exc}")
    if config.position_units != "weight_start_of_bar":
        return _blocked("position_units 'quantity' needs a capital base that is not declared; "
                        "use weight_start_of_bar (weights as a fraction of NAV)")
    unsupported = sorted({e["kind"] for e in config.income if e["kind"] != "dividend"})
    if unsupported:
        return _blocked(f"income kind(s) {unsupported} are not supported yet in S2; unexplained income would "
                        "show up as a hidden gain")
    cols = list(config.adjustment)
    exec_cols = [execution["price_column"]] if "price_column" in execution else list(execution["price_columns"])
    if sorted(exec_cols) != sorted(cols):
        return _blocked(f"execution price column(s) {sorted(exec_cols)} differ from reconstruction.adjustment "
                        f"column(s) {sorted(cols)}")
    dividends = [e["column"] for e in config.income if e["kind"] == "dividend"]
    if dividends and len(cols) != 1:
        return _blocked("dividend income with more than one price column cannot be assigned to a column")
    if periods_per_year is None or not periods_per_year > 0:
        return _blocked("periods_per_year is unknown; set periods_per_year in the hypothesis")
    if result.positions is None or not isinstance(result.positions, pd.DataFrame) or result.positions.empty:
        return _blocked("the run reported no positions to reconstruct PnL from")
    pos = _align_positions(result.positions.astype(float), cols)
    if pos is None:
        return _blocked(f"cannot align positions columns {sorted(map(str, result.positions.columns))} "
                        f"to price columns {cols}")

    lag = int(execution["lag_bars"])
    rep = result.returns.astype(float)
    idx = rep.index.intersection(pos.index).intersection(frame.index)
    px = frame[cols].astype(float)
    rets = px.pct_change()
    for col in dividends:
        rets = rets + (frame[col].astype(float).fillna(0.0).to_frame(cols[0]) / px.shift(1))
    rets = rets.reindex(idx)
    pos, rep = pos.reindex(idx).fillna(0.0), rep.reindex(idx)
    keep = (rets.notna().all(axis=1) & rep.notna()).to_numpy()
    if int(keep.sum()) < MIN_BARS:
        return _blocked(f"only {int(keep.sum())} usable bars; need at least {MIN_BARS}")

    rate = pd.Series(config.min_cost_per_turnover, index=idx, dtype=float)
    if config.cost_series_column is not None:
        series = frame[config.cost_series_column].astype(float).reindex(idx)
        rate = pd.concat([rate, series], axis=1).max(axis=1)
    turnover = pos.diff().abs().sum(axis=1).fillna(pos.abs().sum(axis=1)).shift(lag).fillna(0.0)
    gross = (pos.shift(lag).fillna(0.0) * rets.fillna(0.0)).sum(axis=1)
    explained = (gross - rate * turnover)[keep]
    reported = rep[keep]
    bars = idx[keep]

    pub = pd.to_datetime(frame[config.publication_column].reindex(idx)[keep], utc=True, errors="coerce")
    tz = getattr(bars, "tz", None)
    pub = pub.dt.tz_convert(tz) if tz is not None else pub.dt.tz_localize(None)
    halt = None
    if config.halt_flag_column is not None:
        halt = frame[config.halt_flag_column].reindex(idx)[keep].fillna(False).astype(bool).to_numpy()
    n = int(keep.sum())
    settings = {"bars_per_year": float(periods_per_year), "capital_base": CAPITAL_BASE,
                "newey_west_lag": _newey_west_lag(n), "alpha": ALPHA, "lag_bars": lag}
    verdict = evaluate_pnl_linear(
        reported_pnl=reported.to_numpy(),
        explained_net_pnl=explained.to_numpy(),
        remainder=(reported - explained).to_numpy(),
        frame_columns=set(frame.columns),
        config_block=reconstruction,
        bars_per_year=settings["bars_per_year"],
        capital_base=CAPITAL_BASE,
        newey_west_lag=settings["newey_west_lag"],
        alpha=ALPHA,
        staleness_inputs={
            "bar_times": bars,
            "publication_times": pd.DatetimeIndex(pub),
            "position": pos.abs().sum(axis=1)[keep].reset_index(drop=True),
            "lag": lag,
            "halt_flag": halt,
        },
        built=built,
    )
    verdict.details["settings"] = settings
    return verdict
