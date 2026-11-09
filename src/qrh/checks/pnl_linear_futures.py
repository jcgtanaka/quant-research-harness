"""Futures first slice (F1) of the pnl.linear family: one instrument, whole contracts, daily bars, opt-in.

Opt-in by ``reconstruction.asset_class: futures``. The limits (absolute cap, share, drift, slope) and the staleness
rules are those of ``evaluate_pnl_linear``; this module only builds the series it needs in currency:

    explained gross [t] = contracts[t-lag] x (price[t] - price[t-1]) x multiplier
    explained cost  [t] = cost_per_contract x (|contracts traded at t| + 2 x |contracts[t-lag]| on a roll bar)
    explained net   [t] = gross - cost
    reported        [t] = reported return x declared starting capital
    remainder       [t] = reported - explained net

``pct_change`` is not used. The capital base for the caps is the declared starting capital, so caps stay in bp of it.
A roll counts as two trades (close the old contract, open the new one) for costs.

Series types: ``back_adjusted`` / ``forward_adjusted`` price differences are used as they are. For an unadjusted series
(``adjustment: none``) a ``roll_rule.method`` (calendar, volume, open_interest) and a ``contract`` column are required.
On a roll bar the single continuous column cannot give the held contract's price change, so the optional column
``roll_price_change`` (expiring contract's close minus the previous bar's close) is used when it is present; without it
the roll bars are left out of the comparison and counted in ``details['roll_bars_excluded']``.

BLOCKED (never a fill-forward, never a default): missing or invalid multiplier, starting capital, cost per contract,
currency; contract currency different from account currency; unreadable roll method or missing contract column for an
unadjusted series; fractional contracts; a held bar past its contract's ``expiry`` (optional column) without a roll.

Margin is a sub-check reported in ``details['margin']``: missing ``margin_per_contract`` gives a WARNING on that
sub-check only. Margin is never profit or loss and never changes the PnL verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from qrh.adapters.base import RunResult
from qrh.checks.pnl_linear import PnlLinearVerdict, evaluate_pnl_linear
from qrh.checks.pnl_linear_config import (
    CONTRACT_COLUMN,
    FUTURES,
    ReconstructionConfigError,
    check_frame_columns,
    resolve_broker_cost,
    validate_reconstruction,
)
from qrh.checks.pnl_linear_run import ALPHA, MIN_BARS, _newey_west_lag
from qrh.checks.pnl_reconstruction import _align_positions, validate_execution

EXPIRY_COLUMN = "expiry"
ROLL_CHANGE_COLUMN = "roll_price_change"
_INTEGER_TOL = 1e-9


def _blocked(reason: str) -> PnlLinearVerdict:
    return PnlLinearVerdict("BLOCKED", [f"blocked: {reason}"], {})


def _margin(held: pd.Series, per_contract: float | None, capital: float) -> tuple[dict[str, Any], str | None]:
    if per_contract is None:
        return ({"status": "WARNING", "reason": "margin_per_contract not declared"},
                "warning: margin sub-check not run, margin_per_contract is not declared "
                "(the PnL verdict is unaffected)")
    peak_contracts = float(held.abs().max()) if len(held) else 0.0
    peak = peak_contracts * per_contract
    d = {"status": "OK", "peak_contracts": peak_contracts, "peak_initial_margin": peak, "starting_capital": capital}
    if peak > capital:
        d["status"] = "WARNING"
        return d, (f"warning: peak initial margin {peak:.2f} exceeds the declared starting capital {capital:.2f} "
                   "(the PnL verdict is unaffected)")
    return d, None


def check_pnl_linear_futures(
    result: RunResult,
    frame: pd.DataFrame,
    execution: dict[str, Any] | None,
    reconstruction: Any,
    periods_per_year: float | None,
    commission_records: Sequence[Any] | None = None,
    as_of: date | None = None,
) -> PnlLinearVerdict:
    try:
        config = validate_reconstruction(reconstruction)
        if config.asset_class != FUTURES:
            return _blocked(f"configuration rejected: reconstruction.asset_class must be {FUTURES!r} for this branch")
        check_frame_columns(config, set(frame.columns))
    except ReconstructionConfigError as exc:
        return _blocked(f"configuration rejected: {exc}")
    cost_source: dict[str, Any] = {}
    if config.cost_per_contract is None:  # validation guarantees broker and product are declared in that case
        config.cost_per_contract, cost_source = resolve_broker_cost(config, commission_records, as_of)
        if config.cost_per_contract is None:
            return PnlLinearVerdict("BLOCKED", [f"blocked: {cost_source['reason']}"], {"cost": cost_source})
    if not execution:
        return _blocked("execution is required (execution: {lag_bars: >= 1, price_column: ...}); "
                        "lag_bars has no default")
    try:
        validate_execution(execution)
    except ValueError as exc:
        return _blocked(f"execution rejected: {exc}")
    (col,) = config.adjustment
    exec_cols = [execution["price_column"]] if "price_column" in execution else list(execution["price_columns"])
    if exec_cols != [col]:
        return _blocked(f"execution price column(s) {exec_cols} differ from reconstruction.adjustment column [{col!r}]")
    if periods_per_year is None or not periods_per_year > 0:
        return _blocked("periods_per_year is unknown; set periods_per_year in the hypothesis")
    if result.positions is None or not isinstance(result.positions, pd.DataFrame) or result.positions.empty:
        return _blocked("the run reported no contracts held to reconstruct PnL from")
    pos = _align_positions(result.positions.astype(float), [col])
    if pos is None:
        return _blocked(f"cannot align positions columns {sorted(map(str, result.positions.columns))} to [{col!r}]")
    if not np.all(np.isfinite(pos.to_numpy())) or (np.abs(pos - pos.round()) > _INTEGER_TOL).any().any():
        return _blocked("contracts held must be whole numbers (fixed integer contracts)")

    lag = int(execution["lag_bars"])
    mult, cpc, capital = config.multiplier, config.cost_per_contract, config.starting_capital
    rep_ret = result.returns.astype(float)
    idx = rep_ret.index.intersection(pos.index).intersection(frame.index)
    pos = pos.reindex(idx).fillna(0.0)[col].round()
    held = pos.shift(lag).fillna(0.0)
    price = frame[col].astype(float).reindex(idx)
    diff = price.diff()

    roll_bar = pd.Series(False, index=idx)
    if CONTRACT_COLUMN in frame.columns:
        ct = frame[CONTRACT_COLUMN].reindex(idx)
        roll_bar = (ct != ct.shift(1)) & ct.shift(1).notna()
    unadjusted = config.adjustment[col] == "none"
    excluded = pd.Series(False, index=idx)
    if unadjusted and roll_bar.any():
        if ROLL_CHANGE_COLUMN in frame.columns:
            rc = frame[ROLL_CHANGE_COLUMN].astype(float).reindex(idx)
            diff = diff.where(~roll_bar, rc)
        else:
            excluded = roll_bar.copy()

    if EXPIRY_COLUMN in frame.columns:
        expiry = pd.to_datetime(frame[EXPIRY_COLUMN].reindex(idx), errors="coerce")
        bar_times = pd.Series(pd.DatetimeIndex(idx), index=idx)
        late = (held != 0) & expiry.notna() & (bar_times > expiry)
        if late.any():
            first = idx[late.to_numpy()][0]
            return _blocked(f"contract expired without a roll: {int(late.sum())} held bar(s), first {first}; "
                            "a stale price is never filled forward")

    trades = pos.diff().abs().fillna(pos.abs()).shift(lag).fillna(0.0)
    contracts_traded = trades + 2.0 * held.abs() * roll_bar.astype(float)
    gross = held * diff * mult
    explained_all = gross - cpc * contracts_traded
    reported_all = rep_ret.reindex(idx) * capital
    keep = (diff.notna() & reported_all.notna() & ~excluded).to_numpy()
    if int(keep.sum()) < MIN_BARS:
        return _blocked(f"only {int(keep.sum())} usable bars; need at least {MIN_BARS}")

    explained, reported, bars = explained_all[keep], reported_all[keep], idx[keep]
    pub = pd.to_datetime(frame[config.publication_column].reindex(idx)[keep], utc=True, errors="coerce")
    tz = getattr(bars, "tz", None)
    pub = pub.dt.tz_convert(tz) if tz is not None else pub.dt.tz_localize(None)
    halt = None
    if config.halt_flag_column is not None:
        halt = frame[config.halt_flag_column].reindex(idx)[keep].fillna(False).astype(bool).to_numpy()
    n = int(keep.sum())
    settings = {"bars_per_year": float(periods_per_year), "capital_base": capital,
                "newey_west_lag": _newey_west_lag(n), "alpha": ALPHA, "lag_bars": lag,
                "multiplier": mult, "cost_per_contract": cpc}
    verdict = evaluate_pnl_linear(
        reported_pnl=reported.to_numpy(),
        explained_net_pnl=explained.to_numpy(),
        remainder=(reported - explained).to_numpy(),
        frame_columns=set(frame.columns),
        config_block=reconstruction,
        bars_per_year=settings["bars_per_year"],
        capital_base=capital,
        newey_west_lag=settings["newey_west_lag"],
        alpha=ALPHA,
        staleness_inputs={
            "bar_times": bars,
            "publication_times": pd.DatetimeIndex(pub),
            "position": held.abs()[keep].reset_index(drop=True),
            "lag": lag,
            "halt_flag": halt,
        },
    )
    verdict.details["settings"] = settings
    verdict.details["cost"] = {"status": "OK", **cost_source, "cost_per_contract": cpc}
    if cost_source.get("margin_flagged"):
        verdict.reasons.append("warning: broker extras were not checked; a 25% margin was added to the commission "
                               "(not yet confirmed)")
    verdict.reasons.extend(f"warning: {w}" for w in cost_source.get("warnings", []))
    verdict.details["roll_bars"] = int(roll_bar.sum())
    verdict.details["roll_bars_excluded"] = int(excluded.sum())
    margin, warning = _margin(held, config.margin_per_contract, capital)
    verdict.details["margin"] = margin
    if warning is not None:
        verdict.reasons.append(warning)
    return verdict
