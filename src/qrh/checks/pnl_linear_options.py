"""Options first slice (O1) of the pnl.linear family: one option contract, whole contracts, daily bars, opt-in.

Opt-in by ``reconstruction.asset_class: option`` with ``position_units: quantity``. The limits (absolute cap, share,
drift, slope) and the staleness rules are those of ``evaluate_pnl_linear``; this module only builds the series it
needs in currency:

    explained gross [t] = contracts[t-lag] x (option price[t] - option price[t-1]) x multiplier
    explained cost  [t] = cost_per_contract x |contracts traded at t|
    explained net   [t] = gross - cost
    reported        [t] = reported return x declared starting capital
    remainder       [t] = reported - explained net

``pct_change`` is not used. The capital base for the caps is the declared starting capital.

Decisions (docs/pnl-linear.md, section 9, "Options, first slice (O1)"):

- Price type: declared by the user (mid, settlement or last); ``last`` needs a trade timestamp column, and the
  freshness of a last price is judged on that trade time.
- Series: a full chain is accepted. Rows are selected by the declared strike, expiry and right (the dataset columns
  ``strike``, ``expiry``, ``right``, when present); anything but one row per bar is BLOCKED.
- Expiry: on the expiry bar a held contract is closed at intrinsic value and reported as cash-settled (the intrinsic
  value, not the mark, is the price used). The mark on that bar must match intrinsic value within one currency
  rounding unit, otherwise FAIL. A position held after the expiry bar is BLOCKED; a price is never filled forward.
- Stale held price: a WARNING, not a block. The raw finding stays in ``details['staleness']``; the verdict is marked
  ``details['conditional'] = True``. A future-dated mark still FAILs.
- Cost per contract: declared, never defaulted. Missing is BLOCKED on the cost sub-check (``details['cost']``).
- Point-in-time: ``yes`` or ``unknown``. An unknown (or absent) answer is a warning and an entry in
  ``details['unknown_answers']``, never a block.

BLOCKED (never a default): any missing or invalid declared value, contract currency different from account currency,
fractional contracts, ambiguous chain rows, a held bar past expiry, a missing underlying price when a position is held
at expiry.
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
    OPTION,
    ReconstructionConfig,
    ReconstructionConfigError,
    check_frame_columns,
    has_broker_cost_source,
    infer_option_price_type,
    infer_rounding_unit,
    resolve_broker_cost,
    validate_reconstruction,
)
from qrh.checks.pnl_linear_run import ALPHA, MIN_BARS, _newey_west_lag
from qrh.checks.pnl_reconstruction import _align_positions, validate_execution
from qrh.checks.pnl_staleness import check_mark_freshness

STRIKE_COLUMN = "strike"
EXPIRY_COLUMN = "expiry"
RIGHT_COLUMN = "right"
SETTLEMENT = "cash_settled_at_intrinsic"
_INTEGER_TOL = 1e-9
_RIGHTS = {"call": "call", "c": "call", "put": "put", "p": "put"}


def _blocked(reason: str, details: dict[str, Any] | None = None) -> PnlLinearVerdict:
    return PnlLinearVerdict("BLOCKED", [f"blocked: {reason}"], details or {})


def _select_contract(frame: pd.DataFrame, config: ReconstructionConfig) -> tuple[pd.DataFrame | None, str | None]:
    """The rows of the declared contract, one per bar, or (None, reason). Only dataset columns that exist select."""
    keep = np.ones(len(frame), dtype=bool)
    used: list[str] = []
    if STRIKE_COLUMN in frame.columns:
        strike = pd.to_numeric(frame[STRIKE_COLUMN], errors="coerce").to_numpy(dtype=float)
        keep &= np.isclose(strike, config.strike, rtol=0.0, atol=1e-9)
        used.append(STRIKE_COLUMN)
    if EXPIRY_COLUMN in frame.columns:
        expiry = pd.to_datetime(frame[EXPIRY_COLUMN], errors="coerce").dt.normalize()
        keep &= (expiry == pd.Timestamp(config.expiry)).to_numpy()
        used.append(EXPIRY_COLUMN)
    if RIGHT_COLUMN in frame.columns:
        right = frame[RIGHT_COLUMN].astype(str).str.strip().str.lower().map(_RIGHTS)
        keep &= (right == config.right).to_numpy()
        used.append(RIGHT_COLUMN)
    if used and not keep.any():
        return None, (f"chain input: no row matches the declared strike {config.strike}, expiry {config.expiry} "
                      f"and right {config.right!r} (dataset columns used: {used})")
    selected = frame[keep]
    dup = selected.index.duplicated(keep=False)
    if dup.any():
        bars = selected.index[dup]
        return None, (f"chain input: {bars.nunique()} bar(s) have more than one row after selecting by strike, "
                      f"expiry and right (first bar {bars[0]}; dataset columns used: {used or 'none'}); declare the "
                      f"dataset columns {STRIKE_COLUMN!r}, {EXPIRY_COLUMN!r} and {RIGHT_COLUMN!r} so that exactly "
                      "one contract row remains per bar")
    return selected, None


def check_pnl_linear_options(
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
        if config.asset_class != OPTION:
            return _blocked(f"configuration rejected: reconstruction.asset_class must be {OPTION!r} for this branch")
        check_frame_columns(config, set(frame.columns))
    except ReconstructionConfigError as exc:
        return _blocked(f"configuration rejected: {exc}")
    inferred: dict[str, Any] = {}
    unknown: list[str] = []
    if config.option_price_type is None:
        config.option_price_type = infer_option_price_type(config.adjustment and next(iter(config.adjustment)),
                                                           set(frame.columns))
        if config.option_price_type == "last" and config.trade_time_column is None:
            config.option_price_type = None  # a last price needs a trade timestamp column; ask instead
        if config.option_price_type is None:
            unknown.append("option_price_type")
        else:
            inferred["option_price_type"] = config.option_price_type
    if config.currency_rounding_unit is None:
        config.currency_rounding_unit = infer_rounding_unit(config.account_currency)
        if config.currency_rounding_unit is None:
            unknown.append("currency_rounding_unit")
        else:
            inferred["currency_rounding_unit"] = config.currency_rounding_unit
    cost_source: dict[str, Any] = {}
    if config.cost_per_contract is None and has_broker_cost_source(config):
        config.cost_per_contract, cost_source = resolve_broker_cost(config, commission_records, as_of)
        if config.cost_per_contract is None:
            return _blocked(cost_source["reason"], {"cost": cost_source})
    if config.cost_per_contract is None:
        reason = ("cost sub-check: reconstruction.cost_per_contract is not declared; declare it in account currency, "
                  "or choose a broker so its commissions can be looked up (the lookup is outside this check); "
                  "the cost is never defaulted")
        return _blocked(reason, {"cost": {"status": "BLOCKED", "reason": reason}})
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
    chain, why = _select_contract(frame, config)
    if chain is None:
        return _blocked(why or "contract selection failed")

    lag = int(execution["lag_bars"])
    mult, cpc, capital = config.multiplier, config.cost_per_contract, config.starting_capital
    rep_ret = result.returns.astype(float)
    idx = rep_ret.index.intersection(pos.index).intersection(chain.index)
    pos = pos.reindex(idx).fillna(0.0)[col].round()
    held = pos.shift(lag).fillna(0.0)
    price = chain[col].astype(float).reindex(idx).to_numpy().copy()

    # Expiry: close at intrinsic value, cash-settled; never fill forward.
    bar_days = pd.DatetimeIndex(idx)
    if bar_days.tz is not None:
        bar_days = bar_days.tz_localize(None)
    bar_days = bar_days.normalize()
    expiry = pd.Timestamp(config.expiry)
    is_held = (held != 0).to_numpy()
    late = is_held & (bar_days > expiry)
    if late.any():
        first = idx[late][0]
        return _blocked(f"contract expired on {config.expiry} and is still held: {int(late.sum())} held bar(s), "
                        f"first {first}; the position is closed at intrinsic value at expiry and a stale price is "
                        "never filled forward")
    on_expiry = (bar_days == expiry) & is_held
    expiry_info: dict[str, Any] = {
        "date": config.expiry, "settlement": SETTLEMENT, "held_contracts": 0, "mismatch": False,
        "rounding_unit": config.currency_rounding_unit,
    }
    unit_limit = config.currency_rounding_unit or 0.0  # unknown unit: strict match, and a warning below
    expiry_fail: str | None = None
    if on_expiry.any():
        if config.underlying_column is None:
            return _blocked("underlying_column is required to close a held position at intrinsic value on the "
                            f"expiry bar ({config.expiry}); declare reconstruction.underlying_column")
        und = chain[config.underlying_column].astype(float).reindex(idx).to_numpy()
        if not np.all(np.isfinite(und[on_expiry])):
            return _blocked(f"underlying price missing on the expiry bar ({config.expiry}) in column "
                            f"{config.underlying_column!r}; the intrinsic value cannot be computed")
        sign = 1.0 if config.right == "call" else -1.0
        intrinsic = np.maximum(sign * (und - config.strike), 0.0)
        mark = price[on_expiry]
        gap = np.abs(mark - intrinsic[on_expiry])
        limit = unit_limit * (1.0 + 1e-9) + 1e-12
        bad = np.isfinite(gap) & (gap > limit)
        expiry_info.update(held_contracts=int(np.abs(held.to_numpy()[on_expiry]).max()),
                           mismatch=bool(bad.any()), mark=float(mark[0]), intrinsic=float(intrinsic[on_expiry][0]))
        if bad.any():
            expiry_fail = (f"fail: expiry mark {float(mark[bad][0]):.4f} differs from intrinsic value "
                           f"{float(intrinsic[on_expiry][bad][0]):.4f} by more than the currency rounding unit "
                           f"{config.currency_rounding_unit} on the expiry bar {config.expiry} with a position held")
        price[on_expiry] = intrinsic[on_expiry]
    price_s = pd.Series(price, index=idx)
    diff = price_s.diff()

    trades = pos.diff().abs().fillna(pos.abs()).shift(lag).fillna(0.0)
    gross = held * diff * mult
    explained_all = gross - cpc * trades
    reported_all = rep_ret.reindex(idx) * capital
    keep = (diff.notna() & reported_all.notna()).to_numpy()
    if int(keep.sum()) < MIN_BARS:
        return _blocked(f"only {int(keep.sum())} usable bars; need at least {MIN_BARS}")

    explained, reported, bars = explained_all[keep], reported_all[keep], idx[keep]
    time_col = config.trade_time_column if config.option_price_type == "last" else config.publication_column
    pub = pd.to_datetime(chain[time_col].reindex(idx)[keep], utc=True, errors="coerce")
    tz = getattr(bars, "tz", None)
    pub = pub.dt.tz_convert(tz) if tz is not None else pub.dt.tz_localize(None)
    pub_idx = pd.DatetimeIndex(pub)
    halt = None
    if config.halt_flag_column is not None:
        halt = chain[config.halt_flag_column].reindex(idx)[keep].fillna(False).astype(bool).to_numpy()
    held_keep = held.abs()[keep].reset_index(drop=True)

    # A stale held price is a warning here, not the FAIL of evaluate_pnl_linear: it sees those marks as fresh, and
    # the raw finding replaces its staleness block afterwards.
    raw_fresh = None
    try:
        raw_fresh = check_mark_freshness(bars, pub_idx, held_keep, lag, config.max_staleness, halt_flag=halt)
    except ValueError:
        pass  # evaluate_pnl_linear reports it as BLOCKED
    stale_pos = list(raw_fresh.stale) if raw_fresh is not None else []
    if stale_pos:
        is_stale = np.zeros(len(bars), dtype=bool)
        is_stale[stale_pos] = True
        pub_idx = pub_idx.where(~is_stale, bars)

    n = int(keep.sum())
    settings = {"bars_per_year": float(periods_per_year), "capital_base": capital,
                "newey_west_lag": _newey_west_lag(n), "alpha": ALPHA, "lag_bars": lag,
                "multiplier": mult, "cost_per_contract": cpc, "option_price_type": config.option_price_type or "unknown",
                "strike": config.strike, "expiry": config.expiry, "right": config.right,
                "currency_rounding_unit": config.currency_rounding_unit, "slice": "O1"}
    verdict = evaluate_pnl_linear(
        reported_pnl=reported.to_numpy(),
        explained_net_pnl=explained.to_numpy(),
        remainder=(reported - explained).to_numpy(),
        frame_columns=set(chain.columns),
        config_block=reconstruction,
        bars_per_year=settings["bars_per_year"],
        capital_base=capital,
        newey_west_lag=settings["newey_west_lag"],
        alpha=ALPHA,
        staleness_inputs={
            "bar_times": bars,
            "publication_times": pub_idx,
            "position": held_keep,
            "lag": lag,
            "halt_flag": halt,
        },
    )
    verdict.details["settings"] = settings
    verdict.details["expiry"] = expiry_info
    verdict.details["bars_without_mark"] = int((is_held & ~np.isfinite(price)).sum())
    verdict.details["cost"] = {"status": "OK", **cost_source, "cost_per_contract": cpc,
                               "contracts_traded": float(trades[keep].sum())}
    if verdict.verdict == "BLOCKED":
        return verdict
    if cost_source.get("margin_flagged"):
        verdict.reasons.append("warning: broker extras were not checked; a 25% margin was added to the commission "
                               "(not yet confirmed)")
    verdict.reasons.extend(f"warning: {w}" for w in cost_source.get("warnings", []))
    if raw_fresh is not None:
        verdict.details["staleness"] = raw_fresh
    if expiry_fail is not None:
        verdict.reasons = [expiry_fail] + (verdict.reasons if verdict.verdict == "FAIL" else [])
        verdict.verdict = "FAIL"
    warnings: list[str] = []
    if stale_pos:
        shown = [str(bars[i]) for i in stale_pos]
        msg = (f"stale held price on {len(shown)} bar(s), first {shown[0]} (more than {config.max_staleness} bar(s) "
               "old); this is a warning, not a block")
        warnings.append(msg)
        verdict.details["stale_held_price"] = {"status": "WARNING", "bars": shown}
        verdict.details["conditional"] = True
        verdict.reasons.append(f"warning: {msg}; the result is conditional on those marks")
    verdict.details["point_in_time"] = {"answer": config.point_in_time}
    if config.point_in_time != "yes":
        state = "unknown (recorded as an unknown answer)" if config.point_in_time == "unknown" else "not confirmed"
        msg = f"point-in-time status of the option data is {state}; this is a warning, not a block"
        warnings.append(msg)
        unknown.insert(0, "point_in_time")
        verdict.reasons.append(f"warning: {msg}")
    for name in unknown:
        if name != "point_in_time":
            msg = (f"{name} could not be inferred from the data and is recorded as an unknown answer; please "
                   "declare it; this is a warning, not a block")
            warnings.append(msg)
            verdict.reasons.append(f"warning: {msg}")
    if unknown:
        verdict.details["unknown_answers"] = unknown
    verdict.details["inferred"] = inferred
    verdict.details["warnings"] = warnings
    return verdict
