"""Same-bar execution check, on by default in S2: reported returns must not use the position of the bar they earn.

A strategy that decides on the close of bar t and also fills on that close earns the return of bar t with a
position that was decided from it. That is look-ahead in the fill, and no data perturbation shows it, because the
position itself uses only past data. The check rebuilds the returns from the reported positions and a price column
with the fill one bar after the decision (``qrh.checks.pnl_reconstruction``) and fails when the reported returns
fit an earlier fill instead.

It needs two inputs: the positions the run reported, and the price column(s) to rebuild returns from. Positions
cannot be guessed, so a run without them is BLOCKED. The price column is the one the hypothesis declares in
``execution``; when nothing is declared the check tries each strictly positive numeric column of the dataset
(at most ``MAX_CANDIDATES``) with a lag of one bar, and a run whose returns no column explains is INCONCLUSIVE,
never PASS: declare ``execution`` (lag, price column, cost bound) to settle it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from qrh.adapters.base import RunResult
from qrh.checks.pnl_reconstruction import PnlResult, check_pnl_reconstruction, validate_execution

MAX_CANDIDATES = 12
INFERRED_LAG = 1
_HOW_TO_DECLARE = "declare `execution` in the hypothesis (lag_bars, price_column or price_columns, max_cost_per_turnover)"


def _price_like(frame: pd.DataFrame) -> list[Any]:
    out = []
    for c in frame.columns:
        dt = frame[c].dtype
        if isinstance(dt, np.dtype) and dt.kind in "if":
            v = frame[c].to_numpy(dtype=float)
            if len(v) and np.isfinite(v).all() and (v > 0).all():
                out.append(c)
    return out


def _score(r: PnlResult) -> float:
    m = r.details.get("match_lagged")
    return -1.0 if m is None else float(m)


HELD_DURING = ("positions look like they are reported as held during the bar (they never depend on the data of their "
               "own bar), so the reconstruction cannot tell a same-bar fill from a correctly lagged one; declare "
               "`execution` or report the position held after the bar")


def check_same_bar_execution(
    result: RunResult, frame: pd.DataFrame, execution: dict[str, Any] | None = None,
    own_bar_probe: Callable[[], bool | None] | None = None,
) -> PnlResult:
    """``own_bar_probe`` (see ``qrh.checks.perturbation.make_own_bar_probe``) answers whether the reported position of
    a bar depends on that bar's data. Without a declared ``execution``, an inferred FAIL is turned into INCONCLUSIVE
    when the probe says it does not: the positions are then the position held during the bar, whose returns are
    correctly lagged by construction."""
    positions = result.positions
    if positions is None or not isinstance(positions, pd.DataFrame) or positions.empty:
        return PnlResult("BLOCKED", [
            "same-bar execution cannot be checked: the run reported no positions (RunResult.positions), so the "
            "reported returns cannot be rebuilt; report the position held after each bar"])
    if execution is not None:
        try:
            validate_execution(execution)
        except ValueError as exc:
            return PnlResult("BLOCKED", [f"same-bar execution cannot be checked: {exc}"])
        out = check_pnl_reconstruction(result, frame, execution)
        out.details["declared_execution"] = True
        return out

    prices = _price_like(frame)
    pos_cols = list(positions.columns)
    if positions.shape[1] == 1:
        plans = [{"lag_bars": INFERRED_LAG, "price_column": c} for c in prices[:MAX_CANDIDATES]]
    elif prices and set(pos_cols) <= set(prices):
        plans = [{"lag_bars": INFERRED_LAG, "price_columns": pos_cols}]
    else:
        return PnlResult("BLOCKED", [
            f"same-bar execution cannot be checked: the position columns {sorted(map(str, pos_cols))} do not match "
            f"price columns of the dataset; {_HOW_TO_DECLARE}"])
    if not plans:
        return PnlResult("BLOCKED", [
            "same-bar execution cannot be checked: the dataset has no strictly positive numeric column to rebuild "
            f"returns from; {_HOW_TO_DECLARE}"])
    tried = [p["price_column"] if "price_column" in p else list(p["price_columns"]) for p in plans]
    results = [(p, check_pnl_reconstruction(result, frame, p)) for p in plans]
    base = {"inferred_execution": True, "inferred_lag_bars": INFERRED_LAG, "candidates_tried": [str(t) for t in tried]}

    def named(p: dict[str, Any], r: PnlResult, reasons: list[str] | None = None) -> PnlResult:
        col = p.get("price_column", p.get("price_columns"))
        details = {**r.details, **base, "inferred_price_column" if "price_column" in p else "inferred_price_columns": col}
        return PnlResult(r.verdict, reasons if reasons is not None else list(r.reasons), details)

    fails = [(p, r) for p, r in results if r.verdict == "FAIL"]
    if fails:
        timing = [(p, r) for p, r in fails if "shorter_lag" in r.details]
        p, r = max(timing or fails, key=lambda pr: pr[1].details.get("match_shorter_lag_informative") or 0.0)
        col = p.get("price_column", p.get("price_columns"))
        if own_bar_probe is not None and own_bar_probe() is False:
            return PnlResult("INCONCLUSIVE", [f"cannot verify same-bar execution: {HELD_DURING}"],
                             {**r.details, **base, "held_during_bar": True})
        return named(p, r, [f"{reason} (price column {col!r}, execution lag assumed {INFERRED_LAG} bar)" for reason in r.reasons])
    passes = [(p, r) for p, r in results if r.verdict == "PASS"]
    if passes:
        p, r = max(passes, key=lambda pr: _score(pr[1]))
        return named(p, r)
    inconclusive = [(p, r) for p, r in results if r.verdict == "INCONCLUSIVE"]
    if inconclusive:
        p, r = max(inconclusive, key=lambda pr: _score(pr[1]))
        return named(p, r, [
            f"cannot verify same-bar execution: the reported returns match neither a lag-{INFERRED_LAG} nor a "
            f"same-bar reconstruction from any of the {len(tried)} column(s) tried {base['candidates_tried']} "
            f"(costs above the default bound, leverage or a non-price signal can cause this); {_HOW_TO_DECLARE}"])
    p, r = results[0]
    return named(p, r, [f"same-bar execution cannot be checked: {reason}" for reason in r.reasons])
