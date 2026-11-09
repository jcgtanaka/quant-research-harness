"""Verdict for the pnl.linear method: combines config, bounds and mark-freshness blocks.

Precedence, first match decides the class (all reasons of the deciding class are listed):
    BLOCKED       config invalid, frame columns missing, or a held bar has no publication time.
    FAIL          future-dated mark, stale held mark, absolute cap breached, share breached,
                  significant positive drift,
                  or the slope of reported on explained PnL away from 1 (|b - 1| > K x SE(b)).
    INCONCLUSIVE  share or slope undefined, fewer than MIN_ACTIVE_BARS active bars, downward-only remainder,
                  halt-declared bars, or a cap explicitly set to None (pending).
    PASS          none of the above.
Diagnostics are carried in ``details`` and never change the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qrh.checks.pnl_linear_bounds import (
    EVALUATED,
    NOISE_SCALED,
    UNDEFINED,
    absolute_cap_bound,
    drift_bound,
    share_bound,
    slope_bound,
)
from qrh.checks.pnl_linear_config import (
    ReconstructionConfigError,
    check_frame_columns,
    validate_reconstruction,
)
from qrh.checks.pnl_staleness import check_mark_freshness

MIN_ACTIVE_BARS = 20
BUG_EPS = 1e-9  # a built backtest has no honest noise: a larger unexplained difference is a bug
SHOWN_BUG_DAYS = 5


@dataclass
class PnlLinearVerdict:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def _bug_warnings(rem: np.ndarray, bar_times, lag: int) -> list[str]:
    """One warning naming the first days and amounts of any unexplained difference; empty when there is none.
    The first ``lag`` bars are skipped: their positions come from before the window, which the series cannot see."""
    bad = np.flatnonzero(np.abs(rem) > BUG_EPS)
    bad = bad[bad >= max(0, int(lag))]
    if bad.size == 0:
        return []
    times = list(bar_times)
    shown = ", ".join(f"{str(times[i])[:10]} amount {rem[i]:.6g}" for i in bad[:SHOWN_BUG_DAYS])
    more = f" (and {bad.size - SHOWN_BUG_DAYS} more day(s))" if bad.size > SHOWN_BUG_DAYS else ""
    return [f"warning: bug: unexplained difference between reported and explained PnL on {bad.size} day(s): "
            f"{shown}{more}; a harness-built backtest must have none"]


def compare_with_broker(model_pnl, broker_pnl, capital_base: float, bars_per_year: float) -> PnlLinearVerdict:
    """Comparison with broker records (live results): the only place the noise-scaled tolerance applies.
    The net gap must stay within the noise-scaled absolute cap; zero-mean noise cancels, a steady gap does not."""
    gap = np.asarray(model_pnl, dtype=float) - np.asarray(broker_pnl, dtype=float)
    absolute = absolute_cap_bound(gap, capital_base, bars_per_year, NOISE_SCALED)
    details = {"absolute_cap": absolute}
    if absolute.status == EVALUATED and absolute.breached:
        return PnlLinearVerdict("FAIL", [f"fail: model and broker PnL differ by {absolute.annualised_bp:.2f} bp/year, "
                                         f"above the noise-scaled {absolute.cap_bp_per_year:.2f} bp/year"], details)
    return PnlLinearVerdict("PASS", ["pass: model and broker PnL agree within the noise-scaled tolerance"], details)


def evaluate_pnl_linear(
    reported_pnl,
    explained_net_pnl,
    remainder,
    frame_columns,
    config_block,
    bars_per_year,
    capital_base,
    newey_west_lag,
    alpha,
    staleness_inputs,
    built: bool = False,
) -> PnlLinearVerdict:
    """``staleness_inputs`` keys: bar_times, publication_times, position, lag, and optional
    halt_flag and bar_period. max_staleness comes from the config block.

    ``built=True`` is a backtest the harness built itself: there is no honest noise to forgive, so any unexplained
    difference is a bug shown as a warning (day and amount) in ``details["warnings"]`` instead of a noise-scaled
    cap, share or drift test. The noise-scaled tolerance stays for comparisons with broker records."""
    details: dict[str, Any] = {}

    try:
        config = validate_reconstruction(config_block)
        check_frame_columns(config, frame_columns)
    except ReconstructionConfigError as exc:
        return PnlLinearVerdict("BLOCKED", [f"blocked: configuration rejected: {exc}"], details)

    si = staleness_inputs
    try:
        fresh = check_mark_freshness(
            si["bar_times"], si["publication_times"], si["position"], si["lag"],
            config.max_staleness, halt_flag=si.get("halt_flag"), bar_period=si.get("bar_period"),
        )
    except ValueError as exc:
        return PnlLinearVerdict("BLOCKED", [f"blocked: mark freshness cannot be assessed: {exc}"], details)
    details["staleness"] = fresh

    rep = np.asarray(reported_pnl, dtype=float)
    exp = np.asarray(explained_net_pnl, dtype=float)
    rem = np.asarray(remainder, dtype=float)
    absolute = absolute_cap_bound(rem, capital_base, bars_per_year, config.max_unexplained_bp_per_year)
    share = share_bound(rem, exp, config.max_unexplained_share)
    drift = drift_bound(rem, newey_west_lag, alpha) if rem.size > newey_west_lag else None
    slope = slope_bound(rep, exp, newey_west_lag) if rep.size > newey_west_lag else None
    n_active = int(np.count_nonzero((rep != 0) | (exp != 0)))
    details.update(absolute_cap=absolute, share=share, drift=drift, slope=slope, n_active_bars=n_active)

    if built:
        details["warnings"] = _bug_warnings(rem, si["bar_times"], si["lag"])
    fail: list[str] = []
    if fresh.future_dated:
        fail.append(f"fail: future-dated mark on held bar(s) {fresh.future_dated}")
    if fresh.stale:
        fail.append(f"fail: stale held mark on bar(s) {fresh.stale}")
    if not built and absolute.status == EVALUATED and absolute.breached:
        fail.append(f"fail: absolute cap breached ({absolute.annualised_bp:.2f} bp/year "
                    f"net unexplained, above the {absolute.cap_bp_per_year} bp/year cap)")
    if not built and share.status == EVALUATED and share.breached:
        fail.append(f"fail: share bound breached (share {share.share:.4f} above "
                    f"{share.max_unexplained_share})")
    if not built and drift is not None and drift.significant and drift.mean > 0:
        fail.append(f"fail: significant positive drift in the remainder (p={drift.p_value:.4g} "
                    f"< alpha {drift.alpha})")
    if slope is not None and slope.status == EVALUATED and slope.breached:
        fail.append(f"fail: slope: reported PnL scales as {slope.slope:.4g} against explained PnL, expected 1 "
                    f"(|b - 1| above {slope.limit:.4g})")
    if fail:
        return PnlLinearVerdict("FAIL", fail, details)

    inc: list[str] = []
    if share.status == UNDEFINED:
        inc.append("inconclusive: share bound undefined because net explained PnL is zero")
    if slope is not None and slope.status == UNDEFINED:
        inc.append("inconclusive: slope bound undefined because explained PnL has zero variance")
    if n_active < MIN_ACTIVE_BARS:
        inc.append(f"inconclusive: fewer than {MIN_ACTIVE_BARS} active bars ({n_active})")
    if rem.size and rem.min() < 0 and rem.max() <= 0:
        inc.append("inconclusive: the remainder is downward only (no positive part), "
                   "so the upward bounds carry no information")
    if fresh.halt_declared:
        inc.append(f"inconclusive: halt-declared bar(s) {fresh.halt_declared}; never PASS with halts")
    if config.pending:
        inc.append(f"pending user decision: {', '.join(config.pending)}")
    if inc:
        return PnlLinearVerdict("INCONCLUSIVE", inc, details)

    return PnlLinearVerdict(
        "PASS",
        ["pass: all three bounds and the slope limit evaluated and not breached; no stale or future marks, "
         "enough active bars, no halts"],
        details,
    )
