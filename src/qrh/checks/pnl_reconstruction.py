"""Independent P&L reconstruction (``s2.pnl_reconstruction=pnl_reconstruction@1``, experimental).

Look-ahead in a signal is caught by the truncation test; a *timing* or *units* defect in how an adapter turns
positions into returns is invisible to it, because positions do not change. This check recomputes the strategy
returns itself from the adapter's reported positions and the declared price column(s) with an enforced execution
lag, and compares them with what the adapter reported.

Spec field ``execution`` (recorded at S0, immutable): ``lag_bars`` (integer >= 1, required, no default),
``price_column`` or ``price_columns`` (the column(s) positions are held in; positions are aligned to them by
name, or one-to-one when each has a single column) and ``max_cost_per_turnover`` (fraction of notional per unit
of turnover, default ``DEFAULT_COST_BOUND``).

For lag k the reconstruction is ``sum_i position_i[t-k] * return_i[t]``. A reported return is *consistent* with it
when it is not above it (beyond float noise) and not below it by more than the declared cost bound times the
turnover, so costs may lower the return but nothing may raise it. Verdicts:

- FAIL "same-bar execution (look-ahead)" or "execution lag k < declared": on the bars where lag k and the declared lag
  disagree, the adapter matches the shorter lag and not the declared one;
- FAIL "units mismatch": the reported returns are the lagged (or same-bar) reconstruction times a power of ten
  other than 1 (for example 100, percent against fractions);
- PASS: the reported returns match the lagged reconstruction on at least ``PASS_MATCH`` of the active bars;
- INCONCLUSIVE with the figures: anything else. An unexplained mismatch is never a PASS;
- BLOCKED: the check cannot run (no positions, price column missing, positions not alignable, too few bars, bad spec).

It does not look at the signal, so a peeking signal with a correct execution lag passes here (truncation and lint own that).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from qrh.adapters.base import RunResult

DEFAULT_COST_BOUND = 0.001
PASS_MATCH = 0.98
MIN_BARS = 20
MIN_INFORMATIVE = 5
SHORTER_LAG_MATCH = 0.9
SHORTER_LAG_GAP = 0.5
UNITS_CORR = 0.98
UNITS_TOLERANCE = 0.05  # in log10 units, about 12 percent
_KEYS = {"lag_bars", "price_column", "price_columns", "max_cost_per_turnover"}
_TINY = 1e-12


@dataclass
class PnlResult:
    verdict: str  # PASS | FAIL | INCONCLUSIVE | BLOCKED
    reasons: list[str]
    details: dict[str, Any] = field(default_factory=dict)


def _columns(execution: dict[str, Any]) -> list[str]:
    return [execution["price_column"]] if "price_column" in execution else list(execution["price_columns"])


def validate_execution(execution: Any) -> None:
    """Raises ValueError with the reason if the ``execution`` spec cannot be used."""
    if not isinstance(execution, dict):
        raise ValueError("execution must be a mapping with lag_bars and price_column(s)")
    unknown = sorted(set(execution) - _KEYS)
    if unknown:
        raise ValueError(f"execution: unknown key(s) {unknown}")
    lag = execution.get("lag_bars")
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
        raise ValueError("execution.lag_bars: required, an integer >= 1 (there is no default)")
    one, many = "price_column" in execution, "price_columns" in execution
    if one == many:
        raise ValueError("execution: give exactly one of price_column or price_columns")
    if one and (not isinstance(execution["price_column"], str) or not execution["price_column"]):
        raise ValueError("execution.price_column: must be a column name")
    if many and (
        not isinstance(execution["price_columns"], list) or not execution["price_columns"]
        or not all(isinstance(c, str) and c for c in execution["price_columns"])
    ):
        raise ValueError("execution.price_columns: must be a non-empty list of column names")
    cost = execution.get("max_cost_per_turnover", DEFAULT_COST_BOUND)
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        raise ValueError("execution.max_cost_per_turnover: must be a number >= 0")


def _num(v: float) -> float | None:
    return None if v is None or not math.isfinite(v) else float(v)


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    if len(a) < 3 or np.std(a) <= _TINY or np.std(b) <= _TINY:
        return None, None
    return _num(float(np.corrcoef(a, b)[0, 1])), _num(float(stats.spearmanr(a, b)[0]))


def _blocked(*reasons: str) -> PnlResult:
    return PnlResult("BLOCKED", list(reasons))


def _align_positions(pos: pd.DataFrame, cols: list[str]) -> pd.DataFrame | None:
    if set(pos.columns) == set(cols) and len(pos.columns) == len(cols):
        return pos[cols]
    if len(cols) == 1 and pos.shape[1] == 1:
        return pos.set_axis(cols, axis=1)
    return None


def check_pnl_reconstruction(result: RunResult, frame: pd.DataFrame, execution: dict[str, Any]) -> PnlResult:
    try:
        validate_execution(execution)
    except ValueError as exc:
        return _blocked(str(exc))
    lag, cols = int(execution["lag_bars"]), _columns(execution)
    cost = float(execution.get("max_cost_per_turnover", DEFAULT_COST_BOUND))
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        return _blocked(f"price column(s) {missing} not found in the dataset")
    if result.positions is None or not isinstance(result.positions, pd.DataFrame) or result.positions.empty:
        return _blocked("the run reported no positions to reconstruct returns from")
    pos = _align_positions(result.positions.astype(float), cols)
    if pos is None:
        return _blocked(
            f"cannot align positions columns {sorted(map(str, result.positions.columns))} to price columns {cols}"
        )
    rep = result.returns.astype(float)
    idx = rep.index.intersection(pos.index)
    rets = frame[cols].astype(float).pct_change().reindex(idx)
    pos, rep = pos.reindex(idx).fillna(0.0), rep.reindex(idx)
    valid = np.arange(len(idx)) >= lag
    valid &= rets.notna().all(axis=1).to_numpy() & rep.notna().to_numpy()
    if int(valid.sum()) < MIN_BARS:
        return _blocked(f"only {int(valid.sum())} usable bars; need at least {MIN_BARS}")
    turn = pos.diff().abs().sum(axis=1).fillna(pos.abs().sum(axis=1))
    r = rep.to_numpy()[valid]
    gross, tol = {}, {}
    for k in range(lag + 1):
        gross[k] = (pos.shift(k) * rets).sum(axis=1).to_numpy()[valid]
        tol[k] = (cost * (turn + turn.shift(k).fillna(0.0))).to_numpy()[valid]
    active = np.zeros(len(r), dtype=bool)
    for g in gross.values():
        active |= np.abs(g) > _TINY
    active |= np.abs(r) > _TINY
    if not active.any():
        return PnlResult("INCONCLUSIVE", ["no active bars: positions and returns are all zero, nothing to verify"],
                         {"bars": int(valid.sum()), "active_bars": 0})

    def band(k: int) -> np.ndarray:
        atol = 1e-9 + 1e-6 * np.abs(gross[k])
        return (r >= gross[k] - tol[k] - atol) & (r <= gross[k] + atol)

    match = {k: band(k) for k in gross}
    informative = {k: np.abs(gross[lag] - gross[k]) > 1e-9 for k in range(lag)}
    d: dict[str, Any] = {"bars": int(valid.sum()), "active_bars": int(active.sum()), "lag_bars": lag,
                         "max_cost_per_turnover": cost}
    d["match_lagged"] = _num(float(match[lag][active].mean()))
    d["match_same_bar"] = _num(float(match[0][active].mean()))
    for name, k in (("lagged", lag), ("same_bar", 0)):
        d[f"corr_{name}"], d[f"rank_corr_{name}"] = _corr(r[active], gross[k][active])
    figures = (
        f"figures: match lagged {d['match_lagged']:.3f}, same-bar {d['match_same_bar']:.3f}; correlation lagged "
        f"{d['corr_lagged']}, same-bar {d['corr_same_bar']}; rank correlation lagged {d['rank_corr_lagged']}, "
        f"same-bar {d['rank_corr_same_bar']} over {int(active.sum())} active bars"
    )

    # units: reported = reconstruction x 10^e, e != 0
    for k in (lag, *range(lag)):
        g = gross[k][active]
        if np.dot(g, g) <= _TINY:
            continue
        corr, _ = _corr(r[active], g)
        slope = float(np.dot(r[active], g) / np.dot(g, g))
        if corr is not None and corr >= UNITS_CORR and slope > 0:
            e = math.log10(slope)
            if round(e) != 0 and abs(e - round(e)) < UNITS_TOLERANCE:
                d.update(scale_exponent=int(round(e)), scale_slope=_num(slope))
                what = "lagged" if k == lag else f"lag-{k}"
                reasons = [
                    f"units mismatch (likely percent vs fraction): reported returns are {slope:.4g} times the "
                    f"{what} reconstruction (a factor of 10^{int(round(e))}), correlation {corr:.4f}"
                ]
                if k != lag:
                    reasons.append(_timing_reason(k, lag))
                    d["shorter_lag"] = k  # the reported returns also fit an execution lag shorter than declared
                return PnlResult("FAIL", reasons, d)
        if k == lag:
            continue

    # a shorter execution lag than declared
    best: tuple[float, int] | None = None
    for k in range(lag):
        inf = informative[k]
        if int(inf.sum()) < MIN_INFORMATIVE:
            continue
        m_k, m_l = float(match[k][inf].mean()), float(match[lag][inf].mean())
        c_k, _ = _corr(r[inf], gross[k][inf])
        c_l, _ = _corr(r[inf], gross[lag][inf])
        by_match = m_k >= SHORTER_LAG_MATCH and m_k - m_l >= SHORTER_LAG_GAP
        by_corr = c_k is not None and c_k >= SHORTER_LAG_MATCH and (c_l is None or c_k - c_l >= SHORTER_LAG_GAP)
        if (by_match or by_corr) and (best is None or m_k > best[0]):
            best = (m_k, k)
            d.update(shorter_lag=k, informative_bars=int(inf.sum()), match_shorter_lag_informative=_num(m_k),
                     match_declared_lag_informative=_num(m_l))
    if best is not None:
        return PnlResult("FAIL", [_timing_reason(best[1], lag) + f" ({d['informative_bars']} bars where the lags differ)"], d)

    if d["match_lagged"] >= PASS_MATCH:
        return PnlResult("PASS", [], d)
    return PnlResult(
        "INCONCLUSIVE",
        [f"reported returns match neither the lag-{lag} nor the same-bar reconstruction; cannot verify ({figures})"], d,
    )


def _timing_reason(k: int, lag: int) -> str:
    if k == 0:
        return "same-bar execution (look-ahead): reported returns use the position of the bar they earn"
    return f"execution lag {k} bar(s) is shorter than the declared {lag}: reported returns are better than the lag allows"
