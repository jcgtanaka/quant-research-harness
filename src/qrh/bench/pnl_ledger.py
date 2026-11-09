"""True ledger for the bench: PnL generated from quantities, fills and cash.

Deliberately independent of any reconstruction formula (no weight-times-return
anywhere). NAV is cash plus quantity times price, and the per-bar PnL is the
change in NAV. The breakdown lines are derived separately and checked against
that change, so the ledger is the known-correct answer for the bench.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

_TOL = 1e-8


@dataclass(frozen=True)
class LedgerResult:
    """Ledger output. Arrays are read-only; the ledger is never mutated."""

    pnl: np.ndarray  # change in NAV per bar (bar 0 is relative to starting capital)
    nav: np.ndarray
    quantity: np.ndarray  # quantity held after the bar's fills
    price_pnl: np.ndarray  # prior quantity times price change
    execution_pnl: np.ndarray  # traded quantity times (bar price - fill price)
    cost_pnl: np.ndarray  # negative commissions
    income_pnl: np.ndarray  # declared cash credits
    starting_capital: float


def _frozen(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float, copy=True)
    a.setflags(write=False)
    return a


def run_ledger(
    prices: np.ndarray | pd.Series,
    schedule: Sequence[tuple[int, float, float]],
    starting_capital: float,
    commission_per_trade: float = 0.0,
    income: np.ndarray | pd.Series | None = None,
    seed: int | None = None,
    multiplier: float = 1.0,
    commission_per_contract: float = 0.0,
) -> LedgerResult:
    """Walk a cash-and-quantity ledger bar by bar.

    schedule: (bar, quantity_change, fill_price); several fills per bar allowed,
    applied in list order. Each fill costs ``commission_per_trade`` in cash.
    income: optional per-bar cash credits (e.g. interest, dividends).
    seed: accepted for a uniform bench signature; the ledger itself is fully
    deterministic and draws no random numbers.
    multiplier: currency per price point per unit of quantity (default 1.0, the
    original behaviour). commission_per_contract: extra cash cost per unit of
    absolute quantity traded (default 0.0), on top of commission_per_trade.
    """
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("prices must be a non-empty 1-D series")
    if not np.all(np.isfinite(p)):
        raise ValueError("prices must be finite")
    if not np.isfinite(starting_capital) or starting_capital < 0:
        raise ValueError("starting_capital must be a non-negative finite number")
    if commission_per_trade < 0:
        raise ValueError("commission_per_trade must be non-negative")
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("multiplier must be a positive finite number")
    if not np.isfinite(commission_per_contract) or commission_per_contract < 0:
        raise ValueError("commission_per_contract must be a non-negative finite number")
    n = p.size
    inc = np.zeros(n) if income is None else np.asarray(income, dtype=float)
    if inc.shape != (n,):
        raise ValueError(f"income length {inc.size} does not match prices length {n}")

    fills: dict[int, list[tuple[float, float]]] = {}
    for bar, dq, fp in schedule:
        if not isinstance(bar, (int, np.integer)) or not 0 <= bar < n:
            raise ValueError(f"trade bar {bar!r} outside 0..{n - 1}")
        if not np.isfinite(dq) or not np.isfinite(fp) or fp <= 0:
            raise ValueError("quantity_change must be finite and fill_price positive")
        fills.setdefault(int(bar), []).append((float(dq), float(fp)))

    q = 0.0
    cash = float(starting_capital)
    prev_nav = float(starting_capital)
    prev_price = p[0]
    nav = np.empty(n)
    qty = np.empty(n)
    price_l, exec_l, cost_l = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(n):
        q_prior = q
        price_l[t] = q_prior * multiplier * (p[t] - prev_price)
        for dq, fp in fills.get(t, ()):
            q += dq
            fee = commission_per_trade + commission_per_contract * abs(dq)
            cash -= dq * fp * multiplier
            cash -= fee
            exec_l[t] += dq * multiplier * (p[t] - fp)
            cost_l[t] -= fee
        cash += inc[t]
        nav[t] = cash + q * multiplier * p[t]
        qty[t] = q
        prev_price = p[t]
        step = nav[t] - prev_nav
        parts = price_l[t] + exec_l[t] + cost_l[t] + inc[t]
        if abs(step - parts) > _TOL * max(1.0, abs(nav[t])):
            raise AssertionError(
                f"ledger breakdown {parts!r} does not sum to NAV change {step!r} at bar {t}"
            )
        prev_nav = nav[t]

    pnl = np.diff(nav, prepend=float(starting_capital))
    return LedgerResult(
        pnl=_frozen(pnl),
        nav=_frozen(nav),
        quantity=_frozen(qty),
        price_pnl=_frozen(price_l),
        execution_pnl=_frozen(exec_l),
        cost_pnl=_frozen(cost_l),
        income_pnl=_frozen(inc),
        starting_capital=float(starting_capital),
    )


def plant_hidden_gain(
    ledger: LedgerResult,
    gain_per_bar: float,
    seed: int | None = None,
    noise_sd: float = 0.0,
    gain_series: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (reported_pnl, planted_gain) without touching the ledger.

    planted = gain_series if given, else gain_per_bar plus optional seeded
    Gaussian noise (noise_sd). reported = true ledger pnl + planted.
    """
    n = ledger.pnl.size
    if gain_series is not None:
        planted = np.array(gain_series, dtype=float, copy=True)
        if planted.shape != (n,):
            raise ValueError(f"gain_series length {planted.size} does not match ledger length {n}")
    else:
        if noise_sd < 0:
            raise ValueError("noise_sd must be non-negative")
        planted = np.full(n, float(gain_per_bar))
        if noise_sd > 0:
            planted += np.random.default_rng(seed).normal(0.0, noise_sd, n)
    reported = ledger.pnl + planted  # new array
    return reported, planted
