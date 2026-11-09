"""Tests for the bench-side true ledger (quantities, fills, cash)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qrh.bench.pnl_ledger import LedgerResult, plant_hidden_gain, run_ledger


def _prices(n: int = 50, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))


def test_single_buy_then_hold_matches_price_change_times_quantity():
    p = _prices()
    res = run_ledger(p, [(0, 10.0, p[0])], starting_capital=10_000.0, commission_per_trade=0.0)
    assert isinstance(res, LedgerResult)
    expected = 10.0 * np.diff(p)
    np.testing.assert_allclose(res.pnl[1:], expected, atol=1e-9)
    assert res.pnl[0] == pytest.approx(0.0)
    np.testing.assert_allclose(res.quantity, 10.0)


def test_commissions_reduce_nav():
    p = _prices()
    free = run_ledger(p, [(3, 5.0, p[3])], 10_000.0, commission_per_trade=0.0)
    paid = run_ledger(p, [(3, 5.0, p[3])], 10_000.0, commission_per_trade=2.5)
    assert paid.nav[-1] == pytest.approx(free.nav[-1] - 2.5)
    assert paid.cost_pnl[3] == pytest.approx(-2.5)
    assert paid.cost_pnl.sum() == pytest.approx(-2.5)


@pytest.mark.parametrize("seed", range(5))
def test_breakdown_sums_to_total_change(seed):
    rng = np.random.default_rng(seed)
    p = _prices(80, seed)
    bars = rng.integers(0, 80, size=10)
    sched = [(int(b), float(rng.integers(-5, 6)), float(p[b] * (1 + rng.normal(0, 0.001)))) for b in bars]
    income = rng.normal(0, 0.5, 80)
    res = run_ledger(p, sched, 50_000.0, commission_per_trade=1.0, income=income)
    total = res.price_pnl + res.execution_pnl + res.cost_pnl + res.income_pnl
    np.testing.assert_allclose(total, res.pnl, atol=1e-9)
    np.testing.assert_allclose(res.pnl, np.diff(res.nav, prepend=50_000.0), atol=1e-9)
    np.testing.assert_allclose(res.income_pnl, income, atol=1e-12)


def test_planted_gain_separate_and_ledger_unchanged():
    p = _prices()
    res = run_ledger(p, [(0, 10.0, p[0])], 10_000.0)
    before = res.pnl.copy(), res.nav.copy()
    reported, planted = plant_hidden_gain(res, gain_per_bar=0.7)
    np.testing.assert_array_equal(res.pnl, before[0])
    np.testing.assert_array_equal(res.nav, before[1])
    np.testing.assert_allclose(planted, 0.7)
    np.testing.assert_allclose(reported, res.pnl + planted)
    assert reported is not res.pnl
    with pytest.raises(ValueError):
        res.pnl[0] = 1.0  # ledger arrays are read-only


def test_determinism_same_seed():
    a = run_ledger(_prices(seed=7), [(2, 3.0, 100.0)], 1_000.0, commission_per_trade=0.5)
    b = run_ledger(_prices(seed=7), [(2, 3.0, 100.0)], 1_000.0, commission_per_trade=0.5)
    np.testing.assert_array_equal(a.pnl, b.pnl)
    np.testing.assert_array_equal(a.nav, b.nav)
    p1, g1 = plant_hidden_gain(a, 0.1, seed=3, noise_sd=0.05)
    p2, g2 = plant_hidden_gain(b, 0.1, seed=3, noise_sd=0.05)
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_array_equal(p1, p2)


def test_empty_schedule_zero_pnl_and_series_input():
    p = pd.Series(_prices())
    res = run_ledger(p, [], 1_000.0)
    np.testing.assert_allclose(res.pnl, 0.0)
    np.testing.assert_allclose(res.nav, 1_000.0)


def test_invalid_inputs_raise():
    p = _prices(10)
    with pytest.raises(ValueError, match="starting_capital"):
        run_ledger(p, [], -1.0)
    with pytest.raises(ValueError, match="length"):
        run_ledger(p, [], 100.0, income=np.zeros(5))
    with pytest.raises(ValueError, match="bar"):
        run_ledger(p, [(10, 1.0, 100.0)], 100.0)
    with pytest.raises(ValueError, match="commission"):
        run_ledger(p, [], 100.0, commission_per_trade=-1.0)
    with pytest.raises(ValueError, match="length"):
        plant_hidden_gain(run_ledger(p, [], 100.0), 0.1, gain_series=np.zeros(3))
