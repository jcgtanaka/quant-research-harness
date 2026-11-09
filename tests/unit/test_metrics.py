import math

import numpy as np
import pandas as pd
import pytest

from qrh.validation.metrics import (
    annualized_sharpe,
    cagr,
    infer_periods_per_year,
    max_drawdown,
    sharpe,
)


def test_sharpe_matches_definition_with_ddof_1():
    r = np.array([0.01, -0.02, 0.03, 0.0, 0.01])
    expected = r.mean() / (math.sqrt(((r - r.mean()) ** 2).sum() / (len(r) - 1)))
    assert sharpe(r) == pytest.approx(expected, rel=1e-12)


def test_annualized_sharpe_scales_with_sqrt_periods():
    r = np.array([0.01, -0.02, 0.03, 0.0, 0.01])
    assert annualized_sharpe(r, 100.0) == pytest.approx(sharpe(r) * 10.0, rel=1e-12)


@pytest.mark.parametrize("r", [[0.0, 0.0, 0.0], [0.01] * 5, [0.01], []])
def test_sharpe_undefined_is_none_never_inf(r):
    assert sharpe(np.array(r)) is None
    assert annualized_sharpe(np.array(r), 252.0) is None


def test_sharpe_rejects_non_finite():
    with pytest.raises(ValueError):
        sharpe(np.array([0.01, np.nan, 0.02]))
    with pytest.raises(ValueError):
        sharpe(np.array([0.01, np.inf, 0.02]))


def test_sharpe_accepts_series():
    s = pd.Series([0.01, 0.02, -0.01])
    assert sharpe(s) == sharpe(s.to_numpy())


def test_cagr_closed_form():
    # 2 periods per year, 4 periods of +10% -> 2 years -> annual growth 1.1**2 - 1
    r = np.full(4, 0.10)
    assert cagr(r, 2.0) == pytest.approx(1.1**2 - 1, rel=1e-12)
    # exactly one year of data: CAGR equals total return
    r2 = np.array([0.1, -0.05, 0.02])
    assert cagr(r2, 3.0) == pytest.approx(1.1 * 0.95 * 1.02 - 1, rel=1e-12)


def test_cagr_total_loss_and_empty():
    assert cagr(np.array([0.1, -1.0]), 2.0) == -1.0
    assert cagr(np.array([]), 252.0) is None
    with pytest.raises(ValueError):
        cagr(np.array([0.1]), 0.0)


def test_max_drawdown_known_path():
    # equity: 1 -> 1.1 -> 0.99 -> 0.891 -> 0.9801 ; peak 1.1 trough 0.891
    r = np.array([0.1, -0.1, -0.1, 0.1])
    assert max_drawdown(r) == pytest.approx(0.891 / 1.1 - 1, rel=1e-12)


def test_max_drawdown_monotone_up_is_zero_and_never_positive():
    assert max_drawdown(np.array([0.01, 0.02, 0.0])) == 0.0
    rng = np.random.default_rng(0)
    assert max_drawdown(rng.normal(0, 0.01, 500)) <= 0.0
    assert max_drawdown(np.array([])) is None


def test_max_drawdown_starts_from_initial_capital():
    # first return is a loss: drawdown is measured against the starting equity of 1
    assert max_drawdown(np.array([-0.2, 0.1])) == pytest.approx(-0.2)


def _idx(freq, n=50, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq=freq, tz="UTC")


@pytest.mark.parametrize(
    "freq,per_year",
    [("D", 365.25), ("h", 365.25 * 24), ("min", 365.25 * 24 * 60), ("W", 365.25 / 7)],
)
def test_infer_periods_per_year_on_a_regular_grid(freq, per_year):
    assert infer_periods_per_year(_idx(freq)) == pytest.approx(per_year, rel=1e-12)


def test_infer_calendar_daily_over_three_years():
    assert infer_periods_per_year(_idx("D", 3 * 365)) == pytest.approx(365.25, rel=1e-9)


def test_infer_business_days_reflects_observed_density_not_spacing():
    idx = pd.bdate_range("2020-01-01", periods=3 * 261, tz="UTC")
    assert infer_periods_per_year(idx) == pytest.approx(261, rel=0.01)  # not 365


def test_infer_hourly_24_7():
    assert infer_periods_per_year(_idx("h", 24 * 365 * 2)) == pytest.approx(8766, rel=0.001)


def test_infer_session_bars_on_weekdays():
    # 6.5-hour session of 5-minute bars (78 per day), weekdays only
    days = pd.bdate_range("2021-01-04", periods=261, tz="UTC")
    idx = pd.DatetimeIndex(sorted(
        t for d in days for t in pd.date_range(d + pd.Timedelta(hours=14, minutes=30), periods=78, freq="5min")
    ))
    assert infer_periods_per_year(idx) == pytest.approx(78 * 261, rel=0.02)


def test_infer_density_counts_gaps_unlike_a_median():
    idx = _idx("D", 100).delete(list(range(10, 40)))  # a 30-day hole
    assert infer_periods_per_year(idx) < 365.25 * 0.75


@pytest.mark.parametrize("n", [0, 1, 2])
def test_infer_needs_enough_timestamps(n):
    assert infer_periods_per_year(_idx("D", n)) is None


def test_infer_rejects_unsorted_duplicates_and_non_datetime():
    idx = _idx("D", 10)
    assert infer_periods_per_year(idx[::-1]) is None
    assert infer_periods_per_year(idx.append(idx)) is None
    assert infer_periods_per_year(pd.Index([1, 2, 3, 4])) is None
