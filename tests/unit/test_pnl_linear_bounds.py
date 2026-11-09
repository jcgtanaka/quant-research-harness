"""Tests for the three unexplained-remainder bounds (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pytest

from qrh.checks.pnl_linear_bounds import (
    EVALUATED,
    NOT_EVALUATED,
    UNDEFINED,
    absolute_cap_bound,
    drift_bound,
    share_bound,
)

BPY = 252.0
CAPITAL = 100_000.0
N = 252


def _rng_noise(n=N, scale=1.0, seed=7):
    return np.random.default_rng(seed).normal(0.0, scale, n)


def test_zero_remainder_passes_every_bound():
    rem = np.zeros(N)
    expl = np.full(N, 5.0)
    a = absolute_cap_bound(rem, CAPITAL, BPY, 1.0)
    s = share_bound(rem, expl, 0.01)
    d = drift_bound(rem, 5, 0.05)
    assert (a.status, a.breached, a.annualised_bp) == (EVALUATED, False, 0.0)
    assert (s.status, s.breached) == (EVALUATED, False)
    assert d.status == EVALUATED and d.significant is False


def test_hidden_constant_gain_breaches_cap_and_drift_but_not_large_share_cap():
    gain = 1.0
    rem = np.full(N, gain)
    expl = np.full(N, 100.0)  # net explained = 25200; remainder sum = 252
    a = absolute_cap_bound(rem, CAPITAL, BPY, 10.0)
    # 252 / 100000 * 1e4 = 25.2 bp over exactly one year
    assert a.annualised_bp == pytest.approx(25.2)
    assert a.breached is True
    d = drift_bound(rem, 3, 0.05)
    assert d.significant is True and d.p_value == 0.0
    s = share_bound(rem, expl, 0.5)  # cap sized far larger than the gain
    assert s.breached is False


def test_none_caps_are_not_evaluated_but_measure():
    rem = np.full(N, 1.0)
    expl = np.full(N, 100.0)
    a = absolute_cap_bound(rem, CAPITAL, BPY, None)
    s = share_bound(rem, expl, None)
    assert a.status == NOT_EVALUATED and a.breached is None
    assert a.annualised_bp == pytest.approx(25.2)
    assert s.status == NOT_EVALUATED and s.breached is None
    assert s.share == pytest.approx(252.0 / 25200.0)


def test_zero_net_explained_is_undefined():
    rem = np.full(4, 1.0)
    expl = np.array([1.0, -1.0, 2.0, -2.0])
    s = share_bound(rem, expl, 0.1)
    assert s.status == UNDEFINED and s.breached is None and s.share is None
    # undefined wins even when the cap is None
    assert share_bound(rem, expl, None).status == UNDEFINED


def test_share_breach_and_pass():
    rem = np.full(10, 1.0)
    expl = np.full(10, 10.0)  # share = 0.1
    assert share_bound(rem, expl, 0.05).breached is True
    assert share_bound(rem, expl, 0.1).breached is False


def test_absolute_cap_is_net_over_all_bars():
    rem = np.array([1.0, -5.0, 1.0, -5.0])
    a = absolute_cap_bound(rem, 1000.0, 4.0, 1.0)
    # net sum = -8 ; -8/1000*1e4 = -80 bp over one year (4 bars); a net loss does not breach
    assert a.annualised_bp == pytest.approx(-80.0)
    assert a.breached is False
    up = absolute_cap_bound(np.array([5.0, -1.0, 5.0, -1.0]), 1000.0, 4.0, 1.0)
    assert up.annualised_bp == pytest.approx(80.0)  # net +8
    assert up.breached is True


@pytest.mark.parametrize("noise_bp", [1.0, 5.0])
def test_zero_mean_symmetric_noise_is_within_the_cap(noise_bp):
    # Exactly symmetric noise (each draw paired with its negation) cancels in the net sum.
    half = np.random.default_rng(3).normal(0.0, noise_bp * 1e-4 * CAPITAL, N // 2)
    rem = np.concatenate([half, -half])
    a = absolute_cap_bound(rem, CAPITAL, BPY, 25.0)
    assert abs(a.annualised_bp) < 1e-9 and a.breached is False
    # The positive-parts-only measure would have been far above the cap.
    assert np.clip(rem, 0, None).sum() / CAPITAL * 1e4 > 25.0


def test_absolute_cap_equals_net_mean_annualised():
    rem = np.random.default_rng(5).normal(0.0, 1e-4 * CAPITAL, N)
    a = absolute_cap_bound(rem, CAPITAL, BPY, 25.0)
    assert a.annualised_bp == pytest.approx(rem.mean() / CAPITAL * 1e4 * BPY)


def test_steady_positive_gain_still_breaches_the_cap():
    rem = np.random.default_rng(1).normal(0.0, 5e-4 * CAPITAL, N) + 1e-4 * CAPITAL  # 1 bp/bar gain
    assert absolute_cap_bound(rem, CAPITAL, BPY, 25.0).breached is True


def test_flat_all_zero_remainder_with_zero_net_explained_is_clean():
    s = share_bound(np.zeros(4), np.zeros(4), 0.02)
    assert (s.status, s.breached, s.share) == (EVALUATED, False, 0.0)
    assert share_bound(np.zeros(4), np.array([1.0, -1.0, 2.0, -2.0]), 0.02).breached is False


def test_newey_west_lag_is_required():
    with pytest.raises(TypeError):
        drift_bound(np.ones(10), alpha=0.05)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        drift_bound(np.ones(10), 2)  # type: ignore[call-arg]


def test_drift_p_value_is_one_sided():
    rem = _rng_noise() - 0.5
    d = drift_bound(rem, 5, 0.05)
    assert d.mean < 0 and d.t_stat < 0
    assert d.p_value > 0.95 and d.significant is False
    up = drift_bound(-rem, 5, 0.05)
    assert up.p_value < 0.05 and up.significant is True


def test_newey_west_widens_se_under_positive_autocorrelation():
    e = _rng_noise(500, seed=3)
    x = np.zeros(500)
    for i in range(1, 500):
        x[i] = 0.8 * x[i - 1] + e[i]
    se0 = drift_bound(x, 0, 0.05).std_error
    se10 = drift_bound(x, 10, 0.05).std_error
    assert se10 > se0


def test_lag_zero_matches_plain_standard_error():
    x = _rng_noise(100, seed=11)
    d = drift_bound(x, 0, 0.05)
    expected = np.sqrt(np.mean((x - x.mean()) ** 2) / 100)
    assert d.std_error == pytest.approx(expected)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_bars_per_year_must_be_positive_finite(bad):
    with pytest.raises(ValueError, match="bars_per_year"):
        absolute_cap_bound(np.ones(4), CAPITAL, bad, 1.0)


def test_capital_must_be_positive():
    with pytest.raises(ValueError, match="capital_base"):
        absolute_cap_bound(np.ones(4), 0.0, BPY, 1.0)


def test_length_mismatch_and_non_finite_and_empty():
    with pytest.raises(ValueError, match="equal length"):
        share_bound(np.ones(3), np.ones(4), 0.1)
    with pytest.raises(ValueError, match="finite"):
        absolute_cap_bound(np.array([1.0, np.nan]), CAPITAL, BPY, 1.0)
    with pytest.raises(ValueError, match="finite"):
        share_bound(np.ones(2), np.array([1.0, np.inf]), 0.1)
    with pytest.raises(ValueError, match="non-empty"):
        absolute_cap_bound(np.array([]), CAPITAL, BPY, 1.0)


def test_parameter_validation():
    with pytest.raises(ValueError, match="alpha"):
        drift_bound(np.ones(5), 1, 1.5)
    with pytest.raises(ValueError, match="lag"):
        drift_bound(np.ones(5), -1, 0.05)
    with pytest.raises(ValueError, match="lag"):
        drift_bound(np.ones(5), 5, 0.05)
    with pytest.raises(ValueError, match="cap"):
        absolute_cap_bound(np.ones(5), CAPITAL, BPY, -1.0)
    with pytest.raises(ValueError, match="max_unexplained_share"):
        share_bound(np.ones(5), np.ones(5), -0.1)
