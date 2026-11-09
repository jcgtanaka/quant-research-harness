"""Minimum track record length (Bailey and Lopez de Prado, 2012); synthetic numbers only."""

from __future__ import annotations

import math
from statistics import NormalDist

import pytest

from qrh.validation.min_track_record import min_track_record_length, probabilistic_sharpe


def test_psr_is_half_when_sharpe_equals_benchmark():
    assert probabilistic_sharpe(0.1, 0.1, 100, 0.0, 3.0) == pytest.approx(0.5)


def test_psr_increases_with_n_periods():
    values = [probabilistic_sharpe(0.1, 0.0, n, 0.0, 3.0) for n in (10, 50, 200, 1000)]
    assert values == sorted(values)
    assert len(set(values)) == 4
    assert all(0.0 <= v <= 1.0 for v in values)


def test_min_trl_is_the_smallest_period_count():
    sr, bench, skew, kurt, conf = 0.08, 0.02, -0.3, 4.5, 0.9
    t = min_track_record_length(sr, bench, skew, kurt, conf)
    assert probabilistic_sharpe(sr, bench, t, skew, kurt) >= conf
    assert probabilistic_sharpe(sr, bench, t - 1, skew, kurt) < conf


def test_sr_not_above_benchmark_raises():
    with pytest.raises(ValueError):
        min_track_record_length(0.05, 0.05, 0.0, 3.0, 0.95)
    with pytest.raises(ValueError):
        min_track_record_length(0.01, 0.05, 0.0, 3.0, 0.95)


@pytest.mark.parametrize("conf", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_invalid_confidence_raises(conf):
    with pytest.raises(ValueError):
        min_track_record_length(0.1, 0.0, 0.0, 3.0, conf)


def test_non_finite_inputs_raise():
    with pytest.raises(ValueError):
        probabilistic_sharpe(float("nan"), 0.0, 50, 0.0, 3.0)
    with pytest.raises(ValueError):
        min_track_record_length(0.1, 0.0, float("inf"), 3.0, 0.95)


def test_non_positive_variance_term_raises():
    # 1 - g3*SR + (g4-1)/4*SR^2 = 1 - 5*1 + 0 < 0
    with pytest.raises(ValueError, match="variance"):
        probabilistic_sharpe(1.0, 0.0, 50, 5.0, 1.0)


def test_worked_example_four_decimals():
    # Normal returns (skew 0, kurtosis 3), per-period SR 0.1, benchmark 0, 95% confidence.
    sr, conf = 0.1, 0.95
    radicand = 1 + (3 - 1) / 4 * sr**2  # 1.005
    z = NormalDist().inv_cdf(conf)
    exact = 1 + radicand * (z / sr) ** 2
    assert round(exact, 4) == pytest.approx(272.9, abs=0.05)
    assert min_track_record_length(sr, 0.0, 0.0, 3.0, conf) == math.ceil(exact) == 273
    expected_psr = NormalDist().cdf(sr * math.sqrt(50 - 1) / math.sqrt(radicand))
    assert probabilistic_sharpe(sr, 0.0, 50, 0.0, 3.0) == pytest.approx(expected_psr, abs=5e-5)
    assert round(expected_psr, 4) == round(probabilistic_sharpe(sr, 0.0, 50, 0.0, 3.0), 4)


def test_negative_skew_fat_tails_returns_plain_int():
    t = min_track_record_length(0.15, 0.05, -1.5, 9.0, 0.95)
    assert type(t) is int
    assert t > 2
