"""Slope limit of the pnl.linear bounds (synthetic data only).

Regress reported PnL on explained PnL: reported = a + b x explained + e. An honest run has b near 1; a scale error
(for example a multiplier at half its true value) moves b away from 1 without adding noise-free profit.
The limit is |b - 1| <= K x SE(b), K = NOISE_SCALE_K, SE from the Newey-West estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qrh.checks.pnl_linear_bounds import (
    EVALUATED,
    NOISE_SCALE_K,
    UNDEFINED,
    SlopeResult,
    absolute_cap_bound,
    drift_bound,
    share_bound,
    slope_bound,
)

N = 250
LAG = 5
CAPITAL = 1_000_000.0
SEEDS = range(100)
NOISE = 1e-4 * CAPITAL  # 1 bp of capital per bar
EXPLAINED_SD = 2000.0


def _explained(seed):
    return np.random.default_rng([seed, 1]).normal(0.0, EXPLAINED_SD, N)


def _noise(seed):
    return np.random.default_rng([seed, 2]).normal(0.0, NOISE, N)


def test_honest_run_passes_on_at_least_95_of_100_seeds():
    ok = 0
    for s in SEEDS:
        e = _explained(s)
        r = slope_bound(e + _noise(s), e, LAG)
        assert r.status == EVALUATED
        ok += not r.breached
    assert ok >= 95, ok


@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_scale_error_fails_on_at_least_95_of_100_seeds(factor):
    # factor 0.5: reported is half of explained; factor 2.0: explained is half of the true PnL (half multiplier).
    failed = 0
    for s in SEEDS:
        e = _explained(s)
        r = slope_bound(factor * e + _noise(s), e, LAG)
        assert r.status == EVALUATED
        failed += bool(r.breached)
    assert failed >= 95, failed


def test_zero_variance_explained_is_undefined_not_a_pass():
    r = slope_bound(_noise(1), np.full(N, 100.0), LAG)
    assert r.status == UNDEFINED
    assert r.breached is None
    assert slope_bound(np.zeros(N), np.zeros(N), LAG).status == UNDEFINED


def test_slope_formula_and_newey_west_standard_error():
    e = _explained(3)
    rep = 0.9 * e + 40.0 + _noise(3)
    r = slope_bound(rep, e, LAG)
    xc = e - e.mean()
    b = float(xc @ (rep - rep.mean()) / (xc @ xc))
    a = float(rep.mean() - b * e.mean())
    u = rep - a - b * e
    s = xc * u
    lrv = float(s @ s) / N
    for k in range(1, LAG + 1):
        lrv += 2.0 * (1.0 - k / (LAG + 1.0)) * float(s[k:] @ s[:-k]) / N
    se = math.sqrt(lrv / N) / float(np.mean(xc**2))
    assert r.slope == pytest.approx(b)
    assert r.intercept == pytest.approx(a)
    assert r.std_error == pytest.approx(se)
    assert r.limit == pytest.approx(NOISE_SCALE_K * se)
    assert r.lag == LAG
    assert isinstance(r, SlopeResult)


def test_perfect_reconstruction_is_not_a_breach():
    e = _explained(4)
    r = slope_bound(e.copy(), e, LAG)
    assert r.status == EVALUATED and r.breached is False


def test_input_validation():
    with pytest.raises(ValueError):
        slope_bound(np.zeros(5), np.zeros(6), 1)
    with pytest.raises(ValueError):
        slope_bound(np.zeros(5), np.zeros(5), 5)


def test_existing_bounds_are_unchanged():
    rem = _noise(7)
    e = np.full(N, 100.0)
    a = absolute_cap_bound(rem, CAPITAL, 252.0, 123.0)
    assert a.cap_bp_per_year == 123.0 and a.status == EVALUATED
    assert share_bound(rem, e, 0.5).max_unexplained_share == 0.5
    d = drift_bound(rem, LAG, 0.05)
    assert d.lag == LAG and d.status == EVALUATED
