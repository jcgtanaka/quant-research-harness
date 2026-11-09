"""Noise-scaled limits of the pnl.linear bounds (synthetic data only).

The limits scale with the noise measured in the strategy's own remainder; K is a design constant."""

from __future__ import annotations

import numpy as np
import pytest

from qrh.checks.pnl_linear import evaluate_pnl_linear  # noqa: F401  (import must keep working)
from qrh.checks.pnl_linear_bounds import (
    EVALUATED,
    NOISE_SCALED,
    absolute_cap_bound,
    share_bound,
)
from qrh.checks.pnl_linear_config import NOISE_SCALE_K, validate_reconstruction

BPY = 252.0
CAPITAL = 100_000.0
N = 250
SEEDS = range(100)
EXPLAINED = np.full(N, 100.0)


def _noise(noise_bp, seed):
    return np.random.default_rng([seed, 99]).normal(0.0, noise_bp * 1e-4 * CAPITAL, N)


def _limits_hold(rem):
    a = absolute_cap_bound(rem, CAPITAL, BPY, NOISE_SCALED)
    s = share_bound(rem, EXPLAINED, NOISE_SCALED)
    assert a.status == EVALUATED and s.status == EVALUATED
    return not a.breached and not s.breached


@pytest.mark.parametrize("noise_bp", [1.0, 5.0])
def test_honest_zero_mean_noise_passes_on_at_least_95_percent_of_seeds(noise_bp):
    ok = sum(_limits_hold(_noise(noise_bp, s)) for s in SEEDS)
    assert ok >= 95, ok


def test_steady_hidden_gain_of_one_bp_fails_on_at_least_95_percent_of_seeds():
    gain = 1.0 * 1e-4 * CAPITAL
    failed = sum(not _limits_hold(_noise(1.0, s) + gain) for s in SEEDS)
    assert failed >= 95, failed


def test_k_is_a_named_constant_equal_to_three():
    assert NOISE_SCALE_K == 3


def test_flat_zero_remainder_is_clean():
    rem = np.zeros(N)
    assert _limits_hold(rem)
    a = absolute_cap_bound(rem, CAPITAL, BPY, NOISE_SCALED)
    assert a.annualised_bp == 0.0 and a.cap_bp_per_year == 0.0


def test_scaled_limit_formulas():
    rem = _noise(5.0, 3)
    sigma = float(np.std(rem))
    a = absolute_cap_bound(rem, CAPITAL, BPY, NOISE_SCALED)
    assert a.cap_bp_per_year == pytest.approx(3 * sigma / CAPITAL * 1e4 * np.sqrt(BPY))
    s = share_bound(rem, EXPLAINED, NOISE_SCALED)
    assert s.max_unexplained_share == pytest.approx(3 * sigma * np.sqrt(N) / abs(EXPLAINED.sum()))


def test_numeric_override_still_wins():
    rem = _noise(5.0, 3)
    assert absolute_cap_bound(rem, CAPITAL, BPY, 123.0).cap_bp_per_year == 123.0
    assert share_bound(rem, EXPLAINED, 0.5).max_unexplained_share == 0.5


def test_explicit_none_is_still_pending_and_not_evaluated():
    cfg = validate_reconstruction(_block(max_unexplained_bp_per_year=None))
    assert cfg.pending == ["max_unexplained_bp_per_year"]
    assert absolute_cap_bound(_noise(1.0, 1), CAPITAL, BPY, None).status == "not_evaluated"
    assert share_bound(_noise(1.0, 1), EXPLAINED, None).status == "not_evaluated"


def test_absent_caps_default_to_noise_scaled():
    cfg = validate_reconstruction(_block())
    assert cfg.max_unexplained_bp_per_year == NOISE_SCALED
    assert cfg.max_unexplained_share == NOISE_SCALED
    assert cfg.pending == []


def _block(**over):
    base = {
        "position_units": "quantity",
        "cost_model": {"commission_per_fill": 1.0},
        "asset_class": "equity",
        "min_cost_per_turnover": 0.0,
        "publication_column": "pub",
        "max_staleness": 2,
        "adjustment": {"close": "total_return"},
    }
    return {**base, **over}
