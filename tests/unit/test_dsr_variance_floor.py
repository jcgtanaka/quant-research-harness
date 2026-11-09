"""The deflated Sharpe luck bar cannot be shrunk by near-duplicate variants.

V is the larger of the cross-variant variance of in-sample Sharpe ratios and the sampling variance of the selected
variant's out-of-sample Sharpe estimator, for every grid size.
"""

import numpy as np
import pandas as pd
import pytest

from qrh.validation.deflated_sharpe import deflated_sharpe, expected_max_sharpe
from qrh.validation.overfit import estimator_sharpe_variance, overfitting_statistics
from qrh.validation.pbo_cscv import pbo_cscv

FAMILY = 50  # many variants were tried before, so the luck bar is high
MIN_DSR = 0.95


def oos_series(sharpe_per_period=0.085, n=1000, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(sharpe_per_period * 0.01, 0.01, n)


def stats(matrix, oos, family=FAMILY):
    return overfitting_statistics(matrix, oos, family, 16, pbo_cscv, deflated_sharpe)


def one_strategy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(0.0003, 0.01, n)})


def with_near_copy(matrix):
    """The same rule with 0.2 basis points more cost per period: its Sharpe ratio is almost identical."""
    twin = matrix["a"] - 0.00002
    return pd.concat([matrix, twin.rename("b")], axis=1)


def test_a_near_copy_does_not_change_the_dsr_verdict():
    oos = oos_series()
    alone = stats(one_strategy(), oos)
    twin = stats(with_near_copy(one_strategy()), oos)
    assert (alone.dsr.dsr >= MIN_DSR) == (twin.dsr.dsr >= MIN_DSR)
    assert twin.dsr.dsr == pytest.approx(alone.dsr.dsr, abs=1e-9)
    assert twin.var_sharpe == pytest.approx(alone.var_sharpe)
    assert twin.var_sharpe_source == "estimator"  # the twin's cross-variant variance is far below the sampling variance


def test_the_scenario_is_a_real_discriminator_the_old_rule_let_the_twin_through():
    oos = oos_series()
    twin_matrix = with_near_copy(one_strategy())
    old_v = float(np.var([c.mean() / c.std(ddof=1) for _, c in twin_matrix.items()], ddof=1))
    old = deflated_sharpe(oos, n_trials=FAMILY, var_sharpe=old_v)
    new = stats(twin_matrix, oos)
    assert old.dsr >= MIN_DSR > new.dsr.dsr  # the cross-variant variance alone would have passed it


def test_both_variances_and_the_one_used_are_recorded():
    oos = oos_series()
    out = stats(with_near_copy(one_strategy()), oos)
    assert out.var_sharpe_estimator == pytest.approx(estimator_sharpe_variance(oos))
    assert out.var_sharpe_grid is not None and out.var_sharpe_grid < out.var_sharpe_estimator
    assert out.var_sharpe == pytest.approx(max(out.var_sharpe_grid, out.var_sharpe_estimator))


def test_a_single_combination_has_no_grid_variance():
    out = stats(one_strategy(), oos_series())
    assert out.var_sharpe_grid is None and out.var_sharpe_source == "estimator"
    assert out.var_sharpe == pytest.approx(out.var_sharpe_estimator)


def wide_grid(n=400, seed=1):
    rng = np.random.default_rng(seed)
    means = [-0.0015, -0.0005, 0.0005, 0.0015, 0.0025]
    return pd.DataFrame({f"c{i}": rng.normal(m, 0.01, n) for i, m in enumerate(means)})


def test_a_grid_whose_variance_already_exceeds_the_sampling_variance_is_unchanged():
    oos = oos_series()
    m = wide_grid()
    out = stats(m, oos)
    per_period = [c.mean() / c.std(ddof=1) for _, c in m.items()]
    expected = float(np.var(per_period, ddof=1))
    assert expected > estimator_sharpe_variance(oos)
    assert out.var_sharpe_source == "grid"
    assert out.var_sharpe == pytest.approx(expected)
    assert out.var_sharpe_grid == pytest.approx(expected)
    old = deflated_sharpe(oos, n_trials=FAMILY, var_sharpe=expected)
    assert out.dsr.dsr == pytest.approx(old.dsr, abs=1e-12) and out.dsr.sr0 == pytest.approx(old.sr0)


def test_the_floor_never_lowers_the_luck_bar():
    oos = oos_series()
    for matrix in (one_strategy(), with_near_copy(one_strategy()), wide_grid()):
        out = stats(matrix, oos)
        # an independent computation of the luck bar from the estimator variance alone: the bar actually used cannot be lower
        floor_bar = expected_max_sharpe(FAMILY, estimator_sharpe_variance(oos))
        assert out.dsr.sr0 >= floor_bar - 1e-15
        assert out.dsr.dsr <= deflated_sharpe(oos, FAMILY, estimator_sharpe_variance(oos)).dsr + 1e-12
