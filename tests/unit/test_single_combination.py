"""S5 with exactly one combination: no selection happened, so PBO is not applicable and DSR decides."""

import numpy as np
import pandas as pd
import pytest

from qrh.validation.deflated_sharpe import deflated_sharpe
from qrh.validation.overfit import PBO_NOT_APPLICABLE, overfitting_statistics
from qrh.validation.pbo_cscv import pbo_cscv


def series(mean, n=600, seed=1, sd=0.01):
    return np.random.default_rng(seed).normal(mean, sd, n)


def matrix(cols, n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"c{i}": rng.normal(0, 0.01, n) for i in range(cols)})


def stats(m, oos, family=1):
    return overfitting_statistics(m, oos, family, 16, pbo_cscv, deflated_sharpe)


def test_one_column_reports_pbo_as_not_applicable_and_does_not_raise():
    out = stats(matrix(1), series(0.001))
    assert out.pbo is None
    assert out.pbo_note == PBO_NOT_APPLICABLE
    assert PBO_NOT_APPLICABLE == "not applicable (one combination, no selection among variants)"
    assert 0.0 <= out.dsr.dsr <= 1.0 and out.n_trials == 1


def test_two_columns_are_unchanged():
    out = stats(matrix(2), series(0.001))
    assert out.pbo is not None and out.pbo_note == ""


def test_a_clear_edge_passes_the_deflated_sharpe_with_one_combination():
    assert stats(matrix(1), series(0.0015, seed=2)).dsr.dsr >= 0.95


def test_noise_with_one_combination_fails_the_deflated_sharpe():
    results = [stats(matrix(1), series(0.0, seed=s)).dsr.dsr for s in range(30)]
    assert sum(r < 0.95 for r in results) >= 27  # about 95 percent of null series stay below the 0.95 bar


def test_the_trial_count_actually_charged_still_deflates_a_single_combination():
    oos = series(0.0006, seed=5)
    alone, family = stats(matrix(1), oos, family=1), stats(matrix(1), oos, family=500)
    assert alone.n_trials == 1 and family.n_trials == 500
    assert alone.dsr.sr0 == 0.0 and family.dsr.sr0 > 0.0
    assert family.dsr.dsr < alone.dsr.dsr
    assert family.var_sharpe > 0.0


def test_the_sharpe_variance_for_one_combination_comes_from_the_estimator_not_a_grid():
    out = stats(matrix(1), series(0.001, n=800, seed=7))
    assert out.var_sharpe_source == "estimator"
    assert out.var_sharpe == pytest.approx(1.0 / 799, rel=0.2)  # about 1 / (T - 1) for a near-zero Sharpe
    # changed with the variance floor: for a grid, V is max(grid, estimator) and the result says which one was larger
    three = stats(matrix(3), series(0.001))
    assert three.var_sharpe == pytest.approx(max(three.var_sharpe_grid, three.var_sharpe_estimator))
    assert three.var_sharpe_source == ("grid" if three.var_sharpe_grid >= three.var_sharpe_estimator else "estimator")


def test_zero_columns_is_still_an_error_with_a_plain_message():
    with pytest.raises(ValueError, match="at least one combination"):
        stats(pd.DataFrame(index=range(100)), series(0.001))
