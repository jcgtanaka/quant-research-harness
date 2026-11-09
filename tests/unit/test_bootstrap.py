import math

import numpy as np
import pytest

from qrh.validation.bootstrap import (
    bootstrap_ci,
    default_block_length,
    stationary_bootstrap_indices,
)
from qrh.validation.metrics import annualized_sharpe, cagr, max_drawdown

PPY = 252.0


def series(seed, t=500, mu=0.0008, sd=0.01):
    return np.random.default_rng(seed).normal(mu, sd, t)


def test_default_block_length_is_ceil_cube_root():
    assert default_block_length(1000) == 10
    assert default_block_length(1001) == 11
    assert default_block_length(1) == 1


def test_indices_shape_range_and_determinism():
    a = stationary_bootstrap_indices(200, 50, 5.0, np.random.default_rng(1))
    b = stationary_bootstrap_indices(200, 50, 5.0, np.random.default_rng(1))
    c = stationary_bootstrap_indices(200, 50, 5.0, np.random.default_rng(2))
    assert a.shape == (50, 200) and a.min() >= 0 and a.max() < 200
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_block_length_one_is_iid_resampling():
    n, reps = 50, 4000
    idx = stationary_bootstrap_indices(n, reps, 1.0, np.random.default_rng(3))
    # successor of index i is i+1 only by chance: probability 1/n (iid), not ~1
    follows = (idx[:, 1:] == (idx[:, :-1] + 1) % n).mean()
    assert abs(follows - 1 / n) < 0.004
    # marginal is uniform: chi-square statistic (n-1 dof, mean 49, sd ~9.9) far below 49 + 6 sd
    counts = np.bincount(idx.ravel(), minlength=n)
    expected = idx.size / n
    chi2 = ((counts - expected) ** 2 / expected).sum()
    assert chi2 < 49 + 6 * math.sqrt(2 * 49)
    # consecutive draws are independent: correlation of successive indices ~ 0
    assert abs(np.corrcoef(idx[:, :-1].ravel(), idx[:, 1:].ravel())[0, 1]) < 0.02


def test_mean_block_length_is_respected():
    n, reps, mean_len = 400, 2000, 8.0
    idx = stationary_bootstrap_indices(n, reps, mean_len, np.random.default_rng(4))
    cont = (idx[:, 1:] == (idx[:, :-1] + 1) % n).mean()
    # P(continue) = 1 - 1/L, plus the chance 1/L * 1/n that a fresh start lands on the successor
    expected = (1 - 1 / mean_len) + (1 / mean_len) / n
    assert abs(cont - expected) < 0.005


def test_circular_wrap_happens():
    idx = stationary_bootstrap_indices(10, 500, 1e9, np.random.default_rng(5))  # one long block
    assert ((idx[:, 1:] - idx[:, :-1]) % 10 == 1).all()
    assert (idx[:, 1:] < idx[:, :-1]).any()  # wraps from 9 to 0


@pytest.mark.parametrize("bad", [dict(n=0), dict(reps=0), dict(mean_len=0.5)])
def test_indices_validation(bad):
    kw = dict(n=10, reps=5, mean_len=2.0) | bad
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(kw["n"], kw["reps"], kw["mean_len"], np.random.default_rng(0))


def test_ci_deterministic_by_seed():
    r = series(1)
    a = bootstrap_ci(r, PPY, reps=300, seed=7)
    b = bootstrap_ci(r, PPY, reps=300, seed=7)
    c = bootstrap_ci(r, PPY, reps=300, seed=8)
    assert a == b and a != c


def test_ci_contains_point_estimate_and_orders_bounds():
    r = series(2)
    res = bootstrap_ci(r, PPY, reps=1000, seed=0)
    for name, point in (("sharpe", annualized_sharpe(r, PPY)), ("cagr", cagr(r, PPY)), ("max_drawdown", max_drawdown(r))):
        lo, hi = res.intervals[name]
        assert lo <= hi
        assert res.point[name] == pytest.approx(point)
    assert res.intervals["sharpe"][0] <= res.point["sharpe"] <= res.intervals["sharpe"][1]
    assert res.intervals["cagr"][0] <= res.point["cagr"] <= res.intervals["cagr"][1]
    assert res.intervals["max_drawdown"][1] <= 0.0


def test_ci_width_shrinks_with_longer_series():
    widths = []
    for t in (250, 1000, 4000):
        r = series(3, t=t)
        lo, hi = bootstrap_ci(r, PPY, reps=800, seed=1).intervals["sharpe"]
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]
    # roughly 1/sqrt(T): quadrupling T halves the width (loose bounds)
    assert 0.35 < widths[1] / widths[0] < 0.75


def test_ci_coverage_of_mean_for_iid_data_is_near_nominal():
    # 95% CI for the Sharpe of iid N(mu, sd) data should cover the true value ~95% of the time.
    true_sr = 0.0008 / 0.01 * math.sqrt(PPY)
    hit, reps = 0, 100
    for k in range(reps):
        lo, hi = bootstrap_ci(series(100 + k, t=500), PPY, reps=400, seed=k).intervals["sharpe"]
        hit += lo <= true_sr <= hi
    assert 0.85 <= hit / reps <= 1.0, hit / reps


def test_level_changes_width_and_is_recorded():
    r = series(4)
    narrow = bootstrap_ci(r, PPY, reps=500, level=0.5, seed=1)
    wide = bootstrap_ci(r, PPY, reps=500, level=0.95, seed=1)
    nw = narrow.intervals["sharpe"][1] - narrow.intervals["sharpe"][0]
    ww = wide.intervals["sharpe"][1] - wide.intervals["sharpe"][0]
    assert nw < ww and wide.level == 0.95 and wide.seed == 1 and wide.reps == 500
    assert wide.mean_block_length == default_block_length(len(r))


@pytest.mark.parametrize(
    "kw",
    [dict(reps=0), dict(level=1.0), dict(level=0.0), dict(mean_block_length=0.2), dict(ppy=0.0)],
)
def test_ci_validation(kw):
    args = dict(reps=100, level=0.95, mean_block_length=None, ppy=PPY) | kw
    with pytest.raises(ValueError):
        bootstrap_ci(series(1, t=100), args["ppy"], reps=args["reps"], level=args["level"],
                     mean_block_length=args["mean_block_length"], seed=0)


def test_ci_needs_enough_data_and_finite_values():
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([0.01, 0.02]), PPY, reps=100, seed=0)
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([0.01, np.nan, 0.02, 0.0, 0.01]), PPY, reps=100, seed=0)


def test_undefined_sharpe_everywhere_gives_none_interval():
    res = bootstrap_ci(np.zeros(50), PPY, reps=100, seed=0)
    assert res.point["sharpe"] is None and res.intervals["sharpe"] is None


def test_samples_of_every_metric_are_exposed_and_reproduce_the_interval():
    res = bootstrap_ci(series(5), PPY, reps=300, seed=2)
    assert set(res.samples) == {"sharpe", "cagr", "max_drawdown"}
    for name, sample in res.samples.items():
        assert sample.shape == (300,)
        ok = sample[~np.isnan(sample)]
        lo, hi = np.percentile(ok, [2.5, 97.5])
        assert (lo, hi) == pytest.approx(res.intervals[name])
