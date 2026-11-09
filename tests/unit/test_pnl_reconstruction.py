import numpy as np
import pandas as pd
import pytest

from qrh.adapters.base import RunResult
from qrh.checks.pnl_reconstruction import check_pnl_reconstruction, validate_execution

EXEC = {"lag_bars": 1, "price_column": "close"}


def frame(n=300, seed=1, cols=("close",)):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC", name="timestamp")
    return pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for c in cols}, index=idx)


def positions(df, kind="trend", seed=0):
    px = df[df.columns[0]]
    if kind == "sparse":
        return np.sign(px.rolling(10).mean() - px.rolling(60).mean()).fillna(0.0)
    return np.sign(px.pct_change().rolling(3).mean()).fillna(0.0)


def run_result(df, lag=1, scale=1.0, cost=0.0, kind="trend", col="close", window=60):
    ret = df[col].pct_change()
    pos = positions(df, kind)
    turnover = pos.diff().abs().fillna(pos.abs())
    gross = pos.shift(lag) * ret if lag else pos * ret
    net = (gross - cost * turnover.shift(lag if lag else 0)).fillna(0.0) * scale
    w = slice(df.index[window], None)
    return RunResult(net.loc[w], pd.DataFrame(), pos.to_frame("weight").loc[w])


def test_validate_execution_requires_a_lag_of_at_least_one_bar():
    validate_execution(EXEC)
    validate_execution({"lag_bars": 2, "price_columns": ["a", "b"], "max_cost_per_turnover": 0.0005})
    for bad in (None, {}, {"price_column": "close"}, {"lag_bars": 0, "price_column": "close"},
                {"lag_bars": 1.0, "price_column": "close"}, {"lag_bars": True, "price_column": "close"},
                {"lag_bars": 1}, {"lag_bars": 1, "price_column": "close", "price_columns": ["close"]},
                {"lag_bars": 1, "price_column": "close", "bogus": 1},
                {"lag_bars": 1, "price_column": "close", "max_cost_per_turnover": -1},
                {"lag_bars": 1, "price_columns": []}):
        with pytest.raises(ValueError):
            validate_execution(bad)


def test_a_lagged_strategy_passes_with_the_reconstruction_matching_exactly():
    df = frame()
    r = check_pnl_reconstruction(run_result(df), df, EXEC)
    assert r.verdict == "PASS", r.reasons
    assert r.details["match_lagged"] >= 0.99 and r.details["corr_lagged"] > 0.999


def test_costs_below_the_declared_bound_are_consistent_but_larger_gaps_are_not():
    df = frame()
    assert check_pnl_reconstruction(run_result(df, cost=0.0005), df, EXEC).verdict == "PASS"
    r = check_pnl_reconstruction(run_result(df, cost=0.01), df, EXEC)
    assert r.verdict == "INCONCLUSIVE" and "figures" in " ".join(r.reasons)
    wide = {**EXEC, "max_cost_per_turnover": 0.02}
    assert check_pnl_reconstruction(run_result(df, cost=0.01), df, wide).verdict == "PASS"


def test_returns_better_than_the_gross_reconstruction_are_never_explained_by_costs():
    df = frame()
    res = run_result(df)
    boosted = RunResult(res.returns + 0.002 * res.positions["weight"].abs(), res.trades, res.positions)
    assert check_pnl_reconstruction(boosted, df, EXEC).verdict == "INCONCLUSIVE"


@pytest.mark.parametrize("kind", ["trend", "sparse"])
def test_same_bar_execution_fails(kind):
    df = frame(400)
    r = check_pnl_reconstruction(run_result(df, lag=0, kind=kind), df, EXEC)
    assert r.verdict == "FAIL" and "same-bar execution (look-ahead)" in " ".join(r.reasons)
    assert r.details["match_same_bar"] > r.details["match_lagged"]


def test_a_shorter_lag_than_declared_fails():
    df = frame(400)
    r = check_pnl_reconstruction(run_result(df, lag=1), df, {**EXEC, "lag_bars": 2})
    assert r.verdict == "FAIL" and "lag" in " ".join(r.reasons) and "1" in " ".join(r.reasons)


@pytest.mark.parametrize("scale", [100.0, 10.0, 1000.0, 0.01])
def test_a_power_of_ten_scale_is_a_units_mismatch(scale):
    df = frame()
    r = check_pnl_reconstruction(run_result(df, scale=scale), df, EXEC)
    assert r.verdict == "FAIL" and "units mismatch" in " ".join(r.reasons)
    assert r.details["scale_exponent"] == int(round(np.log10(scale)))


def test_percent_units_with_costs_are_still_a_units_mismatch():
    df = frame()
    assert check_pnl_reconstruction(run_result(df, scale=100.0, cost=0.0005), df, EXEC).verdict == "FAIL"


def test_a_non_power_of_ten_scale_is_inconclusive_not_a_pass():
    df = frame()
    r = check_pnl_reconstruction(run_result(df, scale=3.0), df, EXEC)
    assert r.verdict == "INCONCLUSIVE"


def test_unrelated_returns_are_inconclusive_with_the_figures():
    df = frame()
    res = run_result(df)
    noise = pd.Series(np.random.default_rng(9).normal(0, 0.01, len(res.returns)), index=res.returns.index)
    r = check_pnl_reconstruction(RunResult(noise, res.trades, res.positions), df, EXEC)
    assert r.verdict == "INCONCLUSIVE"
    assert {"match_lagged", "match_same_bar", "corr_lagged", "corr_same_bar", "rank_corr_lagged", "bars"} <= set(r.details)


def test_a_strategy_that_never_holds_a_position_cannot_be_verified():
    df = frame()
    res = run_result(df)
    flat = RunResult(res.returns * 0.0, res.trades, res.positions * 0.0)
    r = check_pnl_reconstruction(flat, df, EXEC)
    assert r.verdict == "INCONCLUSIVE" and "no active" in " ".join(r.reasons)


def test_positions_that_peek_are_not_this_methods_business():
    df = frame()
    ret = df["close"].pct_change()
    pos = np.sign(ret.shift(-1)).fillna(0.0)  # look-ahead in the signal, execution lag respected
    net = (pos.shift(1) * ret).fillna(0.0)
    w = slice(df.index[60], None)
    r = check_pnl_reconstruction(RunResult(net.loc[w], pd.DataFrame(), pos.to_frame("weight").loc[w]), df, EXEC)
    assert r.verdict == "PASS"  # truncation and lint own this class


def test_multi_asset_positions_are_aligned_to_price_columns():
    df = frame(300, cols=("a", "b"))
    ret = df.pct_change()
    pos = pd.DataFrame({c: np.sign(ret[c].rolling(3).mean()) for c in ("a", "b")}).fillna(0.0) * 0.5
    net = (pos.shift(1) * ret).sum(axis=1).fillna(0.0)
    w = slice(df.index[60], None)
    ex = {"lag_bars": 1, "price_columns": ["a", "b"]}
    ok = RunResult(net.loc[w], pd.DataFrame(), pos.loc[w])
    assert check_pnl_reconstruction(ok, df, ex).verdict == "PASS"
    same = (pos * ret).sum(axis=1).fillna(0.0)
    assert check_pnl_reconstruction(RunResult(same.loc[w], pd.DataFrame(), pos.loc[w]), df, ex).verdict == "FAIL"
    swapped = pos.rename(columns={"a": "b", "b": "a"})
    # same names, different order of columns is fine; unmatched names are not
    assert check_pnl_reconstruction(RunResult(net.loc[w], pd.DataFrame(), pos[["b", "a"]].loc[w]), df, ex).verdict == "PASS"
    assert check_pnl_reconstruction(RunResult(net.loc[w], pd.DataFrame(), pos.rename(columns={"a": "x"}).loc[w]), df, ex).verdict == "BLOCKED"
    assert swapped is not None


@pytest.mark.parametrize("mutate,text", [
    (lambda r, df: (RunResult(r.returns, r.trades, None), df), "positions"),
    (lambda r, df: (r, df.drop(columns="close")), "price column"),
    (lambda r, df: (RunResult(r.returns, r.trades, pd.DataFrame({"a": 1.0, "b": 1.0}, index=r.returns.index)), df), "align"),
    (lambda r, df: (RunResult(r.returns.iloc[:3], r.trades, r.positions.iloc[:3]), df), "bars"),
])
def test_what_cannot_be_checked_is_blocked(mutate, text):
    df = frame()
    res, df2 = mutate(run_result(df), df)
    r = check_pnl_reconstruction(res, df2, EXEC)
    assert r.verdict == "BLOCKED" and text in " ".join(r.reasons)


def test_invalid_execution_spec_is_blocked_by_the_check_too():
    df = frame()
    r = check_pnl_reconstruction(run_result(df), df, {"price_column": "close"})
    assert r.verdict == "BLOCKED" and "lag_bars" in " ".join(r.reasons)
