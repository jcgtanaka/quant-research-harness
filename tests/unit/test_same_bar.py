"""The default same-bar execution check: reported returns must not use the position of the bar they earn."""

import numpy as np
import pandas as pd

from lookahead_helpers import adapter, prices, window
from qrh.adapters.base import DataView, RunResult
from qrh.checks.same_bar import MAX_CANDIDATES, check_same_bar_execution

PARAMS = {"window": 20, "k": 2.0}


def run(variant, df=None, k=2.0):
    df = prices() if df is None else df
    s, e = window(df)
    return df, adapter(variant).run({**PARAMS, "k": k}, DataView(df), s, e)


def test_a_decide_and_fill_on_the_same_close_strategy_fails_without_any_declaration():
    df, res = run("sameday")
    out = check_same_bar_execution(res, df, None)
    assert out.verdict == "FAIL", (out.reasons, out.details)
    assert "same-bar execution" in " ".join(out.reasons)
    assert out.details["inferred_price_column"] == "close"


def test_a_lagged_strategy_passes_without_any_declaration():
    df, res = run("clean")
    out = check_same_bar_execution(res, df, None)
    assert out.verdict == "PASS", (out.reasons, out.details)
    assert out.details["inferred_price_column"] == "close"


def test_the_declared_execution_is_used_when_given():
    df, res = run("sameday")
    out = check_same_bar_execution(res, df, {"lag_bars": 1, "price_column": "close"})
    assert out.verdict == "FAIL" and "inferred_price_column" not in out.details


def test_the_price_column_is_inferred_among_several_columns():
    df = prices()
    df["volume"] = np.random.default_rng(0).uniform(1e3, 5e3, len(df))
    df["spread"] = np.random.default_rng(1).uniform(0.01, 0.05, len(df))
    _, res = run("sameday", df)
    out = check_same_bar_execution(res, df, None)
    assert out.verdict == "FAIL" and out.details["inferred_price_column"] == "close"
    _, ok = run("clean", df)
    assert check_same_bar_execution(ok, df, None).verdict == "PASS"


def test_no_positions_is_blocked_with_a_plain_reason_never_passed():
    df, res = run("clean")
    for positions in (None, pd.DataFrame()):
        out = check_same_bar_execution(RunResult(res.returns, res.trades, positions), df, None)
        assert out.verdict == "BLOCKED"
        assert "positions" in " ".join(out.reasons) and "same-bar" in " ".join(out.reasons)


def test_returns_that_no_price_column_explains_are_inconclusive_not_a_pass():
    df, res = run("clean")
    noisy = RunResult(pd.Series(np.random.default_rng(0).normal(0, 0.01, len(res.returns)), index=res.returns.index),
                      res.trades, res.positions)
    out = check_same_bar_execution(noisy, df, None)
    assert out.verdict == "INCONCLUSIVE" and "cannot verify" in " ".join(out.reasons)


def test_positions_that_cannot_be_matched_to_a_price_column_are_blocked_and_say_how_to_fix_it():
    df, res = run("clean")
    wide = RunResult(res.returns, res.trades, pd.DataFrame({"a": res.positions["weight"], "b": res.positions["weight"]}))
    out = check_same_bar_execution(wide, df, None)
    assert out.verdict == "BLOCKED" and "execution" in " ".join(out.reasons)


def test_several_position_columns_named_like_price_columns_are_reconstructed_together():
    rng = np.random.default_rng(4)
    idx = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
    df = pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))) for c in ("aa", "bb")}, index=idx)
    pos = np.sign(df.pct_change().rolling(3).mean()).fillna(0.0) * 0.5
    ret = df.pct_change()
    lagged = (pos.shift(1) * ret).sum(axis=1).fillna(0.0)
    same = (pos * ret).sum(axis=1).fillna(0.0)
    assert check_same_bar_execution(RunResult(lagged, pd.DataFrame(), pos), df, None).verdict == "PASS"
    assert check_same_bar_execution(RunResult(same, pd.DataFrame(), pos), df, None).verdict == "FAIL"


def test_only_a_bounded_number_of_columns_is_tried():
    df, res = run("clean")
    for i in range(MAX_CANDIDATES + 10):
        df[f"x{i}"] = df["close"] * (1 + i)
    out = check_same_bar_execution(res, df, None)
    assert len(out.details["candidates_tried"]) <= MAX_CANDIDATES
