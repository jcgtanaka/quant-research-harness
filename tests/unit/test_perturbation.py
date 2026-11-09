"""Future-perturbation invariance: the S2 runtime proof that a run does not read data after a decision time."""

import numpy as np
import pandas as pd
import pytest

from lookahead_helpers import EXPANDING, adapter, prices, window
from qrh.adapters.base import RunResult
from qrh.checks.perturbation import DEFAULT_CUTOFFS, perturb_future, run_perturbation

PARAMS = {"window": 20, "k": 2.0}


def check(variant, df=None, **kw):
    df = prices() if df is None else df
    s, e = window(df)
    return run_perturbation(adapter(variant), df, PARAMS, s, e, **kw)


def test_a_clean_sparse_strategy_passes_at_every_cutoff():
    res = check("clean")
    assert res.verdict == "PASS", res.reasons
    assert DEFAULT_CUTOFFS == 20
    assert [c["kind"] for c in res.details["cutoffs"]].count("spread") == 20  # plus every change bar and the bar before
    assert all(c["status"] == "PASS" for c in res.details["cutoffs"])


@pytest.mark.parametrize("variant", ["roll", "negvar", "fullframe"])
def test_sparse_lookahead_that_the_old_truncation_check_missed_is_caught(variant):
    res = check(variant)
    assert res.verdict == "FAIL", res.details
    text = " ".join(res.reasons)
    assert "cutoff" in text and "first differing timestamp" in text
    assert res.details["first_divergence"]["timestamp"]


def test_the_first_divergent_timestamp_is_at_or_before_the_cutoff():
    res = check("roll")
    bad = [c for c in res.details["cutoffs"] if c["status"] == "FAIL"]
    assert bad
    for c in bad:
        assert pd.Timestamp(c["first_divergence"]) <= pd.Timestamp(c["cutoff"])


def test_a_global_statistic_leak_is_caught():
    df = prices()
    s, e = window(df)
    res = run_perturbation(adapter("globalstd"), df, {"window": 20, "k": 0.3}, s, e)
    assert res.verdict == "FAIL", res.details


def test_expanding_window_statistics_are_not_a_false_positive():
    ns: dict = {}
    exec(compile(EXPANDING, "<expanding>", "exec"), ns)
    df = prices()
    s, e = window(df)
    res = run_perturbation(ns["Expanding"](), df, {"w": 1}, s, e)
    assert res.verdict == "PASS", res.reasons


def test_it_is_deterministic():
    a, b = check("roll"), check("roll")
    assert a.details == b.details and a.reasons == b.reasons
    assert [c["cutoff"] for c in check("clean").details["cutoffs"]] == [c["cutoff"] for c in check("clean").details["cutoffs"]]


def test_the_cutoffs_include_bars_where_the_position_is_nonzero_or_changes():
    df = prices()
    s, e = window(df)
    ad = adapter("clean")
    full = ad.run(PARAMS, __import__("qrh.adapters.base", fromlist=["DataView"]).DataView(df), s, e)
    pos = full.positions["weight"]
    res = run_perturbation(ad, df, PARAMS, s, e, full_result=full)
    chosen = {pd.Timestamp(c["cutoff"]) for c in res.details["cutoffs"]}
    active = set(pos.index[pos.diff().abs().fillna(0) > 0]) | set(pos.index[pos > 0])
    assert active and chosen & active
    assert any(c["kind"] == "change" for c in res.details["cutoffs"])


def test_the_number_of_adapter_runs_is_the_cutoff_count_plus_the_full_run():
    class Counting:
        calls = 0

        def __init__(self):
            self.inner = adapter("clean")

        def parameter_space(self):
            return self.inner.parameter_space()

        def required_data(self):
            return []

        def run(self, params, data, start, end, cost_multiplier=1.0):
            Counting.calls += 1
            return self.inner.run(params, data, start, end)

    df = prices()
    s, e = window(df)
    res = run_perturbation(Counting(), df, PARAMS, s, e, n_cutoffs=7)
    assert res.verdict == "PASS" and Counting.calls == 1 + len(res.details["cutoffs"])
    assert [c["kind"] for c in res.details["cutoffs"]].count("spread") == 7


def test_a_supplied_full_result_is_not_recomputed():
    class Once(type(adapter("clean"))):
        calls = 0

        def run(self, *a, **k):
            Once.calls += 1
            return super().run(*a, **k)

    df = prices()
    s, e = window(df)
    full = adapter("clean").run(PARAMS, __import__("qrh.adapters.base", fromlist=["DataView"]).DataView(df), s, e)
    res = run_perturbation(Once(), df, PARAMS, s, e, n_cutoffs=4, full_result=full)
    assert Once.calls == len(res.details["cutoffs"])  # one run per cutoff, none for the full result


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, None])
def test_an_invalid_cutoff_count_is_blocked(bad):
    res = check("clean", n_cutoffs=bad)
    assert res.verdict == "BLOCKED" and "perturbation_cutoffs" in " ".join(res.reasons)


def test_an_adapter_that_fails_on_the_perturbed_data_is_blocked_not_passed():
    class Fragile(type(adapter("clean"))):
        def run(self, params, data, start, end, cost_multiplier=1.0):
            if data.as_of(end)["close"].pct_change().abs().max() > 0.1:
                raise RuntimeError("returns look wrong")
            return super().run(params, data, start, end)

    df = prices()
    s, e = window(df)
    res = run_perturbation(Fragile(), df, PARAMS, s, e)
    assert res.verdict == "BLOCKED"
    assert "returns look wrong" in " ".join(res.reasons) and "perturbed" in " ".join(res.reasons)


def test_a_random_adapter_is_blocked_as_not_reproducible_rather_than_failed():
    class Random(type(adapter("clean"))):
        def run(self, params, data, start, end, cost_multiplier=1.0):
            r = super().run(params, data, start, end)
            noise = np.random.default_rng().random(len(r.positions)) < 0.3  # unseeded
            pos = r.positions.copy()
            pos["weight"] = np.where(noise, 1.0 - pos["weight"], pos["weight"])
            return RunResult(r.returns, r.trades, pos)

    df = prices()
    s, e = window(df)
    res = run_perturbation(Random(), df, PARAMS, s, e)
    assert res.verdict == "BLOCKED" and "not reproducible" in " ".join(res.reasons)


def test_a_tiny_window_is_blocked():
    df = prices()
    res = run_perturbation(adapter("clean"), df, PARAMS, df.index[0], df.index[2])
    assert res.verdict == "BLOCKED"


def test_reading_a_whole_row_at_its_index_time_under_a_publication_column_is_flagged():
    # the row stamped t is only published at t + 2 days; deciding at t from it is look-ahead by publication lag
    df = prices()
    df["published"] = df.index + pd.Timedelta(days=2)
    s, e = window(df)
    res = run_perturbation(adapter("clean"), df, PARAMS, s, e, publication_column="published")
    assert res.verdict == "FAIL", res.reasons


# --- the perturbation itself -----------------------------------------------------------------------------------


def two_column_frame():
    df = prices(120)
    df["ref_close"] = df["close"] * 1.002
    df["volume"] = np.random.default_rng(1).integers(100, 1000, len(df)).astype(float)
    df["note"] = "x"
    return df


def test_the_perturbation_leaves_the_past_untouched_and_changes_the_future():
    df = two_column_frame()
    cutoff = df.index[60]
    out = perturb_future(df, cutoff, np.random.default_rng(0))
    assert out.index.equals(df.index) and list(out.columns) == list(df.columns)
    pd.testing.assert_frame_equal(out.loc[:cutoff], df.loc[:cutoff])
    assert not np.allclose(out.loc[df.index > cutoff, "close"], df.loc[df.index > cutoff, "close"])
    assert (out["note"] == "x").all()


def test_the_perturbed_data_stays_valid_for_typical_data_rules():
    df = two_column_frame()
    out = perturb_future(df, df.index[60], np.random.default_rng(5))
    assert np.isfinite(out[["close", "ref_close", "volume"]].to_numpy()).all()
    assert (out[["close", "ref_close", "volume"]] > 0).all().all()
    future = out.index > df.index[60]
    ratio = (out["ref_close"] / out["close"])[future]
    assert np.allclose(ratio, 1.002)  # a relationship between columns survives (rows move together)


def test_the_first_future_bar_return_changes_sign():
    df = prices(120)
    cutoff = df.index[60]
    out = perturb_future(df, cutoff, np.random.default_rng(2))
    before = df["close"].iloc[61] / df["close"].iloc[60] - 1
    after = out["close"].iloc[61] / out["close"].iloc[60] - 1
    assert np.sign(before) == -np.sign(after)


def test_a_cutoff_with_fewer_than_two_future_rows_is_not_perturbed():
    df = prices(50)
    assert perturb_future(df, df.index[-2], np.random.default_rng(0)) is None
    assert perturb_future(df, df.index[-1], np.random.default_rng(0)) is None
