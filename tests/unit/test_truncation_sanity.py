import numpy as np
import pandas as pd

from qrh.adapters.base import DataView, RunResult
from qrh.checks.sanity import check_sanity
from qrh.checks.truncation import run_truncation


def frame(n=200, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC", name="timestamp")
    return pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))}, index=idx)


class Adapter:
    def __init__(self, lookahead=False, noisy_trades=False):
        self.lookahead, self.noisy_trades = lookahead, noisy_trades

    def parameter_space(self):
        return {"k": [1]}

    def required_data(self):
        return []

    def run(self, params, data, start, end):
        px = data.as_of(end)["close"]
        ret = px.pct_change()
        sig = np.sign(ret.shift(-1)) if self.lookahead else np.sign(ret)
        pos = sig.fillna(0.0).to_frame("w")
        strat = (pos["w"].shift(1).fillna(0.0) * ret.fillna(0.0)).loc[start:end]
        trades = pd.DataFrame({"qty": pos["w"].diff().fillna(0.0)})
        return RunResult(strat, trades, pos.loc[start:end])


def window(df):
    return df.index[10], df.index[-1]


def test_clean_adapter_passes():
    df = frame()
    s, e = window(df)
    res = run_truncation(Adapter(), df, {"k": 1}, s, e)
    assert res.verdict == "PASS", res.reasons
    assert len(res.details["cutoffs"]) == 3


def test_lookahead_adapter_fails_with_first_timestamp_and_cutoff():
    df = frame()
    s, e = window(df)
    res = run_truncation(Adapter(lookahead=True), df, {"k": 1}, s, e)
    assert res.verdict == "FAIL"
    text = " ".join(res.reasons)
    assert "cutoff" in text and "first differing timestamp" in text


def test_custom_cutoffs():
    df = frame()
    s, e = window(df)
    res = run_truncation(Adapter(), df, {"k": 1}, s, e, cutoffs=(0.5,))
    assert len(res.details["cutoffs"]) == 1


def test_adapter_exception_is_blocked():
    class Boom(Adapter):
        def run(self, params, data, start, end):
            if len(data.as_of(end)) < len(frame()):
                raise RuntimeError("needs all data")
            return super().run(params, data, start, end)

    df = frame()
    s, e = window(df)
    res = run_truncation(Boom(), df, {"k": 1}, s, e)
    assert res.verdict == "BLOCKED" and "needs all data" in " ".join(res.reasons)


def test_tiny_window_is_blocked():
    df = frame()
    res = run_truncation(Adapter(), df, {"k": 1}, df.index[0], df.index[2])
    assert res.verdict == "BLOCKED"


def test_trade_difference_detected():
    class T(Adapter):
        def run(self, params, data, start, end):
            r = super().run(params, data, start, end)
            full = len(data.as_of(end)) == len(frame())
            if full:
                r.trades.iloc[40, 0] += 1.0
            return r

    df = frame()
    s, e = window(df)
    res = run_truncation(T(), df, {"k": 1}, s, e)
    assert res.verdict == "FAIL" and "trades" in " ".join(res.reasons)


def test_series_for_each_cutoff_are_aligned_before_the_cutoff_and_equal_when_clean():
    df = frame()
    s, e = window(df)
    res = run_truncation(Adapter(), df, {"k": 1}, s, e)
    assert [x["fraction"] for x in res.series] == [0.25, 0.5, 0.75]
    for entry in res.series:
        cutoff = pd.Timestamp(entry["cutoff"])
        assert entry["first_divergence"] is None and entry["status"] == "PASS"
        assert entry["column"] == "w" and entry["columns_total"] == 1
        assert len(entry["timestamps"]) == len(entry["full"]) == len(entry["truncated"]) > 10
        assert all(pd.Timestamp(t) < cutoff for t in entry["timestamps"])
        assert entry["full"] == entry["truncated"]


def test_series_mark_the_first_divergence_of_a_lookahead_adapter():
    df = frame()
    s, e = window(df)
    res = run_truncation(Adapter(lookahead=True), df, {"k": 1}, s, e)
    entry = res.series[0]
    assert entry["status"] == "FAIL"
    div = entry["first_divergence"]
    assert div["what"] == "positions" and div["timestamp"] in entry["timestamps"]
    assert entry["full"] != entry["truncated"]
    assert div["timestamp"] in " ".join(res.reasons)


def test_trade_only_divergence_is_marked_with_its_source():
    class T(Adapter):
        def run(self, params, data, start, end):
            r = super().run(params, data, start, end)
            if len(data.as_of(end)) == len(frame()):
                r.trades.iloc[40, 0] += 1.0
            return r

    df = frame()
    s, e = window(df)
    res = run_truncation(T(), df, {"k": 1}, s, e)
    assert any(x["first_divergence"] and x["first_divergence"]["what"] == "trades" for x in res.series)


def test_long_series_are_decimated_and_the_stride_is_recorded():
    df = frame(3000)
    s, e = window(df)
    res = run_truncation(Adapter(), df, {"k": 1}, s, e, cutoffs=(0.9,))
    entry = res.series[0]
    assert entry["stride"] > 1 and len(entry["timestamps"]) <= 2000


def test_series_do_not_enter_the_details_stored_in_receipts():
    df = frame()
    s, e = window(df)
    res = run_truncation(Adapter(), df, {"k": 1}, s, e)
    assert "timestamps" not in str(res.details)


# sanity -------------------------------------------------------------------
BOUNDS = {"max_abs_period_return": 0.5, "max_gross_leverage": 1.0, "max_position_weight": 1.0}


def result(rets, weights):
    idx = pd.date_range("2021-01-01", periods=len(rets), freq="D", tz="UTC")
    return RunResult(pd.Series(rets, index=idx), pd.DataFrame(), pd.DataFrame(weights, index=idx))


def test_sanity_pass_with_metrics():
    r = result([0.01, -0.02, 0.0], {"a": [1.0, -0.5, 0.0], "b": [0.0, 0.5, 0.2]})
    s = check_sanity(r, BOUNDS)
    assert s.verdict == "PASS", s.reasons
    assert s.metrics["max_abs_period_return"] == 0.02
    assert s.metrics["max_gross_leverage"] == 1.0
    assert s.metrics["max_position_weight"] == 1.0


def test_sanity_violations():
    r = result([0.01, 0.9], {"a": [1.5, 1.0], "b": [0.5, 0.0]})
    s = check_sanity(r, BOUNDS)
    assert s.verdict == "FAIL"
    text = " ".join(s.reasons)
    assert "max_abs_period_return" in text and "max_gross_leverage" in text and "max_position_weight" in text


def test_sanity_percent_units_hint():
    r = result([1.2, -0.9, 0.8, 1.1], {"a": [1.0, 1.0, 1.0, 1.0]})
    s = check_sanity(r, {**BOUNDS, "max_abs_period_return": 100})
    assert s.verdict == "FAIL" and "returns may be in percent, expected fractions" in " ".join(s.reasons)


def test_sanity_nonfinite_and_missing_bounds_and_empty():
    assert check_sanity(result([np.nan, 0.1], {"a": [1.0, 1.0]}), BOUNDS).verdict == "PASS"
    assert check_sanity(result([np.inf, 0.1], {"a": [1.0, 1.0]}), BOUNDS).verdict == "FAIL"
    assert check_sanity(result([0.1], {"a": [1.0]}), {}).verdict == "BLOCKED"
    empty = RunResult(pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame())
    assert check_sanity(empty, BOUNDS).verdict == "BLOCKED"
    no_pos = RunResult(result([0.1], {"a": [1.0]}).returns, pd.DataFrame(), pd.DataFrame())
    assert check_sanity(no_pos, BOUNDS).verdict == "BLOCKED"


def test_sanity_rows_pair_each_metric_with_its_bound_and_the_percent_check():
    from qrh.checks.sanity import sanity_rows

    r = result([0.01, 0.9], {"a": [1.5, 1.0], "b": [0.5, 0.0]})
    s = check_sanity(r, BOUNDS)
    rows = {x["name"]: x for x in sanity_rows(s.metrics, BOUNDS)}
    assert rows["max_gross_leverage"] == {"name": "max_gross_leverage", "value": 2.0, "bound": 1.0, "ok": False}
    assert rows["max_abs_period_return"]["ok"] is False and rows["max_position_weight"]["bound"] == 1.0
    assert rows["median_abs_period_return"]["bound"] == 0.5 and rows["median_abs_period_return"]["ok"] is True
