"""Hardening of the future-perturbation and same-bar checks found by an adversarial review."""

import numpy as np
import pandas as pd
import pytest

from qrh.adapters.base import DataView, DatasetRef, RunResult
from qrh.checks.perturbation import make_own_bar_probe, perturb_future, run_perturbation
from qrh.checks.same_bar import check_same_bar_execution


def walk(n=1000, seed=0, extra=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame({"close": 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))}, index=idx)
    for k, v in (extra or {}).items():
        df[k] = v(rng, idx)
    return df


def result(pos, ret, w):
    ch = pos.diff().fillna(pos)
    return RunResult(ret.loc[w], pd.DataFrame({"qty": ch[ch != 0]}).loc[w], pos.to_frame("weight").loc[w])


class AlwaysInMarketLeak:
    """Always long or short (momentum sign); on the rare bars before a big move it takes that move's sign."""

    def parameter_space(self):
        return {}

    def required_data(self):
        return [DatasetRef("prices")]

    def run(self, params, data, start, end):
        px = data.as_of(end)["close"]
        r = px.pct_change()
        nxt = np.roll(r.to_numpy(), -1)
        base = np.sign(r.rolling(20).mean().fillna(0).to_numpy() + 1e-12)
        pos = pd.Series(np.where(np.abs(nxt) > 0.025, np.sign(nxt), base), index=px.index)
        return result(pos, (pos.shift(1) * r).fillna(0.0), slice(start, end))


# --- finding 2: always in the market ------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_a_sparse_leak_in_an_always_in_the_market_strategy_is_caught(seed):
    df = walk(seed=seed)
    res = run_perturbation(AlwaysInMarketLeak(), df, {}, df.index[0], df.index[-1])
    assert res.verdict == "FAIL", res.details


def test_coverage_of_the_change_bars_is_recorded():
    df = walk()
    res = run_perturbation(AlwaysInMarketLeak(), df, {}, df.index[0], df.index[-1], max_change_bars=5)
    d = res.details
    assert d["change_bars_total"] > 5 and d["change_bars_tested"] == 5
    assert d["coverage"] == pytest.approx(5 / d["change_bars_total"])
    full = run_perturbation(AlwaysInMarketLeak(), df, {}, df.index[0], df.index[-1], max_change_bars=500)
    assert full.details["coverage"] == 1.0 and full.details["change_bars_tested"] == full.details["change_bars_total"]


def test_the_bar_before_each_tested_change_is_also_a_cutoff():
    df = walk()
    res = run_perturbation(AlwaysInMarketLeak(), df, {}, df.index[0], df.index[-1], max_change_bars=3)
    kinds = [c["kind"] for c in res.details["cutoffs"]]
    assert kinds.count("change") == 3 and kinds.count("before-change") <= 3 and "spread" in kinds


# --- finding 3: memoizing adapters --------------------------------------------------------------------------------

_GLOBAL_CACHE: dict = {}


class Memo(AlwaysInMarketLeak):
    """Caches the whole run by (params, window): a common speed-up that hides every perturbation."""

    def __init__(self, shared):
        self.cache = _GLOBAL_CACHE if shared else {}

    def run(self, params, data, start, end):
        key = (start, end)
        if key not in self.cache:
            self.cache[key] = super().run(params, data, start, end)
        return self.cache[key]


def test_a_memoizing_adapter_is_blocked_when_the_perturbed_data_never_reaches_it():
    _GLOBAL_CACHE.clear()
    df = walk()
    res = run_perturbation(Memo(True), df, {}, df.index[0], df.index[-1])
    assert res.verdict == "BLOCKED" and "perturbed data did not reach the strategy" in " ".join(res.reasons)


def test_a_fresh_adapter_per_run_defeats_instance_level_memoization():
    df = walk()
    res = run_perturbation(Memo(False), df, {}, df.index[0], df.index[-1], adapter_factory=lambda: Memo(False))
    assert res.verdict == "FAIL"


# --- finding 4: publication lag -----------------------------------------------------------------------------------

FAR = pd.Timestamp("2100-01-01", tz="UTC")


class Fundamentals:
    def __init__(self, careless, asof_far=False):
        self.careless, self.asof_far = careless, asof_far

    def parameter_space(self):
        return {}

    def required_data(self):
        return [DatasetRef("prices")]

    def run(self, params, data, start, end):
        df = data.as_of(FAR if self.asof_far else end)
        px = df["close"]
        r = px.pct_change()
        if self.careless:  # uses the value stamped t at decision t although it is published a week later
            eps = df["eps"]
        else:  # a value is known only from its publication time on
            eps = pd.Series(df["eps"].to_numpy(), index=pd.DatetimeIndex(df["pub"])).reindex(px.index, method="ffill")
        pos = (eps.diff() > 0).astype(float)
        return result(pos, (pos.shift(1) * r).fillna(0.0), slice(start, end))


def pub_frame():
    return walk(600, extra={
        "eps": lambda rng, idx: np.cumsum(rng.standard_normal(len(idx))) + 50,
        "pub": lambda rng, idx: idx + pd.Timedelta(days=7)})


@pytest.mark.parametrize("far", [False, True])
def test_a_value_used_before_its_publication_time_fails(far):
    df = pub_frame()
    res = run_perturbation(Fundamentals(True, far), df, {}, df.index[0], df.index[-20], "pub")
    assert res.verdict == "FAIL", res.details


def test_a_strategy_that_respects_the_publication_time_passes():
    df = pub_frame()
    res = run_perturbation(Fundamentals(False), df, {}, df.index[0], df.index[-20], "pub")
    assert res.verdict == "PASS", res.reasons


def test_bool_and_text_columns_are_permuted_and_keep_their_dtype():
    df = walk(80, extra={
        "flag": lambda rng, idx: rng.random(len(idx)) > 0.5,
        "tag": lambda rng, idx: np.where(rng.random(len(idx)) > 0.5, "a", "b").astype(object)})
    df["cat"] = pd.Categorical(df["tag"])
    out = perturb_future(df, df.index[40], np.random.default_rng(0))
    assert out["flag"].dtype == bool and out["cat"].dtype == df["cat"].dtype
    assert not (out["flag"].to_numpy()[41:] == df["flag"].to_numpy()[41:]).all()
    assert (out["flag"].to_numpy()[:41] == df["flag"].to_numpy()[:41]).all()
    assert sorted(out["flag"].to_numpy()[41:]) == sorted(df["flag"].to_numpy()[41:])  # a permutation


def test_rows_published_after_the_cutoff_are_perturbed_even_when_stamped_before_it():
    df = pub_frame()
    cutoff = df.index[100]
    out = perturb_future(df, cutoff, np.random.default_rng(0), "pub")
    assert (out["pub"] == df["pub"]).all()
    late = (df.index <= cutoff) & (df["pub"] > cutoff)
    assert late.sum() > 0 and not np.allclose(out.loc[late, "eps"], df.loc[late, "eps"])
    assert np.allclose(out.loc[(df.index <= cutoff) & ~late, "eps"], df.loc[(df.index <= cutoff) & ~late, "eps"])


# --- finding 5: "position held during the bar" ---------------------------------------------------------------------


class HeldDuring:
    """Reports the position held DURING each bar (the signal of the previous close); returns are correctly lagged."""

    def parameter_space(self):
        return {}

    def required_data(self):
        return [DatasetRef("prices")]

    def run(self, params, data, start, end):
        px = data.as_of(end)["close"]
        r = px.pct_change()
        signal = (px > px.rolling(20).mean()).astype(float)
        held = signal.shift(1).fillna(0.0)
        return result(held, (held * r).fillna(0.0), slice(start, end))


class SameClose(HeldDuring):
    """Decides on the close of bar t and earns bar t's return with it."""

    def run(self, params, data, start, end):
        px = data.as_of(end)["close"]
        r = px.pct_change()
        signal = (px > px.rolling(20).mean()).astype(float)
        return result(signal, (signal * r).fillna(0.0), slice(start, end))


def probed(adapter, df):
    s, e = df.index[30], df.index[-1]
    full = adapter.run({}, DataView(df), s, e)
    return full, make_own_bar_probe(adapter, df, {}, s, e, None, full)


def test_a_held_during_the_bar_report_is_inconclusive_not_a_same_bar_fail():
    df = walk(500)
    full, probe = probed(HeldDuring(), df)
    out = check_same_bar_execution(full, df, None, probe)
    assert out.verdict == "INCONCLUSIVE", (out.verdict, out.reasons)
    assert "held during the bar" in " ".join(out.reasons) and "declare `execution`" in " ".join(out.reasons)
    assert check_same_bar_execution(full, df, None).verdict == "FAIL"  # without the probe the old behaviour stands


def test_a_real_same_close_fill_still_fails_with_the_probe():
    df = walk(500)
    full, probe = probed(SameClose(), df)
    out = check_same_bar_execution(full, df, None, probe)
    assert out.verdict == "FAIL" and "same-bar execution" in " ".join(out.reasons)


def test_the_probe_is_not_used_when_the_user_declared_the_execution():
    df = walk(500)
    full, probe = probed(HeldDuring(), df)
    out = check_same_bar_execution(full, df, {"lag_bars": 1, "price_column": "close"}, probe)
    assert out.verdict == "FAIL"
