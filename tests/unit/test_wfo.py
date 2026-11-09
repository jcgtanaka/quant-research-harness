import numpy as np
import pandas as pd
import pytest

from qrh.adapters.base import DataView, DatasetRef, RunResult
from qrh.validation.wfo import (
    make_folds,
    parse_walk_forward,
    run_walk_forward,
    select_best,
)

PPY = 252.0


def spec(**kw):
    return parse_walk_forward({"train": 100, "test": 20} | kw)


def test_parse_defaults_and_fractions():
    s = parse_walk_forward({"train": 0.5, "test": 0.25})
    assert (s.train, s.test, s.step, s.anchored) == (0.5, 0.25, None, False)
    assert spec(step=10, anchored=True).anchored is True


@pytest.mark.parametrize(
    "raw",
    [
        None, "x", {}, {"train": 100}, {"test": 20}, {"train": 0, "test": 20}, {"train": 100, "test": -1},
        {"train": 1.5, "test": 20}, {"train": True, "test": 20}, {"train": 100, "test": 20, "anchored": "yes"},
        {"train": 100, "test": 20, "bogus": 1}, {"train": 100, "test": 20, "step": 0},
    ],
)
def test_parse_rejects_bad_specs(raw):
    with pytest.raises(ValueError):
        parse_walk_forward(raw)


def test_rolling_folds_boundaries():
    folds = make_folds(200, spec())  # train 100, test 20, step 20
    assert [(f.train_start, f.train_end, f.test_start, f.test_end) for f in folds] == [
        (0, 100, 100, 120), (20, 120, 120, 140), (40, 140, 140, 160), (60, 160, 160, 180), (80, 180, 180, 200),
    ]


def test_anchored_folds_boundaries():
    folds = make_folds(200, spec(anchored=True))
    assert [(f.train_start, f.train_end, f.test_start, f.test_end) for f in folds] == [
        (0, 100, 100, 120), (0, 120, 120, 140), (0, 140, 140, 160), (0, 160, 160, 180), (0, 180, 180, 200),
    ]


def test_fractions_are_of_actual_bars():
    folds = make_folds(400, parse_walk_forward({"train": 0.5, "test": 0.25}))
    assert [(f.train_end, f.test_end) for f in folds] == [(200, 300), (300, 400)]
    # the same fractions on a different number of bars give different boundaries
    folds2 = make_folds(100, parse_walk_forward({"train": 0.5, "test": 0.25}))
    assert [(f.train_end, f.test_end) for f in folds2] == [(50, 75), (75, 100)]


@pytest.mark.parametrize("anchored", [False, True])
@pytest.mark.parametrize("n,train,test", [(137, 40, 13), (200, 50, 50), (95, 10, 7)])
def test_no_train_test_overlap_and_tests_contiguous(anchored, n, train, test):
    folds = make_folds(n, spec(train=train, test=test, anchored=anchored))
    assert folds
    for f in folds:
        assert f.train_end == f.test_start and f.train_start < f.train_end < f.test_end <= n
        assert f.test_end - f.test_start == test
    for a, b in zip(folds, folds[1:]):
        assert a.test_end == b.test_start  # contiguous, non-overlapping
    assert folds[0].test_start == train
    assert n - folds[-1].test_end < test  # only a partial remainder is left over


def test_step_larger_than_test_leaves_gaps_but_never_overlaps():
    folds = make_folds(200, spec(step=30))
    for a, b in zip(folds, folds[1:]):
        assert b.test_start >= a.test_end


def test_step_smaller_than_test_is_rejected():
    with pytest.raises(ValueError, match="step"):
        make_folds(200, spec(step=10))


def test_window_too_short_is_an_error():
    with pytest.raises(ValueError, match="too short"):
        make_folds(110, spec())


def _matrix(n=200, cols=("a", "b", "c"), seed=0):
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.01, (n, len(cols))), index=idx, columns=list(cols))


def test_selection_picks_best_in_sample_sharpe():
    m = _matrix()
    m["b"] += 0.004
    f = make_folds(200, spec())[0]
    h, sr = select_best(m, f, PPY)
    assert h == "b" and sr > 0


def test_selection_uses_only_train_data_planted_test():
    # "x" is superb only inside one fold's own test span; that fold must never select it.
    folds = make_folds(200, spec())
    for f in folds:
        m = _matrix(cols=("a", "b", "x"))
        m["x"] = m["a"] - 0.002  # strictly worse than "a" on every bar outside the planted span
        m.iloc[f.test_start : f.test_end, 2] = m["a"].iloc[f.test_start : f.test_end] + 0.05
        assert select_best(m, f, PPY)[0] != "x"
    # Sanity: the same planted column placed inside the train span is selected.
    f = folds[0]
    m = _matrix(cols=("a", "b", "x"))
    m["x"] = m["a"] - 0.002
    m.iloc[f.train_start : f.train_end, 2] = m["a"].iloc[f.train_start : f.train_end] + 0.01  # inside the train span
    assert select_best(m, f, PPY)[0] == "x"


def test_selection_is_independent_of_values_outside_the_train_span():
    m = _matrix()
    for f in make_folds(200, spec()):
        base = select_best(m, f, PPY)
        mm = m.copy()
        mm.iloc[f.test_start :, :] = np.random.default_rng(9).normal(0.5, 0.01, mm.iloc[f.test_start :, :].shape)
        mm.iloc[: f.train_start, :] = -3.0
        assert select_best(mm, f, PPY) == base


def test_ties_go_to_lowest_combo_hash_regardless_of_column_order():
    m = _matrix(cols=("zz", "aa", "mm"))
    m["aa"] = m["zz"]  # identical returns -> identical Sharpe
    m["mm"] -= 0.01  # clearly worse
    f = make_folds(200, spec())[0]
    for order in (["zz", "aa", "mm"], ["mm", "zz", "aa"], ["aa", "mm", "zz"]):
        assert select_best(m[order], f, PPY)[0] == "aa"


def test_selection_ignores_undefined_sharpe_and_errors_when_none_defined():
    m = _matrix(cols=("flat", "b"))
    m["flat"] = 0.0
    f = make_folds(200, spec())[0]
    assert select_best(m, f, PPY)[0] == "b"
    m["b"] = 0.0
    with pytest.raises(ValueError, match="no strategy"):
        select_best(m, f, PPY)


class FakeAdapter:
    """returns = param * (a deterministic wave); records the spans it was asked for."""

    def __init__(self):
        self.calls = []

    def parameter_space(self):
        return {"k": [1, 2]}

    def required_data(self):
        return [DatasetRef("prices")]

    def run(self, params, data, start, end, cost_multiplier=1.0):
        self.calls.append((params["k"], start, end, cost_multiplier))
        px = data.as_of(end)["close"]
        r = (px.pct_change().fillna(0.0) * params["k"] * (2.0 - cost_multiplier)).loc[start:end]
        return RunResult(r, pd.DataFrame(), pd.DataFrame())


def _frame(n=200):
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"close": 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.001, 0.01, n)))}, index=idx)


def test_run_walk_forward_stitches_contiguous_oos_and_runs_adapter_on_test_spans_only():
    frame = _frame()
    bars = frame.index
    ad = FakeAdapter()
    params = {"h1": {"k": 1}, "h2": {"k": 2}}
    # h2 is best in-sample everywhere
    m = pd.DataFrame({"h1": np.random.default_rng(0).normal(0, 0.01, 200), "h2": 0.0}, index=bars)
    m["h2"] = np.random.default_rng(1).normal(0.004, 0.01, 200)
    res = run_walk_forward(ad, DataView(frame), bars, m, params, spec(), PPY)
    folds = make_folds(200, spec())
    assert len(res.folds) == len(folds) == len(ad.calls)
    assert all(f.combo_hash == "h2" for f in res.folds)
    for (k, s, e, cm), f in zip(ad.calls, folds):
        assert (s, e) == (bars[f.test_start], bars[f.test_end - 1]) and k == 2 and cm == 1.0
    # stitched OOS covers exactly the union of test spans, in order, once
    assert res.oos_returns.index.equals(bars[folds[0].test_start : folds[-1].test_end])
    assert res.oos_returns.index.is_unique and res.oos_returns.index.is_monotonic_increasing


def test_run_walk_forward_passes_extra_kwargs_and_selection_unchanged():
    frame = _frame()
    bars = frame.index
    params = {"h1": {"k": 1}, "h2": {"k": 2}}
    m = pd.DataFrame(np.random.default_rng(2).normal(0, 0.01, (200, 2)), index=bars, columns=["h1", "h2"])
    a = run_walk_forward(FakeAdapter(), DataView(frame), bars, m, params, spec(), PPY)
    ad = FakeAdapter()
    b = run_walk_forward(ad, DataView(frame), bars, m, params, spec(), PPY, run_kwargs={"cost_multiplier": 2.0})
    assert [f.combo_hash for f in a.folds] == [f.combo_hash for f in b.folds]
    assert all(c[3] == 2.0 for c in ad.calls)
    assert (b.oos_returns == 0.0).all()  # the fake's cost stress zeroes returns


def test_run_walk_forward_blocks_on_misaligned_adapter_output():
    class Short(FakeAdapter):
        def run(self, params, data, start, end, **kw):
            res = super().run(params, data, start, end)
            return RunResult(res.returns.iloc[1:], res.trades, res.positions)

    frame = _frame()
    m = pd.DataFrame(np.random.default_rng(3).normal(0, 0.01, (200, 2)), index=frame.index, columns=["h1", "h2"])
    with pytest.raises(ValueError, match="bars"):
        run_walk_forward(Short(), DataView(frame), frame.index, m, {"h1": {"k": 1}, "h2": {"k": 2}}, spec(), PPY)
