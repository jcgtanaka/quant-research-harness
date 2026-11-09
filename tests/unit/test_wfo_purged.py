import numpy as np
import pandas as pd
import pytest

from qrh.adapters.base import DataView, RunResult
from qrh.validation.wfo import make_folds, parse_walk_forward, run_rolling_anchored, run_walk_forward
from qrh.validation.wfo_purged import (
    DEFAULT_EMBARGO,
    DEFAULT_PURGE,
    make_folds_purged,
    parse_purged,
    run_purged_embargo,
    validate_purged_embargo,
)

PPY = 252.0
RAW = {"train": 100, "test": 20, "purge": 6, "embargo": 3}


class Replay:
    """Returns the stored column of the matrix; records the spans it was asked for."""

    def __init__(self, matrix):
        self.matrix, self.calls = matrix, []

    def run(self, params, data, start, end, cost_multiplier=1.0):
        self.calls.append((start, end))
        return RunResult(self.matrix[params["k"]].loc[start:end].astype(float), pd.DataFrame(), pd.DataFrame())


def _setup(n=200, seed=0):
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    m = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=idx, columns=["a", "b", "c"])
    frame = pd.DataFrame({"close": np.linspace(100, 120, n)}, index=idx)
    return idx, m, frame, {c: {"k": c} for c in m.columns}


def test_parse_defaults_and_validation():
    s = parse_purged({"train": 100, "test": 20})
    assert (s.purge, s.embargo) == (DEFAULT_PURGE, DEFAULT_EMBARGO)
    s = parse_purged(RAW)
    assert (s.purge, s.embargo, s.base.train, s.base.test) == (6, 3, 100, 20)
    assert parse_purged({**RAW, "purge": 0, "embargo": 0}).purge == 0
    for bad in ({**RAW, "purge": -1}, {**RAW, "embargo": 1.5}, {**RAW, "purge": True}, {**RAW, "bogus": 1}):
        with pytest.raises(ValueError):
            parse_purged(bad)


def test_the_default_method_rejects_purge_and_embargo_keys():
    with pytest.raises(ValueError, match="unknown key"):
        parse_walk_forward(RAW)


def test_folds_have_a_gap_of_purge_before_and_embargo_after():
    base = make_folds(200, parse_walk_forward({"train": 100, "test": 20}))
    folds = make_folds_purged(200, parse_purged(RAW))
    assert len(folds) == len(base)
    for b, f in zip(base, folds):
        assert f.train_start == b.train_start and f.train_end == b.train_end - 6
        assert f.test_start == b.test_start + 3 and f.test_end == b.test_end
        assert f.test_start - f.train_end == 9


def test_fraction_sizes_are_of_actual_bars():
    folds = make_folds_purged(400, parse_purged({"train": 0.5, "test": 0.25, "purge": 0.05, "embargo": 0.01}))
    assert [(f.train_end, f.test_start) for f in folds] == [(180, 204), (280, 304)]


def test_gap_that_swallows_the_train_or_test_span_is_an_error():
    with pytest.raises(ValueError, match="purge"):
        make_folds_purged(200, parse_purged({"train": 100, "test": 20, "purge": 99}))
    with pytest.raises(ValueError, match="embargo"):
        make_folds_purged(200, parse_purged({"train": 100, "test": 20, "embargo": 20}))


def test_zero_gap_reproduces_the_default_method_exactly():
    idx, m, frame, params = _setup()
    raw0 = {"train": 100, "test": 20, "purge": 0, "embargo": 0}
    a = run_purged_embargo(Replay(m), DataView(frame), idx, m, params, raw0, PPY)
    b = run_rolling_anchored(Replay(m), DataView(frame), idx, m, params, {"train": 100, "test": 20}, PPY)
    assert a.oos_returns.equals(b.oos_returns)
    assert [f.combo_hash for f in a.folds] == [f.combo_hash for f in b.folds]


def test_selection_ignores_values_inside_the_purged_gap():
    idx, m, frame, params = _setup(n=125)  # exactly one fold: train 0-100, test 100-120
    poisoned = m.copy()
    poisoned.iloc[94:100, 2] = 0.5  # "c" looks superb, but only inside the purged tail of the train span
    clean = run_purged_embargo(Replay(m), DataView(frame), idx, m, params, RAW, PPY)
    swayed = run_purged_embargo(Replay(poisoned), DataView(frame), idx, poisoned, params, RAW, PPY)
    assert [f.combo_hash for f in swayed.folds] == [f.combo_hash for f in clean.folds]
    assert [f.train_sharpe for f in swayed.folds] == [f.train_sharpe for f in clean.folds]
    # the default method has no purge and does select the poisoned column
    default = run_rolling_anchored(Replay(poisoned), DataView(frame), idx, poisoned, params, {"train": 100, "test": 20}, PPY)
    assert [f.combo_hash for f in default.folds] == ["c"]


def test_embargoed_head_of_each_test_span_is_not_scored_or_run():
    idx, m, frame, params = _setup()
    ad = Replay(m)
    res = run_purged_embargo(ad, DataView(frame), idx, m, params, RAW, PPY)
    folds = make_folds_purged(200, parse_purged(RAW))
    assert ad.calls == [(idx[f.test_start], idx[f.test_end - 1]) for f in folds]
    expected = idx[np.concatenate([np.arange(f.test_start, f.test_end) for f in folds])]
    assert res.oos_returns.index.equals(expected)
    base_starts = [make_folds(200, parse_walk_forward({"train": 100, "test": 20}))[i].test_start for i in range(len(folds))]
    assert all(idx[b] not in res.oos_returns.index for b in base_starts)  # first bar of each original test span is dropped


def test_run_walk_forward_still_accepts_explicit_folds():
    idx, m, frame, params = _setup()
    folds = make_folds(200, parse_walk_forward({"train": 100, "test": 20}))[:2]
    res = run_walk_forward(
        Replay(m), DataView(frame), idx, m, params, parse_walk_forward({"train": 100, "test": 20}), PPY, folds=folds
    )
    assert len(res.folds) == 2


def test_validate_checks_the_spec_against_the_window():
    validate_purged_embargo(RAW, 200)
    with pytest.raises(ValueError):
        validate_purged_embargo(RAW, 110)
    with pytest.raises(ValueError):
        validate_purged_embargo({"train": 100}, 200)
