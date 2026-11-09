"""Unit tests for the pnl.linear mark-freshness rule (synthetic data only)."""

from __future__ import annotations

import pandas as pd
import pytest

from qrh.checks.pnl_staleness import (
    StalenessResult,
    held_mask,
    mark_staleness,
    check_mark_freshness,
)


def _bars(n: int = 10) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D")


def _pubs_current(bars: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(bars, index=range(len(bars)))


def _pos(values) -> pd.Series:
    return pd.Series(values, dtype=float)


# ---------------------------------------------------------------- held mask


def test_held_mask_lag0_covers_base_and_exit():
    # position non-zero on bars 3 and 4 only
    pos = _pos([0, 0, 0, 1, 1, 0, 0, 0])
    m = held_mask(pos, 0)
    # base bar of entry return (2) through exit bar (5)
    assert list(m[m].index) == [2, 3, 4, 5]


def test_held_mask_lag2_extends_to_base_and_late_exit():
    pos = _pos([0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
    m = held_mask(pos, 2)
    # return at t=5 earned by pos[3]; base bar 4; but pos[t-1] also non-zero
    # from bar 4, so the mask is conservative: bars 2..7 are not all needed,
    # check the guaranteed ones.
    for i in (3, 4, 5, 6):
        assert m[i]
    # exit: last non-zero at 4, earned returns at 6 (lag 2); exit bar 6 held
    assert m[6]
    assert not m[0] and not m[9]


def test_held_mask_lag1_entry_base_bar_included():
    pos = _pos([0, 0, 1, 1, 0, 0])
    m = held_mask(pos, 1)
    assert m[2]  # entry bar is the price base of the first earned return
    assert m[4]  # exit bar of the last earned return (pos[3] earns at t=4)
    assert not m[0]


def test_held_mask_all_flat_is_empty():
    assert not held_mask(_pos([0, 0, 0, 0]), 1).any()


def test_held_mask_rejects_bad_lag():
    with pytest.raises(ValueError):
        held_mask(_pos([0, 1]), -1)
    with pytest.raises(ValueError):
        held_mask(_pos([0, 1]), 1.5)


def test_held_mask_nan_position_is_not_held():
    assert not held_mask(_pos([float("nan")] * 4), 1).any()


# ------------------------------------------------------- staleness in time


def test_staleness_is_in_time_units():
    bars = _bars(4)
    pubs = pd.Series([bars[0], bars[0], bars[2], bars[3]])
    s = mark_staleness(bars, pubs)
    assert list(s) == [
        pd.Timedelta(0),
        pd.Timedelta(days=1),
        pd.Timedelta(0),
        pd.Timedelta(0),
    ]


def test_frozen_price_with_current_publication_is_not_stale():
    bars = _bars(6)
    pubs = _pubs_current(bars)  # current publication every bar
    pos = _pos([1] * 6)
    # prices could be identical every bar; only publication time matters
    res = check_mark_freshness(bars, pubs, pos, lag=1, max_staleness=1)
    assert res.stale == [] and res.future_dated == []


def test_late_publication_on_held_bar_is_flagged():
    bars = _bars(6)
    pubs = _pubs_current(bars)
    pubs.iloc[3] = bars[0]  # three bars late
    pos = _pos([1] * 6)
    res = check_mark_freshness(bars, pubs, pos, lag=1, max_staleness=2)
    assert res.stale == [3]


def test_staleness_at_threshold_is_not_flagged():
    bars = _bars(6)
    pubs = _pubs_current(bars)
    pubs.iloc[3] = bars[1]  # exactly 2 bars
    res = check_mark_freshness(bars, pubs, _pos([1] * 6), lag=1, max_staleness=2)
    assert res.stale == []


def test_stale_on_flat_bar_is_ignored():
    bars = _bars(8)
    pubs = _pubs_current(bars)
    pubs.iloc[7] = bars[0]
    pos = _pos([1, 1, 0, 0, 0, 0, 0, 0])
    res = check_mark_freshness(bars, pubs, pos, lag=1, max_staleness=1)
    assert res.stale == []


def test_stale_price_at_entry_base_bar_is_caught():
    # off-by-one: position first non-zero at bar 4, lag 1; the first earned
    # return at 5 uses price at 4 as base. Stale mark at 4 must be caught.
    bars = _bars(8)
    pubs = _pubs_current(bars)
    pubs.iloc[4] = bars[0]
    pos = _pos([0, 0, 0, 0, 1, 1, 0, 0])
    res = check_mark_freshness(bars, pubs, pos, lag=1, max_staleness=1)
    assert 4 in res.stale


def test_stale_price_at_exit_bar_is_caught():
    bars = _bars(8)
    pubs = _pubs_current(bars)
    pubs.iloc[6] = bars[0]  # exit bar: pos[5] earns the return at 6
    pos = _pos([0, 0, 0, 0, 1, 1, 0, 0])
    res = check_mark_freshness(bars, pubs, pos, lag=1, max_staleness=1)
    assert 6 in res.stale


def test_lag0_and_lag2_flag_positions():
    bars = _bars(10)
    pubs = _pubs_current(bars)
    pubs.iloc[2] = bars[0]
    pos = _pos([0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
    r0 = check_mark_freshness(bars, pubs, pos, lag=0, max_staleness=1)
    assert r0.stale == [2]  # base bar of the lag-0 entry return
    pubs2 = _pubs_current(bars)
    pubs2.iloc[6] = bars[0]
    r2 = check_mark_freshness(bars, pubs2, pos, lag=2, max_staleness=1)
    assert r2.stale == [6]  # late exit bar under lag 2
    r2b = check_mark_freshness(bars, pubs2, pos, lag=0, max_staleness=1)
    assert r2b.stale == []


# ------------------------------------------------------------ future dated


def test_future_dated_mark_flagged_separately():
    bars = _bars(6)
    pubs = _pubs_current(bars)
    pubs.iloc[3] = bars[3] + pd.Timedelta(hours=1)
    res = check_mark_freshness(bars, pubs, _pos([1] * 6), lag=1, max_staleness=1)
    assert res.future_dated == [3]
    assert res.stale == []


# -------------------------------------------------------------------- halt


def test_halt_flag_classifies_but_does_not_remove():
    bars = _bars(6)
    pubs = _pubs_current(bars)
    pubs.iloc[3] = bars[0]
    pubs.iloc[4] = bars[0]
    halt = pd.Series([False, False, False, True, False, False])
    res = check_mark_freshness(
        bars, pubs, _pos([1] * 6), lag=1, max_staleness=1, halt_flag=halt
    )
    assert res.halt_declared == [3]
    assert res.stale == [4]
    # never turned into a pass: the bar is still reported in the union
    assert res.all_stale == [3, 4]
    assert isinstance(res, StalenessResult)


def test_halt_flag_on_non_stale_bar_changes_nothing():
    bars = _bars(4)
    pubs = _pubs_current(bars)
    halt = pd.Series([True] * 4)
    res = check_mark_freshness(
        bars, pubs, _pos([1] * 4), lag=1, max_staleness=1, halt_flag=halt
    )
    assert res.all_stale == [] and res.halt_declared == []


# ------------------------------------------------------------------ errors


def test_missing_publication_time_on_needed_bar_raises():
    bars = _bars(6)
    pubs = _pubs_current(bars)
    pubs.iloc[3] = pd.NaT
    with pytest.raises(ValueError, match="publication"):
        check_mark_freshness(bars, pubs, _pos([1] * 6), lag=1, max_staleness=1)


def test_missing_publication_time_on_unneeded_bar_is_fine():
    bars = _bars(8)
    pubs = _pubs_current(bars)
    pubs.iloc[7] = pd.NaT
    pos = _pos([1, 1, 0, 0, 0, 0, 0, 0])
    res = check_mark_freshness(bars, pubs, pos, lag=1, max_staleness=1)
    assert res.all_stale == []


def test_invalid_max_staleness_raises():
    bars = _bars(3)
    with pytest.raises(ValueError):
        check_mark_freshness(
            bars, _pubs_current(bars), _pos([1] * 3), lag=1, max_staleness=0
        )


def test_length_mismatch_raises():
    bars = _bars(3)
    with pytest.raises(ValueError):
        check_mark_freshness(
            bars, _pubs_current(bars), _pos([1] * 4), lag=1, max_staleness=1
        )
