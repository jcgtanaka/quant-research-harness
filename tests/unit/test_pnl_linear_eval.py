"""Tests for the pnl.linear verdict module (synthetic data only; the ledger is not imported)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qrh.checks.pnl_linear import PnlLinearVerdict, evaluate_pnl_linear

N = 250
CAPITAL = 100_000.0
BPY = 252.0
COLUMNS = {"close", "pub", "halt"}


def _block(**over):
    block = {
        "position_units": "quantity",
        "cost_model": {"bps": 1.0},
        "asset_class": "equity",
        "min_cost_per_turnover": 0.0,
        "publication_column": "pub",
        "max_staleness": 2,
        "adjustment": {"close": "total_return"},
        "max_unexplained_bp_per_year": 50.0,
        "max_unexplained_share": 0.2,
    }
    block.update(over)
    return block


def _series(seed=7, n=N):
    rng = np.random.default_rng(seed)
    explained = rng.normal(5.0, 20.0, n)
    remainder = rng.normal(0.0, 1.0, n)
    return explained + remainder, explained, remainder


def _stale_inputs(n=N, **over):
    times = pd.date_range("2024-01-01", periods=n, freq="D")
    inp = {
        "bar_times": times,
        "publication_times": times.copy(),
        "position": pd.Series(np.ones(n)),
        "lag": 1,
        "halt_flag": None,
    }
    inp.update(over)
    return inp


def _run(series=None, block=None, columns=COLUMNS, staleness=None, n=N):
    reported, explained, remainder = series if series is not None else _series(n=n)
    return evaluate_pnl_linear(
        reported_pnl=reported,
        explained_net_pnl=explained,
        remainder=remainder,
        frame_columns=columns,
        config_block=block if block is not None else _block(),
        bars_per_year=BPY,
        capital_base=CAPITAL,
        newey_west_lag=2,
        alpha=0.05,
        staleness_inputs=staleness if staleness is not None else _stale_inputs(n=len(reported)),
    )


def _text(v):
    return " ".join(v.reasons).lower()


def test_clean_honest_series_passes():
    v = _run()
    assert isinstance(v, PnlLinearVerdict)
    assert v.verdict == "PASS", v.reasons
    assert v.details["absolute_cap"].breached is False
    assert v.details["share"].breached is False
    assert v.details["drift"].significant is False


def test_hidden_constant_gain_fails():
    reported, explained, remainder = _series()
    remainder = remainder + 5.0
    v = _run((explained + remainder, explained, remainder))
    assert v.verdict == "FAIL"
    assert "absolute cap" in _text(v) or "drift" in _text(v)
    assert v.details["absolute_cap"].breached is True


def test_drift_alone_fails_with_positive_mean():
    reported, explained, remainder = _series()
    remainder = remainder + 0.8  # small gain: under generous caps, still significant drift
    v = _run((explained + remainder, explained, remainder),
             block=_block(max_unexplained_bp_per_year=1e6, max_unexplained_share=1.0))
    assert v.verdict == "FAIL"
    assert "drift" in _text(v)


def test_share_breach_fails():
    reported, explained, remainder = _series()
    v = _run((reported, explained, remainder),
             block=_block(max_unexplained_bp_per_year=1e6, max_unexplained_share=1e-6))
    assert v.verdict == "FAIL"
    assert "share" in _text(v)


def test_omitted_caps_use_noise_scaled_limits_and_can_pass():
    block = {k: x for k, x in _block().items() if not k.startswith("max_unexplained")}
    rng = np.random.default_rng(7)
    explained = rng.normal(500.0, 20.0, N)  # large net explained so the 0.02 share cap is not tripped by chance
    remainder = rng.normal(0.0, 1.0, N)
    v = _run((explained + remainder, explained, remainder), block=block)
    assert v.verdict == "PASS", v.reasons
    assert v.details["absolute_cap"].cap_bp_per_year > 0  # noise-scaled, not a fixed number
    assert v.details["share"].max_unexplained_share > 0


def test_explicit_none_cap_is_inconclusive_and_named():
    v = _run(block=_block(max_unexplained_bp_per_year=None))
    assert v.verdict == "INCONCLUSIVE"
    assert "pending user decision: max_unexplained_bp_per_year" in v.reasons


def test_both_caps_pending_named():
    v = _run(block=_block(max_unexplained_bp_per_year=None, max_unexplained_share=None))
    assert v.verdict == "INCONCLUSIVE"
    assert any("max_unexplained_bp_per_year" in r and "max_unexplained_share" in r
               and r.startswith("pending user decision:") for r in v.reasons)


def test_pending_cap_never_hides_a_fail():
    reported, explained, remainder = _series()
    remainder = remainder + 5.0
    v = _run((explained + remainder, explained, remainder),
             block=_block(max_unexplained_bp_per_year=None))
    assert v.verdict == "FAIL"


def test_stale_held_mark_fails():
    times = pd.date_range("2024-01-01", periods=N, freq="D")
    pub = times.copy()
    pub = pub.where(np.arange(N) != 100, times[100] - pd.Timedelta(days=10))
    v = _run(staleness=_stale_inputs(publication_times=pub))
    assert v.verdict == "FAIL"
    assert "stale" in _text(v)


def test_future_mark_fails():
    times = pd.date_range("2024-01-01", periods=N, freq="D")
    pub = times.copy()
    pub = pub.where(np.arange(N) != 100, times[100] + pd.Timedelta(days=3))
    v = _run(staleness=_stale_inputs(publication_times=pub))
    assert v.verdict == "FAIL"
    assert "future" in _text(v)


def test_halt_is_inconclusive():
    times = pd.date_range("2024-01-01", periods=N, freq="D")
    pub = times.copy()
    pub = pub.where(np.arange(N) != 100, times[100] - pd.Timedelta(days=10))
    halt = pd.Series(np.arange(N) == 100)
    v = _run(staleness=_stale_inputs(publication_times=pub, halt_flag=halt))
    assert v.verdict == "INCONCLUSIVE"
    assert "halt" in _text(v)


def test_missing_publication_time_on_held_bar_is_blocked():
    times = pd.date_range("2024-01-01", periods=N, freq="D")
    pub = pd.Series(times)
    pub.iloc[50] = pd.NaT
    v = _run(staleness=_stale_inputs(publication_times=pub))
    assert v.verdict == "BLOCKED"
    assert "publication" in _text(v)


def test_invalid_config_is_blocked_with_validator_message():
    v = _run(block=_block(position_units="bogus"))
    assert v.verdict == "BLOCKED"
    assert "reconstruction.position_units" in " ".join(v.reasons)


def test_missing_frame_column_is_blocked():
    v = _run(columns={"close"})
    assert v.verdict == "BLOCKED"
    assert "pub" in " ".join(v.reasons)


def test_zero_net_explained_is_inconclusive():
    reported, explained, remainder = _series()
    explained = np.zeros(N)
    explained[0], explained[1] = 3.0, -3.0
    v = _run((explained + remainder, explained, remainder))
    assert v.verdict == "INCONCLUSIVE"
    assert "net explained" in _text(v)


def test_too_few_active_bars_is_inconclusive():
    v = _run(series=_series(n=15),
             block=_block(max_unexplained_bp_per_year=1e6, max_unexplained_share=1.0))
    assert v.verdict == "INCONCLUSIVE"
    assert "fewer than 20" in _text(v)


def test_downward_only_remainder_is_inconclusive():
    reported, explained, remainder = _series()
    remainder = -np.abs(remainder) - 0.1
    v = _run((explained + remainder, explained, remainder),
             block=_block(max_unexplained_share=1.0))
    assert v.verdict == "INCONCLUSIVE"
    assert "downward" in _text(v)


def test_reasons_name_deciding_rule_for_pass():
    v = _run()
    assert v.reasons and "all three bounds" in _text(v)
