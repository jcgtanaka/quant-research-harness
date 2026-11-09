"""Owner rules of 2026-10-27 for the pnl.linear check. Synthetic data only.

1. Adjustment: a backtest the harness builds needs no declared adjustment; the receipt records the type.
   A backtest the user brings must be asked; an unknown answer is a warning.
2. Truncation: required for a backtest the user brings, a silent self-check (bug warning) for a built one.
3. Cap: for a built backtest any unexplained difference is a bug warning (day and amount); the noise-scaled
   tolerance applies only to a comparison with broker records.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "owner_rules_helpers", Path(__file__).with_name("test_s2_pnl_linear_optin.py"))
_opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_opt)
run_s2, text, BLOCK = _opt.run_s2, _opt.text, _opt.BLOCK

PEEK = "np.sign(trend).fillna(0.0)"
PEEKING = "np.sign(px.shift(-1) - px).fillna(0.0)"  # uses the next bar: the truncation test must catch it
HIDDEN_GAIN = "(position.shift(1) * ret - cost.shift(1) + 0.0005)"


def _as_user(spec):
    spec["origin"] = "user"


def _run(tmp_path, monkeypatch, *, block=BLOCK, user=False, peek=False, gain=False):
    kwargs = {"reconstruction": block}
    if user:
        kwargs["spec_edit"] = _as_user
    if gain:
        kwargs["adapter_edit"] = HIDDEN_GAIN
    s2 = run_s2(tmp_path, monkeypatch, **kwargs) if not peek else _run_peek(tmp_path, monkeypatch, block, user)
    return s2


def _run_peek(tmp_path, monkeypatch, block, user):
    """Same run, with the example adapter changed so that its position peeks one bar ahead."""
    original = _opt._build

    def build_peeking(tmp):
        d, data = original(tmp)
        a = d / "adapter.py"
        a.write_text(a.read_text().replace(PEEK, PEEKING))
        return d, data

    monkeypatch.setattr(_opt, "_build", build_peeking)
    try:
        return run_s2(tmp_path, monkeypatch, reconstruction=block, spec_edit=_as_user if user else None)
    finally:
        monkeypatch.setattr(_opt, "_build", original)


def _linear(s2):
    return s2.params["pnl_linear"]


def _warnings(s2):
    return [r for r in s2.reasons if "warning" in r.lower()]


# (a) a built backtest needs no declared adjustment, and the receipt records the type
def test_built_backtest_needs_no_declared_adjustment_and_receipt_records_the_type(tmp_path, monkeypatch):
    block = {k: v for k, v in BLOCK.items() if k != "adjustment"}
    s2 = _run(tmp_path, monkeypatch, block=block)
    assert _linear(s2)["verdict"] == "PASS", text(s2)
    assert _linear(s2)["adjustment"] == {"type": "total_return", "source": "harness"}


def test_built_backtest_records_the_type_even_when_the_block_declares_one(tmp_path, monkeypatch):
    s2 = _run(tmp_path, monkeypatch)
    assert _linear(s2)["adjustment"] == {"type": "total_return", "source": "harness"}


# (b) a user-brought backtest with an unknown adjustment gives a warning
def test_user_backtest_with_unknown_adjustment_gives_a_warning(tmp_path, monkeypatch):
    s2 = _run(tmp_path, monkeypatch, user=True)
    assert _linear(s2)["adjustment"] == {"type": "unknown", "source": "unanswered"}
    assert any("price adjustment" in w and "unknown" in w for w in _warnings(s2)), text(s2)


# (c) a user-brought backtest whose truncation test fails cannot PASS
def test_user_backtest_whose_truncation_fails_cannot_pass(tmp_path, monkeypatch):
    s2 = _run(tmp_path, monkeypatch, user=True, peek=True)
    assert s2.params["truncation"]["verdict"] == "FAIL"
    assert _linear(s2)["verdict"] != "PASS"
    assert "truncation test did not pass" in text(s2)


# (d) a built backtest whose truncation self-check fails: bug warning, profit check unchanged
def test_built_backtest_truncation_failure_is_a_bug_warning_and_does_not_change_the_profit_check(
        tmp_path, monkeypatch):
    s2 = _run(tmp_path, monkeypatch, peek=True)
    assert s2.params["truncation"]["verdict"] == "FAIL"
    assert any("bug" in w and "truncation" in w for w in _warnings(s2)), text(s2)
    # the planted peek also makes a real profit bug, so the owner's correction rule applies
    assert _linear(s2)["verdict"] == "INCONCLUSIVE", text(s2)
    assert "a bug was found; a correction has been requested and the run is re-checked after it is recorded" in text(s2)
    requests = list((tmp_path / "qrh_home" / "trials").glob("*/S2/correction/correction_request.txt"))
    assert len(requests) == 1, requests


# (e) built: an unexplained difference is a bug warning with day and amount
def test_built_backtest_unexplained_difference_is_a_bug_warning_with_day_and_amount(tmp_path, monkeypatch):
    s2 = _run(tmp_path, monkeypatch, gain=True)
    bugs = [w for w in _warnings(s2) if "bug" in w and "unexplained" in w]
    assert bugs, text(s2)
    assert "0.0005" in bugs[0] and "20" in bugs[0]  # amount and a day (a year starting 20xx)


def test_built_backtest_without_a_difference_has_no_bug_warning(tmp_path, monkeypatch):
    s2 = _run(tmp_path, monkeypatch)
    assert not [w for w in _warnings(s2) if "bug" in w], text(s2)


# (e, second half) the noise-scaled tolerance applies to a comparison with broker records
def test_broker_comparison_within_noise_tolerance_passes():
    from qrh.checks.pnl_linear import compare_with_broker

    rng = np.random.default_rng(7)
    model = rng.normal(0.0, 0.01, 500)
    broker = model + rng.normal(0.0, 0.0001, 500)  # honest zero-mean noise
    res = compare_with_broker(model, broker, capital_base=1.0, bars_per_year=252.0)
    assert res.verdict == "PASS", res.reasons


def test_broker_comparison_with_a_steady_gap_fails():
    from qrh.checks.pnl_linear import compare_with_broker

    rng = np.random.default_rng(7)
    model = rng.normal(0.0, 0.01, 500)
    broker = model - 0.0005 + rng.normal(0.0, 0.0001, 500)
    res = compare_with_broker(model, broker, capital_base=1.0, bars_per_year=252.0)
    assert res.verdict == "FAIL", res.reasons
