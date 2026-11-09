"""S2 dispatch of the opt-in pnl.linear check to the futures branch.

A `reconstruction` block with position_units `quantity` and asset_class `futures` is checked by
check_pnl_linear_futures; every other block keeps the behaviour of check_pnl_linear. Synthetic data only: example 01
with an adapter edit that reports currency-denominated PnL of whole contracts (the position is sign +-1).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from qrh.adapters.base import RunResult
from qrh.checks.pnl_linear_run import check_pnl_linear
from qrh.pipeline.receipt import Verdict

_spec = importlib.util.spec_from_file_location(
    "s2_pnl_linear_optin_helpers", Path(__file__).with_name("test_s2_pnl_linear_optin.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
BLOCK, run_s2, text = _helpers.BLOCK, _helpers.run_s2, _helpers.text

LINEAR = "s2.pnl_linear"
LINEAR_VER = "pnl_linear@1"
MULT, CPC, CAPITAL = 5.0, 2.0, 100000.0
FUTURES_BLOCK = {
    "position_units": "quantity",
    "asset_class": "futures",
    "adjustment": {"close": "back_adjusted"},
    "multiplier": MULT,
    "cost_per_contract": CPC,
    "starting_capital": CAPITAL,
    "contract_currency": "USD",
    "account_currency": "USD",
    "publication_column": "pub",
    "max_staleness": 2,
    "max_unexplained_bp_per_year": 50.0,
    "max_unexplained_share": 0.2,
}
# reported return = (contracts held * price change * multiplier - cost per contract * contracts traded) / capital
FUTURES_EDIT = (f"((position.shift(1) * px.diff() * {MULT} - {CPC} * change.abs().shift(1)) / {CAPITAL})")


def _drop_methods(spec):
    spec.pop("methods", None)  # the other opt-in check reads weights; keep it out of the futures runs


def _futures(tmp_path, monkeypatch, block):
    return run_s2(tmp_path, monkeypatch, reconstruction=block, adapter_edit=FUTURES_EDIT, spec_edit=_drop_methods)


def test_futures_quantity_block_runs_the_futures_check_and_records_method_and_version(tmp_path, monkeypatch):
    s2 = _futures(tmp_path, monkeypatch, FUTURES_BLOCK)
    p = s2.params["pnl_linear"]
    assert p["verdict"] != "BLOCKED", text(s2)
    assert s2.verdict is Verdict.PASS, text(s2)
    assert s2.methods[LINEAR] == LINEAR_VER
    assert s2.params["non_default_methods"][LINEAR] == LINEAR_VER
    assert p["method"] == LINEAR and p["version"] == LINEAR_VER
    # fields that only the futures branch writes
    assert p["settings"]["multiplier"] == MULT and p["settings"]["cost_per_contract"] == CPC
    assert p["settings"]["capital_base"] == CAPITAL
    assert "margin" in p and "roll_bars" in p and "roll_bars_excluded" in p
    assert "pnl_linear" in s2.params["artifacts"]


def test_futures_block_missing_the_multiplier_is_blocked_naming_the_field(tmp_path, monkeypatch):
    block = {k: v for k, v in FUTURES_BLOCK.items() if k != "multiplier"}
    s2 = _futures(tmp_path, monkeypatch, block)
    assert s2.verdict is Verdict.BLOCKED
    assert s2.params["pnl_linear"]["verdict"] == "BLOCKED"
    assert "multiplier" in text(s2)


def test_non_futures_quantity_block_keeps_the_existing_blocked(tmp_path, monkeypatch):
    block = {**BLOCK, "position_units": "quantity"}
    s2 = run_s2(tmp_path, monkeypatch, reconstruction=block)
    assert s2.verdict is Verdict.BLOCKED
    assert "position_units 'quantity' needs a capital base that is not declared" in text(s2)
    assert "multiplier" not in s2.params["pnl_linear"].get("settings", {})


def test_equity_weights_block_is_unchanged(tmp_path, monkeypatch):
    s2 = run_s2(tmp_path, monkeypatch, reconstruction=BLOCK)
    assert s2.verdict is Verdict.PASS, text(s2)
    p = s2.params["pnl_linear"]
    assert p["settings"]["capital_base"] == 1.0
    assert "multiplier" not in p["settings"] and "margin" not in p and "roll_bars" not in p


def test_dispatch_is_only_for_the_futures_quantity_pair(monkeypatch):
    calls = []
    import qrh.checks.pnl_linear_futures as fut

    def spy(*args):
        calls.append(args)
        return "futures-verdict"

    monkeypatch.setattr(fut, "check_pnl_linear_futures", spy)
    res = RunResult(pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame())
    frame = pd.DataFrame()
    check_pnl_linear(res, frame, None, FUTURES_BLOCK, 252.0)
    assert len(calls) == 1
    # futures with weights, and quantity with another asset class, stay on the weights path
    check_pnl_linear(res, frame, None, {**FUTURES_BLOCK, "position_units": "weight_start_of_bar"}, 252.0)
    check_pnl_linear(res, frame, None, {**BLOCK, "position_units": "quantity"}, 252.0)
    check_pnl_linear(res, frame, None, None, 252.0)
    assert len(calls) == 1
