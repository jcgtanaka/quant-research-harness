"""End to end: the S2 cost comes from a confirmed broker record when cost_per_contract is not declared.

Each test runs the real S0 to S2 command line pipeline (example 01 copy, `reconstruction` block) and reads the S2
receipt. Synthetic broker names only. See the notes on the injected records below: the S2 stage does not pass
commission records to the check on its own, so the "record present" tests supply them by wrapping the registered
method entry (the same call S2 makes, with one extra argument).
"""

from __future__ import annotations

import dataclasses
import importlib.util
from datetime import date
from pathlib import Path

import pytest

from qrh.costs.broker_commission import parse_record
from qrh.methods.builtin import BUILTIN
from qrh.pipeline.receipt import Verdict

_spec = importlib.util.spec_from_file_location(
    "e2e_broker_futures_helpers", Path(__file__).with_name("test_s2_pnl_futures_optin.py"))
_fut = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fut)
run_s2, text = _fut.run_s2, _fut.text
FUTURES_BLOCK, FUTURES_EDIT, CPC = _fut.FUTURES_BLOCK, _fut.FUTURES_EDIT, _fut.CPC

BROKER = "Brokerage Alpha"
FUT_PRODUCT = "synthetic index future"
OPT_PRODUCT = "synthetic index option"
TODAY = date.today().isoformat()
LINEAR = "s2.pnl_linear"


def record(product, rate=CPC, broker=BROKER, status="confirmed"):
    return parse_record({
        "broker": broker, "product": product, "commission_per_contract": rate, "per_order_minimum": None,
        "currency": "USD", "last_updated_date": TODAY, "source_url": "https://example.invalid/commissions",
        "retrieved_on": TODAY, "tier_table": "flat", "extras": [{"name": "fees", "amount": 0.0}],
        "status": status, "confirmed_on": TODAY if status == "confirmed" else None, "confidence": "high",
    })


def inject_records(monkeypatch, records):
    """Make the S2 stage's call to the pnl.linear entry carry commission records (the stage itself passes none)."""
    method = BUILTIN.get(LINEAR, "pnl_linear@1")
    original = method.entry

    def with_records(*args, **kwargs):
        return original(*args, commission_records=records, **kwargs)

    # Method is frozen, so swap the registry slot for a copy that has the wrapped entry (undone after the test)
    monkeypatch.setitem(BUILTIN._methods, (LINEAR, "pnl_linear@1"), dataclasses.replace(method, entry=with_records))


def futures_block(**over):
    b = {k: v for k, v in FUTURES_BLOCK.items() if k != "cost_per_contract"}
    b.update(broker=BROKER, product=FUT_PRODUCT)
    b.update(over)
    return b


def options_block(**over):
    b = {
        "position_units": "quantity", "asset_class": "option", "publication_column": "pub", "max_staleness": 2,
        "adjustment": {"close": "none"}, "strike": 100.0, "expiry": "2099-12-19", "right": "call",
        "option_price_type": "mid", "multiplier": 100.0, "starting_capital": 100000.0,
        "contract_currency": "USD", "account_currency": "USD", "currency_rounding_unit": 0.01,
        "broker": BROKER, "product": OPT_PRODUCT, "point_in_time": "yes",
    }
    b.update(over)
    return b


def _futures(tmp_path, monkeypatch, block):
    return _fut._futures(tmp_path, monkeypatch, block)


def _options(tmp_path, monkeypatch, block):
    return run_s2(tmp_path, monkeypatch, reconstruction=block, spec_edit=_fut._drop_methods)


# ------------------------------------------------------------------ futures


def test_futures_without_a_record_is_blocked_naming_broker_and_product(tmp_path, monkeypatch):
    s2 = _futures(tmp_path, monkeypatch, futures_block())
    assert s2.verdict is Verdict.BLOCKED, text(s2)
    cost = s2.params["pnl_linear"]["cost"]
    assert cost["status"] == "BLOCKED" and cost["source"] == "broker_record"
    assert BROKER in cost["reason"] and FUT_PRODUCT in cost["reason"]
    assert BROKER in text(s2) and FUT_PRODUCT in text(s2)


def test_futures_cost_comes_from_the_confirmed_record(tmp_path, monkeypatch):
    inject_records(monkeypatch, [record(FUT_PRODUCT)])
    s2 = _futures(tmp_path, monkeypatch, futures_block())
    assert s2.verdict is Verdict.PASS, text(s2)
    p = s2.params["pnl_linear"]
    assert p["cost"]["status"] == "OK" and p["cost"]["source"] == "broker_record"
    assert p["cost"]["broker"] == BROKER and p["cost"]["product"] == FUT_PRODUCT
    assert p["cost"]["cost_per_contract"] == CPC and p["settings"]["cost_per_contract"] == CPC


def test_futures_record_of_another_broker_is_not_used(tmp_path, monkeypatch):
    inject_records(monkeypatch, [record(FUT_PRODUCT, broker="Brokerage Beta")])
    s2 = _futures(tmp_path, monkeypatch, futures_block())
    assert s2.verdict is Verdict.BLOCKED
    assert BROKER in s2.params["pnl_linear"]["cost"]["reason"]


def test_futures_unconfirmed_record_is_not_used(tmp_path, monkeypatch):
    inject_records(monkeypatch, [record(FUT_PRODUCT, status="proposed")])
    s2 = _futures(tmp_path, monkeypatch, futures_block())
    assert s2.verdict is Verdict.BLOCKED
    assert "not confirmed" in s2.params["pnl_linear"]["cost"]["reason"]


def test_futures_declared_cost_wins_over_a_record(tmp_path, monkeypatch):
    inject_records(monkeypatch, [record(FUT_PRODUCT, rate=9.0)])
    s2 = _futures(tmp_path, monkeypatch, futures_block(cost_per_contract=CPC))
    assert s2.verdict is Verdict.PASS, text(s2)
    assert s2.params["pnl_linear"]["settings"]["cost_per_contract"] == CPC


# ------------------------------------------------------------------ options


def test_options_without_a_record_is_blocked_naming_broker_and_product(tmp_path, monkeypatch):
    s2 = _options(tmp_path, monkeypatch, options_block())
    p = s2.params["pnl_linear"]
    assert s2.verdict is Verdict.BLOCKED, text(s2)
    assert p["cost"]["status"] == "BLOCKED", text(s2)
    assert BROKER in p["cost"]["reason"] and OPT_PRODUCT in p["cost"]["reason"]


def test_options_cost_check_uses_the_confirmed_record(tmp_path, monkeypatch):
    """The cost sub-check reads the record. The whole run is not expected to pass: example 01 has no option chain."""
    inject_records(monkeypatch, [record(OPT_PRODUCT, rate=0.65)])
    s2 = _options(tmp_path, monkeypatch, options_block())
    cost = s2.params["pnl_linear"]["cost"]
    assert cost["status"] == "OK" and cost["source"] == "broker_record", text(s2)
    assert cost["cost_per_contract"] == pytest.approx(0.65)
    assert cost["broker"] == BROKER and cost["product"] == OPT_PRODUCT


# ------------------------------------------- real S2 path, records read from <QRH_HOME>/costs (no injection)


def _write_cost_file(tmp_path, rec_dict):
    import json

    d = tmp_path / "qrh_home" / "costs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "alpha_future.json").write_text(json.dumps(rec_dict), encoding="utf-8")


def test_real_s2_reads_the_confirmed_record_from_the_costs_folder(tmp_path, monkeypatch):
    _write_cost_file(tmp_path, {
        "broker": BROKER, "product": FUT_PRODUCT, "commission_per_contract": CPC, "per_order_minimum": None,
        "currency": "USD", "last_updated_date": TODAY, "source_url": "https://example.invalid/commissions",
        "retrieved_on": TODAY, "tier_table": "flat", "extras": [{"name": "fees", "amount": 0.0}],
        "status": "confirmed", "confirmed_on": TODAY, "confidence": "high",
    })
    s2 = _futures(tmp_path, monkeypatch, futures_block())
    assert s2.verdict is Verdict.PASS, text(s2)
    cost = s2.params["pnl_linear"]["cost"]
    assert cost["status"] == "OK" and cost["source"] == "broker_record" and cost["cost_per_contract"] == CPC


def test_real_s2_shows_a_warning_for_an_unreadable_costs_file(tmp_path, monkeypatch):
    _write_cost_file(tmp_path, {
        "broker": BROKER, "product": FUT_PRODUCT, "commission_per_contract": CPC, "per_order_minimum": None,
        "currency": "USD", "last_updated_date": TODAY, "source_url": "https://example.invalid/commissions",
        "retrieved_on": TODAY, "tier_table": "flat", "extras": [{"name": "fees", "amount": 0.0}],
        "status": "confirmed", "confirmed_on": TODAY, "confidence": "high",
    })
    (tmp_path / "qrh_home" / "costs" / "broken_record.json").write_text("{ not json", encoding="utf-8")
    s2 = _futures(tmp_path, monkeypatch, futures_block())
    assert s2.verdict is Verdict.PASS, text(s2)  # the valid record is still used
    warnings = s2.params["pnl_linear"]["warnings"]
    assert any("broken_record.json" in w and "skipped" in w for w in warnings), warnings
    assert "broken_record.json" in text(s2)


def test_real_s2_without_a_costs_file_is_blocked(tmp_path, monkeypatch):
    s2 = _futures(tmp_path, monkeypatch, futures_block())
    assert s2.verdict is Verdict.BLOCKED, text(s2)
    assert BROKER in s2.params["pnl_linear"]["cost"]["reason"]
