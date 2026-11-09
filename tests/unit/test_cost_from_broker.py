"""Options and futures checks take the cost per contract from a confirmed broker commission record.

Precedence: a declared ``cost_per_contract`` wins and the records are not consulted. Without it, a declared broker and
product are resolved with ``resolve_cost`` against the records the caller supplies; no usable record keeps the cost
sub-check BLOCKED with the reason from ``resolve_cost``. The cost is never defaulted and never taken from another
broker or product. Synthetic broker names only; the check reads no files.
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

from qrh.checks.pnl_linear_run import check_pnl_linear
from qrh.checks.pnl_linear_futures import check_pnl_linear_futures
from qrh.checks.pnl_linear_options import check_pnl_linear_options
from qrh.costs.broker_commission import parse_record

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_cfb_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OPT = _load("test_pnl_options")
FUT = _load("test_pnl_futures")

BROKER, PRODUCT = "Brokerage Alpha", "synthetic index option"
FUT_PRODUCT = "synthetic index future"
RATE = 0.65  # equal to the synthetic ledger's cost, so the check passes
TODAY = date.today().isoformat()


def record(broker=BROKER, product=PRODUCT, rate=RATE, status="confirmed", extras=(), **over):
    d = {
        "broker": broker, "product": product, "commission_per_contract": rate, "per_order_minimum": None,
        "currency": "USD", "last_updated_date": TODAY, "source_url": "https://example.invalid/commissions",
        "retrieved_on": TODAY, "tier_table": "flat", "extras": extras if isinstance(extras, str) else [dict(e) for e in extras], "status": status,
        "confirmed_on": TODAY if status == "confirmed" else None, "confidence": "high",
    }
    d.update(over)
    return parse_record(d)


def run_opt(records, **over):
    result, frame, _ = OPT.make_run()
    block = OPT.block(cost_per_contract=OPT._DROP, broker=BROKER, product=PRODUCT, **over)
    return check_pnl_linear_options(result, frame, OPT.EXECUTION, block, OPT.PPY, commission_records=records)


def run_fut(records, **over):
    result, frame, _ = FUT.make_run(cpc=RATE)
    block = FUT.block(cost_per_contract=FUT._DROP, broker=BROKER, product=FUT_PRODUCT, **over)
    return check_pnl_linear_futures(result, frame, FUT.EXECUTION, block, FUT.PPY, commission_records=records)


# ------------------------------------------------------------------ options


def test_options_cost_from_confirmed_record_passes():
    v = run_opt([record(extras=[{"name": "fees", "amount": 0.0}])])
    assert v.verdict == "PASS", v.reasons
    assert v.details["cost"]["status"] == "OK"
    assert v.details["cost"]["cost_per_contract"] == RATE
    assert v.details["cost"]["source"] == "broker_record"
    assert v.details["cost"]["broker"] == BROKER


def test_options_unlisted_extras_margin_is_flagged_in_result():
    v = run_opt([record(extras="not checked", rate=RATE / 1.25)])
    assert v.details["cost"]["margin_flagged"] is True
    assert abs(v.details["cost"]["cost_per_contract"] - RATE) < 1e-9
    assert any("margin" in r.lower() for r in v.reasons)


def test_options_no_record_is_blocked_naming_broker_and_product():
    v = run_opt([])
    assert v.verdict == "BLOCKED"
    assert v.details["cost"]["status"] == "BLOCKED"
    assert BROKER in v.details["cost"]["reason"] and PRODUCT in v.details["cost"]["reason"]
    assert v.details["cost"].get("cost_per_contract") is None


def test_options_record_for_another_broker_is_not_used():
    v = run_opt([record(broker="Brokerage Beta")])
    assert v.verdict == "BLOCKED"
    assert v.details["cost"]["status"] == "BLOCKED"


def test_options_unconfirmed_record_is_not_used():
    v = run_opt([record(status="proposed")])
    assert v.verdict == "BLOCKED"
    assert "not confirmed" in v.details["cost"]["reason"]


def test_options_no_broker_and_no_cost_still_blocked():
    result, frame, _ = OPT.make_run()
    block = OPT.block(cost_per_contract=OPT._DROP)
    v = check_pnl_linear_options(result, frame, OPT.EXECUTION, block, OPT.PPY, commission_records=[record()])
    assert v.verdict == "BLOCKED"
    assert v.details["cost"]["status"] == "BLOCKED"


def test_options_declared_cost_takes_precedence():
    # a record with a wildly different rate must be ignored when cost_per_contract is declared
    result, frame, _ = OPT.make_run()
    block = OPT.block(broker=BROKER, product=PRODUCT)
    v = check_pnl_linear_options(result, frame, OPT.EXECUTION, block, OPT.PPY, commission_records=[record(rate=9.0)])
    assert v.verdict == "PASS", v.reasons
    assert v.details["cost"]["cost_per_contract"] == OPT.CPC
    assert v.details["cost"].get("source") != "broker_record"


# ------------------------------------------------------------------ futures


def test_futures_cost_from_confirmed_record_passes():
    v = run_fut([record(product=FUT_PRODUCT, extras=[{"name": "fees", "amount": 0.0}])])
    assert v.verdict == "PASS", v.reasons
    assert v.details["cost"]["status"] == "OK"
    assert v.details["cost"]["cost_per_contract"] == RATE
    assert v.details["cost"]["broker"] == BROKER


def test_futures_no_record_is_blocked_naming_broker_and_product():
    v = run_fut([])
    assert v.verdict == "BLOCKED"
    assert v.details["cost"]["status"] == "BLOCKED"
    assert BROKER in v.details["cost"]["reason"] and FUT_PRODUCT in v.details["cost"]["reason"]


def test_futures_record_for_another_broker_is_not_used():
    v = run_fut([record(broker="Brokerage Beta", product=FUT_PRODUCT)])
    assert v.verdict == "BLOCKED"


def test_futures_declared_cost_takes_precedence():
    result, frame, _ = FUT.make_run()
    block = FUT.block(broker=BROKER, product=FUT_PRODUCT)
    recs = [record(product=FUT_PRODUCT, rate=9.0)]
    v = check_pnl_linear_futures(result, frame, FUT.EXECUTION, block, FUT.PPY, commission_records=recs)
    assert v.verdict == "PASS", v.reasons
    assert v.details["settings"]["cost_per_contract"] == FUT.CPC


# ------------------------------------------------------------------ entry point forwarding


def _entry_opt(**kw):
    result, frame, _ = OPT.make_run()
    block = OPT.block(cost_per_contract=OPT._DROP, broker=BROKER, product=PRODUCT)
    return check_pnl_linear(result, frame, OPT.EXECUTION, block, OPT.PPY, **kw)


def _entry_fut(**kw):
    result, frame, _ = FUT.make_run(cpc=RATE)
    block = FUT.block(cost_per_contract=FUT._DROP, broker=BROKER, product=FUT_PRODUCT)
    return check_pnl_linear(result, frame, FUT.EXECUTION, block, FUT.PPY, **kw)


def test_entry_point_forwards_records_to_options():
    v = _entry_opt(commission_records=[record(extras=[{"name": "fees", "amount": 0.0}])], as_of=date.today())
    assert v.verdict == "PASS", v.reasons
    assert v.details["cost"]["source"] == "broker_record"


def test_entry_point_forwards_records_to_futures():
    v = _entry_fut(commission_records=[record(product=FUT_PRODUCT, extras=[{"name": "fees", "amount": 0.0}])])
    assert v.verdict == "PASS", v.reasons
    assert v.details["cost"]["source"] == "broker_record"


def test_entry_point_without_records_is_blocked_naming_broker_and_product():
    for v, product in ((_entry_opt(), PRODUCT), (_entry_fut(), FUT_PRODUCT)):
        assert v.verdict == "BLOCKED"
        assert BROKER in v.details["cost"]["reason"] and product in v.details["cost"]["reason"]


def test_futures_without_cost_or_broker_is_still_a_config_error():
    result, frame, _ = FUT.make_run()
    block = FUT.block(cost_per_contract=FUT._DROP)
    v = check_pnl_linear_futures(result, frame, FUT.EXECUTION, block, FUT.PPY, commission_records=[record()])
    assert v.verdict == "BLOCKED"
    assert "cost_per_contract" in " ".join(v.reasons)
