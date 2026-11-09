"""Tests for the broker commission record validator and store (synthetic data only)."""

from __future__ import annotations

import ast
import json
import socket
from datetime import date
from pathlib import Path

import pytest

from qrh.costs import broker_commission as bc

AS_OF = date(2026, 10, 1)


def rec(**over):
    base = {
        "broker": "Broker A",
        "product": "Product X",
        "commission_per_contract": 1.0,
        "per_order_minimum": 1.5,
        "currency": "USD",
        "last_updated_date": "2026-09-01",
        "source_url": "https://broker-a.example/commissions",
        "retrieved_on": "2026-10-01",
        "tier_table": "flat",
        "extras": [{"name": "exchange", "amount": 0.2}, {"name": "clearing", "amount": 0.1}],
        "status": "confirmed",
        "confirmed_on": "2026-10-01",
        "confidence": "high",
        "source_snapshot_sha256": "a" * 64,
        "confirmed_value": 1.0,
    }
    base.update(over)
    return bc.parse_record(base)


def resolve(records, equity=10_000, **kw):
    return bc.resolve_cost(records, "Broker A", "Product X", equity, AS_OF, **kw)


def test_valid_confirmed_record_is_usable():
    r = resolve([rec()])
    assert r.usable
    assert r.commission_per_contract == 1.0
    assert r.extras_total == pytest.approx(0.3)
    assert r.margin_flagged is False
    assert r.all_in_per_contract == pytest.approx(1.3)
    assert r.warnings == ()
    assert r.block_reasons == ()


def test_proposed_record_is_not_usable():
    r = resolve([rec(status="proposed", confirmed_on=None, confirmed_value=None)])
    assert not r.usable
    assert r.commission_per_contract is None
    assert any("not confirmed" in x for x in r.block_reasons)


def test_confirmed_without_confirmation_date_is_rejected():
    with pytest.raises(bc.RecordError):
        rec(confirmed_on=None)


def test_stale_beyond_12_months_is_blocked():
    r = resolve([rec(last_updated_date="2025-09-30")])
    assert not r.usable
    assert any("stale" in x for x in r.block_reasons)


def test_exactly_12_months_is_still_usable_with_warning():
    r = resolve([rec(last_updated_date="2025-10-01")])
    assert r.usable
    assert any("6 months" in w for w in r.warnings)


def test_warn_at_6_months_but_usable():
    r = resolve([rec(last_updated_date="2026-04-01")])
    assert r.usable
    assert any("6 months" in w for w in r.warnings)


def test_under_6_months_no_warning():
    assert resolve([rec(last_updated_date="2026-04-02")]).warnings == ()


def test_extras_not_checked_adds_flagged_margin():
    r = resolve([rec(extras="not checked")])
    assert r.usable
    assert r.margin_flagged is True
    assert r.extras_margin == pytest.approx(0.25)
    assert r.all_in_per_contract == pytest.approx(1.25)
    assert any("not checked" in w for w in r.warnings)


def test_extras_margin_is_configurable_and_recorded():
    r = resolve([rec(extras="not checked")], margin_fraction=0.5)
    assert r.extras_margin == pytest.approx(0.5)
    assert bc.audit_entry(r)["extras_margin"] == pytest.approx(0.5)
    assert bc.audit_entry(r)["margin_flagged"] is True


TIERS = [
    {"min_equity": 0, "max_equity": 50_000, "commission_per_contract": 1.0},
    {"min_equity": 50_000, "max_equity": 200_000, "commission_per_contract": 0.8},
]


def tiered(**over):
    return rec(tier_table=TIERS, confirmed_value=None, **over)


def test_tier_inferred_from_declared_equity():
    assert resolve([tiered()], equity=10_000).commission_per_contract == 1.0
    assert resolve([tiered()], equity=50_000).commission_per_contract == 0.8
    assert resolve([tiered()], equity=120_000).commission_per_contract == 0.8


def test_no_tier_match_blocks_and_asks_user():
    r = resolve([tiered()], equity=500_000)
    assert not r.usable
    assert r.commission_per_contract is None
    assert r.ask_user is True
    assert any("tier" in x for x in r.block_reasons)


def test_missing_equity_blocks_tiered_cost():
    r = resolve([tiered()], equity=None)
    assert not r.usable and r.ask_user


def test_overlapping_tiers_block():
    overlap = [
        {"min_equity": 0, "max_equity": 60_000, "commission_per_contract": 1.0},
        {"min_equity": 50_000, "max_equity": 200_000, "commission_per_contract": 0.8},
    ]
    r = resolve([rec(tier_table=overlap, confirmed_value=None)], equity=55_000)
    assert not r.usable


def test_conflicting_records_block_show_both_and_never_apply():
    a = rec(source_url="https://broker-a.example/one")
    b = rec(source_url="https://broker-a.example/two", commission_per_contract=1.4, confirmed_value=1.4)
    r = resolve([a, b])
    assert not r.usable
    assert r.commission_per_contract is None
    assert r.all_in_per_contract is None
    assert {c["source_url"] for c in r.conflicts} == {
        "https://broker-a.example/one",
        "https://broker-a.example/two",
    }
    assert r.placeholder_higher == 1.4


def test_identical_figures_from_two_records_are_not_a_conflict():
    a = rec(source_url="https://broker-a.example/one")
    b = rec(source_url="https://broker-a.example/two")
    assert resolve([a, b]).usable


def test_unknown_last_updated_blocks():
    r = resolve([rec(last_updated_date="unknown")])
    assert not r.usable
    assert any("last-updated" in x for x in r.block_reasons)


def test_missing_record_blocks_never_zero():
    r = resolve([])
    assert not r.usable
    assert r.commission_per_contract is None
    assert r.all_in_per_contract is None


def test_other_brokers_figure_never_reused():
    other = rec(broker="Broker B")
    r = resolve([other])
    assert not r.usable
    assert r.commission_per_contract is None
    other_product = rec(product="Product Y")
    assert not resolve([other_product]).usable


def test_schedule_change_since_confirmation_requires_reconfirm():
    r = resolve([rec(commission_per_contract=1.2, confirmed_value=1.0)])
    assert not r.usable
    assert r.ask_user
    assert any("re-confirm" in x for x in r.block_reasons)


def test_confirmed_record_stores_confirmation_date_and_value():
    r = resolve([rec()])
    e = bc.audit_entry(r)
    assert e["record"]["confirmed_on"] == "2026-10-01"
    assert e["record"]["confirmed_value"] == 1.0
    assert e["source_snapshot_sha256"] == "a" * 64


def test_invalid_records_rejected():
    for bad in (
        {"commission_per_contract": "1.0"},
        {"commission_per_contract": -1},
        {"status": "maybe"},
        {"confidence": "certain"},
        {"extras": "none"},
        {"currency": ""},
        {"last_updated_date": "01/09/2026"},
    ):
        with pytest.raises(bc.RecordError):
            rec(**bad)


def test_store_run_audit_writes_confirmed_record_and_hash(tmp_path: Path):
    r = resolve([rec()])
    path = bc.store_run_audit(tmp_path, r)
    data = json.loads(path.read_text())
    assert data["source_snapshot_sha256"] == "a" * 64
    assert data["record"]["broker"] == "Broker A"
    assert path.parent == tmp_path


def test_store_run_audit_refuses_blocked_result(tmp_path: Path):
    with pytest.raises(bc.RecordError):
        bc.store_run_audit(tmp_path, resolve([]))


def test_snapshot_hash_helper():
    assert bc.snapshot_sha256(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_harness_never_fetches(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    assert resolve([rec()]).usable

    tree = ast.parse(Path(bc.__file__).read_text())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    assert not imported & {"socket", "urllib", "http", "requests", "httpx", "aiohttp", "ssl"}
