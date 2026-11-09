"""Loading confirmed broker commission records from <QRH_HOME>/costs/*.json. Synthetic broker names only."""

from __future__ import annotations

import json
from datetime import date

from qrh.costs.store import load_commission_records

TODAY = date.today().isoformat()


def raw(product="synthetic index future", broker="Brokerage Alpha", status="confirmed"):
    return {
        "broker": broker, "product": product, "commission_per_contract": 2.0, "per_order_minimum": None,
        "currency": "USD", "last_updated_date": TODAY, "source_url": "https://example.invalid/commissions",
        "retrieved_on": TODAY, "tier_table": "flat", "extras": [{"name": "fees", "amount": 0.0}],
        "status": status, "confirmed_on": TODAY if status == "confirmed" else None, "confidence": "high",
    }


def put(home, name, data):
    d = home / "costs"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")


def test_missing_folder_gives_no_records_and_no_warnings(tmp_path):
    assert load_commission_records(tmp_path) == ([], [])


def test_confirmed_records_are_loaded_in_name_order(tmp_path):
    put(tmp_path, "b.json", raw(product="synthetic index option"))
    put(tmp_path, "a.json", raw())
    records, warnings = load_commission_records(tmp_path)
    assert [r.product for r in records] == ["synthetic index future", "synthetic index option"]
    assert all(r.status == "confirmed" for r in records) and warnings == []


def test_unreadable_file_is_skipped_with_a_warning(tmp_path):
    put(tmp_path, "bad.json", "{not json")
    put(tmp_path, "good.json", raw())
    records, warnings = load_commission_records(tmp_path)
    assert len(records) == 1
    assert len(warnings) == 1 and "bad.json" in warnings[0]


def test_invalid_record_is_skipped_with_a_warning(tmp_path):
    bad = raw()
    del bad["currency"]
    put(tmp_path, "invalid.json", bad)
    records, warnings = load_commission_records(tmp_path)
    assert records == [] and "invalid.json" in warnings[0]


def test_json_that_is_not_an_object_is_skipped_with_a_warning(tmp_path):
    put(tmp_path, "list.json", "[1, 2]")
    records, warnings = load_commission_records(tmp_path)
    assert records == [] and "list.json" in warnings[0]


def test_unconfirmed_record_is_not_returned_and_is_named_in_a_warning(tmp_path):
    put(tmp_path, "proposed.json", raw(status="proposed"))
    records, warnings = load_commission_records(tmp_path)
    assert records == [] and "proposed.json" in warnings[0] and "not confirmed" in warnings[0]


def test_other_file_types_are_ignored(tmp_path):
    put(tmp_path, "notes.txt", "hello")
    assert load_commission_records(tmp_path) == ([], [])
