import json
import math

import pytest
from pydantic import ValidationError

from qrh.memory.registry import HypothesisRecord, Registry
from qrh.pipeline.receipt import (
    Override,
    Receipt,
    ReceiptTamperedError,
    Stage,
    Verdict,
    collect_environment,
    load_receipt,
    write_receipt,
)


def make_receipt(**kw):
    base = dict(
        trial_id="t1",
        stage=Stage.S0,
        hypothesis_hash="h" * 64,
        code_hash="c" * 64,
        data_hashes={"d": "x"},
        params={"a": 1},
        windows={"train": ("2020-01-01", "2020-06-30")},
        metrics={"m": 0.5},
        verdict=Verdict.PASS,
        reasons=[],
        environment={"python": "3.x"},
    )
    base.update(kw)
    return Receipt(**base)


def test_hash_excludes_created_at():
    a = make_receipt(created_at="2020-01-01T00:00:00+00:00")
    b = make_receipt(created_at="2030-01-01T00:00:00+00:00")
    assert a.receipt_hash() == b.receipt_hash()


def test_identical_inputs_identical_hash_and_changes_detected():
    assert make_receipt().receipt_hash() == make_receipt().receipt_hash()
    assert make_receipt().receipt_hash() != make_receipt(verdict=Verdict.FAIL).receipt_hash()
    assert make_receipt().receipt_hash() != make_receipt(parent_receipt_hash="p").receipt_hash()


def test_nan_metric_cannot_be_hashed():
    with pytest.raises(ValueError):
        make_receipt(metrics={"m": math.nan}).receipt_hash()


def test_bad_window_rejected():
    with pytest.raises(ValidationError):
        make_receipt(windows={"w": ("not-a-date", "2020-01-02")})


def test_override_needs_reason_and_who():
    with pytest.raises(ValidationError):
        Override(reason=" ", who="me")
    with pytest.raises(ValidationError):
        Override(reason="because", who="")


def test_environment_has_python_and_seeds():
    env = collect_environment(seeds={"main": 7})
    assert env["python"] and env["seeds"] == {"main": 7} and "pandas" in env["libraries"]


def test_write_and_load_round_trip(tmp_path):
    r = make_receipt()
    path = write_receipt(r, tmp_path)
    assert path == tmp_path / "receipts" / "t1" / "S0.json"
    assert load_receipt(path) == r


def test_tampered_receipt_detected(tmp_path):
    path = write_receipt(make_receipt(), tmp_path)
    doc = json.loads(path.read_text())
    doc["receipt"]["verdict"] = "FAIL"
    path.write_text(json.dumps(doc))
    with pytest.raises(ReceiptTamperedError):
        load_receipt(path)


def test_tampered_hash_field_detected(tmp_path):
    path = write_receipt(make_receipt(), tmp_path)
    doc = json.loads(path.read_text())
    doc["receipt_hash"] = "0" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(ReceiptTamperedError):
        load_receipt(path)


def test_garbage_file_is_tamper_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(ReceiptTamperedError):
        load_receipt(p)


def test_write_indexes_and_logs_override(tmp_path):
    reg = Registry(tmp_path / "r.sqlite3", id_factory=lambda: "t1")
    reg.register_hypothesis(
        HypothesisRecord(
            text="t", strategy_family="s", asset_class="a", frequency="f", parameter_space_size=1,
            thresholds={"k": 1}, difference_statement="d",
        )
    )
    r = make_receipt(override=Override(reason="documented", who="reviewer"))
    path = write_receipt(r, tmp_path, registry=reg)
    entry = reg.get_receipt_entry("t1", "S0")
    assert entry["receipt_hash"] == r.receipt_hash() and entry["verdict"] == "PASS"
    assert entry["path"] == str(path)
    assert reg.list_overrides("t1")[0]["reason"] == "documented"
    assert reg.list_overrides("t1")[0]["who"] == "reviewer"
    reg.close()


def test_diagnostics_are_excluded_from_the_hash_but_stored(tmp_path):
    a = make_receipt()
    b = make_receipt(diagnostics={"truncation_cached": True, "note": "run dependent"})
    assert a.receipt_hash() == b.receipt_hash()
    path = write_receipt(b, tmp_path)
    assert load_receipt(path).diagnostics == {"truncation_cached": True, "note": "run dependent"}
    assert load_receipt(path).receipt_hash() == a.receipt_hash()


def test_a_receipt_written_before_diagnostics_existed_still_verifies(tmp_path):
    path = write_receipt(make_receipt(), tmp_path)
    doc = json.loads(path.read_text())
    doc["receipt"].pop("diagnostics")  # the form on disk before the field existed
    path.write_text(json.dumps(doc))
    r = load_receipt(path)
    assert r.diagnostics == {} and r.receipt_hash() == doc["receipt_hash"]
