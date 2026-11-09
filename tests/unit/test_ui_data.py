"""Read-only dashboard data layer: everything comes from the registry and stored receipts."""

from __future__ import annotations

import pytest

from qrh.memory.registry import HypothesisRecord, Registry
from qrh.pipeline.receipt import Receipt, Stage, Verdict, receipt_path, write_receipt
from qrh.ui import data


def _record() -> HypothesisRecord:
    return HypothesisRecord(
        text="synthetic momentum on synthetic bars",
        strategy_family="momentum",
        asset_class="synthetic",
        frequency="1d",
        parameter_space_size=4,
        thresholds={"min_sharpe": 0.5},
        difference_statement="synthetic only",
    )


def _receipt(tid: str, stage: Stage, verdict: Verdict, reasons: list[str]) -> Receipt:
    return Receipt(trial_id=tid, stage=stage, hypothesis_hash="h" * 8, code_hash="c" * 8,
                   verdict=verdict, reasons=reasons)


@pytest.fixture
def home(tmp_path):
    reg = Registry(tmp_path / "registry.sqlite3")
    reg.register_hypothesis(_record(), trial_id="t1")
    reg.register_hypothesis(_record(), trial_id="t2")
    write_receipt(_receipt("t1", Stage.S0, Verdict.PASS, []), tmp_path, reg)
    write_receipt(_receipt("t1", Stage.S1, Verdict.FAIL, ["synthetic gap"]), tmp_path, reg)
    reg.conn.commit()
    reg.close()
    return tmp_path


def test_list_trials_shape(home):
    rows = data.list_trials(home)
    assert {r["trial_id"] for r in rows} == {"t1", "t2"}
    t1 = next(r for r in rows if r["trial_id"] == "t1")
    assert t1["family"] == "momentum|synthetic|1d"
    assert t1["strategy_family"] == "momentum"
    assert t1["created_at"]
    assert t1["stage_verdicts"] == {"S0": "PASS", "S1": "FAIL"}
    t2 = next(r for r in rows if r["trial_id"] == "t2")
    assert t2["stage_verdicts"] == {}


def test_trial_detail_reads_receipts(home):
    d = data.trial_detail(home, "t1")
    assert d["trial_id"] == "t1"
    assert d["stages"]["S0"]["verdict"] == "PASS"
    assert d["stages"]["S1"] == {"verdict": "FAIL", "reasons": ["synthetic gap"], "error": None}


def test_trial_detail_tampered_receipt_is_an_error_not_a_verdict(home):
    path = receipt_path(home, "t1", Stage.S1)
    path.write_text(path.read_text().replace("synthetic gap", "edited"), encoding="utf-8")
    d = data.trial_detail(home, "t1")
    assert d["stages"]["S1"]["verdict"] is None
    assert d["stages"]["S1"]["error"]
    assert d["stages"]["S0"]["verdict"] == "PASS"


def test_unknown_and_invalid_trial_ids(home):
    for tid in ("nope", "../etc", "a b", ".hidden"):
        with pytest.raises(data.DataError) as exc:
            data.trial_detail(home, tid)
        assert exc.value.status == 404


def test_warnings_reuse_readiness_collect(home):
    w = data.trial_warnings(home, "t1")
    assert w["trial_id"] == "t1"
    assert w["verdict"] == "NOT RUN"  # no S7 receipt: reported as readiness reports it
    assert isinstance(w["warnings"], list) and w["warnings"]
    assert {"id", "stage", "severity", "message"} <= set(w["warnings"][0])
    with pytest.raises(data.DataError) as exc:
        data.trial_warnings(home, "nope")
    assert exc.value.status == 404


def test_missing_registry_is_clear_error_and_not_created(tmp_path):
    with pytest.raises(data.DataError) as exc:
        data.list_trials(tmp_path)
    assert exc.value.status == 503
    assert "registry" in exc.value.message
    assert not (tmp_path / "registry.sqlite3").exists()
