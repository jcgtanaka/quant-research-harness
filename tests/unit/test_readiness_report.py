"""Readiness report: data and hash, chart documents, deterministic PDF, manifest, automatic generation at S7."""

import json

import pytest

from readiness_helpers import TID, clean_run, make_registry, put_receipt
from qrh.pipeline.receipt import Verdict
from qrh.readiness import collect as rc
from qrh.readiness import data as rd
from qrh.readiness import report as rr
from qrh.reports import manifest


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    return h


@pytest.fixture
def reg(home):
    r = make_registry(home)
    yield r
    r.close()


def messy(home, reg):
    """Conditional pass with warnings at several stages and severities."""
    clean_run(home, reg, s2_methods={}, s2_params={"lint": {"verdict": "PASS", "warnings": [
        {"rule": "LA005", "at": "adapter.py:9", "snippet": "x / x.mean()"}], "suppressed": []}})
    cond = [{"question_id": "rf", "scope": "dataset:prices", "stage": "S1", "text": "How was it made?",
             "status": "unknown", "risk": "Possible look-ahead in the input."}]
    put_receipt(home, reg, "S7", verdict=Verdict.PASS_CONDITIONAL, params={"conditions": cond})
    reg.log_override(TID, "S3", "accepted the thin sample knowingly", "alice")


def data_of(home, reg):
    return rd.build_data(rc.collect(home, TID, reg))


# --- data and hash -------------------------------------------------------------------------------------

def test_data_has_verdict_counts_and_every_warning(home, reg):
    messy(home, reg)
    d = data_of(home, reg)
    assert d.doc["final_verdict"] == "PASS_CONDITIONAL" and d.doc["final_verdict_display"] == "PASS · CONDITIONAL"
    n = len(d.doc["warnings"])
    assert sum(sum(v.values()) for v in d.doc["counts"].values()) == n == sum(d.doc["totals"].values())
    assert d.doc["counts"]["S2"]["medium"] >= 2 and d.doc["counts"]["S7"]["high"] == 1
    assert set(d.doc["counts"]) >= {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "ALL"}
    assert len(d.hash) == 64 and d.warning_ids == sorted(d.warning_ids)


def test_the_hash_is_deterministic_and_changes_with_any_warning_change(home, reg):
    messy(home, reg)
    a, b = data_of(home, reg), data_of(home, reg)
    assert a.hash == b.hash and a.doc == b.doc
    reg.log_override(TID, "S4", "one more knowing exception", "bob")
    c = data_of(home, reg)
    assert c.hash != a.hash and len(c.doc["warnings"]) == len(a.doc["warnings"]) + 1


def test_a_changed_message_or_verdict_changes_the_hash_even_with_the_same_ids(home, reg):
    messy(home, reg)
    a = data_of(home, reg)
    put_receipt(home, reg, "S7", verdict=Verdict.PASS_CONDITIONAL, params={"conditions": [
        {"question_id": "rf", "scope": "dataset:prices", "stage": "S1", "text": "How was it made?",
         "status": "unknown", "risk": "A different risk text."}]})
    b = data_of(home, reg)
    assert a.warning_ids == b.warning_ids and a.hash != b.hash


# --- chart documents and figures --------------------------------------------------------------------------

def test_page_one_states_the_verdict_counts_and_acknowledgement_status(home, reg):
    messy(home, reg)
    d = data_of(home, reg)
    docs = rr.chart_docs(d, (False, "no acknowledgement has been signed"))
    first = docs[0]
    assert first["chart_type"] == "readiness_summary" and first["verdict_display"] == "PASS · CONDITIONAL"
    assert first["ack"] == {"acknowledged": False, "detail": "no acknowledgement has been signed"}
    assert first["report_hash"] == d.hash and first["counts"] == d.doc["counts"]


def test_every_warning_appears_exactly_once_grouped_by_stage_in_order(home, reg):
    messy(home, reg)
    d = data_of(home, reg)
    docs = rr.chart_docs(d, (False, "x"))
    seen = [e["warning"]["id"] for p in docs[1:] for e in p["entries"] if e["type"] == "warning"]
    assert sorted(seen) == d.warning_ids and len(seen) == len(set(seen))
    stages = [e["stage"] for p in docs[1:] for e in p["entries"] if e["type"] == "header"]
    assert stages == sorted(stages, key=lambda s: ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "ALL"].index(s))
    assert all(p["pages"] == len(docs) - 1 for p in docs[1:])


def test_many_warnings_paginate_and_no_page_overflows(home, reg):
    clean_run(home, reg)
    for i in range(60):
        reg.log_override(TID, "S3", f"synthetic exception number {i} " + "very long reason " * 8, "alice")
    docs = rr.chart_docs(data_of(home, reg), (False, "x"))
    assert len(docs) > 3 and all(p["used"] <= rr.PAGE_UNITS for p in docs[1:])


@pytest.mark.parametrize("ack", [(False, "pending"), (True, "signed")])
def test_figures_carry_footer_marker_and_status_by_text(home, reg, ack):
    messy(home, reg)
    d = data_of(home, reg)
    figs = rr.readiness_figures(rr.chart_docs(d, ack))
    assert len(figs) >= 2
    for fig in figs:
        texts = [t.get_text() for t in fig.texts]
        assert any(TID in t and d.hash[:12] in t for t in texts), "footer: trial id and report hash"
        assert any("not investment advice" in t for t in texts)
    page1 = [t.get_text() for t in figs[0].texts]
    assert any("PASS · CONDITIONAL" in t for t in page1)
    assert any(("ACKNOWLEDGED" if ack[0] else "PENDING") in t for t in page1)
    body = " ".join(t.get_text() for f in figs[1:] for t in f.texts)
    assert "▲ HIGH" in body and "◆ MEDIUM" in body and "● LOW" in body  # marker plus word, never colour alone
    assert "Resolve:" in body


# --- render ---------------------------------------------------------------------------------------------------

def test_render_writes_data_chart_json_and_pdf_and_records_the_manifest(home, reg):
    messy(home, reg)
    files = rr.render_readiness(TID, home, reg)
    base = home / "reports" / TID
    assert (base / "readiness.pdf").read_bytes().startswith(b"%PDF")
    assert json.loads((base / "readiness.json").read_text())["trial_id"] == TID
    chart = json.loads((base / "readiness.chart.json").read_text())
    assert chart["stage"] == "readiness" and chart["figures"][0]["chart_type"] == "readiness_summary"
    entries = {(f["kind"], f["path"]) for f in manifest.read_manifest(home, TID)["files"] if f["stage"] == "readiness"}
    assert entries == {("pdf", "readiness.pdf"), ("chart_data", "readiness.chart.json"), ("data", "readiness.json")}
    assert {f.kind for f in files} == {"pdf", "chart_data", "data"}


def test_render_is_byte_deterministic(home, reg):
    messy(home, reg)
    rr.render_readiness(TID, home, reg)
    base = home / "reports" / TID
    first = {n: (base / n).read_bytes() for n in ("readiness.pdf", "readiness.chart.json", "readiness.json")}
    rr.render_readiness(TID, home, reg)
    assert first == {n: (base / n).read_bytes() for n in first}


def test_clearing_reports_for_a_fresh_run_drops_the_readiness_files(home, reg):
    messy(home, reg)
    rr.render_readiness(TID, home, reg)
    from qrh.pipeline.receipt import Stage

    manifest.clear_from(home, TID, Stage.S1)
    assert not (home / "reports" / TID / "readiness.pdf").exists()
    assert not [f for f in manifest.read_manifest(home, TID)["files"] if f["stage"] == "readiness"]


def test_s7_rendering_generates_the_readiness_report_and_a_failure_never_changes_the_verdict(home, reg, monkeypatch):
    from qrh.reports.render import render_after_stage

    messy(home, reg)
    s7 = put_receipt(home, reg, "S7", verdict=Verdict.PASS_CONDITIONAL, params={"conditions": []})
    render_after_stage(home, reg, s7)
    assert (home / "reports" / TID / "readiness.pdf").is_file()

    def boom(*a, **k):
        raise RuntimeError("drawing failed")

    monkeypatch.setattr(rr, "render_readiness", boom)
    render_after_stage(home, reg, s7)  # must not raise
    doc = manifest.read_manifest(home, TID)
    assert any(e["stage"] == "readiness" and "drawing failed" in e["error"] for e in doc["errors"])
    assert reg.get_receipt_entry(TID, "S7")["verdict"] == "PASS_CONDITIONAL"


def test_the_hash_ignores_where_files_live_but_not_the_receipts(home, reg, tmp_path):
    messy(home, reg)
    a = data_of(home, reg)
    moved = {**a.doc, "warnings": [{**w, "receipt_path": "/elsewhere/" + str(i)} for i, w in enumerate(a.doc["warnings"])]}
    from qrh.hashing import hash_json

    assert hash_json(rd._hashed(moved)) == a.hash
    put_receipt(home, reg, "S2", methods={}, params={"lint": {"verdict": "PASS", "warnings": [
        {"rule": "LA005", "at": "adapter.py:9", "snippet": "x / x.mean()"}], "suppressed": []}, "again": 1})
    b = data_of(home, reg)
    assert a.warning_ids == b.warning_ids and a.hash != b.hash   # a new receipt is new evidence
