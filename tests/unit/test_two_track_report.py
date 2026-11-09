"""Track 2 answers in the readiness report: an information section only, from a stored file; no verdict change."""

import json

import pytest

from readiness_helpers import TID, clean_run, make_registry
from qrh.checks.two_track import QUESTIONS
from qrh.readiness import collect as rc
from qrh.readiness import data as rd
from qrh.readiness import report as rr

SETTLED = {
    "price_source": "broker", "price_adjustment": "total_return", "price_timing": "next_open",
    "bar_type": "time", "trade_size_rounding": "whole_units", "cost_basis": "per_trade",
    "cash_interest": "included", "borrow_cost": "included", "dividends": "included",
    "halts_and_stale": "handled_declared",
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    return h


@pytest.fixture
def reg(home):
    r = make_registry(home)
    clean_run(home, r)
    yield r
    r.close()


def put_answers(home, answers):
    d = home / "trials" / TID
    d.mkdir(parents=True, exist_ok=True)
    (d / "two_track_answers.json").write_text(json.dumps(answers), encoding="utf-8")


def data_of(home, reg):
    return rd.build_data(rc.collect(home, TID, reg))


def test_without_an_answer_file_the_report_is_unchanged(home, reg):
    d = data_of(home, reg)
    assert "two_track" not in d.doc
    docs = rr.chart_docs(d, (False, "pending"))
    assert [x["key"] for x in docs] == ["readiness_summary"] + [x["key"] for x in docs[1:]]
    assert not any(x["chart_type"] == "readiness_two_track" for x in docs)
    # adding then removing the file leaves hash and document identical
    put_answers(home, SETTLED)
    (home / "trials" / TID / "two_track_answers.json").unlink()
    again = data_of(home, reg)
    assert again.hash == d.hash and again.doc == d.doc


def test_settled_answers_list_the_ten_questions_with_pass(home, reg):
    put_answers(home, SETTLED)
    d = data_of(home, reg)
    sec = d.doc["two_track"]
    assert sec["overall"] == "PASS"
    assert [r["question_id"] for r in sec["results"]] == [q.id for q in QUESTIONS] and len(sec["results"]) == 10
    assert all(r["status"] == "PASS" for r in sec["results"])
    docs = rr.chart_docs(d, (False, "pending"))
    page = [x for x in docs if x["chart_type"] == "readiness_two_track"]
    assert len(page) == 1 and page[0]["results"] == sec["results"]
    assert len(rr.readiness_figures(docs)) == len(docs)


def test_an_unknown_answer_shows_warning_for_that_question(home, reg):
    put_answers(home, {**SETTLED, "borrow_cost": "unknown"})
    sec = data_of(home, reg).doc["two_track"]
    by = {r["question_id"]: r["status"] for r in sec["results"]}
    assert by["borrow_cost"] == "WARNING" and sec["overall"] == "WARNING"
    assert sum(1 for s in by.values() if s == "PASS") == 9


@pytest.mark.parametrize("answers", [SETTLED, {**SETTLED, "borrow_cost": "unknown"}, {"nonsense": "x"}])
def test_the_section_never_changes_verdict_or_adds_a_warning(home, reg, answers):
    base = data_of(home, reg)
    put_answers(home, answers)
    with_file = data_of(home, reg)
    for key in ("final_verdict", "final_verdict_display", "failing_stage", "counts", "totals", "warnings", "receipts"):
        assert with_file.doc[key] == base.doc[key]
    assert not any(w["severity"] == "high" for w in with_file.doc["warnings"] if w not in base.doc["warnings"])


def test_an_unreadable_file_is_reported_in_the_section_not_as_a_warning(home, reg):
    d = home / "trials" / TID
    d.mkdir(parents=True, exist_ok=True)
    (d / "two_track_answers.json").write_text("{not json", encoding="utf-8")
    base_warnings = data_of(home, reg).doc["warnings"]
    sec = data_of(home, reg).doc["two_track"]
    assert sec["error"] and sec["results"] == []
    put_answers(home, {"nonsense": "x"})
    assert data_of(home, reg).doc["two_track"]["error"]
    assert data_of(home, reg).doc["warnings"] == base_warnings
