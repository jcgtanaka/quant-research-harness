"""Interview report pages, the S7 conditions block and the PASS · CONDITIONAL label."""

import json

import pytest

from qrh.pipeline.receipt import Stage, Verdict
from qrh.reports import chartdata, charts, theme
from qrh.reports.inputs import StageData
from qrh.reports.interview import interview_chart_docs, interview_figure, interview_figures, render_interview
from report_data import TRIAL, _receipt, build


def row(qid, status, **kw):
    base = {"qid": qid, "scope": "dataset:prices", "stage": "S1", "text": f"Question {qid}?", "why": "Because it matters.",
            "critical": False, "origin": "pack", "status": status, "stale": False}
    if status != "pending":
        base.update(answer="pit", label="Point-in-time", version=1, risk="", unknown_risk="", mapped={},
                    answered_by="user via session s", provenance="transcript-verified", answered_at="x")
    base.update(kw)
    return base


def snap(rows_by_stage=None):
    rows_by_stage = rows_by_stage or {"S1": [
        row("a", "answered"), row("b", "unknown", answer="unknown", label="unknown", unknown_risk="Possible look-ahead.", critical=True),
        row("c", "pending", critical=True, why="Needs an answer."), row("d", "answered", risk="lookahead", mapped={"risk": "lookahead"})]}
    stages = {}
    for st, rows in rows_by_stage.items():
        counts = {k: sum(1 for r in rows if r["status"] == k) for k in ("answered", "unknown", "pending")}
        stages[st] = {"rows": rows, "counts": counts, "total": len(rows), "light": "red"}
    return {"trial_id": TRIAL, "stages": stages, "conditions": [], "note": "n"}


def test_verdict_label_reads_pass_conditional():
    assert theme.verdict_label("PASS_CONDITIONAL") == "◆ PASS · CONDITIONAL"
    assert theme.verdict_color("PASS_CONDITIONAL") == theme.STATUS["warning"]


def test_chart_docs_hold_progress_light_and_rows():
    docs = interview_chart_docs(snap(), "S1")
    assert len(docs) == 1
    d = docs[0]
    assert d["chart_type"] == "interview" and d["stage"] == "S1" and d["trial_id"] == TRIAL
    assert d["progress"] == {"answered": 2, "unknown": 1, "pending": 1, "total": 4}
    assert d["light"] == "red"
    by = {r["question_id"]: r for r in d["rows"]}
    assert by["b"]["status"] == "unknown" and by["b"]["risk"] == "Possible look-ahead."
    assert by["c"]["status"] == "pending" and by["c"]["why"] == "Needs an answer."
    assert by["d"]["risk"] == "lookahead" and by["a"]["answer"] == "Point-in-time"
    json.dumps(d, allow_nan=False)


def test_many_rows_paginate():
    rows = [row(f"q{i}", "pending") for i in range(23)]
    docs = interview_chart_docs(snap({"S2": rows}), "S2")
    assert len(docs) == 3 and sum(len(d["rows"]) for d in docs) == 23 and docs[1]["key"] == "interview_S2_2"


def test_figures_draw_for_every_status():
    fig = interview_figure(interview_chart_docs(snap(), "S1")[0])
    texts = " ".join(t.get_text() for t in fig.texts) + " ".join(t.get_text() for a in fig.axes for t in a.texts)
    assert "PENDING" in texts and "DON'T KNOW" in texts and "ANSWERED" in texts and "lookahead" in texts


def test_render_writes_pdf_and_chart_json_per_stage_deterministically(tmp_path):
    s = snap({"S1": [row("a", "answered")], "S2": [row("b", "pending", stage="S2")]})
    files = render_interview(tmp_path, TRIAL, s)
    names = sorted(f.path.name for f in files)
    assert names == ["interview.S1.chart.json", "interview.S1.pdf", "interview.S2.chart.json", "interview.S2.pdf"]
    first = {f.path.name: f.path.read_bytes() for f in files}
    render_interview(tmp_path, TRIAL, s)
    assert {f.path.name: f.path.read_bytes() for f in files} == first
    assert first["interview.S1.pdf"].startswith(b"%PDF")
    only = render_interview(tmp_path, TRIAL, s, stage="S2")
    assert sorted(f.path.name for f in only) == ["interview.S2.chart.json", "interview.S2.pdf"]


def test_figures_for_all_stages_in_order():
    figs = interview_figures(snap({"S2": [row("b", "pending")], "S1": [row("a", "answered")]}))
    assert len(figs) == 2


def test_s7_summary_carries_conditions(make_summary):
    data = make_summary(Verdict.PASS_CONDITIONAL, failing=None)
    data.receipt.params["conditions"] = [{"question_id": "q1", "scope": "dataset:p", "stage": "S1", "text": "How?",
                                          "status": "unknown", "risk": "Possible look-ahead."}]
    d = chartdata.s7_summary(data)
    assert d["final"]["verdict"] == "PASS_CONDITIONAL" and d["conditions"][0]["question_id"] == "q1"
    fig = charts.s7_summary(data)
    texts = " ".join(t.get_text() for t in fig.texts) + " ".join(t.get_text() for a in fig.axes for t in a.texts)
    assert "PASS · CONDITIONAL" in texts and "q1" in texts and "Possible look-ahead." in texts


def texts_of(fig):
    return [t.get_text() for t in fig.texts]


def test_a_long_answer_wraps_to_three_lines_instead_of_one_ellipsis():
    long = "The bar close (the row is known at its timestamp) and nothing is revised after the fact at all"
    r = row("a", "answered", label=long, version=2, provenance="transcript-verified")
    texts = texts_of(interview_figure(interview_chart_docs(snap({"S1": [r]}), "S1")[0]))
    got = [t for t in texts if t in " ".join(long.split()) or t.startswith("(v2")]
    assert "…" not in " ".join(texts) and len(got) <= 3
    assert "".join(" ".join(texts).split()).count("nothingisrevised") == 1 and any("v2" in t for t in texts)


def test_the_heading_keeps_the_scope_with_a_short_trial_id():
    r = row("falsification-criterion", "pending", scope="hypothesis:" + "a1b2c3d4" * 4, stage="S0")
    texts = texts_of(interview_figure(interview_chart_docs(snap({"S0": [r]}), "S0")[0]))
    head = next(t for t in texts if t.startswith("falsification-criterion"))
    assert "hypothesis:a1b2c3d4)" in head and "…" not in head


def test_the_conditional_light_is_an_amber_diamond_not_a_red_cross():
    fig = interview_figure(interview_chart_docs(snap(), "S1")[0])
    label = next(t for t in fig.texts if "CONDITIONS" in t.get_text())
    assert label.get_text().startswith("◆") and "✗" not in label.get_text()
    assert label.get_color() == theme.STATUS["warning"]


def test_the_s0_hypothesis_hash_breaks_in_two_clean_halves(make_data):
    data = make_data("S0")
    h = data.artifacts["hypothesis"]["hypothesis_hash"] = "ab12" * 16
    fig = charts.s0_card(data)
    shown = [t.get_text() for a in fig.axes for t in a.texts]
    block = next(t for t in shown if "ab12" in t)
    assert block.split("\n") == [h[:32], h[32:]]
