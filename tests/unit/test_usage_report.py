import json
import re

import pytest

from qrh.usage.harvest import harvest
from qrh.usage.local import append_local
from qrh.usage.privacy import PrivacyError
from qrh.usage.report import build_charts, build_report, render_pdf, render_text, write_chart_files
from qrh.usage.settings import SettingsError
from usage_helpers import SECRET, SECRET_PATH, build_claude, write_settings


@pytest.fixture
def home(tmp_path):
    claude = build_claude(tmp_path / "claude")
    h = tmp_path / "home"
    write_settings(h, thresholds={"cache_cold_pct": 60, "context_high_tokens": 1000, "mechanical_tool_share": 0.8,
                                  "mechanical_min_tools": 5})
    harvest(claude, h)
    append_local(h, {"ts": "2026-03-01T09:00:00+00:00", "model": "local-x", "prompt_tokens": 50,
                     "completion_tokens": 10, "duration_ms": 5.0, "task": "label only"})
    return h


def test_report_totals_plays_and_missing_cost_are_reported(home):
    rep = build_report(home, "model", env={})
    t = rep["totals"]
    assert t["sessions"] == 3 and t["sessions_without_cost"] == 1
    assert t["cost_sources"] == {"estimate": 1, "harness": 1, "none": 1}
    assert t["local"] == {"records": 1, "prompt_tokens": 50, "completion_tokens": 10}
    assert {p["play"] for p in rep["plays"]} == {"high-context"}
    assert rep["plays"][0]["session"] == "S001" and rep["plays"][0]["project"] == "projA"
    models = {r["key"]: r for r in rep["rows"]}
    assert models["model-small"]["cost_usd"] == pytest.approx(0.5 + 1.0)  # harness 0.5 plus estimated subagent
    assert "local-x" in models


def test_text_report_says_how_many_sessions_lack_cost(home):
    text = render_text(build_report(home, "week", env={}))
    assert "1 session(s) have no cost" in text and "estimate" in text and "local" in text.lower()


def test_missing_thresholds_stop_the_report(home):
    (home / "usage" / "thresholds.json").unlink()
    with pytest.raises(SettingsError, match="thresholds.json"):
        build_report(home, "day", env={})


def test_bad_group_is_rejected(home):
    with pytest.raises(ValueError):
        build_report(home, "nope", env={})


def test_charts_are_the_five_documented_ones(home):
    charts = build_charts(build_report(home, "day", env={}))
    assert [c["key"] for c in charts] == [
        "summary", "daily_cost_by_model", "tokens_by_kind", "cache_hit_over_time", "cost_per_trial", "plays"]
    stacked = charts[1]
    assert len(stacked["series"]) <= 3 and stacked["chart_type"] == "stacked_bar"
    assert charts[2]["chart_type"] == "bar" and {b["label"] for b in charts[2]["bars"]} == {
        "input", "output", "cache read", "cache write"}


def test_chart_files_and_pdf_are_deterministic(home, tmp_path):
    outs = []
    for i in range(2):
        rep = build_report(home, "day", env={})
        charts = build_charts(rep)
        files = write_chart_files(home, charts)
        pdf = tmp_path / f"r{i}.pdf"
        render_pdf(charts, pdf)
        outs.append((json.dumps(rep, sort_keys=True), [f.read_bytes() for f in files], pdf.read_bytes()))
    assert outs[0] == outs[1]
    blob = outs[0][2]
    assert blob.startswith(b"%PDF") and len(re.findall(rb"/Type /Page\b(?!s)", blob)) == 6
    assert b"/CreationDate" not in blob
    assert all(f.parent == home / "usage" / "reports" for f in files)
    assert {f.name for f in files} == {f"{k}.chart.json" for k in (
        "summary", "daily_cost_by_model", "tokens_by_kind", "cache_hit_over_time", "cost_per_trial", "plays")}


def all_outputs(home, tmp_path, env):
    rep = build_report(home, "project", env=env)
    charts = build_charts(rep)
    files = write_chart_files(home, charts)
    pdf = tmp_path / "o.pdf"
    render_pdf(charts, pdf)
    return [json.dumps(rep), render_text(rep), pdf.read_bytes().decode("latin-1"),
            *[f.read_text() for f in files]]


def test_outputs_never_contain_text_paths_or_session_ids(home, tmp_path):
    for out in all_outputs(home, tmp_path, {}):
        for forbidden in (SECRET, SECRET_PATH, "/work/", "-work-projA", str(tmp_path), '"session_id"'):
            assert forbidden not in out


def test_protected_labels_are_hidden_in_every_output(home, tmp_path):
    outs = all_outputs(home, tmp_path, {"QRH_PROTECTED_LABELS": "^projA$"})
    assert "private-1" in outs[0] and "private-1" in outs[1]
    for out in outs:
        assert "projA" not in out


def test_invalid_protected_regex_stops_the_report(home):
    with pytest.raises(PrivacyError):
        build_report(home, "day", env={"QRH_PROTECTED_LABELS": "("})


def test_empty_archive_renders_without_error(tmp_path):
    h = tmp_path / "h"
    write_settings(h)
    rep = build_report(h, "day", env={})
    charts = build_charts(rep)
    render_pdf(charts, tmp_path / "e.pdf")
    assert rep["totals"]["sessions"] == 0 and "no sessions" in render_text(rep).lower()


def test_notes_about_unevaluable_rules_carry_no_session_ids(tmp_path):
    from qrh.usage.harvest import harvest as run_harvest
    from usage_helpers import SECRET, assistant, flat, write_jsonl

    claude, h = tmp_path / "c", tmp_path / "h"
    write_settings(h, prices=None)
    write_jsonl(claude / "projects" / "p" / "sess-unique-id.jsonl", flat(
        assistant("a", "model-big", "2026-03-01T10:00:00Z", inp=1, tools=("Read",) * 6)))
    run_harvest(claude, h)
    rep = build_report(h, "day", env={})
    assert rep["notes"] == ["strong-on-mechanical not evaluated: no tier information (prices.json is missing) (for 1 session(s))"]
    assert "sess-unique-id" not in json.dumps(rep) and SECRET not in json.dumps(rep)


def test_token_tables_are_labelled_lower_bound_where_subagents_were_used(home):
    rep = build_report(home, "week", env={})
    assert rep["totals"]["sessions_with_subagents"] == 2
    text = render_text(rep)
    assert "lower bound" in text and "2 of 3 session(s)" in text and "cost comes from" in text
    charts = {c["key"]: c for c in build_charts(rep)}
    assert "lower bound" in charts["tokens_by_kind"]["title"].lower()
    assert charts["tokens_by_kind"]["lower_bound_sessions"] == 2
    assert any("lower bound" in n for n in charts["summary"]["notes"])


def test_no_lower_bound_label_without_subagents(tmp_path):
    from usage_helpers import assistant, flat, write_jsonl

    claude = tmp_path / "c"
    write_jsonl(claude / "projects" / "-w-p" / "a.jsonl", flat(assistant("m1", "model-big", "2026-03-01T10:00:00Z", inp=5)))
    h = tmp_path / "h"
    write_settings(h)
    harvest(claude, h)
    rep = build_report(h, "week", env={})
    assert rep["totals"]["sessions_with_subagents"] == 0 and "lower bound" not in render_text(rep)
    assert "lower bound" not in {c["key"]: c for c in build_charts(rep)}["tokens_by_kind"]["title"].lower()
