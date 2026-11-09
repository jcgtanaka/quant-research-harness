"""The S2 page "LLM contamination gate": a timeline with text labels, built from the llm_gate artifact only."""

import io
import warnings

import pytest
from matplotlib.text import Text
from report_data import TRIAL, build

from qrh.llm.gate import evaluate_gate
from qrh.pipeline.receipt import Stage, Verdict
from qrh.reports import chartdata, charts
from qrh.reports.catalog import figure_specs
from qrh.reports.inputs import StageData

MODEL = {"id": "m1", "provider": "fake", "training_cutoff": "2020-01-31", "cutoff_source": "provider_documented"}


def gate_data(models=None, windows=None, margin=180, retrieval=False, verdict=Verdict.PASS):
    llm = {"models": models or [MODEL], "live_retrieval": retrieval}
    g = evaluate_gate(llm, windows or {"development": ("2021-01-01", "2023-12-31")}, margin)
    receipt, art = build("S2", verdict)
    return StageData(TRIAL, Stage.S2, receipt, {**art, "llm_gate": g.evidence}), g


def strings(fig):
    return [t.get_text() for t in fig.findobj(Text) if t.get_text()]


def test_the_page_exists_only_when_the_stage_recorded_a_gate():
    receipt, art = build("S2")
    plain = [s.key for s in figure_specs(StageData(TRIAL, Stage.S2, receipt, art))]
    assert "s2_llm_gate" not in plain
    data, _ = gate_data()
    keys = [s.key for s in figure_specs(data)]
    assert keys.index("s2_llm_gate") == keys.index("s2_lint") + 1
    assert [k for k in keys if k != "s2_llm_gate"] == plain


def test_chart_data_describes_the_timeline_in_plain_values():
    data, g = gate_data()
    d = chartdata.s2_llm_gate(data)
    assert d["title"] == "S2 LLM contamination gate" and d["chart_type"] == "llm_timeline"
    assert d["margin_days"] == 180 and d["binding_date"] == "2020-07-29" and d["gate_verdict"] == "PASS"
    assert d["rows"] == [{"label": "development", "start": "2021-01-01", "end": "2023-12-31", "status": "PASS",
                          "earliest_contaminated_date": None}]
    roles = {m["role"]: m for m in d["markers"]}
    assert roles["cutoff"]["x"] == "2020-01-31" and "m1" in roles["cutoff"]["label"]
    assert roles["binding"]["x"] == "2020-07-29" and "180" in roles["binding"]["label"]
    assert d["models"][0]["id"] == "m1"


def test_a_failing_window_is_marked_with_its_earliest_contaminated_date():
    data, _ = gate_data(windows={"development": ("2020-03-01", "2023-12-31"), "holdout": ("2022-01-01", "2022-12-31")})
    d = chartdata.s2_llm_gate(data)
    by = {r["label"]: r for r in d["rows"]}
    assert by["development"]["status"] == "FAIL" and by["development"]["earliest_contaminated_date"] == "2020-03-01"
    assert by["holdout"]["status"] == "PASS" and d["gate_verdict"] == "FAIL"


def test_an_unknown_cutoff_has_no_marker_and_is_listed_in_words():
    data, g = gate_data(models=[{**MODEL, "training_cutoff": "unknown"}])
    d = chartdata.s2_llm_gate(data)
    assert d["gate_verdict"] == "BLOCKED" and d["binding_date"] is None
    assert [m["role"] for m in d["markers"]] == []
    fig = charts.s2_llm_gate(data)
    assert any("unknown" in s for s in strings(fig))


@pytest.mark.parametrize("kw", [
    {}, {"windows": {"development": ("2020-03-01", "2023-12-31")}},
    {"models": [{**MODEL, "training_cutoff": "unknown"}]},
    {"models": [MODEL, {**MODEL, "id": "m2", "training_cutoff": "2021-03-31"}]},
    {"retrieval": {"point_in_time_evidence": "vendor snapshots"}},
])
def test_the_figure_renders_to_pdf_without_warnings_and_labels_status_in_text(kw):
    data, g = gate_data(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = charts.s2_llm_gate(data)
        fig.savefig(io.BytesIO(), format="pdf")
    s = strings(fig)
    assert "S2 LLM contamination gate" in s
    assert any(x.endswith(g.verdict) for x in s), "the gate verdict is written, not only colored"
    for row in g.evidence["windows"]:
        assert any(row["status"] in x and x[:1] in "✓✗▲◆" for x in s), row


def test_the_figure_states_the_cutoff_margin_and_binding_date():
    data, _ = gate_data()
    s = " ".join(strings(charts.s2_llm_gate(data)))
    for needle in ("2020-01-31", "180", "2020-07-29", "m1", "development"):
        assert needle in s, needle


def test_live_retrieval_evidence_is_shown_on_the_page():
    data, _ = gate_data(retrieval={"point_in_time_evidence": "vendor snapshots keyed by publication time"})
    assert "vendor snapshots keyed by publication time" in " ".join(strings(charts.s2_llm_gate(data)))


def _drawn(fig):
    fig.draw_without_rendering()
    return fig, fig.bbox.width


def test_the_zone_label_sits_above_the_axes_away_from_the_dashed_lines():
    data, _ = gate_data()
    fig = charts.s2_llm_gate(data)
    label = next(t for t in fig.findobj(Text) if t.get_text() == "Possibly in the training data")
    assert label.xycoords == "axes fraction" and label.xy[1] >= 1.0  # outside the plot area: no line can cross it
    ax = fig.axes[0]
    fig, _w = _drawn(fig)
    top = ax.get_window_extent().y1
    assert label.get_window_extent().y0 >= top - 1


def test_long_evidence_sentences_are_wrapped_inside_the_page():
    long = "vendor snapshots keyed by publication time, checked against the filing calendar for every source " * 3
    data, _ = gate_data(windows={"development": ("2020-03-01", "2023-12-31")}, retrieval={"point_in_time_evidence": long})
    fig, width = _drawn(charts.s2_llm_gate(data))
    for t in fig.findobj(Text):
        if t.get_text() and t.get_visible():
            assert t.get_window_extent().x1 <= width * 0.955, t.get_text()[:60]
    assert "\n" in next(t.get_text() for t in fig.findobj(Text) if "vendor snapshots" in t.get_text())
