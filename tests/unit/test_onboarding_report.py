"""The S1 dataset card and onboarding table pages: figures, chart data, plain-text rendering."""

import io

import pytest

from report_data import TRIAL, build
from qrh.onboarding.card import profile_card
from qrh.onboarding.profile import build_profile
from qrh.onboarding.store import field_rows
from qrh.onboarding.model import model_from_proposal, parse_proposal
from qrh.pipeline.receipt import Stage, Verdict
from qrh.reports import chartdata, charts
from qrh.reports.catalog import figure_specs
from qrh.reports.inputs import StageData
from qrh.synthetic import instruments as inst


def s1_data(verdict=Verdict.PASS, evidence="ok", column="volume", status="verified"):
    receipt, art = build("S1", verdict)
    frame, proposal = inst.equity_daily()
    frame = frame.rename(columns={"volume": column})  # a data column may be named anything, a claim may not use $
    model = model_from_proposal(parse_proposal(proposal))
    for f, c in model.claims().items():
        setattr(model, f, c.model_copy(update={"status": status}))
    art["onboarding"] = {"datasets": [{
        "name": "prices", "privacy": "public", "state": "VERIFIED_WITH_GAPS", "version": 2, "content_hash": "ab" * 32,
        "model_hash": "cd" * 32, "card": profile_card(build_profile(frame)), "fields": [{**r, "evidence": evidence} for r in field_rows(model)],
        "pending_questions": [{"id": "onb-adjustments", "critical": True, "status": "pending"}],
    }]}
    return StageData(TRIAL, Stage.S1, receipt, art)


def test_the_catalog_adds_two_pages_per_profiled_dataset():
    keys = [s.key for s in figure_specs(s1_data())]
    assert keys == ["s1_coverage", "s1_rules", "s1_dataset_card_0", "s1_onboarding_0"]
    receipt, art = build("S1", Verdict.PASS)
    assert [s.key for s in figure_specs(StageData(TRIAL, Stage.S1, receipt, art))] == ["s1_coverage", "s1_rules"]


def test_chart_data_carries_state_status_text_and_no_markup_interpretation():
    data = s1_data(evidence="<b>bold</b> **x** [l](http://x)")
    card = chartdata.s1_dataset_card(data, 0)
    table = chartdata.s1_onboarding(data, 0)
    assert card["chart_type"] == "text" and {"Rows", "Inferred frequency"} <= {f["label"] for f in card["facts"]}
    assert table["chart_type"] == "table" and table["state"] == "VERIFIED_WITH_GAPS" and table["status_column"] == "status"
    assert {r["status"] for r in table["rows"]} == {"VERIFIED"}
    assert any("<b>bold</b>" in r["evidence"] for r in table["rows"])  # the text is kept as it is
    assert table["pending_questions"][0]["id"] == "onb-adjustments"
    assert "onboarding VERIFIED_WITH_GAPS (version 2)" in card["subtitle"]


@pytest.mark.parametrize("status", ["verified", "confirmed", "unverified", "contradicted"])
def test_the_pages_draw_for_every_status_and_a_dollar_sign_is_plain_text(status):
    data = s1_data(evidence="costs $5 and $6", column="vol$a$b", status=status)
    for fig in (charts.s1_dataset_card(data, 0), charts.s1_onboarding(data, 0)):
        buf = io.BytesIO()
        fig.savefig(buf, format="png")  # a mathtext parse error would raise here
        assert buf.getvalue().startswith(b"\x89PNG")


def test_every_status_is_shown_as_text_with_a_marker_never_by_colour_alone():
    from qrh.reports import theme

    labels = {s: theme.verdict_label(s.upper()) for s in ("verified", "confirmed", "unverified", "contradicted")}
    assert labels == {"verified": "✓ VERIFIED", "confirmed": "✓ CONFIRMED", "unverified": "◆ UNVERIFIED",
                      "contradicted": "✗ CONTRADICTED"}
