"""Stage reports show the methods used and flag experimental versions."""

import pytest

from qrh.pipeline.receipt import Stage, Verdict
from qrh.reports import chartdata as cd
from qrh.reports import charts
from qrh.reports.inputs import StageData
from report_data import TRIAL, build


def _with_methods(stage, methods, flagged=None):
    receipt, art = build(stage)
    params = {**receipt.params, **({"non_default_methods": flagged} if flagged else {})}
    return StageData(TRIAL, Stage(stage), receipt.model_copy(update={"methods": methods, "params": params}), art)


def _texts(fig):
    return " ".join(t.get_text() for ax in fig.axes for t in ax.texts) + " ".join(t.get_text() for t in fig.texts)


def test_s0_card_says_all_defaults_or_names_the_experimental_ones():
    plain = cd.s0_card(_with_methods("S0", {"s5.pbo": "cscv@1"}))
    assert {f["label"]: f["value"] for f in plain["fields"]}["Methods"] == "all default versions (1 methods)"
    flagged = cd.s0_card(_with_methods("S0", {"s4.walkforward": "purged_embargo@1", "s5.pbo": "cscv@1"},
                                       {"s4.walkforward": "purged_embargo@1"}))
    value = {f["label"]: f["value"] for f in flagged["fields"]}["Methods"]
    assert "EXPERIMENTAL" in value and "s4.walkforward=purged_embargo@1" in value


def test_s0_card_without_recorded_methods_is_unchanged(make_data):
    d = cd.s0_card(make_data("S0"))
    assert "Methods" not in {f["label"] for f in d["fields"]}


def test_s0_page_draws_the_experimental_flag():
    data = _with_methods("S0", {"s4.walkforward": "purged_embargo@1"}, {"s4.walkforward": "purged_embargo@1"})
    assert "EXPERIMENTAL" in _texts(charts.s0_card(data))


def test_s7_summary_carries_and_draws_non_default_methods(make_summary):
    data = make_summary(Verdict.FAIL, "S5")
    flagged = {"s4.walkforward": "purged_embargo@1"}
    s7 = data.receipt.model_copy(update={"params": {**data.receipt.params, "non_default_methods": flagged}})
    data = StageData(data.trial_id, data.stage, s7, data.artifacts, data.receipts)
    d = cd.s7_summary(data)
    assert d["non_default_methods"] == flagged
    assert "EXPERIMENTAL" in _texts(charts.s7_summary(data))
    plain = cd.s7_summary(make_summary(Verdict.FAIL, "S5"))
    assert plain["non_default_methods"] == {}
