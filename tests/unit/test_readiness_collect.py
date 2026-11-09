"""Readiness collector: one test per warning source, severities and the shape of a warning."""

import pytest
from typer.testing import CliRunner

from packs_helpers import make_pack, question
from readiness_helpers import TID, clean_run, codes, make_registry, put_receipt
from qrh.cli import app
from qrh.interview import engine
from qrh.packs.runtime import PackView
from qrh.pipeline.receipt import Override, Verdict
from qrh.readiness import collect as rc
from qrh.readiness.model import KNOWN_LIMITS, SEVERITIES, warning_id


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


def collect(home, reg):
    return rc.collect(home, TID, reg)


def add_pack(tmp_path, *qs):
    d = make_pack(tmp_path, "bank", questions=[list(qs)])
    assert CliRunner().invoke(app, ["pack", "add", str(d)]).exit_code == 0


def answer(home, reg, qid, value, scope="dataset:prices"):
    view = PackView.load(home)
    q, _origin = engine.find_question(reg.conn, view, qid, scope)
    engine.record_answer(reg.conn, q=q, scope_key=scope, identity=engine.identity_for(reg.conn, view, scope),
                         value=value, answered_by="human", provenance="human-cli", session_id=None)


# --- shape ------------------------------------------------------------------------------------------

def test_unknown_trial_is_refused(home, reg):
    with pytest.raises(rc.ReadinessError, match="unknown trial"):
        rc.collect(home, "nope", reg)


def test_a_trial_id_cannot_climb_out_of_the_readiness_folder(home, reg):
    with pytest.raises(rc.ReadinessError, match="not a valid trial id"):
        rc.collect(home, "../x", reg)


def test_a_clean_run_has_only_low_known_limit_warnings(home, reg):
    clean_run(home, reg)
    c = collect(home, reg)
    assert c.verdict == "PASS"
    assert c.warnings and {w.severity for w in c.warnings} == {"low"} and {w.kind for w in c.warnings} == {"limit"}
    for w in c.warnings:
        assert "docs/methods.md" in w.resolution


def test_warning_ids_are_stable_hashes_of_kind_code_and_scope(home, reg):
    clean_run(home, reg)
    a, b = collect(home, reg), collect(home, reg)
    assert [w.id for w in a.warnings] == [w.id for w in b.warnings]
    w = a.warnings[0]
    assert w.id == warning_id(w.kind, w.code, w.scope) and len(w.id) == 16
    assert warning_id("a", "b", "c") != warning_id("a", "b", "d") != warning_id("a", "x", "d")
    assert len({w.id for w in a.warnings}) == len(a.warnings)


def test_every_warning_has_a_resolution_and_a_known_severity(home, reg, tmp_path):
    clean_run(home, reg, s2_methods={}, s2_params={"lint": {"verdict": "PASS", "warnings": [
        {"rule": "LA005", "at": "adapter.py:9", "snippet": "x / x.mean()"}], "suppressed": []}})
    for w in collect(home, reg).warnings:
        assert w.severity in SEVERITIES and w.resolution.strip() and w.message.strip() and w.stage


def test_known_limits_table_is_one_line_each_with_a_methods_pointer():
    assert len(KNOWN_LIMITS) >= 3
    for lim in KNOWN_LIMITS:
        assert "\n" not in lim["text"] and len(lim["text"]) <= 200 and "docs/methods.md" in lim["pointer"]


# --- (h) limits ---------------------------------------------------------------------------------------

def test_same_bar_limit_is_listed_only_when_the_reconstruction_was_not_selected(home, reg):
    clean_run(home, reg)
    assert "same-bar-execution" not in [w.code for w in collect(home, reg).warnings]
    put_receipt(home, reg, "S2", params={"lint": {"verdict": "PASS", "warnings": [], "suppressed": []}})
    assert "same-bar-execution" in [w.code for w in collect(home, reg).warnings if w.kind == "limit"]


# --- (d) exemptions -----------------------------------------------------------------------------------

def test_no_independent_returns_reconstruction_is_medium_with_the_reason(home, reg):
    clean_run(home, reg, s2_methods={})
    c = collect(home, reg)
    w = next(w for w in c.warnings if w.code == "no-reconstruction")
    assert (w.stage, w.severity, w.kind) == ("S2", "medium", "exemption")
    assert "s2.pnl_reconstruction was not selected" in w.message and "s2.pnl_reconstruction" in w.resolution
    assert w.receipt_hash and w.receipt_path.endswith("S2.json")


# --- (c) lint -------------------------------------------------------------------------------------------

def test_lint_warnings_and_suppressions_are_medium_and_carry_the_recorded_reason(home, reg):
    clean_run(home, reg, s2_params={"lint": {"verdict": "PASS", "warnings": [
        {"rule": "LA005", "at": "adapter.py:9", "snippet": "x / x.mean()"}],
        "suppressed": [{"rule": "LA002", "at": "adapter.py:20", "reason": "label column is shifted upstream"},
                       {"rule": "LA003", "at": "adapter.py:30", "reason": None}]}})
    c = collect(home, reg)
    assert sorted(w.code for w in c.warnings if w.kind == "lint") == ["lint-suppressed", "lint-suppressed", "lint-warn"]
    msgs = [w.message for w in c.warnings if w.kind == "lint"]
    assert any("LA005" in m and "adapter.py:9" in m for m in msgs)
    assert any("LA002" in m and "label column is shifted upstream" in m for m in msgs)
    assert any("LA003" in m and "no reason" in m.lower() for m in msgs)
    assert all(w.severity == "medium" and w.stage == "S2" for w in c.warnings if w.kind == "lint")


# --- (e) overrides --------------------------------------------------------------------------------------

def test_overrides_and_hook_approvals_for_the_trial_are_medium(home, reg):
    clean_run(home, reg)
    reg.log_override(TID, "S3", "accepted the thin sample knowingly", "alice")
    reg.log_hook_override("data-gate", "dataset refresh is late, owner decided", "s1", trial_id=TID, approval_id="ab12")
    c = collect(home, reg)
    ov = [w for w in c.warnings if w.kind == "override"]
    assert len(ov) == 2 and {w.severity for w in ov} == {"medium"}
    text = " ".join(w.message for w in ov)
    assert "thin sample" in text and "data-gate" in text and "dataset refresh is late" in text
    assert all(w.message and "20" in w.message for w in ov)  # a date is shown


def test_a_receipt_override_is_listed_once(home, reg):
    clean_run(home, reg)
    put_receipt(home, reg, "S3", override=Override(reason="owner accepted the tail risk", who="bob"))
    ov = [w for w in collect(home, reg).warnings if w.kind == "override"]
    assert len(ov) == 1 and "owner accepted the tail risk" in ov[0].message and ov[0].stage == "S3"


# --- (f) non-default methods ------------------------------------------------------------------------------

def test_non_default_methods_are_medium_once_per_method(home, reg):
    clean_run(home, reg)
    flagged = {"s2.pnl_reconstruction": "pnl_reconstruction@1"}
    put_receipt(home, reg, "S2", methods=flagged, params={"non_default_methods": flagged,
                                                          "lint": {"verdict": "PASS", "warnings": [], "suppressed": []}})
    put_receipt(home, reg, "S7", params={"non_default_methods": flagged})
    nd = [w for w in collect(home, reg).warnings if w.kind == "method"]
    assert len(nd) == 1 and nd[0].severity == "medium" and nd[0].stage == "S2"
    assert "s2.pnl_reconstruction=pnl_reconstruction@1" in nd[0].message


# --- (g) packs ----------------------------------------------------------------------------------------------

def test_a_loosened_gate_from_a_pack_is_high(home, reg):
    clean_run(home, reg)
    put_receipt(home, reg, "S0", params={"pack_layering": {
        "approved_loosening": ["demo:max_pbo=0.5"], "hypothesis_looser_than_pack": ["max_drawdown"]}})
    c = collect(home, reg)
    got = {(w.code, w.severity, w.stage) for w in c.warnings if w.kind == "pack"}
    assert got == {("loosened-gate", "high", "S0"), ("hypothesis-looser-than-pack", "high", "S0")}
    assert any("demo:max_pbo=0.5" in w.message for w in c.warnings)


# --- (b) data -------------------------------------------------------------------------------------------------

def test_data_gaps_failed_rules_and_warn_results_are_medium(home, reg):
    clean_run(home, reg)
    put_receipt(home, reg, "S1", artifacts={"coverage": {"datasets": [
        {"name": "prices", "status": "PASS", "reasons": [], "gap_count": 3, "gaps": [],
         "rules": [{"rule": "range#1", "type": "range", "passed": True, "offenders": [], "reasons": [],
                    "status": "WARN"},
                   {"rule": "gaps#2", "type": "gaps", "passed": False, "offenders": [], "reasons": ["3 gaps"]}]}]}},
        data_hashes={"prices": "hash-prices"})
    got = {w.code: w for w in collect(home, reg).warnings if w.kind == "data"}
    assert {"gaps", "quality-warn", "rule-failed"} <= set(got)
    assert all(got[k].severity == "medium" and got[k].stage == "S1" for k in ("gaps", "quality-warn", "rule-failed"))


def test_a_dataset_that_changed_after_s1_is_high(home, reg):
    clean_run(home, reg)
    reg.conn.execute("UPDATE datasets SET content_hash='something-else' WHERE name='prices'")
    w = next(w for w in collect(home, reg).warnings if w.code == "dataset-changed")
    assert (w.kind, w.severity, w.stage) == ("data", "high", "S1")


def test_a_dataset_no_longer_passing_is_high(home, reg):
    clean_run(home, reg)
    reg.conn.execute("UPDATE datasets SET status='BLOCKED' WHERE name='prices'")
    assert ("dataset-status", "high") in codes(collect(home, reg), "data")


def test_an_unreadable_artifact_is_a_high_warning_not_a_crash(home, reg):
    clean_run(home, reg)
    (home / "trials" / TID / "S1" / "coverage.json").write_text("{}")
    assert ("artifact-unreadable", "high") in codes(collect(home, reg), "receipt")


# --- (a) interview ------------------------------------------------------------------------------------------------

@pytest.fixture
def bank(tmp_path, home, reg):
    add_pack(tmp_path,
             question("rf", critical=True),
             question("tz", critical=False, answer_type="bool", text="Do timestamps mark bar close?", choices=[]),
             question("hyp-rec", stage="S0", scope="hypothesis", critical=False, answer_type="text", choices=[],
                      text="What would make you drop this?"))
    clean_run(home, reg)


def interview(c):
    return {(w.code, w.severity, w.scope) for w in c.warnings if w.kind == "interview"}


def test_pending_critical_is_high_and_pending_recommended_is_medium(home, reg, bank):
    got = interview(collect(home, reg))
    assert ("pending-critical", "high", "rf@dataset:prices") in got
    assert ("pending-recommended", "medium", "tz@dataset:prices") in got
    assert ("pending-recommended", "medium", f"hyp-rec@hypothesis:{TID}") in got
    w = next(w for w in collect(home, reg).warnings if w.code == "pending-critical")
    assert w.stage == "S1" and "qrh interview next --stage S1 --scope dataset:prices --trial " + TID in w.resolution


def test_unknown_is_high_if_critical_else_medium_and_carries_the_risk_text(home, reg, bank):
    answer(home, reg, "rf", "unknown")
    answer(home, reg, "tz", "unknown")
    c = collect(home, reg)
    got = interview(c)
    assert ("unknown-critical", "high", "rf@dataset:prices") in got
    assert ("unknown-recommended", "medium", "tz@dataset:prices") in got
    crit = next(w for w in c.warnings if w.code == "unknown-critical")
    assert "Possible look-ahead in the input." in crit.message and "--include-unknown" in crit.resolution


def test_an_answer_with_a_risk_tag_is_high_and_flags_the_dataset(home, reg, bank):
    answer(home, reg, "rf", "curve")
    c = collect(home, reg)
    assert ("risk-tag", "high", "rf@dataset:prices") in interview(c)
    flag = next(w for w in c.warnings if w.code == "dataset-flag")
    assert (flag.kind, flag.severity, flag.stage) == ("data", "high", "S1") and "lookahead" in flag.message


def test_a_clean_answer_removes_its_warning(home, reg, bank):
    before = interview(collect(home, reg))
    answer(home, reg, "rf", "pit")
    after = interview(collect(home, reg))
    assert ("pending-critical", "high", "rf@dataset:prices") in before and not any(s.startswith("rf@") for _c, _v, s in after)


def test_a_stale_answer_is_flagged_by_rule(home, reg, bank):
    answer(home, reg, "rf", "pit")
    reg.conn.execute("UPDATE datasets SET content_hash='new-content' WHERE name='prices'")  # identity changed
    got = interview(collect(home, reg))
    assert ("stale-answer", "high", "rf@dataset:prices") in got


def test_conditions_of_a_conditional_pass_are_high(home, reg, bank):
    cond = [{"question_id": "rf", "scope": "dataset:prices", "stage": "S1", "text": "How?", "status": "unknown",
             "risk": "Possible look-ahead in the input."}]
    put_receipt(home, reg, "S7", verdict=Verdict.PASS_CONDITIONAL, params={"conditions": cond})
    answer(home, reg, "rf", "unknown")
    c = collect(home, reg)
    assert c.verdict == "PASS_CONDITIONAL"
    w = next(w for w in c.warnings if w.kind == "condition")
    assert (w.code, w.severity, w.stage) == ("open-condition", "high", "S7") and "rf" in w.message


def test_a_verdict_older_than_the_answers_is_marked_stale(home, reg, bank):
    cond = [{"question_id": "rf", "scope": "dataset:prices", "stage": "S1", "text": "How?", "status": "unknown", "risk": ""}]
    put_receipt(home, reg, "S7", verdict=Verdict.PASS_CONDITIONAL, params={"conditions": cond})
    answer(home, reg, "rf", "pit")  # resolved since S7 ran
    assert ("stale-verdict", "high") in codes(collect(home, reg), "interview")


# --- final verdict --------------------------------------------------------------------------------------------------

def test_without_s7_the_verdict_is_not_run_and_a_high_warning_says_so(home, reg):
    put_receipt(home, reg, "S0")
    c = collect(home, reg)
    assert c.verdict == "NOT RUN" and ("no-final-verdict", "high") in codes(c, "receipt")


def test_a_failed_trial_carries_a_high_not_deployable_warning(home, reg):
    clean_run(home, reg)
    put_receipt(home, reg, "S5", verdict=Verdict.FAIL, reasons=["PBO 0.4 exceeds max_pbo 0.2"])
    put_receipt(home, reg, "S7", verdict=Verdict.FAIL, params={"failing_stage": "S5"})
    c = collect(home, reg)
    assert c.verdict == "FAIL" and ("final-not-pass", "high") in codes(c, "receipt")


def test_a_tampered_receipt_is_a_high_warning(home, reg):
    clean_run(home, reg)
    p = home / "receipts" / TID / "S3.json"
    p.write_text(p.read_text().replace('"PASS"', '"FAIL"', 1))
    assert ("receipt-unreadable", "high") in codes(collect(home, reg), "receipt")


def test_warnings_are_sorted_by_stage_then_severity(home, reg, bank):
    c = collect(home, reg)
    order = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "ALL"]
    keys = [(order.index(w.stage), SEVERITIES.index(w.severity)) for w in c.warnings]
    assert keys == sorted(keys)


def test_a_perturbation_test_that_covered_only_some_change_bars_is_a_medium_warning(home, reg):
    clean_run(home, reg)
    put_receipt(home, reg, "S2", params={"lint": {"verdict": "PASS", "warnings": [], "suppressed": []},
                                         "perturbation": {"verdict": "PASS", "coverage": 0.25,
                                                          "change_bars_tested": 25, "change_bars_total": 100}})
    w = next(w for w in collect(home, reg).warnings if w.code == "perturbation-coverage")
    assert (w.stage, w.severity) == ("S2", "medium") and "25 of 100" in w.message
    put_receipt(home, reg, "S2", params={"lint": {"verdict": "PASS", "warnings": [], "suppressed": []},
                                         "perturbation": {"verdict": "PASS", "coverage": 1.0,
                                                          "change_bars_tested": 100, "change_bars_total": 100}})
    assert "perturbation-coverage" not in [w.code for w in collect(home, reg).warnings]
