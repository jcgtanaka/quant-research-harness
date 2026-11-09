"""S7: PASS becomes PASS_CONDITIONAL when a critical question is open or an answer carries a risk tag."""

import itertools
import json
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from packs_helpers import lesson, make_pack, question
from qrh.cli import app
from qrh.memory.registry import Registry
from qrh.pipeline.receipt import Stage, Verdict, display_verdict, is_pass, load_receipt
from qrh.pipeline.stages import Pipeline, RegisterHypothesis, make_receipt, register_new_trial

runner = CliRunner()
SPEC = {
    "text": "a moving average rule", "strategy_family": "trend", "asset_class": "equity", "frequency": "1D",
    "parameter_space_size": 4, "thresholds": {"k": 1}, "difference_statement": "baseline",
}


class Stub:
    def __init__(self, stage, verdict=Verdict.PASS):
        self.id, self.verdict = stage, verdict

    def run(self, ctx):
        return make_receipt(ctx, self.id, self.verdict, ["stub"] if self.verdict is not Verdict.PASS else [])


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("QRH_HOME", str(home))
    counter = itertools.count(1)
    reg = Registry(home / "registry.sqlite3", id_factory=lambda: f"t{next(counter)}")
    yield home, reg, tmp_path
    reg.close()


def bank(tmp, *qs, lessons=()):
    d = make_pack(tmp, "bank", questions=[list(qs)], lessons=list(lessons))
    assert runner.invoke(app, ["pack", "add", str(d)]).exit_code == 0


def run(env, final_stub=None):
    home, reg, _ = env
    res = register_new_trial(SPEC, reg, home)
    tid = res.trial_id
    stages = [RegisterHypothesis()] + [Stub(Stage(f"S{i}"), (final_stub or {}).get(i, Verdict.PASS)) for i in range(1, 7)]
    out = Pipeline(reg, home, stages=stages).run(tid)
    return tid, out


def crit(qid="hq", **kw):
    kw = {"critical": True, **kw}
    return question(qid, stage="S2", scope="hypothesis", answer_type="text", choices=[],
                    text="When is the order filled?", **kw)


def test_display_helpers():
    assert display_verdict(Verdict.PASS_CONDITIONAL) == "PASS · CONDITIONAL" and display_verdict("FAIL") == "FAIL"
    assert is_pass(Verdict.PASS_CONDITIONAL) and is_pass(Verdict.PASS) and not is_pass(Verdict.FAIL)


def test_all_pass_without_questions_stays_pass(env):
    _, out = run(env)
    assert out.final.verdict is Verdict.PASS and "conditions" not in out.final.params


def test_pending_critical_question_makes_a_conditional_pass(env):
    bank(env[2], crit())
    tid, out = run(env)
    assert out.final.verdict is Verdict.PASS_CONDITIONAL
    cond = out.final.params["conditions"]
    assert cond[0]["question_id"] == "hq" and cond[0]["status"] == "pending" and cond[0]["text"]
    assert "PASS · CONDITIONAL" in " ".join(out.final.reasons)
    report = (env[0] / "reports" / f"{tid}.md").read_text()
    assert "PASS · CONDITIONAL" in report and "hq" in report and "When is the order filled?" in report


def test_non_critical_pending_does_not_change_the_verdict_but_is_counted(env):
    bank(env[2], crit(critical=False))
    tid, out = run(env)
    assert out.final.verdict is Verdict.PASS
    s2 = load_receipt(env[0] / "receipts" / tid / "S2.json")
    assert s2.params["interview"]["pending"] == 1 and s2.params["interview"]["pending_ids"] == [f"hq@hypothesis:{tid}"]


def test_stage_receipts_record_interview_status_and_packs(env):
    bank(env[2], crit())
    tid, out = run(env)
    s2 = load_receipt(env[0] / "receipts" / tid / "S2.json")
    assert s2.params["interview"] == {"answered": 0, "unknown": 0, "pending": 1,
                                      "answered_ids": [], "unknown_ids": [], "pending_ids": [f"hq@hypothesis:{tid}"],
                                      "answer_versions": {}}
    assert "interview" not in load_receipt(env[0] / "receipts" / tid / "S3.json").params
    assert "bank" in out.final.packs


def answer_human(tid, qid, value, scope=None):
    res = runner.invoke(app, ["interview", "answer", qid, "--scope", scope or f"hypothesis:{tid}", "--value", value, "--human"])
    assert res.exit_code == 0, res.output


def test_unknown_answer_stays_conditional_and_records_its_risk(env):
    bank(env[2], crit(unknown_risk="Fills may be optimistic."))
    tid, _ = run(env)
    answer_human(tid, "hq", "unknown")
    out = Pipeline(env[1], env[0], stages=[RegisterHypothesis()] + [Stub(Stage(f"S{i}")) for i in range(1, 7)]).run(tid)
    assert out.final.verdict is Verdict.PASS_CONDITIONAL
    c = out.final.params["conditions"][0]
    assert c["status"] == "unknown" and c["risk"] == "Fills may be optimistic."
    s2 = load_receipt(env[0] / "receipts" / tid / "S2.json")
    assert s2.params["interview"]["unknown_ids"] == [f"hq@hypothesis:{tid}"] and s2.params["interview"]["answer_versions"] == {f"hq@hypothesis:{tid}": 1}


def test_a_clean_answer_clears_the_condition_and_versions_are_in_receipts(env):
    bank(env[2], crit())
    tid, first = run(env)
    answer_human(tid, "hq", "next bar open")
    stages = [RegisterHypothesis()] + [Stub(Stage(f"S{i}")) for i in range(1, 7)]
    out = Pipeline(env[1], env[0], stages=stages).run(tid)
    assert out.final.verdict is Verdict.PASS and "conditions" not in out.final.params
    s2 = load_receipt(env[0] / "receipts" / tid / "S2.json")
    assert s2.params["interview"]["answer_versions"] == {f"hq@hypothesis:{tid}": 1}
    assert first.final.receipt_hash() != out.final.receipt_hash()


def test_a_risk_tagged_answer_is_a_condition_even_if_not_critical(env):
    q = question("rf", stage="S2", scope="hypothesis", critical=False)
    bank(env[2], q)
    tid, _ = run(env)
    answer_human(tid, "rf", "curve")
    stages = [RegisterHypothesis()] + [Stub(Stage(f"S{i}")) for i in range(1, 7)]
    out = Pipeline(env[1], env[0], stages=stages).run(tid)
    assert out.final.verdict is Verdict.PASS_CONDITIONAL
    assert out.final.params["conditions"][0]["status"] == "risk" and out.final.params["conditions"][0]["risk"] == "lookahead"


def test_a_lesson_can_make_a_question_critical(env):
    bank(env[2], crit(critical=False), lessons=[lesson("need", action="require", requirement={"interview_question": "hq"})])
    _, out = run(env)
    assert out.final.verdict is Verdict.PASS_CONDITIONAL


@pytest.mark.parametrize("bad", [Verdict.FAIL, Verdict.BLOCKED, Verdict.INCONCLUSIVE])
def test_failures_are_unchanged(env, bad):
    bank(env[2], crit())
    _, out = run(env, {3: bad})
    assert out.final.verdict is bad and "conditions" not in out.final.params


def test_run_command_exits_zero_and_prints_the_conditions(env, monkeypatch):
    bank(env[2], crit())
    spec = env[2] / "h.json"
    spec.write_text(json.dumps(SPEC))
    tid = runner.invoke(app, ["new", str(spec)]).stdout.strip().splitlines()[-1]
    res = runner.invoke(app, ["status", tid])
    assert res.exit_code == 0
    from qrh.pipeline import stages as st

    monkeypatch.setattr(st, "default_stages", lambda: [RegisterHypothesis()] + [Stub(Stage(f"S{i}")) for i in range(1, 7)])
    res = runner.invoke(app, ["run", tid])
    assert res.exit_code == 0, res.output
    assert "S7: PASS · CONDITIONAL" in res.output and "condition: hq" in res.output
    status = runner.invoke(app, ["status", tid]).output
    assert "S7: PASS · CONDITIONAL" in status
