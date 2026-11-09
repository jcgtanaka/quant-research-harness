"""P3d-b1 end to end: example 01 with retail-core and example-trader installed, the interview loop, PASS · CONDITIONAL."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qrh.cli import app
from qrh.memory.registry import Registry
from qrh.pipeline.receipt import Verdict

runner = CliRunner()
PACKS = Path(__file__).resolve().parents[2] / "packs"
SESSION = "sess-accept"


def cli(*args):
    return runner.invoke(app, list(args))


@pytest.fixture(autouse=True)
def fake_user_home(monkeypatch, tmp_path):
    """Transcripts live where Claude Code keeps them, under ~/.claude/projects of a temporary HOME."""
    monkeypatch.setenv("HOME", str(tmp_path / "user"))


def say(tmp_path, text):
    """A synthetic transcript with one user message posted after the question was issued, reported for SESSION by
    the SessionStart hook exactly as Claude Code would (F5: only a recorded transcript is accepted)."""
    from qrh.hooks.runner import run_hook

    ts = (datetime.now(timezone.utc) + timedelta(seconds=3)).isoformat()
    p = tmp_path / "user" / ".claude" / "projects" / "proj" / f"{SESSION}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"type": "user", "timestamp": ts, "sessionId": SESSION,
                             "message": {"role": "user", "content": text}}) + "\n")
    run_hook("session-context", json.dumps({"hook_event_name": "SessionStart", "session_id": SESSION,
                                            "transcript_path": str(p), "source": "startup"}),
             home=tmp_path / "qrh_home", env={})
    return p


def ask_and_answer(tmp_path, tid, stage, scope, qid, value, said=None):
    res = cli("interview", "next", "--stage", stage, "--scope", scope, "--trial", tid, "--include-unknown", "--json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["question"]["id"] == qid
    t = say(tmp_path, said or value)
    res = cli("interview", "answer", qid, "--scope", scope, "--value", value, "--transcript", str(t), "--session", SESSION)
    assert res.exit_code == 0, res.output


@pytest.fixture
def ex(fresh_example, home):
    assert cli("pack", "add", str(PACKS / "retail-core")).exit_code == 0
    assert cli("pack", "add", str(PACKS / "example-trader")).exit_code == 0
    e = fresh_example("01_clean_strategy")
    e.register(e.build_data(), "prices")
    e.new_trial()
    return e


def test_conditional_then_pass_once_the_critical_question_is_answered_cleanly(ex, tmp_path, home):
    tid, fam = ex.trial_id, "trend|synthetic|1D"
    # every critical question is answered except one, which the user does not know
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "bar-type-declared", "time", "Time bars, one day each")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "data-history-revised", "never_revised", "No, it is as first published")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "data-timezone", "utc", "UTC")
    ask_and_answer(tmp_path, tid, "S1", f"dataset:prices", "timestamp-semantics", "bar_close", "the bar close")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "ex-rate-source", "unknown", "honestly I don't know")
    ask_and_answer(tmp_path, tid, "S2", f"strategy:{fam}", "signal-vs-fill", "next_bar", "Signal at one bar, fill at a later bar")
    ask_and_answer(tmp_path, tid, "S3", f"strategy:{fam}", "variants-outside-harness", "0", "I tried 0 variants outside")
    ask_and_answer(tmp_path, tid, "S4", f"strategy:{fam}", "reopt-cadence-live", "fixed", "never, parameters stay fixed")

    res = ex.run()
    assert res.exit_code == 0, res.output
    s7 = ex.receipt("S7")
    assert s7.verdict is Verdict.PASS_CONDITIONAL
    assert [(c["question_id"], c["status"]) for c in s7.params["conditions"]] == [("ex-rate-source", "unknown")]
    assert "PASS · CONDITIONAL" in res.output and "condition: ex-rate-source" in res.output
    for s in ("S0", "S1", "S2", "S3", "S4", "S5", "S6"):
        assert ex.receipt(s).verdict is Verdict.PASS, s
        assert set(ex.receipt(s).packs) == {"retail-core", "example-trader"}
        assert all(len(v["hash"]) == 64 and v["version"] for v in ex.receipt(s).packs.values())
    s1 = ex.receipt("S1").params["interview"]
    assert s1["unknown_ids"] == ["ex-rate-source@dataset:prices"] and s1["pending"] == 3 and s1["answered"] == 4  # pending: the recommended data-licensing, data-source-declared, time-bar-label-confirmed
    assert ex.receipt("S1").params["interview"]["answer_versions"]["ex-rate-source@dataset:prices"] == 1
    # account facts and imported attempts reached the receipts
    s0 = ex.receipt("S0")
    assert s0.params["account"]["pack"] == "example-trader" and s0.params["account"]["capital"] == 25000
    assert s0.params["thresholds"]["max_pbo"] == 0.18
    assert ex.receipt("S5").trial_family_count == 9 + 5  # grid + imported attempts (3 + 2); 0 outside variants
    assert "demo-trend-warning" in [x["id"] for x in s0.params["lessons"]]
    report = (home / "reports" / f"{tid}.md").read_text()
    assert "PASS · CONDITIONAL" in report and "ex-rate-source" in report

    # the user answers the open question with a point-in-time choice: version 2, clean PASS
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "ex-rate-source", "pit_daily", "Point-in-time daily published rates")
    res = ex.run()
    assert res.exit_code == 0, res.output
    s7b = ex.receipt("S7")
    assert s7b.verdict is Verdict.PASS and "conditions" not in s7b.params
    assert ex.receipt("S1").params["interview"]["answer_versions"]["ex-rate-source@dataset:prices"] == 2
    assert ex.receipt("S1").params["interview"]["unknown"] == 0
    assert s7b.receipt_hash() != s7.receipt_hash()


def test_a_risk_tagged_answer_is_a_condition_and_flags_the_dataset(ex, tmp_path, home):
    tid, fam = ex.trial_id, "trend|synthetic|1D"
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "bar-type-declared", "time", "Time bars, one day each")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "data-history-revised", "never_revised", "No, it is as first published")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "data-timezone", "utc", "UTC")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "timestamp-semantics", "bar_close", "the bar close")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "ex-rate-source", "current_curve", "the current curve applied to all of history")
    res = cli("interview", "status", "--trial", tid)
    assert "current curve" in res.output.lower() or "current_curve" in res.output
    reg = Registry(home / "registry.sqlite3")
    from qrh.interview.engine import dataset_flags
    from qrh.packs.runtime import PackView
    assert dataset_flags(reg.conn, PackView.load(home), "prices") == ["lookahead"]
    reg.close()


def test_fabricated_answer_is_refused_and_nothing_is_recorded(ex, tmp_path):
    cli("interview", "next", "--stage", "S1", "--scope", "dataset:prices", "--trial", ex.trial_id, "--json")
    t = say(tmp_path, "please continue with the next step")
    res = cli("interview", "answer", "bar-type-declared", "--scope", "dataset:prices", "--value", "time",
              "--transcript", str(t), "--session", SESSION)
    assert res.exit_code == 2 and "answer not found in a user message" in res.output
    assert "no answers recorded" in cli("interview", "status").output


def test_pack_lint_rule_runs_at_s2(ex, tmp_path, home):
    import yaml

    d = tmp_path / "lintpack"
    (d / "lint").mkdir(parents=True)
    (d / "qrh-pack.yaml").write_text(yaml.safe_dump({"name": "lintpack", "version": "1.0.0", "schema_version": 1,
                                                     "visibility": "private"}))
    (d / "lint" / "r.yaml").write_text(yaml.safe_dump({"rules": [{"id": "PK001", "severity": "BLOCK",
                                                                 "message": "rolling is banned here", "ast": {"call": "rolling"}}]}))
    assert cli("pack", "add", str(d)).exit_code == 0
    ex.new_trial()
    res = ex.run()
    assert res.exit_code == 2
    assert ex.receipt("S2").verdict is Verdict.FAIL and "PK001" in " ".join(ex.receipt("S2").reasons)


def test_interview_report_and_combined_pdf(ex, tmp_path, home, monkeypatch):
    monkeypatch.delenv("QRH_REPORTS", raising=False)
    tid = ex.trial_id
    assert ex.run().exit_code == 0
    res = cli("interview", "report", "--trial", tid, "--stage", "S1")
    assert res.exit_code == 0, res.output
    base = home / "reports" / tid
    assert (base / "interview.S1.pdf").read_bytes().startswith(b"%PDF")
    doc = json.loads((base / "interview.S1.chart.json").read_text())
    assert doc["figures"][0]["progress"]["total"] == 8 and doc["figures"][0]["light"] == "red"
    combined = (base / f"{tid}.pdf").read_bytes()
    assert combined.startswith(b"%PDF")
    s7 = json.loads((base / "S7.chart.json").read_text())
    assert s7["figures"][0]["conditions"]
    assert not cli("interview", "report", "--trial", "nope").exit_code == 0
