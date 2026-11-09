"""Track 2 (user-supplied backtest): the ten fixed questions reach the interview, one at a time, with provenance."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from qrh.checks.two_track import QUESTIONS, WARNING, evaluate_two_track
from qrh.cli import app
from qrh.hooks import sessions
from qrh.interview import engine
from qrh.memory.registry import HypothesisRecord, Registry

runner = CliRunner()
SPEC = {
    "text": "t", "strategy_family": "trend", "asset_class": "equity", "frequency": "1D",
    "parameter_space_size": 1, "thresholds": {"k": 1}, "difference_statement": "d",
}
IDS = [f"t2-{q.id}" for q in QUESTIONS]


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    return h


def register(home, **extra):
    home.mkdir(parents=True, exist_ok=True)
    reg = Registry(home / "registry.sqlite3")
    tid = reg.register_hypothesis(HypothesisRecord(**{**SPEC, **extra}))
    reg.conn.commit()
    reg.close()
    return tid


def cli(*args):
    return runner.invoke(app, ["interview", *args])


def nxt(tid, trial=True):
    args = ["next", "--stage", "S1", "--scope", f"hypothesis:{tid}", "--json"] + (["--trial", tid] if trial else [])
    res = cli(*args)
    assert res.exit_code == 0, res.output
    return json.loads(res.stdout)


def say(tmp_path, home, qid, tid, value, said):
    ts = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
    p = tmp_path / "user" / ".claude" / "projects" / "proj" / "sess.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"type": "user", "timestamp": ts, "sessionId": "sess",
                             "message": {"role": "user", "content": said}}) + "\n")
    sessions.record(home, "sess", str(p), datetime.now(timezone.utc))
    return cli("answer", qid, "--scope", f"hypothesis:{tid}", "--value", value, "--transcript", str(p),
               "--session", "sess")


def stored(home):
    reg = Registry(home / "registry.sqlite3")
    rows = [dict(r) for r in reg.conn.execute("SELECT * FROM interview_answers ORDER BY id")]
    reg.close()
    return rows


def test_user_origin_gets_the_ten_questions_in_order(home, tmp_path):
    tid = register(home, origin="user")
    seen = []
    for qid, value in zip(IDS, ["broker"] * 10):
        out = nxt(tid)
        assert out["question"]["id"] == qid
        seen.append(qid)
        res = cli("answer", qid, "--scope", f"hypothesis:{tid}", "--value", "unknown", "--human")
        assert res.exit_code == 0, res.output
    assert seen == IDS and nxt(tid)["question"] is None


def test_scope_lookup_without_trial_flag_also_finds_them(home):
    tid = register(home, origin="user")
    assert nxt(tid, trial=False)["question"]["id"] == IDS[0]


def test_choices_are_the_track2_options_without_unknown(home):
    tid = register(home, origin="user")
    q = nxt(tid)["question"]
    assert [c["id"] for c in q["choices"]] == ["broker", "data_vendor", "own_file"]


def test_harness_origin_and_absent_origin_get_none(home):
    a = register(home)
    b = register(home, origin="harness")
    assert nxt(a)["question"] is None and nxt(b)["question"] is None


def test_harness_hash_is_unchanged_by_the_new_field(home):
    base = HypothesisRecord(**SPEC)
    assert HypothesisRecord(**SPEC, origin="harness").hypothesis_hash() == base.hypothesis_hash()
    assert HypothesisRecord(**SPEC, origin="user").hypothesis_hash() != base.hypothesis_hash()
    assert "origin" not in base.binding()


def test_bad_origin_is_rejected():
    with pytest.raises(ValueError):
        HypothesisRecord(**SPEC, origin="robot")


def test_answer_is_stored_with_transcript_provenance(home, tmp_path):
    tid = register(home, origin="user")
    nxt(tid)
    res = say(tmp_path, home, IDS[0], tid, "broker", "The prices come from my broker")
    assert res.exit_code == 0, res.output
    row = stored(home)[0]
    assert (row["qid"], row["answer"], row["provenance"], row["version"]) == (IDS[0], "broker", "transcript-verified", 1)
    assert nxt(tid)["question"]["id"] == IDS[1]


def test_unverified_answer_is_refused(home, tmp_path):
    tid = register(home, origin="user")
    nxt(tid)
    res = say(tmp_path, home, IDS[0], tid, "broker", "no idea what you mean")
    assert res.exit_code == 2 and stored(home) == []


def test_invalid_option_is_refused(home):
    tid = register(home, origin="user")
    nxt(tid)
    res = cli("answer", IDS[0], "--scope", f"hypothesis:{tid}", "--value", "bogus", "--human")
    assert res.exit_code == 2 and stored(home) == []


def test_unknown_is_stored_and_takes_the_warning_path(home):
    tid = register(home, origin="user")
    nxt(tid)
    assert cli("answer", IDS[0], "--scope", f"hypothesis:{tid}", "--value", "unknown", "--human").exit_code == 0
    row = stored(home)[0]
    assert row["answer"] == "unknown" and row["provenance"] == "human-cli"
    reg = Registry(home / "registry.sqlite3")
    answers = engine.track2_answers(reg.conn, f"hypothesis:{tid}")
    reg.close()
    assert answers == {"price_source": "unknown"}
    result = evaluate_two_track(answers, {}, {})
    assert result.results[0].status == WARNING and result.overall == WARNING


def test_answered_option_is_passed_to_the_answer_rules(home):
    tid = register(home, origin="user")
    nxt(tid)
    cli("answer", IDS[0], "--scope", f"hypothesis:{tid}", "--value", "own_file", "--human")
    reg = Registry(home / "registry.sqlite3")
    answers = engine.track2_answers(reg.conn, f"hypothesis:{tid}")
    reg.close()
    assert answers == {"price_source": "own_file"}
