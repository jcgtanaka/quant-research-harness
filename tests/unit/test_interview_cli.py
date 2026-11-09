"""qrh interview: next / answer / add-question / status, provenance and reuse rules."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from packs_helpers import lesson, make_pack, question
from qrh.cli import app
from qrh.memory.registry import Registry

runner = CliRunner()
EXAMPLE_TRIAL_SPEC = {
    "text": "t", "strategy_family": "trend", "asset_class": "equity", "frequency": "1D",
    "parameter_space_size": 1, "thresholds": {"k": 1}, "difference_statement": "d",
}


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    monkeypatch.setenv("HOME", str(tmp_path / "user"))  # transcripts live under ~/.claude/projects of a fake home
    return h


@pytest.fixture
def packed(tmp_path):
    d = make_pack(tmp_path, "bank", questions=[[
        question("rf", critical=True),
        question("tz", critical=False, answer_type="bool", text="Do timestamps mark bar close?", choices=[]),
        question("variants", stage="S3", scope="strategy", answer_type="number", critical=True,
                 text="How many variants were tried?", choices=[], maps_to={"kind": "imported_trials"}),
    ]])
    assert runner.invoke(app, ["pack", "add", str(d)]).exit_code == 0
    return d


def cli(*args):
    return runner.invoke(app, ["interview", *args])


def nxt(stage="S1", scope="dataset:prices"):
    res = cli("next", "--stage", stage, "--scope", scope, "--json")
    assert res.exit_code == 0, res.output
    return json.loads(res.stdout)


def transcript(tmp_path, text, ts=None, session="sess", record=True):
    """A transcript where Claude Code keeps it, registered for the session as the SessionStart hook would."""
    from qrh.hooks import sessions

    ts = ts or (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
    p = tmp_path / "user" / ".claude" / "projects" / "proj" / f"{session}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"type": "user", "timestamp": ts, "sessionId": session,
                             "message": {"role": "user", "content": text}}) + "\n")
    if record:
        sessions.record(tmp_path / "qrh_home", session, str(p), datetime.now(timezone.utc))
    return p


def answer(tmp_path, qid, value, scope="dataset:prices", said=None, session="sess"):
    t = transcript(tmp_path, said if said is not None else value, session=session)
    return cli("answer", qid, "--scope", scope, "--value", value, "--transcript", str(t), "--session", session)


def test_next_returns_critical_first_with_why_and_choices(packed):
    out = nxt()
    q = out["question"]
    assert q["id"] == "rf" and q["critical"] and "timing" in q["why_it_matters"]
    assert [c["id"] for c in q["choices"]] == ["pit", "curve"] and out["remaining"] == 2


def test_next_without_a_bank_says_nothing_to_ask(home):
    assert nxt()["question"] is None


def test_answer_needs_the_question_to_have_been_issued(packed, tmp_path):
    res = answer(tmp_path, "rf", "pit", said="pit")
    assert res.exit_code == 2 and "issued" in res.output


def test_fabricated_answer_is_rejected(packed, tmp_path):
    nxt()
    res = answer(tmp_path, "rf", "pit", said="I have no idea what you mean by that word")
    assert res.exit_code == 2 and "answer not found in a user message" in res.output
    assert nxt()["question"]["id"] == "rf"  # still pending


def test_answer_from_a_message_before_the_question_is_rejected(packed, tmp_path):
    nxt()
    t = transcript(tmp_path, "pit", ts=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())
    res = cli("answer", "rf", "--scope", "dataset:prices", "--value", "pit", "--transcript", str(t), "--session", "sess")
    assert res.exit_code == 2


def test_verified_answer_is_recorded_and_not_asked_again(packed, tmp_path, home):
    nxt()
    res = answer(tmp_path, "rf", "pit", said="Point-in-time daily published, from the vendor docs")
    assert res.exit_code == 0, res.output
    assert nxt()["question"]["id"] == "tz"
    reg = Registry(home / "registry.sqlite3")
    row = reg.conn.execute("SELECT * FROM interview_answers").fetchone()
    assert row["provenance"] == "transcript-verified" and row["answer"] == "pit" and row["version"] == 1
    reg.close()


def test_the_choice_label_in_the_user_message_is_enough(packed, tmp_path):
    nxt()
    assert answer(tmp_path, "rf", "curve", said="the Current curve applied to history, sadly").exit_code == 0


def test_unknown_is_always_valid_and_records_the_risk(packed, tmp_path, home):
    nxt()
    res = answer(tmp_path, "rf", "unknown", said="honestly I don't know")
    assert res.exit_code == 0, res.output
    reg = Registry(home / "registry.sqlite3")
    row = reg.conn.execute("SELECT * FROM interview_answers").fetchone()
    assert row["answer"] == "unknown" and "look-ahead" in json.loads(row["mapped"])["unknown_risk"]
    reg.close()


def test_invalid_choice_is_rejected_before_provenance(packed, tmp_path):
    nxt()
    res = answer(tmp_path, "rf", "bogus", said="bogus")
    assert res.exit_code == 2 and "not one of the choices" in res.output


def test_risk_tag_adds_a_dataset_flag_and_number_maps_to_imported_trials(packed, tmp_path, home):
    nxt()
    answer(tmp_path, "rf", "curve", said="current curve applied to history")
    nxt("S3", "strategy:trend|equity|1D")
    assert answer(tmp_path, "variants", "40", scope="strategy:trend|equity|1D", said="about 40").exit_code == 0
    reg = Registry(home / "registry.sqlite3")
    from qrh.interview.engine import dataset_flags
    from qrh.packs.runtime import PackView
    assert dataset_flags(reg.conn, PackView.load(home), "prices") == ["lookahead"]
    assert reg.family_trial_count("trend|equity|1D") == 40
    reg.close()


def test_changing_an_answer_creates_a_new_version(packed, tmp_path, home):
    nxt()
    answer(tmp_path, "rf", "curve", said="current curve applied to history")
    nxt()
    answer(tmp_path, "rf", "pit", said="actually point-in-time daily published")
    reg = Registry(home / "registry.sqlite3")
    rows = reg.conn.execute("SELECT version, answer FROM interview_answers ORDER BY version").fetchall()
    assert [(r[0], r[1]) for r in rows] == [(1, "curve"), (2, "pit")]
    reg.close()


def test_human_mode_records_human_cli_provenance(packed, home):
    res = cli("answer", "rf", "--scope", "dataset:prices", "--value", "pit", "--human")
    assert res.exit_code == 0, res.output
    reg = Registry(home / "registry.sqlite3")
    assert reg.conn.execute("SELECT provenance FROM interview_answers").fetchone()[0] == "human-cli"
    reg.close()


def test_answer_without_transcript_or_human_is_refused(packed):
    res = cli("answer", "rf", "--scope", "dataset:prices", "--value", "pit")
    assert res.exit_code == 2 and "--transcript" in res.output


def test_answer_for_an_unknown_question_or_scope(packed, tmp_path):
    assert answer(tmp_path, "nope", "x").exit_code == 2
    assert cli("answer", "rf", "--scope", "hypothesis:abc", "--value", "pit", "--human").exit_code == 2  # wrong scope kind
    assert cli("answer", "rf", "--scope", "bogus", "--value", "pit", "--human").exit_code == 2


def test_llm_followup_question_is_recorded_and_never_critical(packed, tmp_path):
    res = cli("add-question", "--stage", "S1", "--scope", "dataset:prices", "--text", "Were splits adjusted?",
              "--why", "Unadjusted splits look like crashes.")
    assert res.exit_code == 0, res.output
    qid = res.stdout.strip().splitlines()[-1]
    assert qid.startswith("llm-")
    out = nxt()
    assert out["question"]["id"] == "rf"
    nxt()
    answer(tmp_path, "rf", "pit", said="point-in-time daily published")
    nxt()
    assert answer(tmp_path, "tz", "yes").exit_code == 0
    out = cli("next", "--stage", "S1", "--scope", "dataset:prices", "--json")
    q = json.loads(out.stdout)["question"]
    assert q["id"] == qid and q["origin"] == "llm" and q["critical"] is False


def test_status_lists_answers_per_scope(packed, tmp_path):
    nxt()
    answer(tmp_path, "rf", "pit", said="point-in-time daily published")
    res = cli("status", "--json")
    assert res.exit_code == 0
    doc = json.loads(res.stdout)
    assert doc["answers"][0]["qid"] == "rf" and doc["answers"][0]["provenance"] == "transcript-verified"


def test_include_unknown_offers_a_dont_know_again(packed, tmp_path):
    nxt()
    answer(tmp_path, "rf", "unknown", said="I don't know")
    assert nxt()["question"]["id"] == "tz"
    res = cli("next", "--stage", "S1", "--scope", "dataset:prices", "--include-unknown", "--json")
    assert json.loads(res.stdout)["question"]["id"] == "rf"
    assert answer(tmp_path, "rf", "pit", said="point-in-time daily published").exit_code == 0


# --- F5: the transcript must be the real one of a recorded session --------------------------------

def test_answer_from_an_unrecorded_session_is_refused(packed, tmp_path):
    nxt()
    t = transcript(tmp_path, "pit", session="ghost", record=False)
    res = cli("answer", "rf", "--scope", "dataset:prices", "--value", "pit", "--transcript", str(t), "--session", "ghost")
    assert res.exit_code == 2 and "session" in res.output


def test_answer_with_a_different_transcript_is_refused(packed, tmp_path):
    nxt()
    transcript(tmp_path, "pit")                                    # the real, recorded transcript of "sess"
    forged = tmp_path / "user" / ".claude" / "projects" / "proj" / "forged.jsonl"
    forged.write_text((tmp_path / "user" / ".claude" / "projects" / "proj" / "sess.jsonl").read_text())
    res = cli("answer", "rf", "--scope", "dataset:prices", "--value", "pit", "--transcript", str(forged), "--session", "sess")
    assert res.exit_code == 2 and "transcript" in res.output


def test_answer_with_a_transcript_outside_claude_projects_is_refused(packed, tmp_path):
    from qrh.hooks import sessions

    nxt()
    outside = tmp_path / "elsewhere.jsonl"
    outside.write_text(json.dumps({"type": "user", "timestamp": (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat(),
                                   "sessionId": "s-out", "message": {"role": "user", "content": "pit"}}) + "\n")
    sessions.record(tmp_path / "qrh_home", "s-out", str(outside), datetime.now(timezone.utc))
    res = cli("answer", "rf", "--scope", "dataset:prices", "--value", "pit", "--transcript", str(outside), "--session", "s-out")
    assert res.exit_code == 2 and ".claude/projects" in res.output
