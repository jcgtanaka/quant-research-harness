import json

import pytest
from typer.testing import CliRunner

from qrh.cli import app
from usage_helpers import SECRET, build_claude, write_settings

runner = CliRunner()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("QRH_HOME", str(home))
    monkeypatch.delenv("QRH_PROTECTED_LABELS", raising=False)
    claude = build_claude(tmp_path / "claude")
    return home, claude


def invoke(*args, **kw):
    return runner.invoke(app, ["usage", *args], **kw)


def test_init_writes_examples_then_harvest_report_verify_work(setup):
    home, claude = setup
    r = invoke("init")
    assert r.exit_code == 0 and "thresholds.json" in r.output and "EXAMPLE" in r.output
    r = invoke("harvest", "--claude-dir", str(claude))
    assert r.exit_code == 0, r.output
    assert "3 session(s)" in r.output and "1 malformed" in r.output
    r = invoke("report", "--by", "model")
    assert r.exit_code == 0 and "model-big" in r.output and "2 session(s) have no cost" in r.output
    assert (home / "usage" / "reports" / "plays.chart.json").is_file()
    r = invoke("verify", "--scope", "strict")
    assert r.exit_code == 0, r.output
    assert "includes subagents" in r.output and "inflation" in r.output.lower()


def test_harvest_uses_claude_config_dir_env(setup, monkeypatch):
    home, claude = setup
    write_settings(home)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    assert invoke("harvest").exit_code == 0


def test_harvest_and_report_exit_2_without_thresholds(setup):
    home, claude = setup
    r = invoke("harvest", "--claude-dir", str(claude))
    assert r.exit_code == 2 and "thresholds.json" in r.output
    r = invoke("report")
    assert r.exit_code == 2 and "thresholds.json" in r.output


def test_report_json_and_pdf(setup, tmp_path):
    home, claude = setup
    write_settings(home)
    invoke("harvest", "--claude-dir", str(claude))
    pdf = tmp_path / "out" / "u.pdf"
    r = invoke("report", "--by", "week", "--json", "--pdf", str(pdf))
    assert r.exit_code == 0 and pdf.read_bytes().startswith(b"%PDF")
    doc = json.loads(r.stdout)
    assert doc["by"] == "week" and doc["totals"]["sessions"] == 3
    assert SECRET not in r.stdout


def test_report_rejects_unknown_grouping(setup):
    home, _ = setup
    write_settings(home)
    r = invoke("report", "--by", "nope")
    assert r.exit_code == 2


def test_verify_exit_2_on_mismatch(setup):
    home, claude = setup
    write_settings(home)
    invoke("harvest", "--claude-dir", str(claude))
    path = home / "usage" / "sessions.json"
    doc = json.loads(path.read_text())
    doc["sessions"]["s1"]["harness_usage"]["model-big"]["input"] += 1000
    path.write_text(json.dumps(doc))
    r = invoke("verify", "--scope", "strict")
    assert r.exit_code == 2 and "mismatch" in r.output.lower()


def _plain_session(claude, home):
    """A session with no subagents whose cost-state matches its transcript exactly."""
    from usage_helpers import assistant, cost_state, flat, write_jsonl

    write_jsonl(claude / "projects" / "-work-projA" / "s9.jsonl", flat(
        assistant("p1", "model-big", "2026-03-02T10:00:00.000Z", inp=10, out=5, cr=100, cw5=20),
        cost_state(1.0, {"model-big": {"inp": 10, "out": 5, "cr": 100, "cw": 20, "cost": 1.0}})))


def test_verify_defaults_to_transcript_scope_and_gates_on_sessions_without_subagents(setup):
    home, claude = setup
    write_settings(home)
    _plain_session(claude, home)
    invoke("harvest", "--claude-dir", str(claude))
    r = invoke("verify")
    assert r.exit_code == 0, r.output
    assert "scope               transcript" in r.output and "not gating" in r.output
    path = home / "usage" / "sessions.json"
    doc = json.loads(path.read_text())
    doc["sessions"]["s9"]["harness_usage"]["model-big"]["cache_read"] += 1000
    path.write_text(json.dumps(doc))
    bad = invoke("verify")
    assert bad.exit_code == 2 and "unexplained" in bad.output


def test_verify_with_only_subagent_sessions_has_nothing_to_gate_on(setup):
    home, claude = setup
    write_settings(home)
    invoke("harvest", "--claude-dir", str(claude))
    r = invoke("verify")
    assert r.exit_code == 2 and "nothing to check" in r.output
    assert invoke("verify", "--scope", "strict").exit_code == 0


def test_verify_without_archive_is_blocked_not_a_pass(setup):
    r = invoke("verify")
    assert r.exit_code == 2


def test_log_local_appends_a_record_from_stdin(setup):
    home, _ = setup
    resp = json.dumps({"model": "qwen-x", "prompt_eval_count": 12, "eval_count": 4, "total_duration": 3_000_000_000})
    r = invoke("log-local", "--task", "extract", "--session-id", "s9", input=resp)
    assert r.exit_code == 0
    rec = json.loads((home / "logs" / "local_llm.jsonl").read_text().strip())
    assert rec["model"] == "qwen-x" and rec["prompt_tokens"] == 12 and rec["session_id"] == "s9"
    assert invoke("log-local", input="garbage").exit_code == 2


# --- acknowledging an unexplained session -----------------------------------------------------------

def _two_plain_sessions(claude):
    from usage_helpers import assistant, cost_state, flat, write_jsonl

    for sid, day in (("g1", 2), ("g2", 3)):
        write_jsonl(claude / "projects" / "-work-projA" / f"{sid}.jsonl", flat(
            assistant(f"{sid}m", "model-big", f"2026-03-0{day}T10:00:00.000Z", inp=10, out=5, cr=100, cw5=20),
            cost_state(1.0, {"model-big": {"inp": 10, "out": 5, "cr": 100, "cw": 20, "cost": 1.0}})))


def _break(home, sid, extra=1000):
    path = home / "usage" / "sessions.json"
    doc = json.loads(path.read_text())
    doc["sessions"][sid]["harness_usage"]["model-big"]["cache_read"] += extra
    path.write_text(json.dumps(doc))


def _alias_of(home, sid):
    doc = json.loads((home / "usage" / "sessions.json").read_text())["sessions"]
    order = sorted(doc.values(), key=lambda s: (s.get("start") or "", s["session_id"]))
    return f"S{[s['session_id'] for s in order].index(sid) + 1:03d}"


@pytest.fixture
def broken(setup):
    home, claude = setup
    write_settings(home)
    _two_plain_sessions(claude)
    invoke("harvest", "--claude-dir", str(claude))
    _break(home, "g2")
    assert invoke("verify").exit_code == 2
    return home, claude, _alias_of(home, "g2")


def test_ack_records_and_stops_gating_but_stays_listed(broken):
    home, claude, alias = broken
    r = invoke("verify", "--ack", alias, "--reason", "cause not found after checking stop reasons and gaps")
    assert r.exit_code == 0, r.output
    ack = json.loads((home / "usage" / "verify_acks.json").read_text())
    (entry,) = ack["acks"].values()
    assert entry["alias"] == alias and "stop reasons" in entry["reason"] and entry["ts"]
    assert entry["figures"]["best_diff_pct"] > 1 and "fingerprint" in entry
    out = invoke("verify")
    assert out.exit_code == 0
    assert "acknowledged, unexplained" in out.output and alias in out.output and "not gating" in out.output
    assert "unexplained mismatch(es)" in out.output and " 0 unexplained" in out.output


def test_ack_needs_a_long_enough_reason_a_known_alias_and_an_actual_mismatch(broken):
    home, _, alias = broken
    short = invoke("verify", "--ack", alias, "--reason", "too short")
    assert short.exit_code == 2 and "at least 10" in short.output
    assert invoke("verify", "--ack", "S999", "--reason", "a perfectly long reason").exit_code == 2
    good = invoke("verify", "--ack", "S001", "--reason", "a perfectly long reason")
    assert good.exit_code == 2 and "not a gating mismatch" in good.output
    assert not (home / "usage" / "verify_acks.json").exists()
    assert invoke("verify", "--ack", alias).exit_code != 0                      # --reason is required


def test_ack_expires_when_the_numbers_change(broken):
    home, claude, alias = broken
    invoke("verify", "--ack", alias, "--reason", "cause not found, evidence in docs")
    assert invoke("verify").exit_code == 0
    _break(home, "g2", extra=7)                                                  # a re-harvest would give other totals
    out = invoke("verify")
    assert out.exit_code == 2 and "acknowledgement expired" in out.output and alias in out.output


def test_ack_survives_an_identical_reharvest_and_never_hides_other_sessions(broken):
    home, claude, alias = broken
    invoke("verify", "--ack", alias, "--reason", "cause not found, evidence in docs")
    _break(home, "g1")                                                           # a new problem elsewhere
    out = invoke("verify")
    assert out.exit_code == 2 and out.output.count("[unexplained]") == 1


def test_ack_file_is_corrupt_means_blocked_not_ignored(broken):
    home, _, _ = broken
    (home / "usage" / "verify_acks.json").write_text("{nope")
    assert invoke("verify").exit_code == 2
