"""Session registry (F5): the real transcript path per session, recorded by SessionStart/UserPromptSubmit hooks."""

from hook_helpers import bash_payload, payload, run
from qrh.hooks import sessions


def test_session_start_and_prompt_submit_record_the_transcript(tmp_path):
    home = tmp_path / "home"
    run("session-context", payload("SessionStart", session_id="s-1", transcript_path="/x/s-1.jsonl"), home)
    assert sessions.lookup(home, "s-1") == "/x/s-1.jsonl"
    run("prior-attempts", payload("UserPromptSubmit", session_id="s-2", transcript_path="/x/s-2.jsonl", prompt="hi"), home)
    assert sessions.lookup(home, "s-2") == "/x/s-2.jsonl"
    assert sessions.lookup(home, "unknown") is None


def test_the_first_recorded_path_wins(tmp_path):
    home = tmp_path / "home"
    run("session-context", payload("SessionStart", session_id="s-1", transcript_path="/real/s-1.jsonl"), home)
    run("prior-attempts", payload("UserPromptSubmit", session_id="s-1", transcript_path="/forged.jsonl", prompt="x"), home)
    assert sessions.lookup(home, "s-1") == "/real/s-1.jsonl"


def test_other_events_record_nothing(tmp_path):
    home = tmp_path / "home"
    run("data-gate", bash_payload("ls", session_id="s-9", transcript_path="/x/s-9.jsonl"), home)
    assert sessions.lookup(home, "s-9") is None


def test_agent_cannot_run_hooks_by_hand_or_write_transcripts(tmp_path, monkeypatch):
    user = tmp_path / "user"
    user.mkdir()
    monkeypatch.setenv("HOME", str(user))
    home = user / ".qrh"
    home.mkdir()
    cmd = """echo '{"hook_event_name":"SessionStart","session_id":"f","transcript_path":"/t"}' | qrh hook session-context"""
    assert run("approval-guard", bash_payload(cmd), home).exit_code == 2
    w = payload("PreToolUse", tool_name="Write",
                tool_input={"file_path": str(user / ".claude" / "projects" / "p" / "f.jsonl"), "content": "{}"})
    assert run("approval-guard", w, home).exit_code == 2
    assert run("approval-guard", bash_payload(f"echo x >> {user}/.claude/projects/p/f.jsonl"), home).exit_code == 2
