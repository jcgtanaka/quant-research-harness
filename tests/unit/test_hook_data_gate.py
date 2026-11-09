import pytest

from hook_helpers import bash_payload, hook_log, payload, register_dataset, run, seed_trial, session_trials, write_config

TID = "a1b2c3d4e5f60718293a4b5c6d7e8f90"


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


@pytest.fixture
def ready(tmp_path, home):
    """A trial bound to dataset 'prices' (PASS, unchanged)."""
    csv = register_dataset(home, tmp_path, "prices")
    seed_trial(home, TID, datasets=["prices"])
    return csv


def gate(home, command, env=None, **kw):
    return run("data-gate", bash_payload(command), home, env=env, **kw)


def test_run_with_passing_unchanged_dataset_is_allowed(ready, home):
    out = gate(home, f"qrh run {TID}")
    assert out.exit_code == 0 and out.stderr == ""
    rec = hook_log(home)[-1]
    assert rec["decision"] == "allow" and rec["trial_id"] == TID


def test_uv_run_prefix_and_chained_commands_are_recognized(ready, home, tmp_path):
    register_dataset(home, tmp_path, "prices", status="BLOCKED")
    assert gate(home, f"uv run qrh run {TID}").exit_code == 2
    assert gate(home, f"cd /x && qrh run {TID} | tail -3").exit_code == 2


@pytest.mark.parametrize("status", ["BLOCKED", "UNCHECKED"])
def test_non_pass_dataset_blocks_with_reason(ready, home, tmp_path, status):
    register_dataset(home, tmp_path, "prices", status=status)
    out = gate(home, f"qrh run {TID}")
    assert out.exit_code == 2
    assert "prices" in out.stderr and status in out.stderr


def test_changed_file_blocks(ready, home):
    ready.write_text(ready.read_text() + "2020-02-01T00:00:00Z,999\n")
    out = gate(home, f"qrh run {TID}")
    assert out.exit_code == 2 and "prices" in out.stderr and "changed" in out.stderr


def test_missing_file_blocks(ready, home):
    ready.unlink()
    out = gate(home, f"qrh run {TID}")
    assert out.exit_code == 2 and "prices" in out.stderr and "unreadable" in out.stderr


def test_every_failing_dataset_is_listed(tmp_path, home):
    register_dataset(home, tmp_path, "prices", status="BLOCKED")
    seed_trial(home, TID, datasets=["prices", "ghost"])
    out = gate(home, f"qrh run {TID}")
    assert out.exit_code == 2 and "prices" in out.stderr and "ghost" in out.stderr and "not in the catalog" in out.stderr


def test_unknown_trial_blocks(ready, home):
    out = gate(home, "qrh run deadbeefdeadbeef")
    assert out.exit_code == 2 and "unknown trial" in out.stderr


def test_unique_prefix_of_8_hex_resolves(ready, home):
    assert gate(home, f"qrh run {TID[:8]}").exit_code == 0
    assert gate(home, f"qrh run {TID[:6]}").exit_code == 2  # too short to be a prefix


def test_unrecognized_commands_pass_silently(ready, home):
    for cmd in ("ls -la", "git status", "pytest -q", "qrh status", "echo hello"):
        out = gate(home, cmd)
        assert out.exit_code == 0 and out.stdout == "" and out.stderr == ""
    assert session_trials(home) == []


def test_non_bash_tool_ignored(ready, home):
    data = payload("PreToolUse", tool_name="Edit", tool_input={"file_path": "x.py"})
    assert run("data-gate", data, home).exit_code == 0


def test_trial_without_datasets_is_allowed(home):
    seed_trial(home, TID)
    out = gate(home, f"qrh run {TID}")
    assert out.exit_code == 0


CFG_DATASETS = """
backtest_commands:
  - pattern: 'python\\s+bt\\.py'
    datasets: [prices, fx]
"""


def test_configured_command_maps_to_dataset_names(tmp_path, home):
    register_dataset(home, tmp_path, "prices")
    write_config(home, CFG_DATASETS)
    out = gate(home, "python bt.py --fast")
    assert out.exit_code == 2 and "fx" in out.stderr and "not in the catalog" in out.stderr
    register_dataset(home, tmp_path, "fx")
    assert gate(home, "python bt.py --fast").exit_code == 0


CFG_TRIAL = """
backtest_commands:
  - pattern: 'make bt (?P<trial>\\w+)'
    trial_id: from_args
  - pattern: 'runbt'
    trial_id: from_args
"""


def test_configured_trial_from_args_named_group_and_fallback(ready, home, tmp_path):
    write_config(home, CFG_TRIAL)
    assert gate(home, f"make bt {TID}").exit_code == 0
    assert gate(home, f"runbt --verbose {TID}").exit_code == 0  # first hex-like token after the match
    register_dataset(home, tmp_path, "prices", status="BLOCKED")
    assert gate(home, f"make bt {TID}").exit_code == 2
    out = gate(home, "runbt")
    assert out.exit_code == 2 and "trial" in out.stderr


def test_unreadable_registry_fails_closed(home):
    home.mkdir()
    (home / "registry.sqlite3").write_bytes(b"garbage, not sqlite " * 50)
    out = gate(home, f"qrh run {TID}")
    assert out.exit_code == 2 and "registry" in out.stderr


def test_unrelated_command_does_not_need_the_registry(home):
    home.mkdir()
    (home / "registry.sqlite3").write_bytes(b"garbage, not sqlite " * 50)
    assert gate(home, "ls").exit_code == 0


def test_malformed_config_blocks(ready, home):
    write_config(home, "backtest_commands: nope")
    assert gate(home, "ls").exit_code == 2


def test_human_approval_allows_one_blocked_run(ready, home, tmp_path):
    from hook_helpers import human_approve

    register_dataset(home, tmp_path, "prices", status="BLOCKED")
    first = gate(home, f"qrh run {TID}")
    assert first.exit_code == 2
    human_approve(home, first.stderr, "vendor feed outage, owner approved")
    assert gate(home, f"qrh run {TID}").exit_code == 0
    assert gate(home, f"qrh run {TID}").exit_code == 2
    from qrh.memory.registry import Registry

    reg = Registry(home / "registry.sqlite3")
    try:
        assert reg.list_hook_overrides("1970-01-01T00:00:00+00:00")[0]["trial_id"] == TID
    finally:
        reg.close()


def test_session_trial_tagging(ready, home):
    gate(home, f"qrh run {TID}")
    gate(home, f"qrh report {TID} --stage s4")
    gate(home, f"qrh report --stage S5 {TID}")
    gate(home, "qrh new hyp.yaml")
    recs = session_trials(home)
    assert [(r["kind"], r["trial_id"], r["stage"]) for r in recs] == [
        ("run", TID, None), ("report", TID, "S4"), ("report", TID, "S5"), ("new", None, None)]
    assert all(r["session_id"] == "sess-1" and r["ts"] for r in recs)


def test_mapped_backtest_command_is_tagged_and_blocked_runs_too(tmp_path, home):
    register_dataset(home, tmp_path, "prices", status="BLOCKED")
    seed_trial(home, TID, datasets=["prices"])
    write_config(home, CFG_TRIAL)
    assert gate(home, f"make bt {TID}").exit_code == 2
    assert session_trials(home)[0]["trial_id"] == TID  # intent is logged even when blocked


def test_logged_command_is_redacted(ready, home):
    gate(home, f"qrh run {TID} --token supersecretvalue1")
    raw = (home / "logs" / "hooks.jsonl").read_text()
    assert "supersecretvalue1" not in raw and "qrh run" in raw
