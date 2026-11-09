from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from hook_helpers import DB, bash_payload, hook_log, payload, register_dataset, run, write_config
from qrh.cli import app
from qrh.hooks.pre_restart_guard import SourceHashError, compute_source_hash
from qrh.memory.registry import Registry

CFG = """
engine_start_commands: ['engine\\s+start']
guard:
  source_globs: ["src/**/*.py"]
  build_hash_file: build/HASH
  required_datasets: [prices]
  max_dataset_age: {prices: 3D}
  test_receipt_max_age: 1D
  parity: {disabled: "parity tooling is not built yet"}
"""
NOW = datetime(2020, 1, 11, 12, 0, tzinfo=timezone.utc)  # the synthetic dataset ends 2020-01-10
runner = CliRunner()


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


@pytest.fixture
def proj(tmp_path):
    d = tmp_path / "proj"
    (d / "src").mkdir(parents=True)
    (d / "src" / "a.py").write_text("A = 1\n")
    (d / "src" / "b.py").write_text("B = 2\n")
    (d / "build").mkdir()
    return d


def add_receipt(home, source_hash, exit_code=0, created_at=None):
    reg = Registry(home / DB)
    reg.record_test_receipt(source_hash, "pytest -q", exit_code, 1.5)
    if created_at:
        reg.conn.execute("UPDATE test_receipts SET created_at=?", (created_at,))
    reg.close()


@pytest.fixture
def good(tmp_path, home, proj):
    """Everything satisfied: build hash, test receipt, dataset PASS and fresh, parity disabled with reason."""
    write_config(home, CFG)
    register_dataset(home, tmp_path, "prices")
    h = compute_source_hash(["src/**/*.py"], proj)
    (proj / "build" / "HASH").write_text(h + "\n")
    add_receipt(home, h, created_at=NOW.isoformat())
    return h


def guard(home, proj, command="engine start --paper", env=None, now=NOW):
    data = bash_payload(command)
    data["cwd"] = str(proj)
    return run("pre-restart-guard", data, home, env=env, now=lambda: now)


def test_source_hash_depends_on_content_not_order_and_needs_files(proj):
    h = compute_source_hash(["src/**/*.py"], proj)
    assert h == compute_source_hash(["src/b.py", "src/a.py"], proj)
    (proj / "src" / "a.py").write_text("A = 2\n")
    assert compute_source_hash(["src/**/*.py"], proj) != h
    with pytest.raises(SourceHashError):
        compute_source_hash(["nothing/*.py"], proj)


def test_all_checks_pass(good, home, proj):
    out = guard(home, proj)
    assert out.exit_code == 0 and out.stderr == ""
    rec = hook_log(home)[-1]
    assert rec["decision"] == "allow" and rec["parity_disabled_reason"] == "parity tooling is not built yet"


def test_unrecognized_command_allowed(good, home, proj):
    assert guard(home, proj, "ls").exit_code == 0
    assert run("pre-restart-guard", payload("PreToolUse", tool_name="Edit", tool_input={}), home).exit_code == 0


def test_build_hash_mismatch_blocks(good, home, proj):
    (proj / "build" / "HASH").write_text("0" * 64)
    out = guard(home, proj)
    assert out.exit_code == 2 and "build hash" in out.stderr


def test_build_hash_file_missing_blocks(good, home, proj):
    (proj / "build" / "HASH").unlink()
    out = guard(home, proj)
    assert out.exit_code == 2 and "build hash" in out.stderr


def test_source_edit_after_build_blocks_hash_and_test_receipt(good, home, proj):
    (proj / "src" / "a.py").write_text("A = 99\n")
    out = guard(home, proj)
    assert out.exit_code == 2 and "build hash" in out.stderr and "test receipt" in out.stderr


def test_missing_failing_or_stale_test_receipt_blocks(good, home, proj):
    reg = Registry(home / DB)
    reg.conn.execute("DELETE FROM test_receipts")
    reg.close()
    assert "test receipt" in guard(home, proj).stderr
    add_receipt(home, good, exit_code=1, created_at=NOW.isoformat())
    out = guard(home, proj)
    assert out.exit_code == 2 and "test receipt" in out.stderr
    add_receipt(home, good, created_at="2020-01-05T00:00:00+00:00")  # passing, but older than 1D
    reg = Registry(home / DB)
    reg.conn.execute("UPDATE test_receipts SET created_at='2020-01-05T00:00:00+00:00'")
    reg.close()
    out = guard(home, proj)
    assert out.exit_code == 2 and "older than" in out.stderr


def test_dataset_problems_block(good, home, proj, tmp_path):
    register_dataset(home, tmp_path, "prices", status="BLOCKED")
    out = guard(home, proj)
    assert out.exit_code == 2 and "prices" in out.stderr and "BLOCKED" in out.stderr


def test_stale_dataset_blocks(good, home, proj):
    out = guard(home, proj, now=datetime(2020, 2, 1, tzinfo=timezone.utc))
    assert out.exit_code == 2 and "prices" in out.stderr and "older than 3D" in out.stderr


def test_parity_required_without_report_warns_then_passes_with_one(good, home, proj):
    write_config(home, CFG.replace('{disabled: "parity tooling is not built yet"}', "required"))
    out = guard(home, proj)
    assert out.exit_code == 0 and "parity report missing" in out.stdout
    (home / "parity").mkdir()
    (home / "parity" / "report.json").write_text("{}")
    assert guard(home, proj).exit_code == 0


def test_parity_defaults_to_required(good, home, proj):
    write_config(home, CFG.replace('  parity: {disabled: "parity tooling is not built yet"}\n', ""))
    assert "parity report missing" in guard(home, proj).stdout


def test_every_failed_check_is_listed(good, home, proj, tmp_path):
    (proj / "build" / "HASH").write_text("0" * 64)
    register_dataset(home, tmp_path, "prices", status="BLOCKED")
    write_config(home, CFG.replace('{disabled: "parity tooling is not built yet"}', "required"))
    (proj / "src" / "a.py").write_text("A = 7\n")
    out = guard(home, proj)
    assert out.exit_code == 2
    for needle in ("build hash", "test receipt", "prices", "parity report missing"):
        assert needle in out.stderr, needle


def test_guard_not_configured_blocks(home, proj):
    write_config(home, "engine_start_commands: ['engine\\s+start']\n")
    out = guard(home, proj)
    assert out.exit_code == 2 and "guard" in out.stderr and "not configured" in out.stderr


def test_no_matching_source_files_blocks(good, home, proj):
    write_config(home, CFG.replace("src/**/*.py", "nothing/**/*.py"))
    out = guard(home, proj)
    assert out.exit_code == 2 and "source" in out.stderr


def test_bad_offset_string_blocks(good, home, proj):
    write_config(home, CFG.replace("test_receipt_max_age: 1D", "test_receipt_max_age: soonish"))
    out = guard(home, proj)
    assert out.exit_code == 2 and "soonish" in out.stderr


def test_unreadable_registry_blocks(good, home, proj):
    (home / DB).write_bytes(b"garbage, not sqlite " * 50)
    out = guard(home, proj)
    assert out.exit_code == 2 and "registry" in out.stderr


def test_malformed_config_blocks(good, home, proj):
    write_config(home, "guard: [oops")
    assert guard(home, proj).exit_code == 2


def test_human_approval_allows_one_start(good, home, proj):
    from hook_helpers import human_approve

    (proj / "build" / "HASH").write_text("0" * 64)
    first = guard(home, proj)
    assert first.exit_code == 2
    human_approve(home, first.stderr, "hotfix restart, tests run by hand", now=NOW)
    assert guard(home, proj).exit_code == 0
    assert guard(home, proj).exit_code == 2


# --- qrh test-receipt ---------------------------------------------------------------------------

@pytest.fixture
def in_proj(home, proj, monkeypatch):
    write_config(home, CFG)
    monkeypatch.setenv("QRH_HOME", str(home))
    monkeypatch.chdir(proj)
    return proj


def test_test_receipt_records_pass(in_proj, home):
    res = runner.invoke(app, ["test-receipt", "--", "python", "-c", "import sys; sys.exit(0)"])
    assert res.exit_code == 0, res.output
    h = compute_source_hash(["src/**/*.py"], in_proj)
    reg = Registry(home / DB)
    got = reg.latest_passing_test_receipt(h)
    reg.close()
    assert got and got["exit_code"] == 0 and got["duration_s"] >= 0 and h[:12] in res.output


def test_test_receipt_records_failure_and_exits_2(in_proj, home):
    res = runner.invoke(app, ["test-receipt", "--", "python", "-c", "import sys; sys.exit(3)"])
    assert res.exit_code == 2 and "exit code 3" in res.output
    h = compute_source_hash(["src/**/*.py"], in_proj)
    reg = Registry(home / DB)
    row = reg.conn.execute("SELECT exit_code, passed FROM test_receipts").fetchone()
    assert reg.latest_passing_test_receipt(h) is None
    reg.close()
    assert row["exit_code"] == 3 and row["passed"] == 0


def test_test_receipt_command_not_found_is_recorded_as_failure(in_proj, home):
    res = runner.invoke(app, ["test-receipt", "--", "definitely-not-a-command-xyz"])
    assert res.exit_code == 2
    reg = Registry(home / DB)
    assert reg.conn.execute("SELECT passed FROM test_receipts").fetchone()["passed"] == 0
    reg.close()


def test_test_receipt_redacts_the_stored_command(in_proj, home):
    runner.invoke(app, ["test-receipt", "--", "python", "-c", "pass", "--token", "supersecretvalue1"])
    reg = Registry(home / DB)
    cmd = reg.conn.execute("SELECT command FROM test_receipts").fetchone()["command"]
    reg.close()
    assert "supersecretvalue1" not in cmd


def test_test_receipt_needs_guard_config_and_a_command(home, proj, monkeypatch):
    monkeypatch.setenv("QRH_HOME", str(home))
    monkeypatch.chdir(proj)
    assert runner.invoke(app, ["test-receipt", "--", "python", "-c", "pass"]).exit_code == 2  # no guard section
    write_config(home, CFG)
    assert runner.invoke(app, ["test-receipt"]).exit_code == 2  # no command


def test_hooks_source_hash_command(in_proj):
    res = runner.invoke(app, ["hooks", "source-hash"])
    assert res.exit_code == 0 and res.output.strip() == compute_source_hash(["src/**/*.py"], in_proj)
