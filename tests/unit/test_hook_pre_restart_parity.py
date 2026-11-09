"""Pre-restart guard, parity report check: a missing, unreadable, other-trial or old report only WARNS."""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from hook_helpers import DB, bash_payload, hook_log, register_dataset, run, write_config
from qrh.hooks import pre_restart_guard
from qrh.hooks.common import tag_session_trial
from qrh.hooks.pre_restart_guard import compute_source_hash
from qrh.memory.registry import Registry

CFG = """
engine_start_commands: ['engine\\s+start']
guard:
  source_globs: ["src/**/*.py"]
  build_hash_file: build/HASH
  required_datasets: [prices]
  max_dataset_age: {prices: 3D}
  test_receipt_max_age: 1D
  parity: required
"""
NOW = datetime(2020, 1, 11, 12, 0, tzinfo=timezone.utc)
TRIAL = "trial-A"


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


@pytest.fixture
def proj(tmp_path):
    d = tmp_path / "proj"
    (d / "src").mkdir(parents=True)
    (d / "src" / "a.py").write_text("A = 1\n")
    (d / "build").mkdir()
    return d


@pytest.fixture
def good(tmp_path, home, proj):
    """Every other check passes; the session is tagged with TRIAL; parity is required."""
    write_config(home, CFG)
    register_dataset(home, tmp_path, "prices")
    h = compute_source_hash(["src/**/*.py"], proj)
    (proj / "build" / "HASH").write_text(h + "\n")
    reg = Registry(home / DB)
    reg.record_test_receipt(h, "pytest -q", 0, 1.5)
    reg.conn.execute("UPDATE test_receipts SET created_at=?", (NOW.isoformat(),))
    reg.close()
    tag_session_trial(home, "sess-1", TRIAL, None, "test", ts=NOW)


def guard(home, proj):
    data = bash_payload("engine start --paper")
    data["cwd"] = str(proj)
    return run("pre-restart-guard", data, home, now=lambda: NOW)


def put_report(home, trial, age_days, name=None):
    d = home / "parity"
    d.mkdir(exist_ok=True)
    p = d / (name or f"{trial}.json")
    p.write_text(json.dumps({"trial_id": trial}))
    t = (NOW - timedelta(days=age_days)).timestamp()
    os.utime(p, (t, t))
    return p


def test_max_age_is_a_named_constant_of_seven_days():
    assert pre_restart_guard.PARITY_MAX_AGE == timedelta(days=7)


def test_no_report_warns_and_allows(good, home, proj):
    out = guard(home, proj)
    assert out.exit_code == 0
    assert "parity report missing" in out.stdout
    assert hook_log(home)[-1]["decision"] == "warn"


def test_report_for_another_trial_warns(good, home, proj):
    put_report(home, "trial-B", 0)
    out = guard(home, proj)
    assert out.exit_code == 0 and "trial-A" in out.stdout and "trial-B" not in out.stdout.split("trial-A")[0]
    assert hook_log(home)[-1]["decision"] == "warn"


def test_old_report_warns(good, home, proj):
    put_report(home, TRIAL, 8)
    out = guard(home, proj)
    assert out.exit_code == 0 and "older than 7 days" in out.stdout
    assert hook_log(home)[-1]["decision"] == "warn"


def test_unreadable_report_warns(good, home, proj):
    p = put_report(home, TRIAL, 0)
    p.write_text("not json {")
    out = guard(home, proj)
    assert out.exit_code == 0 and "cannot be read" in out.stdout


def test_recent_report_for_this_trial_is_silent(good, home, proj):
    put_report(home, TRIAL, 2)
    out = guard(home, proj)
    assert out.exit_code == 0 and out.stdout == "" and out.stderr == ""
    assert hook_log(home)[-1]["decision"] == "allow"


def test_warning_is_also_shown_when_another_check_blocks(good, home, proj):
    (proj / "build" / "HASH").write_text("0" * 64)
    out = guard(home, proj)
    assert out.exit_code == 2 and "build hash" in out.stderr and "parity report missing" in out.stderr
