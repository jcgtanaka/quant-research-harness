"""pre-restart-guard: `guard.require_readiness: <trial>` blocks an engine start until the readiness report is acknowledged."""

from datetime import datetime, timezone

import pytest

from hook_helpers import DB, TEST_PASSPHRASE, bash_payload, ensure_test_keys, human_approve, run, write_config
from readiness_helpers import TID, clean_run, make_registry, put_receipt
from qrh import integrity, signing
from qrh.banner import BANNER
from qrh.hooks.pre_restart_guard import compute_source_hash
from qrh.readiness import ack as rack
from qrh.readiness import collect as rc
from qrh.readiness import data as rd

NOW = datetime(2020, 1, 11, 12, 0, tzinfo=timezone.utc)
BASE = """
engine_start_commands: ['engine\\s+start']
guard:
  source_globs: ["src/**/*.py"]
  build_hash_file: build/HASH
  test_receipt_max_age: 1D
  parity: {disabled: "parity tooling is not built yet"}
"""


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


def setup(home, proj, readiness=f"  require_readiness: {TID}\n"):
    """All the ordinary pre-restart checks satisfied; a finished synthetic trial; the config baseline signed."""
    write_config(home, BASE + readiness)
    reg = make_registry(home, datasets=())
    clean_run(home, reg)
    h = compute_source_hash(["src/**/*.py"], proj)
    (proj / "build" / "HASH").write_text(h + "\n")
    reg.record_test_receipt(h, "pytest -q", 0, 1.0)
    reg.conn.execute("UPDATE test_receipts SET created_at=?", (NOW.isoformat(),))
    ensure_test_keys(home)
    integrity.sign_baseline(home, signing.unlock(home, TEST_PASSPHRASE, NOW), [], NOW)
    return reg


def acknowledge(home, reg):
    data = rd.build_data(rc.collect(home, TID, reg))
    rack.sign_ack(home, data, signing.unlock(home, TEST_PASSPHRASE, NOW), NOW)


def guard(home, proj, command="engine start --paper"):
    data = bash_payload(command)
    data["cwd"] = str(proj)
    return run("pre-restart-guard", data, home, now=lambda: NOW)


def test_an_unacknowledged_trial_blocks_with_the_banner_and_the_way_to_acknowledge(home, proj):
    setup(home, proj).close()
    out = guard(home, proj)
    assert out.exit_code == 2 and out.stderr.startswith(BANNER)
    assert "readiness" in out.stderr and "qrh readiness ack " + TID in out.stderr
    assert "no acknowledgement" in out.stderr


def test_an_acknowledged_trial_passes(home, proj):
    reg = setup(home, proj)
    acknowledge(home, reg)
    reg.close()
    out = guard(home, proj)
    assert out.exit_code == 0, out.stderr


def test_any_change_in_the_warnings_blocks_again(home, proj):
    reg = setup(home, proj)
    acknowledge(home, reg)
    assert guard(home, proj).exit_code == 0
    reg.log_override(TID, "S3", "a new knowing exception after the acknowledgement", "alice")
    reg.close()
    out = guard(home, proj)
    assert out.exit_code == 2 and "warnings changed" in out.stderr


def test_other_failed_checks_are_still_listed_with_the_readiness_one(home, proj):
    setup(home, proj).close()
    (proj / "build" / "HASH").write_text("0" * 64)
    out = guard(home, proj)
    assert out.exit_code == 2 and "build hash" in out.stderr and "readiness" in out.stderr


def test_the_block_is_approvable_only_through_the_human_flow(home, proj):
    setup(home, proj).close()
    first = guard(home, proj)
    assert first.exit_code == 2 and "qrh approve" in first.stderr
    human_approve(home, first.stderr, now=NOW)
    assert guard(home, proj).exit_code == 0               # one approved retry
    assert guard(home, proj).exit_code == 2               # single use


def test_an_unknown_trial_blocks(home, proj):
    reg = setup(home, proj, readiness="  require_readiness: no-such-trial\n")
    reg.close()
    out = guard(home, proj)
    assert out.exit_code == 2 and "unknown trial" in out.stderr


def test_an_absent_key_means_the_check_is_not_applied(home, proj):
    reg = setup(home, proj, readiness="")
    reg.close()
    out = guard(home, proj)
    assert out.exit_code == 0, out.stderr


@pytest.mark.parametrize("value", ["5", "'../x'", "''", "[a]", "'a b'"])
def test_an_invalid_trial_id_in_config_blocks_fail_closed(home, proj, value):
    reg = setup(home, proj, readiness=f"  require_readiness: {value}\n")
    reg.close()
    out = guard(home, proj)
    assert out.exit_code == 2 and "require_readiness" in out.stderr


def test_session_context_says_nothing_new_with_or_without_the_key(home, proj):
    reg = setup(home, proj, readiness="")
    reg.close()
    from hook_helpers import payload

    without = run("session-context", payload("SessionStart", source="startup"), home).stdout
    write_config(home, BASE + f"  require_readiness: {TID}\n")
    integrity.sign_baseline(home, signing.unlock(home, TEST_PASSPHRASE, NOW), [], NOW)
    with_key = run("session-context", payload("SessionStart", source="startup"), home).stdout
    assert with_key == without and "readiness" not in with_key.lower()
