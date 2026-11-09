"""Readiness acknowledgement: signing domain, tamper, stale after change, wrong key, terminal and passphrase rules."""

import json
import os
import stat
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from readiness_helpers import TID, clean_run, make_registry, put_receipt
from qrh import human, signing, tty
from qrh.banner import BANNER
from qrh.cli import app
from qrh.pipeline.receipt import Verdict
from qrh.readiness import ack as rack
from qrh.readiness import collect as rc
from qrh.readiness import data as rd
from qrh.readiness import is_acknowledged

PASS = "a test passphrase, runtime only"  # gitleaks:allow
OTHER = "another test passphrase, runtime"  # gitleaks:allow
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    return h


@pytest.fixture
def reg(home):
    r = make_registry(home)
    clean_run(home, r)
    put_receipt(home, r, "S3", params={})
    r.log_override(TID, "S3", "accepted the thin sample knowingly", "alice")
    yield r
    r.close()


@pytest.fixture
def private(home):
    signing.init_keys(home, PASS, T0)
    return signing.unlock(home, PASS, T0)


@pytest.fixture
def terminal(monkeypatch):
    state = {"shown": [], "typed": []}
    monkeypatch.setattr(tty, "has_terminal", lambda: True)
    monkeypatch.setattr(tty, "write", lambda text: state["shown"].append(text))

    def read_secret(prompt):
        state["shown"].append(f"<{prompt}>")
        return state["typed"].pop(0)

    monkeypatch.setattr(tty, "read_secret", read_secret)
    return state


def data_of(home, reg):
    return rd.build_data(rc.collect(home, TID, reg))


def ack_file(home):
    return home / "readiness" / TID / "ack.json"


def edit(home, **changes):
    doc = json.loads(ack_file(home).read_text())
    doc.update(changes)
    ack_file(home).write_text(json.dumps(doc))


def signed(home, reg, private, now=T0):
    return rack.sign_ack(home, data_of(home, reg), private, now)


# --- signing --------------------------------------------------------------------------------------------

def test_the_payload_is_signed_under_its_own_domain_and_covers_the_whole_warning_set(home, reg, private):
    d = data_of(home, reg)
    doc = signed(home, reg, private)
    stored = json.loads(ack_file(home).read_text())
    assert stored["trial_id"] == TID and stored["report_hash"] == d.hash and stored["warning_ids"] == d.warning_ids
    assert stored["high_count"] == d.doc["totals"]["high"] and stored["key_id"] == signing.load_public(home).key_id
    assert stored["created_at"] == T0.isoformat() and stored["signature"] == doc["signature"]
    pub = signing.load_public(home)
    payload = {k: stored[k] for k in ("trial_id", "report_hash", "warning_ids", "high_count", "key_id", "created_at")}
    assert signing.verify(pub, "readiness-ack", {"v": 1, **payload}, stored["signature"])
    assert not signing.verify(pub, "approval", {"v": 1, **payload}, stored["signature"])   # domain separation
    assert not signing.verify(pub, "config-baseline", {"v": 1, **payload}, stored["signature"])
    assert signing.canonical("readiness-ack", {}).startswith(b"qrh/readiness-ack/v1\0")


def test_the_file_and_folder_are_private(home, reg, private):
    signed(home, reg, private)
    assert stat.S_IMODE(os.stat(ack_file(home)).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(ack_file(home).parent).st_mode) == 0o700


def test_a_valid_acknowledgement_is_accepted(home, reg, private):
    assert is_acknowledged(TID, home, reg)[0] is False
    signed(home, reg, private)
    ok, why = is_acknowledged(TID, home, reg)
    assert ok and "HIGH warning(s) accepted" in why


def test_without_an_acknowledgement_it_says_so(home, reg, private):
    ok, why = is_acknowledged(TID, home, reg)
    assert not ok and "no acknowledgement" in why


def test_nothing_to_acknowledge_without_a_final_verdict(home, private):
    r = make_registry(home, "t-nofinal")
    put_receipt(home, r, "S0", tid="t-nofinal")
    with pytest.raises(rack.AckError, match="no final verdict"):
        rack.sign_ack(home, rd.build_data(rc.collect(home, "t-nofinal", r)), private, T0)
    assert not (home / "readiness" / "t-nofinal").exists()
    r.close()


# --- tamper, staleness, keys ---------------------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [("high_count", 7), ("report_hash", "0" * 64), ("warning_ids", []),
                                         ("trial_id", "someone-else"), ("created_at", "2030-01-01T00:00:00+00:00")])
def test_editing_any_signed_field_invalidates_it(home, reg, private, field, value):
    signed(home, reg, private)
    edit(home, **{field: value})
    ok, why = is_acknowledged(TID, home, reg)
    assert not ok and "does not verify" in why


def test_a_dropped_signature_or_garbage_file_is_not_an_acknowledgement(home, reg, private):
    signed(home, reg, private)
    doc = json.loads(ack_file(home).read_text())
    doc.pop("signature")
    ack_file(home).write_text(json.dumps(doc))
    assert not is_acknowledged(TID, home, reg)[0]
    ack_file(home).write_text("not json at all")
    ok, why = is_acknowledged(TID, home, reg)
    assert not ok and "unreadable" in why
    ack_file(home).write_text("[1, 2]")
    assert not is_acknowledged(TID, home, reg)[0]


def test_a_symlinked_acknowledgement_is_ignored(home, reg, private, tmp_path):
    signed(home, reg, private)
    real = tmp_path / "elsewhere.json"
    real.write_text(ack_file(home).read_text())
    ack_file(home).unlink()
    ack_file(home).symlink_to(real)
    ok, why = is_acknowledged(TID, home, reg)
    assert not ok and "regular file" in why


def test_any_change_in_the_warning_set_makes_the_acknowledgement_stale(home, reg, private):
    signed(home, reg, private)
    assert is_acknowledged(TID, home, reg)[0]
    reg.log_override(TID, "S4", "a new knowing exception", "bob")
    ok, why = is_acknowledged(TID, home, reg)
    assert not ok and "warnings changed" in why and "1 new" in why and "acknowledge again" in why


def test_a_resolved_warning_also_invalidates_it(home, reg, private):
    signed(home, reg, private)
    reg.conn.execute("DELETE FROM overrides")
    ok, why = is_acknowledged(TID, home, reg)
    assert not ok and "1 resolved" in why


def test_a_changed_verdict_invalidates_it(home, reg, private):
    signed(home, reg, private)
    put_receipt(home, reg, "S7", verdict=Verdict.PASS_CONDITIONAL, params={"conditions": []})
    assert not is_acknowledged(TID, home, reg)[0]


def test_another_trials_acknowledgement_cannot_be_copied_over(home, reg, private):
    signed(home, reg, private)
    r2 = make_registry(home, "trial-two")
    for s in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"):
        put_receipt(home, r2, s, tid="trial-two")
    (home / "readiness" / "trial-two").mkdir(parents=True)
    (home / "readiness" / "trial-two" / "ack.json").write_text(ack_file(home).read_text())
    ok, why = is_acknowledged("trial-two", home, r2)
    assert not ok and "another trial" in why
    r2.close()


def test_a_different_key_cannot_acknowledge(home, reg, private, tmp_path):
    other = tmp_path / "other_home"
    signing.init_keys(other, OTHER, T0)
    forged = rack.sign_ack(other, data_of(home, reg), signing.unlock(other, OTHER, T0), T0)
    ack_file(home).parent.mkdir(parents=True, exist_ok=True)
    ack_file(home).write_text(json.dumps({"v": 1, **forged}))
    ok, why = is_acknowledged(TID, home, reg)
    assert not ok and "different key" in why
    edit(home, key_id=signing.load_public(home).key_id)          # claiming the real key id does not help
    ok, why = is_acknowledged(TID, home, reg)
    assert not ok and "does not verify" in why


def test_rotating_the_key_drops_every_acknowledgement(home, reg, private):
    signed(home, reg, private)
    signing.rotate_keys(home, PASS, OTHER, T0 + timedelta(days=1))
    ok, why = is_acknowledged(TID, home, reg)
    assert not ok and "different key" in why


def test_without_any_key_nothing_is_acknowledged(home, reg, private):
    signed(home, reg, private)
    signing.pub_path(home).unlink()
    ok, why = is_acknowledged(TID, home, reg)
    assert not ok and "approval key" in why


def test_is_acknowledged_never_raises(home):
    assert is_acknowledged("../../etc", home)[0] is False
    ok, why = is_acknowledged("unknown-trial", home)
    assert not ok and why


# --- the human step ---------------------------------------------------------------------------------------------

def test_the_human_sees_the_banner_verdict_counts_and_every_high_warning_before_the_passphrase(home, reg, private, terminal):
    reg.log_hook_override("data-gate", "dataset refresh is late, owner decided", "s1", trial_id=TID)
    put_receipt(home, reg, "S7", verdict=Verdict.PASS_CONDITIONAL, params={"conditions": [
        {"question_id": "rf", "scope": "dataset:prices", "stage": "S1", "text": "How was it made?",
         "status": "unknown", "risk": "Possible look-ahead in the input."}]})
    terminal["typed"] = [PASS]
    doc = human.acknowledge_readiness(home, TID, T0, registry=reg)
    shown = "".join(terminal["shown"])
    before = shown.split("<")[0]
    assert before.startswith(BANNER)
    assert "PASS · CONDITIONAL" in before and "HIGH" in before and "MEDIUM" in before and "LOW" in before
    d = data_of(home, reg)
    for w in d.doc["warnings"]:
        if w["severity"] == "high":
            assert w["id"][:8] in before and "rf" in before
    assert d.hash[:12] in before and "readiness.pdf" in before
    assert "passphrase" in shown.split("<")[1].lower()
    assert doc["report_hash"] == d.hash
    assert ack_file(home).exists()


def test_the_acknowledgement_is_for_the_report_the_human_was_shown(home, reg, private, terminal):
    terminal["typed"] = [PASS]
    doc = human.acknowledge_readiness(home, TID, T0, registry=reg)
    assert doc["report_hash"] == data_of(home, reg).hash and is_acknowledged(TID, home, reg)[0]


def test_no_terminal_means_no_acknowledgement(home, reg, private, monkeypatch):
    monkeypatch.setattr(tty, "has_terminal", lambda: False)
    for var in ("QRH_PASSPHRASE", "QRH_APPROVE_PASSPHRASE", "CI"):
        monkeypatch.setenv(var, PASS)
    with pytest.raises(human.HumanError, match="terminal"):
        human.acknowledge_readiness(home, TID, T0, registry=reg)
    assert not ack_file(home).exists()


def test_a_wrong_passphrase_acknowledges_nothing(home, reg, private, terminal):
    terminal["typed"] = ["definitely not the passphrase"]
    with pytest.raises(human.HumanError, match="wrong passphrase"):
        human.acknowledge_readiness(home, TID, T0, registry=reg)
    assert not ack_file(home).exists()


def test_without_a_key_or_a_final_verdict_it_refuses_before_asking(home, reg, terminal):
    with pytest.raises(human.HumanError, match="qrh approve --init"):
        human.acknowledge_readiness(home, TID, T0, registry=reg)
    signing.init_keys(home, PASS, T0)
    r = make_registry(home, "t-nofinal")
    with pytest.raises(human.HumanError, match="no final verdict"):
        human.acknowledge_readiness(home, "t-nofinal", T0, registry=r)
    assert not any(s.startswith("<") for s in terminal["shown"])      # no passphrase prompt was shown
    r.close()


def test_an_unknown_trial_is_refused_before_asking(home, reg, private, terminal):
    with pytest.raises(human.HumanError, match="unknown trial"):
        human.acknowledge_readiness(home, "nope", T0, registry=reg)
    assert not any(s.startswith("<") for s in terminal["shown"])


# --- the command ------------------------------------------------------------------------------------------------

def cli(*args):
    return CliRunner().invoke(app, ["readiness", *args])


def test_cli_status_shows_pending_then_acknowledged(home, reg, private, terminal):
    res = cli("status", TID)
    assert res.exit_code == 1 and "PENDING" in res.output and "no acknowledgement" in res.output
    terminal["typed"] = [PASS]
    ok = cli("ack", TID)
    assert ok.exit_code == 0, ok.output
    res = cli("status", TID)
    assert res.exit_code == 0 and "ACKNOWLEDGED" in res.output and "HIGH" in res.output


def test_cli_ack_refuses_without_a_terminal(home, reg, private, monkeypatch):
    monkeypatch.setattr(tty, "has_terminal", lambda: False)
    res = cli("ack", TID)
    assert res.exit_code == 2 and "terminal" in res.output and not ack_file(home).exists()


def test_cli_ack_takes_no_passphrase_option():
    res = cli("ack", TID, "--passphrase", "x")
    assert res.exit_code != 0
    assert "--passphrase" not in CliRunner().invoke(app, ["readiness", "ack", "--help"]).output


def test_cli_status_of_an_unknown_trial_is_blocked(home, reg):
    res = cli("status", "nope")
    assert res.exit_code == 2 and "unknown trial" in res.output


def test_cli_report_writes_the_files_and_prints_the_hash(home, reg, private):
    res = cli("report", TID)
    assert res.exit_code == 0, res.output
    d = data_of(home, reg)
    assert d.hash in res.output and "readiness.pdf" in res.output
    assert (home / "reports" / TID / "readiness.pdf").is_file()
    assert cli("report", "nope").exit_code == 2


def test_the_ack_code_reads_no_environment_variable():
    from pathlib import Path

    text = Path(rack.__file__).read_text(encoding="utf-8")
    assert "getenv" not in text and "environ" not in text
