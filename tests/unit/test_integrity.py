"""Config integrity: a signed baseline of config.yaml and the hooks section of Claude Code settings files."""

import json
from datetime import datetime, timezone

import pytest

from hook_helpers import TEST_PASSPHRASE, bash_payload, ensure_test_keys, payload, run, write_config
from qrh import integrity, signing
from qrh.banner import BANNER
from qrh.hooks.runner import HookResult, HookSpec

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
MESSAGE = "config or hook registration changed since it was signed; run qrh config sign"


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    write_config(h, "detect_strategies: true\n")
    ensure_test_keys(h)
    return h


@pytest.fixture
def settings(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"theme": "dark", "hooks": {"Stop": [{"hooks": [{"type": "command",
                                                                              "command": "qrh hook claim-check"}]}]}}))
    return p


def sign(home, paths):
    key = signing.unlock(home, TEST_PASSPHRASE, NOW)
    return integrity.sign_baseline(home, key, [str(p) for p in paths], NOW)


def test_no_baseline_is_missing_not_a_failure(home):
    assert integrity.check(home).state == "missing"


def test_signed_baseline_verifies_until_config_or_hooks_change(home, settings):
    sign(home, [settings])
    assert integrity.check(home).state == "ok"
    doc = json.loads(settings.read_text())
    doc["theme"] = "light"                        # a non-hook setting: not covered
    settings.write_text(json.dumps(doc))
    assert integrity.check(home).state == "ok"
    doc["hooks"] = {}                             # hooks removed: covered
    settings.write_text(json.dumps(doc))
    st = integrity.check(home)
    assert st.state == "changed" and str(settings) in st.detail and str(settings) in st.settings


def test_config_change_is_detected(home, settings):
    sign(home, [settings])
    write_config(home, "detect_strategies: false\n")
    st = integrity.check(home)
    assert st.state == "changed" and "config.yaml" in st.detail


def test_deleted_settings_file_is_a_change(home, settings):
    sign(home, [settings])
    settings.unlink()
    assert integrity.check(home).state == "changed"


@pytest.mark.parametrize("field", ["config_sha256", "signature", "key_id"])
def test_tampered_baseline_is_invalid(home, settings, field):
    sign(home, [settings])
    path = home / "keys" / "config-baseline.json"
    doc = json.loads(path.read_text())
    doc[field] = "0" * 16 if field == "key_id" else ("A" * 88 if field == "signature" else "f" * 64)
    path.write_text(json.dumps(doc))
    assert integrity.check(home).state == "invalid"


def test_rotated_key_invalidates_the_baseline(home, settings):
    sign(home, [settings])
    signing.rotate_keys(home, TEST_PASSPHRASE, "another runtime passphrase", NOW)
    assert integrity.check(home).state == "invalid"


def test_missing_public_key_with_a_baseline_is_invalid(home, settings):
    sign(home, [settings])
    (home / "keys" / "approval.pub").unlink()
    assert integrity.check(home).state == "invalid"


def _stub(enforcement=True, needs_config=True):
    return {"stub": HookSpec("stub", "PreToolUse", "Bash", 5, enforcement, "unused",
                             run=lambda ctx: HookResult("allow", "fine"), needs_config=needs_config)}


def test_enforcement_hooks_block_on_a_changed_baseline_with_the_banner(home, settings):
    sign(home, [settings])
    write_config(home, "detect_strategies: false\n")
    out = run("stub", bash_payload("ls"), home, hooks=_stub())
    assert out.exit_code == 2 and out.stderr.startswith(BANNER) and MESSAGE in out.stderr
    assert f"qrh config sign --settings {settings}" in out.stderr and "qrh approve" not in out.stderr
    assert not (home / "approvals").exists()                                   # not approvable
    assert run("stub", bash_payload("ls"), home, hooks=_stub(enforcement=False)).exit_code == 0
    assert run("stub", bash_payload("ls"), home, hooks=_stub(needs_config=False)).exit_code == 0


def test_enforcement_hooks_run_normally_with_a_valid_baseline(home, settings):
    sign(home, [settings])
    assert run("stub", bash_payload("ls"), home, hooks=_stub()).exit_code == 0


def test_without_any_key_a_missing_baseline_only_warns(tmp_path):
    bare = tmp_path / "bare"
    write_config(bare, "detect_strategies: true\n")
    assert run("stub", bash_payload("ls"), bare, hooks=_stub()).exit_code == 0


def test_once_a_key_exists_a_missing_baseline_blocks_enforcement_hooks_with_the_banner(home):
    out = run("stub", bash_payload("ls"), home, hooks=_stub())
    assert out.exit_code == 2 and out.stderr.startswith(BANNER)
    assert "no signed config baseline" in out.stderr and "qrh config sign" in out.stderr
    assert "qrh approve" not in out.stderr and not (home / "approvals").exists()   # a human signs; not approvable
    assert run("stub", bash_payload("ls"), home, hooks=_stub(enforcement=False)).exit_code == 0
    assert run("stub", bash_payload("ls"), home, hooks=_stub(needs_config=False)).exit_code == 0


def test_deleting_a_signed_baseline_blocks_until_it_is_signed_again(home, settings):
    sign(home, [settings])
    assert run("stub", bash_payload("ls"), home, hooks=_stub()).exit_code == 0
    integrity.baseline_path(home).unlink()
    out = run("stub", bash_payload("ls"), home, hooks=_stub())
    assert out.exit_code == 2 and "qrh config sign" in out.stderr
    sign(home, [settings])
    assert run("stub", bash_payload("ls"), home, hooks=_stub()).exit_code == 0


def test_a_repeated_stop_is_not_exempt_from_a_missing_baseline(home):
    hooks = {"stub": HookSpec("stub", "Stop", None, 5, True, "unused", run=lambda ctx: HookResult("allow", "x"))}
    assert run("stub", payload("Stop", stop_hook_active=True), home, hooks=hooks).exit_code == 2


def test_a_repeated_stop_is_not_exempt_from_a_changed_baseline(home, settings):
    sign(home, [settings])
    write_config(home, "detect_strategies: false\n")
    hooks = {"stub": HookSpec("stub", "Stop", None, 5, True, "unused", run=lambda ctx: HookResult("allow", "x"))}
    assert run("stub", payload("Stop", stop_hook_active=True), home, hooks=hooks).exit_code == 2


# --- the human command ----------------------------------------------------------------------------

def test_cli_config_sign_needs_a_terminal_and_the_passphrase(home, settings, monkeypatch):
    from typer.testing import CliRunner

    from qrh import tty
    from qrh.cli import app

    monkeypatch.setenv("QRH_HOME", str(home))
    monkeypatch.setattr(tty, "has_terminal", lambda: False)
    res = CliRunner().invoke(app, ["config", "sign", "--settings", str(settings)])
    assert res.exit_code == 2 and "terminal" in res.output and integrity.check(home).state == "missing"

    shown, typed = [], ["wrong passphrase, twelve+"]
    monkeypatch.setattr(tty, "has_terminal", lambda: True)
    monkeypatch.setattr(tty, "write", shown.append)
    monkeypatch.setattr(tty, "read_secret", lambda prompt: (shown.append("<prompt>"), typed.pop(0))[1])
    res = CliRunner().invoke(app, ["config", "sign", "--settings", str(settings)])
    assert res.exit_code == 2 and "wrong passphrase" in res.output and integrity.check(home).state == "missing"

    typed.append(TEST_PASSPHRASE)
    res = CliRunner().invoke(app, ["config", "sign", "--settings", str(settings)])
    assert res.exit_code == 0, res.output
    before_prompt = "".join(shown).split("<prompt>")[-2]
    assert "config.yaml" in before_prompt and str(settings) in before_prompt
    assert integrity.check(home).state == "ok"
