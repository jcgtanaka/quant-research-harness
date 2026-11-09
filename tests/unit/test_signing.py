"""Approval key pair: passphrase-encrypted Ed25519 private key, public-key-only verification, lockout, rotation."""

import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization

from qrh import signing
from qrh.signing import LockedOut, SigningError

T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
PASS = "a test passphrase, runtime only"  # gitleaks:allow
OTHER = "another test passphrase here"  # gitleaks:allow


def test_init_writes_an_encrypted_private_key_0600_and_a_public_key(tmp_path):
    pub = signing.init_keys(tmp_path, PASS, T0)
    enc = tmp_path / "keys" / "approval.key.enc"
    assert enc.is_file() and stat.S_IMODE(enc.stat().st_mode) == 0o600
    data = enc.read_bytes()
    with pytest.raises(TypeError):  # encrypted: unusable without a password
        serialization.load_pem_private_key(data, password=None)
    with pytest.raises(ValueError):  # and the raw passphrase is not the password either (a KDF is in front)
        serialization.load_pem_private_key(data, password=PASS.encode())
    assert PASS.encode() not in data
    doc = json.loads((tmp_path / "keys" / "approval.pub").read_text())
    assert doc["algorithm"] == "ed25519" and doc["key_id"] == pub.key_id and len(pub.key_id) == 16
    assert "PRIVATE" not in (tmp_path / "keys" / "approval.pub").read_text()
    assert signing.load_public(tmp_path).key_id == pub.key_id


def test_short_passphrase_is_refused(tmp_path):
    with pytest.raises(SigningError, match="at least 12"):
        signing.init_keys(tmp_path, "short", T0)
    assert not (tmp_path / "keys" / "approval.key.enc").exists()


def test_init_never_overwrites_an_existing_key(tmp_path):
    first = signing.init_keys(tmp_path, PASS, T0)
    with pytest.raises(SigningError, match="already exists"):
        signing.init_keys(tmp_path, OTHER, T0)
    assert signing.load_public(tmp_path).key_id == first.key_id


def test_sign_and_verify_with_the_public_key_only(tmp_path):
    pub = signing.init_keys(tmp_path, PASS, T0)
    priv = signing.unlock(tmp_path, PASS, T0)
    sig = signing.sign(priv, "test", {"a": 1, "b": "x"})
    assert signing.verify(pub, "test", {"b": "x", "a": 1}, sig)       # canonical: key order does not matter
    assert not signing.verify(pub, "test", {"a": 2, "b": "x"}, sig)   # any change breaks it
    assert not signing.verify(pub, "other-domain", {"a": 1, "b": "x"}, sig)
    assert not signing.verify(pub, "test", {"a": 1, "b": "x"}, "not base64 !!")


def test_wrong_passphrase_delays_exponentially_then_locks_out(tmp_path, monkeypatch):
    signing.init_keys(tmp_path, PASS, T0)
    delays = []
    monkeypatch.setattr(signing, "_sleep", delays.append)
    for i in range(5):
        with pytest.raises(SigningError, match="wrong passphrase"):
            signing.unlock(tmp_path, "wrong passphrase number x", T0 + timedelta(seconds=i))
    assert delays == [1, 2, 4, 8, 16]
    with pytest.raises(LockedOut, match="locked"):
        signing.unlock(tmp_path, PASS, T0 + timedelta(minutes=5))  # even the right one, inside the window
    signing.unlock(tmp_path, PASS, T0 + timedelta(minutes=11))      # the window has passed
    failures = tmp_path / "keys" / "unlock-failures.json"
    assert not failures.exists() or json.loads(failures.read_text())["failures"] == []


def test_old_failures_outside_the_window_do_not_count(tmp_path):
    signing.init_keys(tmp_path, PASS, T0)
    for _ in range(4):
        with pytest.raises(SigningError):
            signing.unlock(tmp_path, "wrong passphrase number x", T0)
    for _ in range(4):
        with pytest.raises(SigningError, match="wrong passphrase"):
            signing.unlock(tmp_path, "wrong passphrase number x", T0 + timedelta(minutes=20))
    signing.unlock(tmp_path, PASS, T0 + timedelta(minutes=20))


def test_unlock_without_a_key_explains_init(tmp_path):
    with pytest.raises(SigningError, match="qrh approve --init"):
        signing.unlock(tmp_path, PASS, T0)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_loose_private_key_permissions_are_refused(tmp_path):
    signing.init_keys(tmp_path, PASS, T0)
    os.chmod(tmp_path / "keys" / "approval.key.enc", 0o644)
    with pytest.raises(SigningError, match="permissions"):
        signing.unlock(tmp_path, PASS, T0)


def test_mismatched_key_files_are_refused(tmp_path):
    signing.init_keys(tmp_path, PASS, T0)
    other = tmp_path / "other"
    signing.init_keys(other, PASS, T0)
    (tmp_path / "keys" / "approval.pub").write_text((other / "keys" / "approval.pub").read_text())
    with pytest.raises(SigningError):
        signing.unlock(tmp_path, PASS, T0)


def test_rotation_needs_the_old_passphrase_and_changes_the_key_id(tmp_path):
    old = signing.init_keys(tmp_path, PASS, T0)
    with pytest.raises(SigningError, match="wrong passphrase"):
        signing.rotate_keys(tmp_path, "not the old passphrase", OTHER, T0)
    assert signing.load_public(tmp_path).key_id == old.key_id
    new = signing.rotate_keys(tmp_path, PASS, OTHER, T0)
    assert new.key_id != old.key_id and signing.load_public(tmp_path).key_id == new.key_id
    signing.unlock(tmp_path, OTHER, T0)
    with pytest.raises(SigningError):
        signing.unlock(tmp_path, PASS, T0)


def test_public_key_and_key_id_need_no_passphrase(tmp_path):
    assert signing.load_public(tmp_path) is None
    signing.init_keys(tmp_path, PASS, T0)
    assert signing.load_public(tmp_path) is not None


def test_no_environment_variable_reaches_the_secret_paths():
    """No env var, flag or switch may replace the terminal or the passphrase (shipped code)."""
    src = Path(signing.__file__).parent
    for name in ("signing.py", "tty.py", "human.py"):
        text = (src / name).read_text(encoding="utf-8")
        assert "environ" not in text and "getenv" not in text, name
