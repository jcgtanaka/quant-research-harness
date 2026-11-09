"""The approval key pair: an Ed25519 private key that only a passphrase unlocks, and a public key for checks.

Layout under ``<QRH_HOME>/keys``::

    approval.key.enc        PKCS8 private key, encrypted (BestAvailableEncryption) with a password derived from
                            the human's passphrase by scrypt; mode 0600
    approval.pub            JSON: public key, key id, scrypt parameters and salt (nothing secret)
    unlock-failures.json    timestamps of wrong passphrases (delay and lockout)

Hooks and every verification path read only ``approval.pub``. Signing needs the passphrase, which is read from
the controlling terminal by ``qrh.human``; this module never reads it from anywhere else.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import secrets
import stat
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

KEY_FILE = "approval.key.enc"
PUB_FILE = "approval.pub"
LEGACY_KEY_FILE = "approval.key"  # the removed HMAC key; still protected, never used
FAILURES_FILE = "unlock-failures.json"
MIN_PASSPHRASE = 12
KDF_N = 2 ** 17  # scrypt cost; tests lower it through this module attribute (a code-level seam, not a switch)
KDF_R = 8
KDF_P = 1
MAX_FAILURES = 5
FAILURE_WINDOW = timedelta(minutes=10)
MAX_DELAY_S = 30

_sleep = time.sleep  # seam for tests


class SigningError(RuntimeError):
    """The key pair cannot be created, unlocked or used (the message says why)."""


class LockedOut(SigningError):
    """Too many wrong passphrases in the window."""


@dataclass(frozen=True)
class PublicKey:
    key: Ed25519PublicKey
    key_id: str
    created_at: str = ""


def keys_dir(home: str | Path) -> Path:
    return Path(home) / "keys"


def key_path(home: str | Path) -> Path:
    return keys_dir(home) / KEY_FILE


def pub_path(home: str | Path) -> Path:
    return keys_dir(home) / PUB_FILE


def _raw(pub: Ed25519PublicKey) -> bytes:
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def key_id_of(pub: Ed25519PublicKey) -> str:
    return hashlib.sha256(_raw(pub)).hexdigest()[:16]


def _chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _ensure_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    _chmod(path, 0o700)


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _chmod(path, 0o600)


# --- public key -----------------------------------------------------------------------------------

def _pub_doc(home: str | Path) -> dict[str, Any] | None:
    path = pub_path(home)
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SigningError(f"cannot read the approval public key: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise SigningError("the approval public key is not a regular file; refusing to use it")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SigningError(f"the approval public key is unreadable: {exc}") from exc
    if not isinstance(doc, dict):
        raise SigningError("the approval public key file is malformed")
    return doc


def load_public(home: str | Path) -> PublicKey | None:
    """The public key, or None if no key pair exists. Raises SigningError if the file is malformed."""
    doc = _pub_doc(home)
    if doc is None:
        return None
    try:
        raw = base64.b64decode(str(doc["public_key"]), validate=True)
        key = Ed25519PublicKey.from_public_bytes(raw)
    except (KeyError, ValueError, binascii.Error) as exc:
        raise SigningError(f"the approval public key is malformed: {exc}") from exc
    kid = key_id_of(key)
    if doc.get("key_id") != kid:
        raise SigningError("the approval public key file does not match its key id")
    return PublicKey(key, kid, str(doc.get("created_at", "")))


def has_keypair(home: str | Path) -> bool:
    return key_path(home).exists() or pub_path(home).exists()


# --- creation -------------------------------------------------------------------------------------

def _derive(passphrase: str, kdf: dict[str, Any]) -> bytes:
    try:
        n, r, p, salt = int(kdf["n"]), int(kdf["r"]), int(kdf["p"]), bytes.fromhex(str(kdf["salt"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise SigningError(f"the key derivation parameters are malformed: {exc}") from exc
    mem = 128 * n * r * p + (1 << 20)
    out = hashlib.scrypt(passphrase.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=2 * mem, dklen=32)
    return out.hex().encode("ascii")


def _check_passphrase(passphrase: str) -> None:
    if len(passphrase or "") < MIN_PASSPHRASE:
        raise SigningError(f"the passphrase must be at least {MIN_PASSPHRASE} characters")


def _create(home: Path, passphrase: str, now: datetime) -> PublicKey:
    _check_passphrase(passphrase)
    _ensure_dir(keys_dir(home))
    private = Ed25519PrivateKey.generate()
    pub = private.public_key()
    kdf = {"name": "scrypt", "n": KDF_N, "r": KDF_R, "p": KDF_P, "salt": secrets.token_hex(16)}
    blob = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                 serialization.BestAvailableEncryption(_derive(passphrase, kdf)))
    kid = key_id_of(pub)
    doc = {"version": 1, "algorithm": "ed25519", "public_key": base64.b64encode(_raw(pub)).decode("ascii"),
           "key_id": kid, "kdf": kdf, "created_at": now.isoformat()}
    _write_atomic(key_path(home), blob)
    _write_atomic(pub_path(home), (json.dumps(doc, sort_keys=True, indent=1) + "\n").encode("utf-8"))
    _chmod(pub_path(home), 0o644)
    (keys_dir(home) / FAILURES_FILE).unlink(missing_ok=True)
    return PublicKey(pub, kid, doc["created_at"])


def init_keys(home: str | Path, passphrase: str, now: datetime) -> PublicKey:
    """Create the key pair. Never overwrites an existing one (rotation needs the old passphrase)."""
    home = Path(home)
    if has_keypair(home):
        raise SigningError("an approval key already exists; use `qrh approve --init --rotate` (needs the old "
                           "passphrase), or see docs/security.md if the passphrase is lost")
    _check_passphrase(passphrase)
    return _create(home, passphrase, now)


def rotate_keys(home: str | Path, old_passphrase: str, new_passphrase: str, now: datetime) -> PublicKey:
    """Replace the key pair after proving the old passphrase. Old signatures stop verifying (new key id)."""
    home = Path(home)
    _check_passphrase(new_passphrase)
    unlock(home, old_passphrase, now)
    return _create(home, new_passphrase, now)


# --- unlock with delay and lockout ----------------------------------------------------------------

def _failures(home: Path) -> list[datetime]:
    try:
        doc = json.loads((keys_dir(home) / FAILURES_FILE).read_text(encoding="utf-8"))
        items = doc.get("failures", []) if isinstance(doc, dict) else []
    except (OSError, ValueError):
        return []
    out = []
    for x in items:
        try:
            out.append(datetime.fromisoformat(str(x)))
        except ValueError:
            continue
    return out


def _save_failures(home: Path, items: list[datetime]) -> None:
    try:
        _write_atomic(keys_dir(home) / FAILURES_FILE,
                      json.dumps({"failures": [x.isoformat() for x in items]}).encode("utf-8"))
    except OSError:
        pass


def _check_private_file(path: Path) -> bytes:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise SigningError("no approval key: create one in your own terminal with `qrh approve --init`") from None
    except OSError as exc:
        raise SigningError(f"cannot read the approval key: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise SigningError("the approval key is not a regular file; refusing to use it")
    if os.name == "posix" and stat.S_IMODE(st.st_mode) & 0o077:
        raise SigningError(f"approval key permissions are too loose ({oct(stat.S_IMODE(st.st_mode))}); "
                           "run `chmod 600` on it")
    return path.read_bytes()


def unlock(home: str | Path, passphrase: str, now: datetime) -> Ed25519PrivateKey:
    """Decrypt the private key. A wrong passphrase waits 1, 2, 4, ... seconds; 5 failures in 10 minutes lock out."""
    home = Path(home)
    blob = _check_private_file(key_path(home))
    doc = _pub_doc(home)
    if doc is None:
        raise SigningError("the approval public key is missing; see docs/security.md")
    recent = [t for t in _failures(home) if now - t < FAILURE_WINDOW]
    if len(recent) >= MAX_FAILURES:
        until = min(recent) + FAILURE_WINDOW
        raise LockedOut(f"too many wrong passphrases: locked until {until.isoformat()}")
    try:
        private = serialization.load_pem_private_key(blob, password=_derive(passphrase, doc.get("kdf") or {}))
    except (ValueError, TypeError):
        recent.append(now)
        _save_failures(home, recent)
        _sleep(min(MAX_DELAY_S, 2 ** (len(recent) - 1)))
        raise SigningError("wrong passphrase") from None
    if not isinstance(private, Ed25519PrivateKey):
        raise SigningError("the approval key is not an Ed25519 key")
    pub = load_public(home)
    if pub is None or key_id_of(private.public_key()) != pub.key_id:
        raise SigningError("the private and public approval keys do not belong together")
    (keys_dir(home) / FAILURES_FILE).unlink(missing_ok=True)
    return private


# --- signatures -----------------------------------------------------------------------------------

def canonical(domain: str, payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"qrh/{domain}/v1\0".encode("ascii") + body.encode("ascii")


def sign(private: Ed25519PrivateKey, domain: str, payload: dict[str, Any]) -> str:
    return base64.b64encode(private.sign(canonical(domain, payload))).decode("ascii")


def verify(pub: PublicKey, domain: str, payload: dict[str, Any], signature: Any) -> bool:
    if not isinstance(signature, str):
        return False
    try:
        raw = base64.b64decode(signature, validate=True)
        pub.key.verify(raw, canonical(domain, payload))
    except (InvalidSignature, ValueError, binascii.Error):
        return False
    return True
