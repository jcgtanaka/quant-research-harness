"""Config integrity: a signed baseline of ``config.yaml`` and the ``hooks`` section of Claude Code settings files.

``qrh config sign [--settings PATH ...]`` (a human at a terminal, with the approval passphrase) records the
SHA-256 of ``<QRH_HOME>/config.yaml`` and of the canonical JSON of each settings file's ``hooks`` section, signed
with the approval key. Enforcement hooks verify the signature with the public key and compare the current hashes;
any difference blocks until a human signs again. A missing baseline blocks too, once an approval key exists (deleting
the baseline must not disable the check); before any key exists, hooks run normally and session-context warns (first
use must stay possible). Hooks block only while a baseline exists and is intact, or a key makes its absence an alarm.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from qrh import signing

BASELINE_FILE = "config-baseline.json"
DOMAIN = "config-baseline"
ABSENT = "absent"
MESSAGE = "config or hook registration changed since it was signed; run qrh config sign"
MISSING_MESSAGE = "the signed config baseline is missing although an approval key exists; run qrh config sign"


@dataclass(frozen=True)
class Status:
    state: str  # missing | ok | changed | invalid
    detail: str = ""
    settings: list[str] = field(default_factory=list)

    def command(self) -> str:
        return "qrh config sign" + "".join(f" --settings {p}" for p in self.settings)


def baseline_path(home: str | Path) -> Path:
    return signing.keys_dir(home) / BASELINE_FILE


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def config_hash(home: str | Path) -> str:
    try:
        return _sha_bytes((Path(home) / "config.yaml").read_bytes())
    except FileNotFoundError:
        return ABSENT


def hooks_hash(path: str | Path) -> str:
    """SHA-256 of the canonical JSON of the file's ``hooks`` section; ``absent`` if the file does not exist."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ABSENT
    except (OSError, ValueError):
        return "unreadable"
    hooks = doc.get("hooks") if isinstance(doc, dict) else None
    return _sha_bytes(json.dumps(hooks, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii"))


def current(home: str | Path, settings: list[str]) -> dict[str, Any]:
    paths = [os.path.abspath(os.path.expanduser(p)) for p in settings]
    return {"config_sha256": config_hash(home),
            "settings": [{"path": p, "hooks_sha256": hooks_hash(p)} for p in paths]}


def _payload(doc: dict[str, Any]) -> dict[str, Any]:
    return {"v": 1, "config_sha256": doc.get("config_sha256"), "settings": doc.get("settings"),
            "signed_at": doc.get("signed_at"), "key_id": doc.get("key_id")}


def sign_baseline(home: str | Path, private_key: Any, settings: list[str], now: datetime) -> dict[str, Any]:
    doc = {**current(home, settings), "signed_at": now.isoformat(),
           "key_id": signing.key_id_of(private_key.public_key())}
    doc["signature"] = signing.sign(private_key, DOMAIN, _payload(doc))
    path = baseline_path(home)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return doc


def check(home: str | Path) -> Status:
    """Verify the baseline signature (public key only) and compare the current hashes. Never raises."""
    path = baseline_path(home)
    if not path.exists():
        return Status("missing", "no signed config baseline yet")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("not an object")
    except (OSError, ValueError) as exc:
        return Status("invalid", f"the config baseline is unreadable: {exc}")
    raw = doc.get("settings")
    if not isinstance(raw, list) or not all(isinstance(s, dict) and isinstance(s.get("path"), str) for s in raw):
        return Status("invalid", "the config baseline is malformed")
    paths = [s["path"] for s in raw]
    try:
        pub = signing.load_public(home)
    except signing.SigningError as exc:
        return Status("invalid", f"cannot verify the config baseline: {exc}", paths)
    if pub is None:
        return Status("invalid", "a config baseline exists but the approval public key is missing", paths)
    if doc.get("key_id") != pub.key_id or not signing.verify(pub, DOMAIN, _payload(doc), doc.get("signature")):
        return Status("invalid", "the config baseline signature does not verify (tampered, or signed by another key)",
                      paths)
    changed: list[str] = []
    if config_hash(home) != doc.get("config_sha256"):
        changed.append(str(Path(home) / "config.yaml"))
    for s in raw:
        if hooks_hash(s["path"]) != s.get("hooks_sha256"):
            changed.append(f"{s['path']} (hooks section)")
    if changed:
        return Status("changed", "changed since it was signed: " + ", ".join(changed), paths)
    return Status("ok", "", paths)
