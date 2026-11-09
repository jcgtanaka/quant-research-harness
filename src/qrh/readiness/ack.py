"""The signed acknowledgement of a readiness report.

``qrh readiness ack <trial>`` (a person at a terminal, with the approval passphrase; see ``qrh.human``) signs, with
the approval key and the distinct domain ``readiness-ack`` (canonical prefix ``qrh/readiness-ack/v1``), a payload of
``{v, trial_id, report_hash, warning_ids (sorted), high_count, key_id, created_at}`` and stores it at
``<QRH_HOME>/readiness/<trial>/ack.json``. Everything here verifies with the PUBLIC key only; nothing in this module
can sign without an unlocked private key handed to ``sign_ack``.

An acknowledgement is valid only for the exact warning set it was given for: any change (a new or resolved warning,
a changed message, a changed verdict) changes the report hash, and the old acknowledgement stops verifying.
"""

from __future__ import annotations

import json
import os
import secrets
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qrh import signing
from qrh.config import qrh_home
from qrh.readiness import collect as rc
from qrh.readiness.data import ReportData, build_data

if TYPE_CHECKING:
    from qrh.memory.registry import Registry

DOMAIN = "readiness-ack"
ACK_FILE = "ack.json"
NO_ACK = "no acknowledgement has been signed for this trial"


class AckError(RuntimeError):
    """An acknowledgement cannot be created."""


def readiness_dir(home: str | Path) -> Path:
    return Path(home) / "readiness"


def ack_path(home: str | Path, trial_id: str) -> Path:
    return readiness_dir(home) / rc.check_trial_id(trial_id) / ACK_FILE


def _payload(doc: dict[str, Any]) -> dict[str, Any]:
    return {"v": 1, "trial_id": doc.get("trial_id"), "report_hash": doc.get("report_hash"),
            "warning_ids": doc.get("warning_ids"), "high_count": doc.get("high_count"),
            "key_id": doc.get("key_id"), "created_at": doc.get("created_at")}


def sign_ack(home: str | Path, data: ReportData, private_key: Any, now: datetime | None = None) -> dict[str, Any]:
    """Sign and store the acknowledgement of exactly this report. Raises AckError when there is no final verdict."""
    now = now or datetime.now(timezone.utc)
    if data.doc["final_verdict"] == rc.NOT_RUN:
        raise AckError("there is no final verdict (S7) for this trial yet: run `qrh run <trial>` first")
    doc: dict[str, Any] = {
        "trial_id": data.doc["trial_id"], "report_hash": data.hash, "warning_ids": data.warning_ids,
        "high_count": data.doc["totals"]["high"], "key_id": signing.key_id_of(private_key.public_key()),
        "created_at": now.isoformat(),
    }
    doc["signature"] = signing.sign(private_key, DOMAIN, _payload(doc))
    path = ack_path(home, doc["trial_id"])
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    for d in (path.parent, readiness_dir(home)):
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"v": 1, **doc}, sort_keys=True, indent=1) + "\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return doc


def _read(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None, NO_ACK
    except OSError as exc:
        return None, f"the acknowledgement cannot be read: {exc}"
    if not stat.S_ISREG(st.st_mode):
        return None, "the acknowledgement is not a regular file; it is ignored"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"the acknowledgement is unreadable: {exc}"
    return (doc, "") if isinstance(doc, dict) else (None, "the acknowledgement is malformed")


def evaluate(home: str | Path, data: ReportData) -> tuple[bool, str]:
    """Whether the stored acknowledgement is valid for exactly this report; never raises."""
    home = Path(home)
    try:
        doc, why = _read(ack_path(home, data.doc["trial_id"]))
        if doc is None:
            return False, why
        try:
            pub = signing.load_public(home)
        except signing.SigningError as exc:
            return False, f"the acknowledgement cannot be verified: {exc}"
        if pub is None:
            return False, "there is no approval key to verify the acknowledgement with (`qrh approve --init`)"
        if doc.get("key_id") != pub.key_id:
            return False, "the acknowledgement was signed by a different key (the approval key was replaced)"
        if not signing.verify(pub, DOMAIN, _payload(doc), doc.get("signature")):
            return False, "the acknowledgement signature does not verify (the file was altered or forged)"
        if doc.get("trial_id") != data.doc["trial_id"]:
            return False, "the acknowledgement belongs to another trial"
        ids, now_ids = doc.get("warning_ids"), data.warning_ids
        if doc.get("report_hash") != data.hash or ids != now_ids or doc.get("high_count") != data.doc["totals"]["high"]:
            new = sorted(set(now_ids) - set(ids if isinstance(ids, list) else []))
            gone = sorted(set(ids if isinstance(ids, list) else []) - set(now_ids))
            extra = (f" new: {', '.join(i[:8] for i in new[:5])};" if new else "") + \
                    (f" resolved: {', '.join(i[:8] for i in gone[:5])};" if gone else "")
            what = "the warnings changed since you acknowledged them" if new or gone else \
                "the report changed since you acknowledged it (same warnings, but the receipts, a message or the " \
                "verdict differ: the evidence changed)"
            return False, (f"{what} (report hash {str(doc.get('report_hash'))[:12]} -> {data.hash[:12]};"
                           f"{extra} {len(new)} new, {len(gone)} resolved): acknowledge again")
        return True, (f"acknowledged on {doc.get('created_at')} with key {pub.key_id}: report {data.hash[:12]}, "
                      f"{data.doc['totals']['high']} HIGH warning(s) accepted")
    except Exception as exc:  # fail closed: anything unexpected is "not acknowledged"
        return False, f"the acknowledgement could not be checked: {type(exc).__name__}: {exc}"


def is_acknowledged(trial_id: str, home: str | Path | None = None, registry: "Registry | None" = None) -> tuple[bool, str]:
    """``(acknowledged, reason)`` for the trial's current warning set. Recomputes the report; never raises."""
    try:
        home_dir = Path(home) if home is not None else qrh_home()
        return evaluate(home_dir, build_data(rc.collect(home_dir, trial_id, registry)))
    except Exception as exc:
        return False, f"readiness cannot be checked: {type(exc).__name__}: {exc}"
