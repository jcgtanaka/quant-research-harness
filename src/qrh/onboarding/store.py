"""Persistence of profiles and onboarding versions (SQLite, through the shared registry connection).

A profile is stored per dataset content hash and mode. An onboarding version belongs to one dataset name and one
content hash; a changed file or changed claims creates a new version, and an old version is never rewritten except
by its own later verification and confirmation (so a verdict recorded against it stays tied to what it saw).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from qrh.hashing import hash_json
from qrh.onboarding.model import CRITICAL, FIELDS, InstrumentModel, claim_text

STATES = ("NOT_ONBOARDED", "PROPOSED", "VERIFIED_WITH_GAPS", "ONBOARDED", "CONTRADICTED")
SPOT_CLASSES = ("equity", "etf", "fx_spot", "crypto_spot", "index", "rate", "macro")
ALWAYS_REQUIRED = ("asset_class", "price_columns", "price_units", "adjustments", "timestamp_semantics")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- profiles -------------------------------------------------------------------------------------------------

def save_profile(conn: sqlite3.Connection, content_hash: str, profile: dict[str, Any], digest: str) -> None:
    mode = "no_values" if profile.get("no_values") else "values"
    conn.execute(
        "INSERT INTO dataset_profiles(content_hash, mode, profile, profile_hash, created_at) VALUES (?,?,?,?,?)"
        " ON CONFLICT(content_hash, mode) DO UPDATE SET profile=excluded.profile,"
        " profile_hash=excluded.profile_hash, created_at=excluded.created_at",
        (content_hash, mode, json.dumps(profile, sort_keys=True), digest, _now()))


def get_profile(conn: sqlite3.Connection, content_hash: str | None, prefer: tuple[str, ...] = ("values", "no_values")
                ) -> tuple[dict[str, Any], str] | None:
    if not content_hash:
        return None
    for mode in prefer:
        row = conn.execute("SELECT profile, profile_hash FROM dataset_profiles WHERE content_hash=? AND mode=?",
                           (content_hash, mode)).fetchone()
        if row is not None:
            return json.loads(row["profile"]), row["profile_hash"]
    return None


# --- versions -------------------------------------------------------------------------------------------------

def _version(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {"name": row["name"], "version": row["version"], "content_hash": row["content_hash"],
            "model": InstrumentModel.model_validate(json.loads(row["model"])), "model_hash": row["model_hash"],
            "claims_hash": row["claims_hash"], "verified_at": row["verified_at"], "created_at": row["created_at"]}


def get_version(conn: sqlite3.Connection, name: str, version: int) -> dict[str, Any] | None:
    return _version(conn.execute("SELECT * FROM onboarding_versions WHERE name=? AND version=?", (name, version)).fetchone())


def latest_version(conn: sqlite3.Connection, name: str, content_hash: str | None = None) -> dict[str, Any] | None:
    """The newest version of a dataset, or of one content hash."""
    sql, args = "SELECT * FROM onboarding_versions WHERE name=?", [name]
    if content_hash is not None:
        sql, args = sql + " AND content_hash=?", args + [content_hash]
    return _version(conn.execute(sql + " ORDER BY version DESC LIMIT 1", args).fetchone())


def current_version(conn: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    """The newest version that was made for the dataset's current content, or None."""
    row = conn.execute("SELECT content_hash FROM datasets WHERE name=?", (name,)).fetchone()
    return latest_version(conn, name, row["content_hash"]) if row and row["content_hash"] else None


def add_version(conn: sqlite3.Connection, name: str, content_hash: str, model: InstrumentModel) -> int:
    last = conn.execute("SELECT MAX(version) FROM onboarding_versions WHERE name=?", (name,)).fetchone()[0] or 0
    conn.execute(
        "INSERT INTO onboarding_versions(name, version, content_hash, model, model_hash, claims_hash, verified_at,"
        " created_at) VALUES (?,?,?,?,?,?,NULL,?)",
        (name, last + 1, content_hash, json.dumps(model.to_doc(), sort_keys=True), model.model_hash(),
         model.claims_hash(), _now()))
    return last + 1


def update_version(conn: sqlite3.Connection, name: str, version: int, model: InstrumentModel,
                   verified: bool | None = None) -> None:
    """Record new statuses on a version (claims must be unchanged: a changed claim is a new version)."""
    row = conn.execute("SELECT claims_hash FROM onboarding_versions WHERE name=? AND version=?", (name, version)).fetchone()
    if row is None or row["claims_hash"] != model.claims_hash():
        raise ValueError("claims changed: add a new version instead")
    when = _now() if verified else None
    conn.execute(
        "UPDATE onboarding_versions SET model=?, model_hash=?, verified_at=COALESCE(?, verified_at) WHERE name=? AND version=?",
        (json.dumps(model.to_doc(), sort_keys=True), model.model_hash(), when, name, version))


# --- state ----------------------------------------------------------------------------------------------------

def required_fields(model: InstrumentModel) -> list[str]:
    """Fields every dataset must settle: five always, and the multiplier unless the asset class is a spot or index one."""
    out = list(ALWAYS_REQUIRED)
    cls = model.asset_class.value if model.asset_class is not None else None
    if cls not in SPOT_CLASSES:
        out.append("multiplier")
    return out


def field_rows(model: InstrumentModel) -> list[dict[str, Any]]:
    """One row per claimed field plus every required field that was not proposed (shown as unverified)."""
    claims = model.claims()
    names = [f for f in FIELDS if f in claims or f in required_fields(model)]
    rows = []
    for f in names:
        c = claims.get(f)
        rows.append({
            "field": f, "critical": f in CRITICAL,
            "claim": claim_text(f, c.value) if c else "(not proposed)",
            "source": c.source if c else "-", "status": c.status if c else "unverified",
            "evidence": c.evidence if c else "no value was proposed for this required field",
            "checked_by": c.checked_by if c else "",
        })
    return rows


def state_of(model: InstrumentModel | None, verified_at: str | None) -> str:
    if model is None:
        return "NOT_ONBOARDED"
    rows = field_rows(model)
    if any(r["status"] == "contradicted" for r in rows):
        return "CONTRADICTED"
    if verified_at is None:
        return "PROPOSED"
    gaps = [r for r in rows if r["critical"] and r["status"] not in ("verified", "confirmed")]
    return "VERIFIED_WITH_GAPS" if gaps else "ONBOARDED"


def contradictions(conn: sqlite3.Connection, name: str) -> list[str]:
    """Reasons (one per contradicted field) of the current onboarding version; empty when none or not onboarded."""
    v = current_version(conn, name)
    if v is None:
        return []
    return [f"onboarding: {r['field']} contradicted by the data ({r['evidence']})"
            for r in field_rows(v["model"]) if r["status"] == "contradicted"]


def dataset_state(conn: sqlite3.Connection, name: str) -> dict[str, Any]:
    """State and identity of the current onboarding of a dataset, in the form S1 records in its receipt."""
    row = conn.execute("SELECT content_hash FROM datasets WHERE name=?", (name,)).fetchone()
    v = current_version(conn, name)
    if v is None:
        return {"state": "NOT_ONBOARDED", "version": None, "content_hash": row["content_hash"] if row else None,
                "model_hash": None}
    return {"state": state_of(v["model"], v["verified_at"]), "version": v["version"], "content_hash": v["content_hash"],
            "model_hash": v["model_hash"]}


def claim_digest(model: InstrumentModel, field: str) -> str:
    c = model.claims().get(field)
    return hash_json([field, c.value, c.source])[:12] if c else "none"
