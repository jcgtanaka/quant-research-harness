"""Pre-harness attempts from packs, kept in the registry and marked ``imported``."""

from __future__ import annotations

import sqlite3
from typing import Any

from qrh.hashing import hash_json
from qrh.packs.model import Attempt


def sync_imported(conn: sqlite3.Connection, pack: str, pack_hash: str, attempts: list[Attempt]) -> int:
    """Make the registry rows of ``pack`` equal to ``attempts`` (idempotent). Returns the row count."""
    wanted = {hash_json(a.model_dump()): a for a in attempts}
    have = {r["attempt_hash"]: r["pack_hash"] for r in conn.execute(
        "SELECT attempt_hash, pack_hash FROM imported_attempts WHERE pack=?", (pack,))}
    if set(have) == set(wanted) and set(have.values()) <= {pack_hash}:
        return len(wanted)
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM imported_attempts WHERE pack=?", (pack,))
        for digest, a in wanted.items():
            conn.execute(
                "INSERT INTO imported_attempts(pack, pack_hash, attempt_hash, text, strategy_family, asset_class,"
                " frequency, family_id, variants_tried, verdict, attempt_date, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (pack, pack_hash, digest, a.text, a.strategy_family, a.asset_class, a.frequency, a.family_id(),
                 a.variants_tried, a.verdict, a.date, a.notes))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return len(wanted)


def remove_imported(conn: sqlite3.Connection, pack: str) -> None:
    conn.execute("DELETE FROM imported_attempts WHERE pack=?", (pack,))


def imported_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT * FROM imported_attempts ORDER BY id")]
