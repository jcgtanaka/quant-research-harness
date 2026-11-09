"""Dataset and trial lookups for hooks, using plain SQLite so no hook has to import pandas."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from qrh.hashing import sha256_file

MIN_PREFIX = 8


def dataset_row(conn: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM datasets WHERE name=?", (name,)).fetchone()
    return dict(row) if row else None


def dataset_problems(conn: sqlite3.Connection, name: str) -> list[str]:
    """Why a dataset may not be used (empty list = PASS and its file is unchanged)."""
    rec = dataset_row(conn, name)
    if rec is None:
        return ["not in the catalog"]
    problems: list[str] = []
    if rec["status"] != "PASS":
        try:
            reasons = "; ".join(json.loads(rec["reasons"]))
        except (ValueError, TypeError):
            reasons = ""
        problems.append(f"status {rec['status']}" + (f" ({reasons})" if reasons else ""))
        return problems
    try:
        actual = sha256_file(rec["path"])
    except OSError:
        return ["file unreadable"]
    if actual != rec["content_hash"]:
        problems.append("file changed since registration (hash mismatch)")
    return problems


def resolve_trial(conn: sqlite3.Connection, token: str) -> str | None:
    """Exact trial id, else the unique trial whose id starts with ``token`` (>= 8 characters)."""
    row = conn.execute("SELECT trial_id FROM hypotheses WHERE trial_id=?", (token,)).fetchone()
    if row:
        return str(row["trial_id"])
    if len(token) < MIN_PREFIX:
        return None
    esc = token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    rows = conn.execute(
        "SELECT trial_id FROM hypotheses WHERE trial_id LIKE ? ESCAPE '\\' LIMIT 2", (esc + "%",)
    ).fetchall()
    return str(rows[0]["trial_id"]) if len(rows) == 1 else None


def trial_datasets(conn: sqlite3.Connection, trial_id: str) -> list[str]:
    row = conn.execute("SELECT binding FROM hypotheses WHERE trial_id=?", (trial_id,)).fetchone()
    if row is None:
        return []
    try:
        return [d["name"] for d in json.loads(row["binding"]).get("datasets", [])]
    except (ValueError, TypeError, KeyError):
        raise RuntimeError(f"trial {trial_id} has an unreadable dataset binding") from None
