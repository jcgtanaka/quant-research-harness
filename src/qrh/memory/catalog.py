"""Data catalog: registered datasets, their profiles and check status.

Fail closed: anything that cannot be read or profiled is BLOCKED, and an
unknown dataset is BLOCKED.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from qrh.hashing import sha256_file
from qrh.memory import db

STATUSES = ("UNCHECKED", "PASS", "BLOCKED")
PRIVACY = ("private", "public")


def _parquet_available() -> bool:
    return any(importlib.util.find_spec(m) is not None for m in ("pyarrow", "fastparquet"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        if len(df.columns) and pd.api.types.is_string_dtype(df.dtypes.iloc[0]):
            try:
                df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], utc=True)
            except (ValueError, TypeError):
                pass
        return df
    if suffix in (".parquet", ".pq"):
        if not _parquet_available():
            raise _Blocked("parquet support unavailable: install pyarrow")
        df = pd.read_parquet(path)
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        return df
    raise _Blocked(f"unsupported file type {suffix!r} (expected .csv or .parquet)")


class _Blocked(Exception):
    pass


def load_indexed_frame(path: str | Path) -> pd.DataFrame:
    """Read a dataset and index it by its first datetime column (kept timezone-aware)."""
    df = _read(Path(path))
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return df.set_index(col)
    raise _Blocked("no datetime column found to use as the index")


def _time_range(df: pd.DataFrame) -> tuple[str | None, str | None]:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) and df[col].notna().any():
            return df[col].min().isoformat(), df[col].max().isoformat()
    return None, None


class Catalog:
    def __init__(self, path: str | Path):
        self.home = Path(path).parent
        self.conn = db.connect(path)

    def close(self) -> None:
        self.conn.close()

    def register_dataset(
        self, path: str | Path, name: str | None = None, rules: str | Path | None = None,
        privacy: str | None = None,
    ) -> dict[str, Any]:
        if privacy is not None and privacy not in PRIVACY:
            raise ValueError(f"privacy must be one of {PRIVACY}, got {privacy!r}")
        p = Path(path)
        stored = str(p.resolve())  # absolute: the data-guard hook runs from another cwd and must find the same file
        rules_path = str(Path(rules)) if rules is not None else None
        name = name or p.stem
        reasons: list[str] = []
        content_hash: str | None = None
        schema: dict[str, str] = {}
        null_rates: dict[str, float] = {}
        start = end = None
        rows: int | None = None
        try:
            content_hash = sha256_file(p)
            df = _read(p)
            schema = {str(c): str(t) for c, t in df.dtypes.items()}
            null_rates = {str(c): float(df[c].isna().mean()) if len(df) else 0.0 for c in df.columns}
            start, end = _time_range(df)
            rows = int(len(df))
        except _Blocked as exc:
            reasons.append(str(exc))
        except Exception as exc:  # unreadable input must not pass
            reasons.append(f"{type(exc).__name__}: {exc}")
        status = "BLOCKED" if reasons else "UNCHECKED"

        existing = self.conn.execute(
            "SELECT content_hash, status, rules_path FROM datasets WHERE name=?", (name,)
        ).fetchone()
        if existing and status != "BLOCKED" and existing["content_hash"] == content_hash:
            # Unchanged content: refresh the path only, keep the recorded status
            # (new rules invalidate it, because they have not been run yet).
            self.conn.execute("UPDATE datasets SET path=? WHERE name=?", (stored, name))
            if rules_path is not None and rules_path != existing["rules_path"]:
                self.conn.execute(
                    "UPDATE datasets SET rules_path=?, status='UNCHECKED', reasons='[]' WHERE name=?",
                    (rules_path, name),
                )
        else:
            if rules_path is None and existing:
                rules_path = existing["rules_path"]
            self.conn.execute(
                "INSERT INTO datasets(name, path, content_hash, schema, start, end, row_count, null_rates,"
                " status, last_checked, reasons, rules_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(name) DO UPDATE SET path=excluded.path, content_hash=excluded.content_hash,"
                " schema=excluded.schema, start=excluded.start, end=excluded.end,"
                " row_count=excluded.row_count, null_rates=excluded.null_rates, status=excluded.status,"
                " last_checked=excluded.last_checked, reasons=excluded.reasons, rules_path=excluded.rules_path",
                (name, stored, content_hash, json.dumps(schema), start, end, rows,
                 json.dumps(null_rates), status, _now(), json.dumps(reasons), rules_path),
            )
        if privacy is not None:  # default private; only an explicit flag (synthetic data) makes a dataset public
            self.conn.execute("UPDATE datasets SET privacy=? WHERE name=?", (privacy, name))
        return self.get_dataset(name)  # type: ignore[return-value]

    def rules_for(self, name: str) -> Path | None:
        """Rules file for a dataset: the registered path, else ``<home>/rules/<name>.rules.yaml``."""
        rec = self.get_dataset(name)
        if rec is not None and rec.get("rules_path"):
            return Path(rec["rules_path"])
        candidate = self.home / "rules" / f"{name}.rules.yaml"
        return candidate if candidate.is_file() else None

    def get_dataset(self, name: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM datasets WHERE name=?", (name,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        for k in ("schema", "null_rates", "reasons"):
            d[k] = json.loads(d[k])
        return d

    def list_datasets(self) -> list[dict[str, Any]]:
        names = [r["name"] for r in self.conn.execute("SELECT name FROM datasets ORDER BY name")]
        return [self.get_dataset(n) for n in names]  # type: ignore[misc]

    def get_status(self, name: str) -> tuple[str, list[str]]:
        rec = self.get_dataset(name)
        if rec is None:
            return "BLOCKED", [f"unknown dataset {name!r}"]
        return rec["status"], rec["reasons"]

    def set_status(self, name: str, status: str, reasons: list[str]) -> None:
        if status not in STATUSES:
            raise ValueError(f"invalid status {status!r}")
        if self.get_dataset(name) is None:
            raise KeyError(name)
        self.conn.execute(
            "UPDATE datasets SET status=?, reasons=?, last_checked=? WHERE name=?",
            (status, json.dumps(reasons), _now(), name),
        )
