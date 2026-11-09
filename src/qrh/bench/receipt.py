"""Bench receipts: hash-addressed records of what a benchmark ran, with which methods, and what it found.

Stored as ``<QRH_HOME>/bench/<bench_id>/receipt.json`` (``bench_id`` = ``bench-`` + the first 12 characters of the
receipt hash). Like stage receipts the hash covers everything except ``created_at``, so identical inputs give
the same id, and loading re-verifies it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from qrh.hashing import hash_json

SCHEMA_VERSION = 1
BENCH_DIR = "bench"


class BenchReceiptError(ValueError):
    """A bench receipt is missing, ambiguous, unreadable or does not match its recorded hash."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BenchReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    profile: str
    suites: list[str]
    seed: int
    config: dict[str, Any]
    methods: dict[str, str]  # every method id -> the version the candidate arm used
    overrides: dict[str, str] = Field(default_factory=dict)
    non_default_methods: dict[str, str] = Field(default_factory=dict)
    baseline_methods: dict[str, str] | None = None
    results: dict[str, Any]
    baseline_results: dict[str, Any] | None = None
    hashes: dict[str, str] = Field(default_factory=dict)
    code_hash: str
    environment: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)
    # Written by new runs only (``qrh.bench.verdict``). Receipts made before these fields existed have neither key,
    # and the keys are left out of the hashed and stored form while None, so their hash is unchanged.
    verdict: str | None = None
    suite_verdicts: dict[str, str] | None = None

    def _dump(self, **kw: Any) -> dict[str, Any]:
        absent = {k for k in ("verdict", "suite_verdicts") if getattr(self, k) is None}
        return self.model_dump(mode="json", exclude=absent | set(kw.pop("exclude", set())), **kw)

    def receipt_hash(self) -> str:
        return hash_json(self._dump(exclude={"created_at"}))

    def bench_id(self) -> str:
        return f"bench-{self.receipt_hash()[:12]}"


def bench_root(home: str | Path) -> Path:
    return Path(home) / BENCH_DIR


def write_bench(receipt: BenchReceipt, home: str | Path) -> Path:
    """Write the receipt and return its folder; the reports are written beside it by ``qrh.bench.render``."""
    digest = receipt.receipt_hash()  # raises on NaN or inf before anything is written
    folder = bench_root(home) / receipt.bench_id()
    folder.mkdir(parents=True, exist_ok=True)
    doc = {"receipt_hash": digest, "receipt": receipt._dump()}
    tmp = folder / f"receipt.json.{os.getpid()}.tmp"
    try:
        tmp.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, folder / "receipt.json")
    finally:
        tmp.unlink(missing_ok=True)
    return folder


def load_bench_receipt(path: str | Path) -> BenchReceipt:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        receipt = BenchReceipt.model_validate(doc["receipt"])
        stored, actual = doc["receipt_hash"], receipt.receipt_hash()
    except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
        raise BenchReceiptError(f"bench receipt {str(path)!r} is unreadable or invalid: {exc}") from exc
    if stored != actual:
        raise BenchReceiptError(f"bench receipt {str(path)!r} does not match its recorded hash")
    return receipt


def list_receipts(home: str | Path) -> list[tuple[str, Path]]:
    root = bench_root(home)
    if not root.is_dir():
        return []
    return sorted((p.name, p / "receipt.json") for p in root.iterdir() if (p / "receipt.json").is_file())


def resolve_receipt(home: str | Path, ref: str) -> BenchReceipt:
    """A receipt by bench id, unique id prefix, or path to its ``receipt.json`` (or its folder)."""
    path = Path(ref)
    if path.is_dir():
        path = path / "receipt.json"
    if path.is_file():
        return load_bench_receipt(path)
    known = list_receipts(home)
    exact = [p for i, p in known if i == ref]
    matches = exact or [p for i, p in known if i.startswith(ref)]
    if not matches:
        raise BenchReceiptError(f"no bench receipt matches {ref!r}")
    if len(matches) > 1:
        raise BenchReceiptError(f"{ref!r} matches {len(matches)} bench receipts; give more characters")
    return load_bench_receipt(matches[0])
