"""Receipts: immutable, hash-addressed records of what a stage decided and why."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from qrh.hashing import hash_json

if TYPE_CHECKING:
    from qrh.memory.registry import Registry

SCHEMA_VERSION = 1
KEY_LIBRARIES = ("pandas", "numpy", "pydantic")


class Stage(str, Enum):
    S0 = "S0"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    S5 = "S5"
    S6 = "S6"
    S7 = "S7"


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    INCONCLUSIVE = "INCONCLUSIVE"
    PASS_CONDITIONAL = "PASS_CONDITIONAL"  # S7 only: every gate passed, but open conditions remain (see interviews)


def display_verdict(verdict: "Verdict | str") -> str:
    """How a verdict is shown to people: PASS_CONDITIONAL reads as ``PASS · CONDITIONAL`` everywhere."""
    value = verdict.value if isinstance(verdict, Verdict) else str(verdict)
    return value.replace("PASS_CONDITIONAL", "PASS · CONDITIONAL")


def is_pass(verdict: "Verdict | str") -> bool:
    """PASS or PASS_CONDITIONAL: every gate passed (a conditional one still carries open conditions)."""
    value = verdict.value if isinstance(verdict, Verdict) else str(verdict)
    return value in ("PASS", "PASS_CONDITIONAL")


class ReceiptTamperedError(ValueError):
    """A stored receipt is unreadable or does not match its recorded hash."""


class Override(BaseModel):
    """An explicit, attributed exception. Never silent."""

    model_config = ConfigDict(extra="forbid")

    reason: str
    who: str

    @field_validator("reason", "who")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-empty")
        return v


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    trial_id: str
    stage: Stage
    hypothesis_hash: str
    code_hash: str
    data_hashes: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    windows: dict[str, tuple[str, str]] = Field(default_factory=dict)
    trial_family_count: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    verdict: Verdict
    reasons: list[str] = Field(default_factory=list)
    environment: dict[str, Any] = Field(default_factory=dict)
    parent_receipt_hash: str | None = None
    methods: dict[str, str] = Field(default_factory=dict)  # method id -> version actually used
    packs: dict[str, dict[str, Any]] = Field(default_factory=dict)  # active experience packs: {name: {version, hash}}
    override: Override | None = None
    # Facts about HOW this run went (a cache hit, timings): stored, never hashed. Anything that can differ between
    # two runs of identical inputs belongs here and not in ``params``, so identical inputs give identical hashes.
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)

    @field_validator("windows")
    @classmethod
    def _iso_windows(cls, v: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
        for name, (start, end) in v.items():
            for ts in (start, end):
                try:
                    datetime.fromisoformat(ts)
                except ValueError as exc:
                    raise ValueError(f"window {name!r}: {ts!r} is not an ISO timestamp") from exc
        return v

    def receipt_hash(self) -> str:
        """Canonical-JSON hash of everything except ``created_at`` and ``diagnostics``.

        ``methods`` and ``packs`` join the hash only when non-empty, so receipts written before they existed still verify.
        """
        exclude = {"created_at", "diagnostics"} | ({"methods"} if not self.methods else set()) | ({"packs"} if not self.packs else set())
        return hash_json(self.model_dump(mode="json", exclude=exclude))


def collect_environment(seeds: dict[str, int] | None = None) -> dict[str, Any]:
    libs: dict[str, str] = {}
    for name in KEY_LIBRARIES:
        try:
            libs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            libs[name] = "unknown"
    return {"python": platform.python_version(), "libraries": libs, "seeds": dict(seeds or {})}


def receipt_path(home: str | Path, trial_id: str, stage: Stage | str) -> Path:
    return Path(home) / "receipts" / trial_id / f"{Stage(stage).value}.json"


def write_receipt(receipt: Receipt, home: str | Path, registry: Registry | None = None) -> Path:
    """Write ``<home>/receipts/<trial>/<stage>.json``; index it and log any override."""
    digest = receipt.receipt_hash()  # raises on NaN/inf before anything is written
    path = receipt_path(home, receipt.trial_id, receipt.stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"receipt_hash": digest, "receipt": receipt.model_dump(mode="json")}
    path.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if registry is not None:
        registry.index_receipt(receipt.trial_id, receipt.stage.value, digest, receipt.verdict.value, str(path))
        if receipt.override is not None:
            registry.log_override(
                receipt.trial_id, receipt.stage.value, receipt.override.reason, receipt.override.who
            )
    return path


def load_receipt(path: str | Path) -> Receipt:
    """Load a receipt and verify its stored hash; any mismatch raises."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        receipt = Receipt.model_validate(doc["receipt"])
        stored = doc["receipt_hash"]
        actual = receipt.receipt_hash()
    except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
        raise ReceiptTamperedError(f"receipt {str(path)!r} is unreadable or invalid: {exc}") from exc
    if stored != actual:
        raise ReceiptTamperedError(f"receipt {str(path)!r} does not match its recorded hash")
    return receipt
