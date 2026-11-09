"""The data document of a readiness report and its hash.

The hash covers the verdict, the counts, every warning (id, stage, severity, kind, code, message, resolution and the
hash of the receipt it comes from) and the hash of every stage receipt. It has no timestamps, no file locations and no
acknowledgement status, so the same trial state always gives the same hash on any machine. Any change in the warning
set gives a different one, and so does changed evidence (a replaced adapter, changed data, a new answer): the
acknowledgement is of the evidence the human was shown. Receipts are deterministic (run-dependent facts live in
``Receipt.diagnostics``, which is never hashed), so an identical re-run keeps the hash. The acknowledgement signs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qrh.hashing import hash_json
from qrh.pipeline.receipt import display_verdict
from qrh.readiness.collect import NOT_RUN, Collected

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReportData:
    doc: dict[str, Any]
    hash: str

    @property
    def warning_ids(self) -> list[str]:
        return sorted(w["id"] for w in self.doc["warnings"])


def build_data(c: Collected) -> ReportData:
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "trial_id": c.trial_id,
        "final_verdict": c.verdict,
        "final_verdict_display": c.verdict_display if c.verdict != NOT_RUN else NOT_RUN,
        "failing_stage": c.failing_stage,
        "counts": c.counts(),
        "totals": c.totals(),
        "warnings": [w.doc() for w in c.warnings],
        "receipts": dict(sorted(c.receipt_hashes.items())),
    }
    if c.two_track is not None:  # information only; absent keeps the document and hash of a trial without the file
        doc["two_track"] = c.two_track
    return ReportData(doc, hash_json(_hashed(doc)))


def _hashed(doc: dict[str, Any]) -> dict[str, Any]:
    """The document without file locations (receipt paths): where a file lives is not evidence."""
    return {**doc, "warnings": [{k: v for k, v in w.items() if k != "receipt_path"} for w in doc["warnings"]]}


__all__ = ["ReportData", "build_data", "display_verdict"]
