"""The per-trial report manifest: every rendered file with its sha256 and the receipt hash it was built from.

Stored at ``<home>/reports/<trial_id>/manifest.json``. A rendering failure is recorded
as an error entry (and surfaces in ``qrh status``); it never changes a stage verdict.
The manifest is canonical JSON without timestamps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qrh.pipeline.receipt import Stage

SCHEMA_VERSION = 1
COMBINED = "combined"
READINESS = "readiness"  # the deployment readiness report (qrh.readiness)


def reports_dir(home: str | Path, trial_id: str) -> Path:
    return Path(home) / "reports" / trial_id


def manifest_path(home: str | Path, trial_id: str) -> Path:
    return reports_dir(home, trial_id) / "manifest.json"


def read_manifest(home: str | Path, trial_id: str) -> dict[str, Any]:
    """The stored manifest, or an empty one. An unreadable manifest is an error entry, never a crash."""
    path = manifest_path(home, trial_id)
    empty = {"schema_version": SCHEMA_VERSION, "trial_id": trial_id, "files": [], "errors": []}
    if not path.is_file():
        return empty
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc.get("files"), list) or not isinstance(doc.get("errors"), list):
            raise ValueError("malformed manifest")
        return doc
    except (OSError, ValueError, AttributeError) as exc:
        return {**empty, "errors": [{"stage": "manifest", "error": f"manifest unreadable: {exc}"}]}


def _write(home: str | Path, trial_id: str, doc: dict[str, Any]) -> None:
    doc["files"] = sorted(doc["files"], key=lambda f: (f["path"], f["kind"]))
    doc["errors"] = sorted(doc["errors"], key=lambda e: (e["stage"], e["error"]))
    path = manifest_path(home, trial_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _paths(trial_id: str, stage_key: str) -> list[str]:
    if stage_key == COMBINED:
        return [f"{trial_id}.pdf"]
    if stage_key == READINESS:
        return ["readiness.pdf", "readiness.chart.json", "readiness.json"]
    return [f"{stage_key}.pdf", f"{stage_key}.chart.json"]


def _drop(home: str | Path, trial_id: str, doc: dict[str, Any], stage_key: str) -> None:
    """Remove a stage's files from disk and the manifest, and any earlier error for it."""
    for rel in _paths(trial_id, stage_key):
        (reports_dir(home, trial_id) / rel).unlink(missing_ok=True)
    doc["files"] = [f for f in doc["files"] if f["stage"] != stage_key]
    doc["errors"] = [e for e in doc["errors"] if e["stage"] != stage_key]


def record_files(home: str | Path, trial_id: str, stage_key: str, files: list[dict[str, str]]) -> None:
    doc = read_manifest(home, trial_id)
    doc["files"] = [f for f in doc["files"] if f["stage"] != stage_key] + files
    doc["errors"] = [e for e in doc["errors"] if e["stage"] != stage_key]
    _write(home, trial_id, doc)


def record_error(home: str | Path, trial_id: str, stage_key: str, message: str) -> None:
    """Fail closed: outputs of the failed stage are removed, so no PDF built from other data stays behind."""
    doc = read_manifest(home, trial_id)
    _drop(home, trial_id, doc, stage_key)
    doc["errors"].append({"stage": stage_key, "error": message})
    _write(home, trial_id, doc)


def clear_from(home: str | Path, trial_id: str, stage: Stage) -> None:
    """Remove outputs of ``stage`` and every later stage (and the combined PDF), for a fresh run."""
    if not manifest_path(home, trial_id).is_file() and not reports_dir(home, trial_id).is_dir():
        return
    doc = read_manifest(home, trial_id)
    for s in Stage:
        if s.value >= stage.value:
            _drop(home, trial_id, doc, s.value)
    for key in sorted({f["stage"] for f in doc["files"] if f["stage"].startswith("interview.")}):
        _drop(home, trial_id, doc, key)  # interview pages are built from answers that may have changed
    _drop(home, trial_id, doc, COMBINED)
    _drop(home, trial_id, doc, READINESS)  # built from the receipts and answers of the run being replaced
    _write(home, trial_id, doc)
