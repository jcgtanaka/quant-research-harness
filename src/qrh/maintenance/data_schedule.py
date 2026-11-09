"""Schedule-ready data check: run the existing data rules and keep a plain report.

It only ever WARNS. Nothing here blocks the user's work, and nothing runs on its own: call
``run_data_check`` before a run or on demand. Each dataset gets ``<home>/maintenance/data-<name>.json``
and ``.txt``, written atomically (the old report stays whole if a write fails).

Verdicts: PASS (all good), WARN (look at it: old data, missing data, unknown name), FAIL (a data
rule found a problem). A dataset's rules file may add a top-level ``max_age_days: N`` to say how old
its last date may be.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from qrh.checks.data_quality import check_dataset_report
from qrh.memory.catalog import Catalog

DB_NAME = "registry.sqlite3"
_ADVICE = (
    ("missing timestamp", "Some dates are missing. Fetch the missing rows or fix the source file."),
    ("duplicate", "Some dates appear twice. Remove the repeated rows from the data file."),
    ("not sorted", "The dates are out of order. Sort the data file by date."),
    ("out of range", "Some values look wrong. Check them at the source and correct the file."),
    ("column", "A column is missing or has the wrong type. Fix the data file or the rule."),
)


def _advice(reason: str) -> str:
    low = reason.lower()
    todo = next((t for key, t in _ADVICE if key in low), "Look at the data file and the rule, then fix whichever is wrong.")
    return f"{reason}. What to do: {todo}"


def _max_age(rules_path: Path | None) -> int | None:
    if rules_path is None:
        return None
    try:
        doc = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        value = doc.get("max_age_days") if isinstance(doc, dict) else None
        return int(value) if value is not None else None
    except Exception:
        return None


def _check_one(catalog: Catalog | None, name: str, today: date, max_age_days: int | None) -> dict[str, Any]:
    rep: dict[str, Any] = {"dataset": name, "verdict": "PASS", "last_date": None, "age_days": None,
                           "checked_on": today.isoformat(), "findings": []}
    findings: list[str] = rep["findings"]
    if catalog is None or catalog.get_dataset(name) is None:
        rep["verdict"] = "WARN"
        findings.append(f"Dataset {name!r} is not registered, so it was not checked. "
                        "What to do: register it first, or check the name for typos.")
        return rep
    rec = catalog.get_dataset(name) or {}
    status, reasons, report, cov = check_dataset_report(catalog, name)
    end = (cov or {}).get("end") or rec.get("end")
    if end:
        rep["last_date"] = pd.Timestamp(end).date().isoformat()
        rep["age_days"] = (today - pd.Timestamp(end).date()).days
    rule_failed = report is not None and report.verdict == "FAIL"
    if status == "PASS":
        pass
    elif report is not None and report.verdict in ("PASS", "FAIL"):
        findings.extend(_advice(r) for r in reasons)
    elif status == "UNCHECKED":
        findings.append("No data rules are declared for this dataset, so it was not checked. "
                        "What to do: add a rules file for it.")
    else:
        findings.extend(f"{r}. What to do: fix or re-register the data file, then run the check again." for r in reasons)
    limit = max_age_days if max_age_days is not None else _max_age(catalog.rules_for(name))
    if limit is not None and rep["age_days"] is not None and rep["age_days"] > limit:
        findings.append(f"The data is {rep['age_days']} days old and the limit is {limit} days. "
                        "What to do: update the data.")
    if rule_failed:
        rep["verdict"] = "FAIL"
    elif findings:
        rep["verdict"] = "WARN"
    return rep


def _render(rep: dict[str, Any]) -> str:
    lines = [f"Data check for {rep['dataset']}: {rep['verdict']}",
             f"Checked on: {rep['checked_on']}",
             f"Last date in the data: {rep['last_date'] or 'unknown'}",
             f"Age in days: {rep['age_days'] if rep['age_days'] is not None else 'unknown'}"]
    lines += [f"- {f}" for f in rep["findings"]] or ["- Nothing to do."]
    lines.append("This is a warning only. Nothing is blocked.")
    return "\n".join(lines) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".data-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name)).strip(".") or "dataset"


def run_data_check(
    home: str | Path, dataset_names: list[str], today: date, max_age_days: int | None = None
) -> dict[str, dict[str, Any]]:
    """Check each dataset and write its report. Returns ``{name: report}``. Never raises for bad data."""
    home = Path(home)
    out: dict[str, dict[str, Any]] = {}
    db = home / DB_NAME
    try:
        catalog: Catalog | None = Catalog(db) if db.is_file() else None
    except Exception:
        catalog = None
    try:
        for name in dataset_names:
            try:
                rep = _check_one(catalog, name, today, max_age_days)
            except Exception as exc:  # a broken dataset must warn, not stop the caller
                rep = {"dataset": name, "verdict": "WARN", "last_date": None, "age_days": None,
                       "checked_on": today.isoformat(),
                       "findings": [f"The check could not run ({type(exc).__name__}: {exc}). "
                                    "What to do: look at the data file, then run the check again."]}
            out[name] = rep
            try:
                base = home / "maintenance" / f"data-{_safe(name)}"
                _write_atomic(base.with_suffix(".json"), json.dumps(rep, indent=2, sort_keys=True) + "\n")
                _write_atomic(base.with_suffix(".txt"), _render(rep))
            except OSError:
                pass  # the report is still returned; a full disk must not stop the user's work
    finally:
        if catalog is not None:
            with closing(catalog.conn):
                pass
    return out
