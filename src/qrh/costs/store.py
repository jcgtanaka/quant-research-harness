"""Read confirmed broker commission records from ``<QRH_HOME>/costs/*.json``.

The folder holds one record per file. Each file is checked by ``parse_record``. A file that cannot be read, is not
valid, or is not confirmed is skipped with a plain warning. Nothing is invented and nothing here raises.
"""

from __future__ import annotations

import json
from pathlib import Path

from qrh.costs.broker_commission import CommissionRecord, RecordError, parse_record

COSTS_DIRNAME = "costs"


def load_commission_records(home: Path) -> tuple[list[CommissionRecord], list[str]]:
    """Return (confirmed records in file-name order, warnings for the files that were skipped)."""
    folder = Path(home) / COSTS_DIRNAME
    if not folder.is_dir():
        return [], []
    records: list[CommissionRecord] = []
    warnings: list[str] = []
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            warnings.append(f"{path.name}: skipped, the file cannot be read ({type(exc).__name__})")
            continue
        if not isinstance(data, dict):
            warnings.append(f"{path.name}: skipped, the file must hold one JSON object")
            continue
        try:
            record = parse_record(data)
        except RecordError as exc:
            warnings.append(f"{path.name}: skipped, the record is not valid ({exc})")
            continue
        if record.status != "confirmed":
            warnings.append(f"{path.name}: skipped, the record is not confirmed")
            continue
        records.append(record)
    return records, warnings
