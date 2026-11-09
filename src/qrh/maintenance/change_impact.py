"""Change impact: after source files change, which trials' receipts are out of date?

A receipt records the code hash it was made with. If that hash no longer matches the current code hash, the receipt
is stale. This is information only: it blocks nothing and never raises for missing or broken data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

DB_NAME = "registry.sqlite3"


@dataclass
class ChangeImpact:
    """``trials`` has one entry per trial: ``trial_id``, ``stale`` (bool), ``sentence``. ``note`` explains gaps."""

    trials: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""


def change_impact(home: str | Path, changed_paths: Iterable[str | Path] | None,
                  current_hash: str | None = None) -> ChangeImpact:
    """Compare each trial's latest receipt code hash with the current one (``package_code_hash`` by default)."""
    try:
        return _change_impact(Path(home), changed_paths, current_hash)
    except Exception as exc:  # information only: never raise
        return ChangeImpact([], f"Could not check change impact: {exc}")


def _change_impact(home: Path, changed_paths: Iterable[str | Path] | None, current_hash: str | None) -> ChangeImpact:
    from qrh.memory.registry import Registry
    from qrh.pipeline.receipt import load_receipt
    from qrh.pipeline.stages import package_code_hash

    db_path = home / DB_NAME
    if not db_path.is_file():
        return ChangeImpact([], "No registry was found, so there are no trials to check.")
    current = current_hash if current_hash is not None else package_code_hash()
    changed = ", ".join(str(p) for p in (changed_paths or [])) or "no files listed"
    registry = Registry(db_path)
    try:
        trials = registry.list_trials()
        out: list[dict[str, Any]] = []
        for trial in trials:
            tid = trial["trial_id"]
            recorded = _latest_code_hash(registry, tid, load_receipt)
            if recorded is None:
                out.append({"trial_id": tid, "stale": False, "recorded_hash": None,
                            "sentence": f"Trial {tid} has no readable receipt, so it cannot be compared."})
            elif recorded != current:
                out.append({"trial_id": tid, "stale": True, "recorded_hash": recorded,
                            "sentence": f"Trial {tid} is out of date: its receipt was made with older code "
                                        f"(changed files: {changed})."})
            else:
                out.append({"trial_id": tid, "stale": False, "recorded_hash": recorded,
                            "sentence": f"Trial {tid} is up to date: its receipt matches the current code."})
    finally:
        registry.close()
    return ChangeImpact(out, "" if out else "The registry has no trials.")


def _latest_code_hash(registry: Any, trial_id: str, load_receipt: Any) -> str | None:
    for row in reversed(registry.list_receipts(trial_id)):
        try:
            return load_receipt(row["path"]).code_hash
        except Exception:
            continue
    return None
