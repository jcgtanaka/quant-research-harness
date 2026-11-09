"""Read functions for the dashboard. Every value comes from the registry or a hash-verified stored receipt.

Nothing here writes, and no verdict is computed: verdicts are read back from receipts. A receipt that cannot be
verified is reported as an error for that stage, never as a verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qrh.memory.registry import Registry
from qrh.pipeline.receipt import Stage, load_receipt, receipt_path
from qrh.readiness.collect import ReadinessError, check_trial_id, collect

REGISTRY_NAME = "registry.sqlite3"


class DataError(Exception):
    """A request that cannot be answered; ``status`` is the HTTP status, ``message`` says why."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _open(home: Path) -> Registry:
    path = Path(home) / REGISTRY_NAME
    if not path.is_file():  # never create an empty registry as a side effect of a read
        raise DataError(503, f"registry not found at {str(path)!r}")
    try:
        return Registry(path)
    except Exception as exc:
        raise DataError(503, f"registry unreadable: {type(exc).__name__}: {exc}") from exc


def _known_trial(reg: Registry, trial_id: str) -> dict[str, Any]:
    try:
        check_trial_id(trial_id)
    except ReadinessError:
        raise DataError(404, f"unknown trial {trial_id!r}") from None
    trial = reg.get_trial(trial_id)
    if trial is None:
        raise DataError(404, f"unknown trial {trial_id!r}")
    return trial


def list_trials(home: Path) -> list[dict[str, Any]]:
    reg = _open(home)
    try:
        out = []
        for t in reg.list_trials():
            receipts = reg.list_receipts(t["trial_id"])
            out.append({
                "trial_id": t["trial_id"],
                "family": t["family_id"],
                "strategy_family": t["strategy_family"],
                "asset_class": t["asset_class"],
                "frequency": t["frequency"],
                "created_at": t["created_at"],
                "stage_verdicts": {r["stage"]: r["verdict"] for r in receipts},
            })
        return out
    except DataError:
        raise
    except Exception as exc:
        raise DataError(503, f"registry unreadable: {type(exc).__name__}: {exc}") from exc
    finally:
        reg.close()


def _cost_source(cost: Any) -> dict[str, Any] | None:
    """The recorded cost source as stored (never worked out here), or None when the run recorded none."""
    if not isinstance(cost, dict) or not cost:
        return None
    return {"source": cost.get("source"), "broker": cost.get("broker"), "status": cost.get("status"),
            "reason": cost.get("reason")}


def _read_json_folder(folder: Path, pattern: str) -> list[dict[str, Any]]:
    """Every stored JSON file matching ``pattern``, in file-name order. A missing folder is an empty list; a file
    that cannot be read becomes an item with an ``error`` so it is shown, never skipped."""
    if not folder.exists():
        return []
    if not folder.is_dir():
        raise DataError(503, f"folder {str(folder)!r} is not a folder")
    try:
        files = sorted(p for p in folder.glob(pattern) if p.is_file())
    except OSError as exc:
        raise DataError(503, f"folder unreadable: {type(exc).__name__}: {exc}") from exc
    items: list[dict[str, Any]] = []
    for f in files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(doc, dict):
                raise ValueError("the file does not hold a report")
            items.append(doc)
        except (OSError, ValueError) as exc:
            items.append({"file": f.name, "error": f"this file could not be read: {exc}"})
    return items


def list_parity(home: Path) -> list[dict[str, Any]]:
    return _read_json_folder(Path(home) / "parity", "*.json")


def list_data_checks(home: Path) -> list[dict[str, Any]]:
    return _read_json_folder(Path(home) / "maintenance", "data-*.json")


def trial_detail(home: Path, trial_id: str) -> dict[str, Any]:
    reg = _open(home)
    try:
        trial = _known_trial(reg, trial_id)
        stages: dict[str, Any] = {}
        cost_source = None
        for entry in reg.list_receipts(trial_id):
            stage = entry["stage"]
            try:
                r = load_receipt(receipt_path(home, trial_id, Stage(stage)))
                stages[stage] = {"verdict": r.verdict.value, "reasons": list(r.reasons), "error": None}
                linear = r.params.get("pnl_linear")
                if cost_source is None and isinstance(linear, dict):
                    cost_source = _cost_source(linear.get("cost"))
            except Exception as exc:
                stages[stage] = {"verdict": None, "reasons": [], "error": str(exc)}
        recon = trial.get("reconstruction")
        adjustment = recon.get("adjustment") if isinstance(recon, dict) else None
        return {
            "trial_id": trial_id,
            "family": trial["family_id"],
            "strategy_family": trial["strategy_family"],
            "asset_class": trial["asset_class"],
            "frequency": trial["frequency"],
            "created_at": trial["created_at"],
            "hypothesis": trial["text"],
            "stages": stages,
            "cost_source": cost_source,
            "adjustment": adjustment if isinstance(adjustment, dict) and adjustment else None,
        }
    except DataError:
        raise
    except Exception as exc:
        raise DataError(503, f"registry unreadable: {type(exc).__name__}: {exc}") from exc
    finally:
        reg.close()


def trial_warnings(home: Path, trial_id: str) -> dict[str, Any]:
    """Warnings from ``qrh.readiness.collect.collect``, which is documented read-only (it only reads receipts,
    artifacts and the registry)."""
    reg = _open(home)
    try:
        _known_trial(reg, trial_id)
        c = collect(home, trial_id, reg)
        return {
            "trial_id": trial_id,
            "verdict": c.verdict,
            "failing_stage": c.failing_stage,
            "totals": c.totals(),
            "warnings": [w.doc() for w in c.warnings],
        }
    except DataError:
        raise
    except ReadinessError as exc:
        raise DataError(404, str(exc)) from exc
    except Exception as exc:
        raise DataError(503, f"warnings unavailable: {type(exc).__name__}: {exc}") from exc
    finally:
        reg.close()
