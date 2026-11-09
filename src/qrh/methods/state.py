"""Local method state: status changes made on this machine (promotions, demotions, deprecations).

Public defaults ship in code (``builtin.py``). A local promotion is recorded in
``<QRH_HOME>/methods/state.json`` with who/why/which benchmark, and overrides the shipped status for this
home only. Reading never creates the file; a malformed file is an error, never a silent fall back to the
shipped defaults.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from qrh.methods.registry import STATUSES, Method, MethodError

SCHEMA_VERSION = 1


def state_path(home: str | Path) -> Path:
    return Path(home) / "methods" / "state.json"


def empty_state() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "statuses": {}, "history": []}


def load_state(home: str | Path) -> dict[str, Any]:
    path = state_path(home)
    if not path.exists():
        return empty_state()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MethodError(f"methods state {str(path)!r} is unreadable: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema_version") != SCHEMA_VERSION:
        raise MethodError(f"methods state {str(path)!r} has an unsupported schema")
    statuses, history = doc.get("statuses"), doc.get("history")
    if not isinstance(statuses, dict) or not isinstance(history, list):
        raise MethodError(f"methods state {str(path)!r} is malformed")
    for mid, versions in statuses.items():
        if not isinstance(versions, dict) or any(s not in STATUSES for s in versions.values()):
            raise MethodError(f"methods state {str(path)!r}: invalid statuses for {mid!r}")
    return doc


def save_state(home: str | Path, state: dict[str, Any]) -> Path:
    path = state_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def effective_status(method: Method, state: dict[str, Any]) -> str:
    return state.get("statuses", {}).get(method.id, {}).get(method.version, method.status)
