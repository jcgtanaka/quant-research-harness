"""Usage of local models (for example Ollama), logged to <QRH_HOME>/logs/local_llm.jsonl.

One JSON object per line: ``ts``, ``session_id`` (optional), ``model``, ``prompt_tokens``,
``completion_tokens``, ``duration_ms``, ``task`` (a short label you choose). Tokens only; the cost is 0.
Prompts and replies are never stored.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qrh.hooks.common import _append_json, redact

TASK_LIMIT = 80


class LocalLogError(ValueError):
    """The response cannot be turned into a usage record."""


def local_log_path(home: str | Path) -> Path:
    return Path(home) / "logs" / "local_llm.jsonl"


def _objects(text: str) -> list[dict[str, Any]]:
    try:
        whole = json.loads(text)
        found = [whole]
    except ValueError:
        found = []
        for line in text.splitlines():
            try:
                found.append(json.loads(line))
            except ValueError:
                continue
    return [o for o in found if isinstance(o, dict)]


def record_from_ollama(
    text: str, *, task: str = "", session_id: str | None = None, model: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a record from an Ollama /api/generate or /api/chat response (one object, or streamed lines)."""
    final = [o for o in _objects(text) if "eval_count" in o or "prompt_eval_count" in o]
    if not final:
        raise LocalLogError("no Ollama response with prompt_eval_count/eval_count found on stdin")
    obj = final[-1]
    name = model or obj.get("model")
    if not isinstance(name, str) or not name:
        raise LocalLogError("the response has no model name; pass --model")

    def count(key: str) -> int:
        v = obj.get(key)
        return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0

    nanos = obj.get("total_duration")
    duration = round(nanos / 1_000_000, 3) if isinstance(nanos, (int, float)) and not isinstance(nanos, bool) else 0.0
    rec: dict[str, Any] = {"ts": (now or datetime.now(timezone.utc)).isoformat()}
    if session_id:
        rec["session_id"] = session_id
    rec.update({
        "model": name, "prompt_tokens": count("prompt_eval_count"), "completion_tokens": count("eval_count"),
        "duration_ms": duration, "task": redact(" ".join(task.split()))[:TASK_LIMIT],
    })
    return rec


def append_local(home: str | Path, record: dict[str, Any]) -> None:
    _append_json(local_log_path(home), record)


def load_local(home: str | Path) -> list[dict[str, Any]]:
    path = local_log_path(home)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and isinstance(rec.get("model"), str):
                out.append(rec)
    return out
