"""Session registry (F5): the transcript path Claude Code reported for each session id.

The runner records ``transcript_path`` from SessionStart and UserPromptSubmit payloads into
``<QRH_HOME>/sessions/`` (write-protected by approval-guard). The first recorded path wins, so a later payload
cannot redirect a session to another file. ``qrh interview answer`` accepts only a transcript that equals the
recorded one and lives under ``~/.claude/projects``.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

EVENTS = ("SessionStart", "UserPromptSubmit")


def sessions_dir(home: str | Path) -> Path:
    return Path(home) / "sessions"


def _path(home: str | Path, session_id: str) -> Path:
    return sessions_dir(home) / (hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32] + ".json")


def record(home: str | Path, session_id: str, transcript_path: str, now: datetime) -> None:
    """Remember the transcript of a session; an existing different path is kept and the conflict noted."""
    if not session_id or not transcript_path:
        return
    path = _path(home, session_id)
    try:
        doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        doc = None
    if isinstance(doc, dict) and doc.get("session_id") == session_id and isinstance(doc.get("transcript_path"), str):
        if doc["transcript_path"] == transcript_path:
            return
        conflicts = doc.setdefault("conflicting_paths", [])
        if transcript_path in conflicts:
            return
        conflicts.append(transcript_path)
    else:
        doc = {"session_id": session_id, "transcript_path": transcript_path, "first_seen": now.isoformat()}
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def lookup(home: str | Path, session_id: str) -> str | None:
    """The recorded transcript path of a session, or None if the session was never seen."""
    if not session_id:
        return None
    try:
        doc = json.loads(_path(home, session_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("session_id") != session_id:
        return None
    value = doc.get("transcript_path")
    return value if isinstance(value, str) else None
