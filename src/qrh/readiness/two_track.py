"""Information section of the readiness report: the Track 2 answers a person stored for the trial.

Read-only. The answers come only from ``<home>/trials/<trial>/two_track_answers.json`` (a JSON object
``{question_id: option}``). No file means no section. The section is information: it never becomes a
readiness warning and never changes a verdict. A file that cannot be read or validated is described in the
section itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qrh.checks.two_track import evaluate_two_track

FILE_NAME = "two_track_answers.json"


def answers_path(home: str | Path, trial_id: str) -> Path:
    return Path(home) / "trials" / trial_id / FILE_NAME


def load_section(home: str | Path, trial_id: str) -> dict[str, Any] | None:
    """The section document, or None when the trial has no answer file."""
    path = answers_path(home, trial_id)
    if not path.is_file():
        return None
    try:
        answers = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(answers, dict):
            raise ValueError("the file must hold a JSON object {question_id: option}")
        result = evaluate_two_track(answers, {}, {})
    except (OSError, ValueError) as exc:
        return {"overall": None, "results": [], "error": f"{FILE_NAME} cannot be used: {exc}"}
    return {"overall": result.overall, "error": None,
            "results": [{"question_id": r.question_id, "status": r.status, "reason": r.reason}
                        for r in result.results]}
