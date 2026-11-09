"""Answers live in <home>/intake/answers.json, a plain JSON object written atomically."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from qrh.intake.questions import QUESTIONS, QUESTION_IDS, AnswerError, Question, validate_answer


def answers_path(home: str | Path) -> Path:
    return Path(home) / "intake" / "answers.json"


def load_answers(home: str | Path) -> dict[str, Any]:
    p = answers_path(home)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise AnswerError(f"The answers file {p} cannot be read. Fix or remove it.") from None
    if not isinstance(data, dict):
        raise AnswerError(f"The answers file {p} is not a JSON object. Fix or remove it.")
    return data


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".answers.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_answer(home: str | Path, qid: str, value: Any) -> None:
    """Store one already-validated answer under a known question id."""
    if qid not in QUESTION_IDS:
        raise AnswerError(f"There is no intake question called '{qid}'.")
    data = load_answers(home)
    data[qid] = value
    _write_atomic(answers_path(home), data)


def run_intake(
    ask: Callable[[Question], Any],
    home: str | Path,
    notify: Callable[[str], None] | None = None,
    redo: Iterable[str] = (),
) -> dict[str, Any]:
    """Ask each question one at a time; refuse bad answers and ask again.

    Already-answered questions are skipped unless named in `redo`. Only what the
    user answers (via `ask`) is ever stored.
    """
    redo_set = set(redo)
    for qid in redo_set:
        if qid not in QUESTION_IDS:
            raise AnswerError(f"There is no intake question called '{qid}'.")
    done = load_answers(home)
    for question in QUESTIONS:
        if question.id in done and question.id not in redo_set:
            continue
        while True:
            try:
                value = validate_answer(question, ask(question))
                break
            except AnswerError as exc:
                if notify is not None:
                    notify(str(exc))
        save_answer(home, question.id, value)
    return load_answers(home)
