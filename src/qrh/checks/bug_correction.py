"""One-attempt bug correction flow.

When the harness finds a bug in a backtest it built, the LLM that built the
file fixes it once. If the bug is still there afterwards, the profit check is
INCONCLUSIVE. There is no loop.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REQUEST_NAME = "correction_request.txt"
DONE_NAME = "correction_done.json"
EXHAUSTED_NAME = "correction_exhausted.json"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def correction_state(run_dir) -> str:
    """Return none, requested, done or exhausted."""
    run = Path(run_dir)
    if (run / EXHAUSTED_NAME).exists():
        return "exhausted"
    if (run / DONE_NAME).exists():
        return "done"
    if (run / REQUEST_NAME).exists():
        return "requested"
    return "none"


def correction_request(run_dir, bugs):
    """Write the plain-English request for the LLM. Returns its path.

    With no bugs nothing is written and None is returned. If the single
    attempt is already used, no new request is written and the state becomes
    "exhausted".
    """
    if not bugs:
        return None
    run = Path(run_dir)
    if correction_state(run) in ("done", "exhausted"):
        _atomic_write(
            run / EXHAUSTED_NAME,
            json.dumps({"reason": "the single correction attempt is already used"}),
        )
        return None
    lines = ["The checks found a difference (a bug) in the backtest file you built.", ""]
    for bug in bugs:
        lines.append(f"- Day {bug['day']}: the difference is {bug['amount']}.")
    lines += [
        "",
        "What to do: fix the backtest file so that the difference disappears,",
        "then record the fix with a short note.",
        "You have one attempt only. If the difference is still there after it,",
        "the profit check will be marked INCONCLUSIVE.",
        "",
    ]
    path = run / REQUEST_NAME
    _atomic_write(path, "\n".join(lines))
    return path


def record_correction(run_dir, note):
    """Record the one correction. Refuses a second record for the same run."""
    run = Path(run_dir)
    if (run / DONE_NAME).exists():
        raise ValueError("A correction was already recorded for this run; only one attempt is allowed.")
    data = {
        "note": note,
        "time": datetime.now(timezone.utc).isoformat(),
        "attempt": 1,
    }
    path = run / DONE_NAME
    _atomic_write(path, json.dumps(data, indent=2))
    return path


def verdict_after_correction(bugs_before, bugs_after, run_dir):
    """Return (verdict, reason). PASS here means the caller keeps its own verdict."""
    if bugs_after:
        first = bugs_after[0]
        where = f"day {first['day']}, amount {first['amount']}"
        if correction_state(run_dir) in ("done", "exhausted"):
            return (
                "INCONCLUSIVE",
                f"The bug is still there after the one correction attempt (first: {where}). "
                "The profit check cannot be trusted.",
            )
        return (
            "INCONCLUSIVE",
            f"The bug has not been corrected yet (first: {where}). "
            "Fix the backtest file once, record it, then check again.",
        )
    if bugs_before:
        return (
            "PASS",
            "The bug was corrected once and the run is re-checked.",
        )
    return ("PASS", "No bugs found, so the verdict is not changed here.")
