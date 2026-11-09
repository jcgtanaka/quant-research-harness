"""Interview status of a trial: per stage rows and counts, conditions for S7, and the compact form for receipts."""

from __future__ import annotations

import sqlite3
from typing import Any

from qrh.interview.engine import Item, items_for_trial, state_of
from qrh.packs.model import STAGES
from qrh.packs.runtime import PackView

NOTE = "Unanswered questions never block a stage; they surface as conditions on a final PASS."


def _light(rows: list[dict[str, Any]]) -> str:
    if any((r["critical"] and r["status"] != "answered") or r.get("risk") for r in rows):
        return "red"
    return "amber" if any(r["status"] != "answered" for r in rows) else "green"


def build_snapshot(conn: sqlite3.Connection, view: PackView, trial: dict[str, Any], stage: str | None = None) -> dict[str, Any]:
    by_stage: dict[str, list[Item]] = {}
    for it in items_for_trial(conn, view, trial):
        if stage is None or it.question.stage == stage:
            by_stage.setdefault(it.question.stage, []).append(it)
    stages: dict[str, Any] = {}
    for st in STAGES:
        if st not in by_stage:
            continue
        rows = [state_of(conn, it) for it in by_stage[st]]
        counts = {k: sum(1 for r in rows if r["status"] == k) for k in ("answered", "unknown", "pending")}
        stages[st] = {"rows": rows, "counts": counts, "total": len(rows), "light": _light(rows)}
    return {"trial_id": trial["trial_id"], "stages": stages, "conditions": conditions(stages), "note": NOTE}


def conditions(stages: dict[str, Any]) -> list[dict[str, Any]]:
    """Reasons a clean PASS becomes PASS_CONDITIONAL: a critical question pending/unknown, or any risk-tagged answer."""
    out: list[dict[str, Any]] = []
    for st, body in stages.items():
        for r in body["rows"]:
            if r["status"] == "answered" and r.get("risk"):
                out.append({"question_id": r["qid"], "scope": r["scope"], "stage": st, "text": r["text"],
                            "status": "risk", "risk": r["risk"]})
            elif r["critical"] and r["status"] in ("pending", "unknown"):
                out.append({"question_id": r["qid"], "scope": r["scope"], "stage": st, "text": r["text"],
                            "status": r["status"], "risk": r.get("unknown_risk", "")})
    return out


def receipt_status(conn: sqlite3.Connection, view: PackView, trial: dict[str, Any], stage: str) -> dict[str, Any] | None:
    """Compact status for a stage receipt (None when no question concerns the stage, so old hashes are untouched)."""
    body = build_snapshot(conn, view, trial, stage)["stages"].get(stage)
    if body is None:
        return None
    rows = body["rows"]
    ids = lambda s: sorted(f"{r['qid']}@{r['scope']}" for r in rows if r["status"] == s)  # noqa: E731
    return {
        **body["counts"], "answered_ids": ids("answered"), "unknown_ids": ids("unknown"), "pending_ids": ids("pending"),
        "answer_versions": {f"{r['qid']}@{r['scope']}": r["version"] for r in rows if "version" in r},
    }
