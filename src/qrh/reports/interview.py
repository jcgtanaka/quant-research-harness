"""Interview pages: per stage, what was asked, answered, unknown or still pending, and why it matters.

Built from an interview snapshot (``qrh.interview.snapshot``). The S7 receipt records that snapshot as the
hash-verified ``interview`` artifact, and the combined PDF is drawn from it, so the one-source rule holds; the
``qrh interview report`` command draws the same pages from the live registry state.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from qrh.hashing import hash_json, sha256_file
from qrh.reports import manifest, theme
from qrh.reports.chartdata import SCHEMA_VERSION
from qrh.reports.pdf import write_pdf
from qrh.reports.render import KEYWORDS, RenderedFile

ROWS_PER_PAGE = 9
LIGHT = {"green": ("good", "✓", "ALL ANSWERED"), "amber": ("warning", "◆", "OPEN QUESTIONS"),
         "red": ("warning", "◆", "CONDITIONS")}  # amber diamond, as PASS · CONDITIONAL
STATUS_STYLE = {"answered": ("good", "✓", "ANSWERED"), "unknown": ("warning", "◆", "DON'T KNOW"),
                "pending": (None, "○", "PENDING")}


def _row_doc(r: dict[str, Any]) -> dict[str, Any]:
    mapped = r.get("mapped") or {}
    maps = []
    if mapped.get("dataset_flag"):
        maps.append(f"flag {mapped['dataset_flag']['flag']} on {mapped['dataset_flag']['dataset']}")
    if mapped.get("imported_trials"):
        maps.append(f"+{mapped['imported_trials']} family trials")
    for kind in ("param", "threshold", "flag", "rule"):
        if mapped.get(kind):
            value = mapped[kind]["value"]
            maps.append(f"{kind} {mapped[kind]['name']}={'default' if value is None else value}")
    return {
        "question_id": r["qid"], "scope": r["scope"], "question": r["text"], "why": r.get("why", ""),
        "critical": bool(r["critical"]), "origin": r.get("origin", "pack"), "status": r["status"],
        "answer": r.get("label", ""), "version": r.get("version"), "maps_to": maps,
        "risk": r.get("risk") or r.get("unknown_risk") or "", "provenance": r.get("provenance", ""),
    }


def interview_chart_docs(snapshot: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    body = snapshot["stages"].get(stage)
    if body is None:
        return []
    rows = [_row_doc(r) for r in body["rows"]]
    pages = [rows[i:i + ROWS_PER_PAGE] for i in range(0, len(rows), ROWS_PER_PAGE)] or [[]]
    digest = hash_json(snapshot)
    return [{
        "schema_version": SCHEMA_VERSION, "key": f"interview_{stage}_{n}", "title": f"{stage} Interview",
        "stage": stage, "trial_id": snapshot["trial_id"], "chart_type": "interview", "page": n, "pages": len(pages),
        "progress": {**body["counts"], "total": body["total"]}, "light": body["light"], "rows": chunk,
        "snapshot_hash": digest, "note": snapshot.get("note", ""), "disclaimer": theme.DISCLAIMER,
    } for n, chunk in enumerate(pages, 1)]


def _wrap(text: str, width: int, lines: int) -> list[str]:
    out = textwrap.wrap(" ".join(str(text).split()), width) or [""]
    if len(out) > lines:
        out = out[:lines]
        out[-1] = textwrap.shorten(out[-1] + " …", width, placeholder="…")
    return out


@theme.themed
def interview_figure(doc: dict[str, Any]) -> Figure:
    stage = doc["stage"]
    footer = f"Trial {doc['trial_id']}  |  Interview {stage}  |  Snapshot {doc['snapshot_hash'][:12]}"
    fig = theme.new_report_page(f"{stage} Interview", f"Page {doc['page']} of {doc['pages']}. " + doc["note"], footer)
    p = doc["progress"]
    status, icon, word = LIGHT[doc["light"]]
    fig.text(0.95, 0.955, f"{icon} {word}", fontsize=12, fontweight="bold", ha="right", va="center",
             color=theme.STATUS[status])
    # progress bar: answered | don't know | pending
    total = max(p["total"], 1)
    ax = fig.add_axes((0.05, 0.835, 0.90, 0.028))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x = 0.0
    for key, color in (("answered", theme.STATUS["good"]), ("unknown", theme.STATUS["warning"]), ("pending", theme.GRID)):
        w = p[key] / total
        if w:
            ax.add_patch(Rectangle((x, 0), w, 1, facecolor=color, edgecolor=theme.AXIS, linewidth=0.6))
        x += w
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=theme.AXIS, linewidth=0.8))
    fig.text(0.05, 0.815, f"{p['answered']} of {p['total']} answered, {p['unknown']} don't know, {p['pending']} pending",
             fontsize=9, va="top", color=theme.TEXT)
    if not doc["rows"]:
        fig.text(0.5, 0.5, "No interview questions apply to this stage.", ha="center", color=theme.TEXT_MUTED)
        return fig
    y = 0.775
    height = 0.083
    for c, x0 in (("Status", 0.05), ("Question", 0.20), ("Answer and what it changed", 0.62)):
        fig.text(x0, y, c, fontsize=9, fontweight="bold", va="center", color=theme.TEXT)
    fig.add_artist(Line2D([0.05, 0.95], [y - 0.012, y - 0.012], color=theme.AXIS, linewidth=0.8))
    y -= 0.03
    for r in doc["rows"]:
        color_key, icon, word = STATUS_STYLE[r["status"]]
        color = theme.STATUS[color_key] if color_key else theme.TEXT_MUTED
        fig.text(0.05, y, f"{icon} {word}", fontsize=8.5, fontweight="bold", va="top", color=color)
        if r["critical"]:
            fig.text(0.05, y - 0.022, "CRITICAL", fontsize=7.5, va="top", color=theme.TEXT_MUTED)
        scope = re.sub(r"(:[0-9a-f]{8})[0-9a-f]{8,}", r"\1", r["scope"])  # a trial id is shown by its first 8 hex
        head = f"{r['question_id']}  ({scope})"
        fig.text(0.20, y, textwrap.shorten(head, 60, placeholder="…"), fontsize=7.5, va="top", color=theme.TEXT_MUTED)
        for i, line in enumerate(_wrap(r["question"], 62, 2)):
            fig.text(0.20, y - 0.018 - 0.017 * i, line, fontsize=8.5, va="top", color=theme.TEXT)
        if r["status"] == "pending":
            right = _wrap("Why: " + (r["why"] or "no reason recorded"), 52, 3)
        elif r["status"] == "unknown":
            right = ["Risk: " + (r["risk"] or "unknown")]
            right = _wrap(right[0], 52, 3)
        else:
            extras = []
            if r["risk"]:
                extras += _wrap(f"{theme.VERDICT_ICON['BLOCKED']} Risk tag: {r['risk']}", 52, 1)
            if r["maps_to"]:
                extras += _wrap("Mapped: " + "; ".join(r["maps_to"]), 52, 1)
            answer = r["answer"] + (f" (v{r['version']}, {r['provenance']})" if r["version"] else "")
            right = _wrap(answer, 52, 3 - len(extras)) + extras
        for i, line in enumerate(right[:3]):
            fig.text(0.62, y - 0.017 * i, line, fontsize=8.5, va="top",
                     color=theme.STATUS["serious"] if line.startswith(theme.VERDICT_ICON["BLOCKED"]) else theme.TEXT)
        y -= height
    return fig


def interview_figures(snapshot: dict[str, Any]) -> list[Figure]:
    from qrh.packs.model import STAGES

    return [interview_figure(d) for st in STAGES for d in interview_chart_docs(snapshot, st)]


def render_interview(home: str | Path, trial_id: str, snapshot: dict[str, Any], stage: str | None = None) -> list[RenderedFile]:
    """Write ``interview.<stage>.pdf`` and ``.chart.json`` for one stage, or every stage that has questions."""
    base = manifest.reports_dir(home, trial_id)
    base.mkdir(parents=True, exist_ok=True)
    out: list[RenderedFile] = []
    for st in [stage] if stage else list(snapshot["stages"]):
        docs = interview_chart_docs(snapshot, st)
        if not docs:
            continue
        key = f"interview.{st}"
        pdf_path, json_path = base / f"{key}.pdf", base / f"{key}.chart.json"
        write_pdf(pdf_path, [interview_figure(d) for d in docs], title=f"{trial_id} {st} interview", keywords=KEYWORDS,
                  subject=f"trial {trial_id} interview {st} snapshot {docs[0]['snapshot_hash']}")
        json_path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "trial_id": trial_id, "stage": st,
                                         "snapshot_hash": docs[0]["snapshot_hash"], "figures": docs},
                                        sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        files = [RenderedFile(key, "pdf", pdf_path, sha256_file(pdf_path), docs[0]["snapshot_hash"]),
                 RenderedFile(key, "chart_data", json_path, sha256_file(json_path), docs[0]["snapshot_hash"])]
        manifest.record_files(home, trial_id, key, [f.entry(base) for f in files])
        out += files
    return out
