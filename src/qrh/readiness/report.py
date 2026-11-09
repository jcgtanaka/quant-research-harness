"""The readiness report: data JSON, chart.json and a PDF in the report theme.

Page 1: trial, final verdict (PASS · CONDITIONAL spelled out), a stage x severity count table and the acknowledgement
status. Following pages: warnings grouped by stage, each with severity, message, short receipt hash and the way to
resolve it. Status is carried by a marker and a word, never by color alone. Every page footer has the trial id, the
report hash and the research-only disclaimer. Output is deterministic (no timestamps); rendering never changes a
verdict, and a failure is recorded in the report manifest.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from qrh.config import ensure_home
from qrh.hashing import sha256_file
from qrh.pipeline.report import STAGE_TITLES
from qrh.readiness import collect as rc
from qrh.readiness.data import SCHEMA_VERSION, ReportData, build_data
from qrh.readiness.model import GENERAL, SEVERITIES, STAGE_ORDER
from qrh.reports import manifest, theme
from qrh.reports.pdf import write_pdf
from qrh.reports.render import KEYWORDS, RenderedFile

if TYPE_CHECKING:
    from qrh.memory.registry import Registry

PAGE_UNITS = 44  # vertical budget of a warnings page, in text lines
UNIT = 0.0175  # figure-height fraction of one line
TOP = 0.865
WRAP = 118
MSG_LINES = 4
RES_LINES = 4
SEV_STYLE = {"high": ("▲", "HIGH", "critical"), "medium": ("◆", "MEDIUM", "warning"), "low": ("●", "LOW", None)}
STAGE_NAMES = {**{s.value: t for s, t in STAGE_TITLES.items()}, "S7": "Final report", GENERAL: "General (whole workflow)"}


def _stage_name(stage: str) -> str:
    return f"{stage} {STAGE_NAMES[stage]}" if stage in STAGE_NAMES and stage != GENERAL else STAGE_NAMES.get(stage, stage)


def _lines(text: str, width: int, cap: int) -> list[str]:
    out = textwrap.wrap(" ".join(str(text).split()), width) or [""]
    if len(out) > cap:
        out = out[:cap]
        out[-1] = textwrap.shorten(out[-1] + " …", width, placeholder="…")
    return out


def _height(w: dict[str, Any]) -> int:
    return 1 + len(_lines(w["message"], WRAP, MSG_LINES)) + len(_lines("Resolve: " + w["resolution"], WRAP, RES_LINES)) + 1


def chart_docs(data: ReportData, ack: tuple[bool, str]) -> list[dict[str, Any]]:
    """Page documents: the summary first, then the warning pages (the figures are drawn from these only)."""
    d = data.doc
    base = {"schema_version": SCHEMA_VERSION, "trial_id": d["trial_id"], "report_hash": data.hash,
            "disclaimer": theme.DISCLAIMER}
    summary = {**base, "key": "readiness_summary", "title": "Deployment readiness", "chart_type": "readiness_summary",
               "verdict": d["final_verdict"], "verdict_display": d["final_verdict_display"],
               "failing_stage": d["failing_stage"], "counts": d["counts"], "totals": d["totals"],
               "ack": {"acknowledged": bool(ack[0]), "detail": ack[1]}}
    pages: list[list[dict[str, Any]]] = [[]]
    used = [0]
    for stage in [s for s in STAGE_ORDER if any(w["stage"] == s for w in d["warnings"])]:
        group = [w for w in d["warnings"] if w["stage"] == stage]
        header = {"type": "header", "stage": stage, "title": _stage_name(stage), "count": len(group)}
        first = True
        for w in group:
            need = _height(w) + (2 if first else 0)
            if used[-1] + need > PAGE_UNITS and pages[-1]:
                pages.append([])
                used.append(0)
                first = True
                need = _height(w) + 2
            if first:
                pages[-1].append({**header, "continued": any(e["type"] == "warning" and e["stage"] == stage
                                                             for p in pages for e in p)})
                used[-1] += 2
                first = False
            pages[-1].append({"type": "warning", "stage": stage, "warning": w})
            used[-1] += _height(w)
    pages = [p for p in pages if p]
    docs = [summary]
    if d.get("two_track") is not None:  # information page; never a warning
        docs.append({**base, "key": "readiness_two_track", "title": "Track 2 answers (information)",
                     "chart_type": "readiness_two_track", "overall": d["two_track"]["overall"],
                     "error": d["two_track"]["error"], "results": d["two_track"]["results"]})
    for n, entries in enumerate(pages, 1):
        docs.append({**base, "key": f"readiness_warnings_{n}", "title": "Warnings by stage",
                     "chart_type": "readiness_warnings", "page": n, "pages": len(pages), "entries": entries,
                     "used": sum(2 if e["type"] == "header" else _height(e["warning"]) for e in entries)})
    return docs


def _footer(doc: dict[str, Any]) -> str:
    return f"Trial {doc['trial_id']}  |  Readiness report {doc['report_hash'][:12]}"


def _color(key: str | None) -> str:
    return theme.STATUS[key] if key else theme.TEXT_MUTED


def _summary_figure(doc: dict[str, Any]) -> Figure:
    fig = theme.new_report_page("Deployment readiness", f"Trial {doc['trial_id']}", _footer(doc))
    verdict, shown = doc["verdict"], doc["verdict_display"]
    status = theme.VERDICT_STATUS.get(verdict)
    label = f"{theme.VERDICT_ICON.get(verdict, '•')} {shown}"
    fig.text(0.05, 0.85, f"Final verdict: {label}", fontsize=14, fontweight="bold", va="center",
             color=theme.STATUS[status] if status else theme.TEXT)
    note = {"PASS": "Every gate passed.", "PASS_CONDITIONAL": "Every gate passed, but open conditions remain; "
            "they are listed as HIGH warnings."}.get(verdict, f"Not a pass ({shown}): this strategy is not deployable as it stands.")
    fig.text(0.05, 0.815, note, fontsize=9.5, va="center", color=theme.TEXT_MUTED)
    fig.text(0.05, 0.785, f"Report hash {doc['report_hash']}", fontsize=8, va="center", color=theme.TEXT_MUTED)
    cols = (("Stage", 0.05), ("▲ HIGH", 0.46), ("◆ MEDIUM", 0.58), ("● LOW", 0.72), ("Total", 0.84))
    y = 0.735
    for name, x in cols:
        fig.text(x, y, name, fontsize=9, fontweight="bold", va="center", color=theme.TEXT)
    fig.add_artist(Line2D([0.05, 0.95], [y - 0.017, y - 0.017], color=theme.AXIS, linewidth=0.8))
    y -= 0.04
    counts = doc["counts"]
    for stage in [s for s in STAGE_ORDER if s in counts]:
        row = counts[stage]
        fig.text(0.05, y, _stage_name(stage), fontsize=8.5, va="center", color=theme.TEXT)
        for sev, x in zip(SEVERITIES, (0.46, 0.58, 0.72)):
            fig.text(x, y, str(row[sev]), fontsize=8.5, va="center", color=theme.TEXT,
                     fontweight="bold" if row[sev] and sev == "high" else "normal")
        fig.text(0.84, y, str(sum(row.values())), fontsize=8.5, va="center", color=theme.TEXT)
        y -= 0.034
    fig.add_artist(Line2D([0.05, 0.95], [y + 0.017, y + 0.017], color=theme.AXIS, linewidth=0.8))
    t = doc["totals"]
    fig.text(0.05, y - 0.004, "All stages", fontsize=8.5, fontweight="bold", va="center", color=theme.TEXT)
    for sev, x in zip(SEVERITIES, (0.46, 0.58, 0.72)):
        fig.text(x, y - 0.004, str(t[sev]), fontsize=8.5, fontweight="bold", va="center", color=theme.TEXT)
    fig.text(0.84, y - 0.004, str(sum(t.values())), fontsize=8.5, fontweight="bold", va="center", color=theme.TEXT)
    ack = doc["ack"]
    y -= 0.065
    mark, word, key = ("✓", "ACKNOWLEDGED", "good") if ack["acknowledged"] else ("○", "PENDING", "warning")
    fig.text(0.05, y, f"Acknowledgement: {mark} {word}", fontsize=11, fontweight="bold", va="center",
             color=theme.STATUS[key])
    for i, line in enumerate(_lines(ack["detail"], 140, 3)):
        fig.text(0.05, y - 0.03 - 0.019 * i, line, fontsize=8.5, va="center", color=theme.TEXT)
    fig.text(0.05, 0.075, "Only a person can acknowledge: `qrh readiness ack <trial>` in your own terminal, "
             "with the approval passphrase. Any change in the warnings invalidates it.", fontsize=8, va="center",
             color=theme.TEXT_MUTED)
    return fig


def _warnings_figure(doc: dict[str, Any]) -> Figure:
    fig = theme.new_report_page(
        "Warnings by stage",
        f"Page {doc['page']} of {doc['pages']}. Severity is a marker and a word: ▲ HIGH, ◆ MEDIUM, ● LOW.", _footer(doc))
    y = TOP
    for e in doc["entries"]:
        if e["type"] == "header":
            cont = " (continued)" if e.get("continued") else ""
            fig.text(0.05, y, f"{e['title']}{cont}  ({e['count']})", fontsize=10.5, fontweight="bold", va="top",
                     color=theme.TEXT)
            fig.add_artist(Line2D([0.05, 0.95], [y - UNIT * 1.25, y - UNIT * 1.25], color=theme.AXIS, linewidth=0.7))
            y -= 2 * UNIT
            continue
        w = e["warning"]
        mark, word, key = SEV_STYLE[w["severity"]]
        fig.text(0.05, y, f"{mark} {word}", fontsize=8.5, fontweight="bold", va="top", color=_color(key))
        meta = f"{w['stage']} · {w['kind']} · {w['code']}   id {w['id'][:8]}   receipt {(w['receipt_hash'] or '-')[:12]}"
        fig.text(0.16, y, meta, fontsize=7.5, va="top", color=theme.TEXT_MUTED)
        y -= UNIT
        for line in _lines(w["message"], WRAP, MSG_LINES):
            fig.text(0.05, y, line, fontsize=8.5, va="top", color=theme.TEXT)
            y -= UNIT
        for line in _lines("Resolve: " + w["resolution"], WRAP, RES_LINES):
            fig.text(0.05, y, line, fontsize=8, va="top", color=theme.TEXT_MUTED)
            y -= UNIT
        y -= UNIT
    return fig


def _two_track_figure(doc: dict[str, Any]) -> Figure:
    fig = theme.new_report_page(
        "Track 2 answers (information)",
        "Information only: these answers never change the verdict and are not counted as warnings.", _footer(doc))
    y = TOP
    if doc["error"]:
        for line in _lines(doc["error"], WRAP, 4):
            fig.text(0.05, y, line, fontsize=8.5, va="top", color=theme.TEXT)
            y -= UNIT
        return fig
    fig.text(0.05, y, f"Overall: {doc['overall']}", fontsize=10.5, fontweight="bold", va="top", color=theme.TEXT)
    y -= 2 * UNIT
    for r in doc["results"]:
        mark = "✓" if r["status"] == "PASS" else "▲" if r["status"] == "FAIL" else "◆"
        fig.text(0.05, y, f"{mark} {r['status']}", fontsize=8.5, fontweight="bold", va="top",
                 color=_color({"PASS": "good", "WARNING": "warning", "FAIL": "critical"}.get(r["status"])))
        for i, line in enumerate(_lines(r["reason"], 100, 2)):
            fig.text(0.16, y, line, fontsize=8.5, va="top", color=theme.TEXT)
            y -= UNIT
        y -= UNIT * 0.5
    return fig


@theme.themed
def readiness_figure(doc: dict[str, Any]) -> Figure:
    if doc["chart_type"] == "readiness_summary":
        return _summary_figure(doc)
    return _two_track_figure(doc) if doc["chart_type"] == "readiness_two_track" else _warnings_figure(doc)


def readiness_figures(docs: list[dict[str, Any]]) -> list[Figure]:
    return [readiness_figure(d) for d in docs]


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def render_readiness(trial_id: str, home: str | Path | None = None, registry: "Registry | None" = None) -> list[RenderedFile]:
    """Write ``readiness.json``, ``readiness.chart.json`` and ``readiness.pdf`` under the trial's reports folder."""
    from qrh.readiness import ack as rack

    home_dir = Path(home) if home is not None else ensure_home()
    try:
        collected = rc.collect(home_dir, trial_id, registry)
        data = build_data(collected)
        docs = chart_docs(data, rack.evaluate(home_dir, data))
        figs = readiness_figures(docs)
        base = manifest.reports_dir(home_dir, trial_id)
        base.mkdir(parents=True, exist_ok=True)
        pdf, chart, raw = base / "readiness.pdf", base / "readiness.chart.json", base / "readiness.json"
        write_pdf(pdf, figs, title=f"{trial_id} readiness report", keywords=KEYWORDS,
                  subject=f"trial {trial_id} readiness report {data.hash}")
        _write(chart, json.dumps({"schema_version": SCHEMA_VERSION, "trial_id": trial_id, "stage": "readiness",
                                  "report_hash": data.hash, "figures": docs}, sort_keys=True, indent=2,
                                 allow_nan=False) + "\n")
        _write(raw, json.dumps({**data.doc, "report_hash": data.hash}, sort_keys=True, indent=2, allow_nan=False) + "\n")
        s7 = collected.receipt_hashes.get("S7", "")
        files = [RenderedFile(manifest.READINESS, kind, p, sha256_file(p), s7)
                 for kind, p in (("pdf", pdf), ("chart_data", chart), ("data", raw))]
        manifest.record_files(home_dir, trial_id, manifest.READINESS, [f.entry(base) for f in files])
        return files
    except rc.ReadinessError as exc:
        _record(home_dir, trial_id, exc)
        raise
    except Exception as exc:
        _record(home_dir, trial_id, exc)
        raise rc.ReadinessError(f"readiness report failed: {type(exc).__name__}: {exc}") from exc


def _record(home: Path, trial_id: str, exc: Exception) -> None:
    try:
        manifest.record_error(home, trial_id, manifest.READINESS, f"{type(exc).__name__}: {exc}")
    except Exception:  # an unwritable reports directory must not hide the original error
        pass
