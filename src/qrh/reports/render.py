"""Render reports from persisted receipts and artifacts.

``render_stage`` writes ``<home>/reports/<trial_id>/<stage>.pdf`` and ``<stage>.chart.json``;
``render_trial`` renders every stage that has a receipt plus the combined PDF
``<trial_id>.pdf``. Both fail closed (ReportError) when a receipt or artifact does not
verify, and record the failure in the manifest. Rendering never changes a verdict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from matplotlib.figure import Figure

from qrh.config import ensure_home
from qrh.hashing import sha256_file
from qrh.pipeline.receipt import Receipt, Stage
from qrh.reports import manifest
from qrh.reports.catalog import figure_specs
from qrh.reports.chartdata import SCHEMA_VERSION
from qrh.reports.inputs import ReportError, ReportInputError, StageData, available_stages, load_stage_data
from qrh.reports.pdf import write_pdf

if TYPE_CHECKING:
    from qrh.memory.registry import Registry

KEYWORDS = "synthetic research use only, not investment advice"


class ReportRenderError(ReportError):
    """Drawing or writing a report failed for a reason other than its inputs."""


@dataclass(frozen=True)
class RenderedFile:
    stage: str
    kind: str  # pdf | chart_data
    path: Path
    sha256: str
    receipt_hash: str

    def entry(self, base: Path) -> dict[str, str]:
        return {
            "stage": self.stage, "kind": self.kind, "path": self.path.relative_to(base).as_posix(),
            "sha256": self.sha256, "receipt_hash": self.receipt_hash,
        }


def _home(home: str | Path | None) -> Path:
    return Path(home) if home is not None else ensure_home()


def _build(data: StageData) -> tuple[list[Figure], list[dict[str, Any]]]:
    try:
        specs = figure_specs(data)
        return [s.chart() for s in specs], [s.data() for s in specs]
    except ReportError:
        raise
    except Exception as exc:
        raise ReportRenderError(f"{data.stage.value}: figure failed: {type(exc).__name__}: {exc}") from exc


def _write_stage(home: Path, data: StageData, figs: list[Figure], docs: list[dict[str, Any]]) -> list[RenderedFile]:
    base = manifest.reports_dir(home, data.trial_id)
    base.mkdir(parents=True, exist_ok=True)
    stage = data.stage.value
    pdf_path, json_path = base / f"{stage}.pdf", base / f"{stage}.chart.json"
    write_pdf(
        pdf_path, figs, title=f"{data.trial_id} {stage} report", keywords=KEYWORDS,
        subject=f"trial {data.trial_id} stage {stage} receipt {data.receipt_hash}",
    )
    doc = {
        "schema_version": SCHEMA_VERSION, "trial_id": data.trial_id, "stage": stage,
        "receipt_hash": data.receipt_hash, "figures": docs,
    }
    json_path.write_text(json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return [
        RenderedFile(stage, "pdf", pdf_path, sha256_file(pdf_path), data.receipt_hash),
        RenderedFile(stage, "chart_data", json_path, sha256_file(json_path), data.receipt_hash),
    ]


def _write_combined(home: Path, trial_id: str, final: StageData, figs: list[Figure]) -> RenderedFile:
    path = manifest.reports_dir(home, trial_id) / f"{trial_id}.pdf"
    write_pdf(
        path, figs, title=f"{trial_id} research report", keywords=KEYWORDS,
        subject=f"trial {trial_id} combined report, S7 receipt {final.receipt_hash}",
    )
    return RenderedFile(manifest.COMBINED, "pdf", path, sha256_file(path), final.receipt_hash)


def _record(home: Path, trial_id: str, stage_key: str, exc: Exception) -> None:
    try:
        manifest.record_error(home, trial_id, stage_key, f"{type(exc).__name__}: {exc}")
    except Exception:  # an unwritable reports directory must not hide the original error
        pass


def render_stage(
    trial_id: str, stage: Stage | str, home: str | Path | None = None, registry: Registry | None = None
) -> list[RenderedFile]:
    """Render one stage from its verified receipt and artifacts; raises ReportError (recorded in the manifest)."""
    home_dir, stage = _home(home), Stage(stage)
    try:
        data = load_stage_data(home_dir, trial_id, stage, registry)
        figs, docs = _build(data)
        files = _write_stage(home_dir, data, figs, docs)
        manifest.record_files(home_dir, trial_id, stage.value, [f.entry(manifest.reports_dir(home_dir, trial_id)) for f in files])
        return files
    except ReportError as exc:
        _record(home_dir, trial_id, stage.value, exc)
        raise
    except Exception as exc:
        _record(home_dir, trial_id, stage.value, exc)
        raise ReportRenderError(f"{stage.value}: {type(exc).__name__}: {exc}") from exc


def _combined_figures(per_stage: dict[Stage, list[Figure]], final: StageData | None = None) -> list[Figure]:
    """Every stage's pages in order, then the interview pages recorded (hash-verified) in the S7 receipt."""
    figs = [fig for s in sorted(per_stage, key=lambda s: s.value) for fig in per_stage[s]]
    snapshot = (final.artifacts.get("interview") if final is not None else None)
    if snapshot:
        from qrh.reports.interview import interview_figures

        figs += interview_figures(snapshot)
    return figs


def render_combined(
    trial_id: str, home: str | Path | None = None, registry: Registry | None = None
) -> RenderedFile:
    """Build ``<trial_id>.pdf`` with every stage's pages in order (needs the S7 receipt)."""
    home_dir = _home(home)
    try:
        per_stage: dict[Stage, list[Figure]] = {}
        final: StageData | None = None
        for stage in available_stages(home_dir, trial_id):
            data = load_stage_data(home_dir, trial_id, stage, registry)
            per_stage[stage] = _build(data)[0]
            final = data if stage is Stage.S7 else final
        if final is None:
            raise ReportInputError("S7: no receipt for the combined report")
        file = _write_combined(home_dir, trial_id, final, _combined_figures(per_stage, final))
        manifest.record_files(home_dir, trial_id, manifest.COMBINED, [file.entry(manifest.reports_dir(home_dir, trial_id))])
        return file
    except ReportError as exc:
        _record(home_dir, trial_id, manifest.COMBINED, exc)
        raise
    except Exception as exc:
        _record(home_dir, trial_id, manifest.COMBINED, exc)
        raise ReportRenderError(f"combined: {type(exc).__name__}: {exc}") from exc


def render_trial(
    trial_id: str, home: str | Path | None = None, registry: Registry | None = None
) -> list[RenderedFile]:
    """Render every stage with a receipt, then the combined PDF if S7 exists.

    Every stage is attempted; if any fails the others are still written, the combined PDF is
    withheld, and a ReportError listing the failures is raised.
    """
    home_dir = _home(home)
    base = manifest.reports_dir(home_dir, trial_id)
    stages = available_stages(home_dir, trial_id)
    if not stages:
        raise ReportInputError(f"no receipts found for trial {trial_id!r}")
    out: list[RenderedFile] = []
    per_stage: dict[Stage, list[Figure]] = {}
    finals: StageData | None = None
    failures: list[str] = []
    for stage in stages:
        try:
            data = load_stage_data(home_dir, trial_id, stage, registry)
            figs, docs = _build(data)
            files = _write_stage(home_dir, data, figs, docs)
            manifest.record_files(home_dir, trial_id, stage.value, [f.entry(base) for f in files])
            out += files
            per_stage[stage] = figs
            finals = data if stage is Stage.S7 else finals
        except Exception as exc:
            _record(home_dir, trial_id, stage.value, exc)
            failures.append(f"{stage.value}: {exc}")
    if not failures and finals is not None:
        try:
            file = _write_combined(home_dir, trial_id, finals, _combined_figures(per_stage, finals))
            manifest.record_files(home_dir, trial_id, manifest.COMBINED, [file.entry(base)])
            out.append(file)
        except Exception as exc:
            _record(home_dir, trial_id, manifest.COMBINED, exc)
            failures.append(f"combined: {exc}")
    elif failures:
        _record(home_dir, trial_id, manifest.COMBINED, ReportError("not built because a stage report failed"))
    if finals is not None and registry is not None and registry.get_trial(trial_id) is not None:
        try:  # the deployment readiness report needs no rendered pages, only the verified receipts and the registry
            from qrh.readiness import report as readiness_report

            out += readiness_report.render_readiness(trial_id, home_dir, registry)
        except Exception as exc:
            failures.append(f"readiness: {exc}")
    if failures:
        raise ReportError("; ".join(failures))
    return out


def render_after_stage(home: str | Path, registry: Registry | None, receipt: Receipt) -> None:
    """Pipeline hook: render this stage (and the combined PDF after S7). Never raises, never touches a verdict."""
    try:
        render_stage(receipt.trial_id, receipt.stage, home, registry)
        if receipt.stage is Stage.S7:
            render_combined(receipt.trial_id, home, registry)
    except Exception:
        pass  # already recorded in the manifest where possible
    if receipt.stage is Stage.S7:  # the deployment readiness report gathers every warning of the finished run
        try:
            from qrh.readiness import report as readiness_report

            readiness_report.render_readiness(receipt.trial_id, home, registry)
        except Exception as exc:
            _record(Path(home), receipt.trial_id, manifest.READINESS, exc)
