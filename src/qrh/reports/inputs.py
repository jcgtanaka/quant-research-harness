"""Verified inputs for reports: receipts and the artifacts they record, nothing else."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qrh.pipeline.artifacts import ARTIFACTS_KEY, ArtifactError, read_artifact
from qrh.pipeline.receipt import Receipt, Stage, load_receipt, receipt_path

if TYPE_CHECKING:
    from qrh.memory.registry import Registry


class ReportError(ValueError):
    """A report could not be produced. Reports fail closed: nothing is rendered from data that does not verify."""


class ReportInputError(ReportError):
    """A receipt or artifact is missing, unreadable or does not match its recorded hash."""


@dataclass
class StageData:
    """One stage's verified receipt and artifacts (S7 also carries every stage receipt)."""

    trial_id: str
    stage: Stage
    receipt: Receipt
    artifacts: dict[str, Any] = field(default_factory=dict)
    receipts: dict[Stage, Receipt] = field(default_factory=dict)

    @property
    def receipt_hash(self) -> str:
        return self.receipt.receipt_hash()


def _load_receipt(home: Path, trial_id: str, stage: Stage, registry: Registry | None) -> Receipt:
    path = receipt_path(home, trial_id, stage)
    if not path.is_file():
        raise ReportInputError(f"{stage.value}: no receipt for trial {trial_id!r}")
    try:
        receipt = load_receipt(path)
    except Exception as exc:
        raise ReportInputError(f"{stage.value}: receipt cannot be verified: {exc}") from exc
    if receipt.trial_id != trial_id or receipt.stage is not stage:
        raise ReportInputError(f"{stage.value}: receipt belongs to {receipt.trial_id}/{receipt.stage.value}")
    if registry is not None:
        entry = registry.get_receipt_entry(trial_id, stage.value)
        if entry is not None and entry["receipt_hash"] != receipt.receipt_hash():
            raise ReportInputError(f"{stage.value}: receipt does not match the registry index")
    return receipt


def available_stages(home: str | Path, trial_id: str) -> list[Stage]:
    """Stages that have a receipt file on disk, in order."""
    return [s for s in Stage if receipt_path(home, trial_id, s).is_file()]


def load_stage_data(
    home: str | Path, trial_id: str, stage: Stage | str, registry: Registry | None = None
) -> StageData:
    """Load and verify a stage's receipt and every artifact it records; any mismatch raises."""
    home, stage = Path(home), Stage(stage)
    receipt = _load_receipt(home, trial_id, stage, registry)
    artifacts: dict[str, Any] = {}
    try:
        for name in sorted(receipt.params.get(ARTIFACTS_KEY) or {}):
            artifacts[name] = read_artifact(home, receipt, name)
    except ArtifactError as exc:
        raise ReportInputError(str(exc)) from exc
    receipts: dict[Stage, Receipt] = {}
    if stage is Stage.S7:
        for s in available_stages(home, trial_id):
            receipts[s] = receipt if s is stage else _load_receipt(home, trial_id, s, registry)
    return StageData(trial_id, stage, receipt, artifacts, receipts)
