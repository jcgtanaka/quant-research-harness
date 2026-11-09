"""Gates: pure decisions about whether work may proceed and what the overall verdict is."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from pydantic import ValidationError

from qrh.memory.registry import HypothesisRecord
from qrh.pipeline.receipt import Stage, Verdict, load_receipt

if TYPE_CHECKING:
    from qrh.memory.registry import Registry

_TEXT_FIELDS = ("text", "strategy_family", "asset_class", "frequency", "difference_statement")


def check_hypothesis_spec(raw: Any) -> tuple[HypothesisRecord | None, list[str]]:
    """Validate a hypothesis spec. Returns (record, []) or (None, reasons)."""
    if not isinstance(raw, dict):
        return None, ["hypothesis spec must be a mapping of fields"]
    reasons: list[str] = []
    rec: HypothesisRecord | None = None
    try:
        rec = HypothesisRecord.model_validate(raw)
    except ValidationError as exc:
        reasons = [f"{'.'.join(str(p) for p in e['loc']) or 'spec'}: {e['msg']}" for e in exc.errors()]
    # Semantic checks run on the raw input too, so every problem is reported at once.
    for f in _TEXT_FIELDS:
        v = raw.get(f)
        if isinstance(v, str) and not v.strip() and not any(r.startswith(f"{f}:") for r in reasons):
            reasons.append(f"{f}: must not be blank")
    if isinstance(raw.get("thresholds"), dict) and not raw["thresholds"]:
        reasons.append("thresholds: must be non-empty")
    if reasons or rec is None:
        return None, reasons
    return rec, []


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reason: str | None = None
    parent_hash: str | None = None


def previous_stage(stage: Stage, order: Sequence[Stage]) -> Stage | None:
    i = list(order).index(stage)
    return order[i - 1] if i > 0 else None


def require_previous_pass(
    registry: Registry, trial_id: str, stage: Stage, order: Sequence[Stage]
) -> GateResult:
    """Stage N may run only if the stored, hash-verified receipt of N-1 is PASS."""
    prev = previous_stage(stage, order)
    if prev is None:
        return GateResult(True)
    entry = registry.get_receipt_entry(trial_id, prev.value)
    if entry is None:
        return GateResult(False, f"no receipt for previous stage {prev.value}")
    try:
        receipt = load_receipt(entry["path"])
    except Exception as exc:
        return GateResult(False, f"receipt for {prev.value} cannot be verified: {exc}")
    if receipt.receipt_hash() != entry["receipt_hash"]:
        return GateResult(False, f"receipt for {prev.value} does not match the registry index")
    if receipt.verdict is not Verdict.PASS:
        return GateResult(False, f"previous stage {prev.value} verdict is {receipt.verdict.value}, not PASS")
    return GateResult(True, parent_hash=entry["receipt_hash"])


def overall_verdict(
    verdicts: dict[Stage, Verdict], expected: Sequence[Stage]
) -> tuple[Verdict, list[str]]:
    """PASS only if every expected stage PASSed; otherwise BLOCKED > FAIL > INCONCLUSIVE."""
    reasons: list[str] = []
    missing = [s for s in expected if s not in verdicts]
    reasons.extend(f"{s.value}: not run" for s in missing)
    reasons.extend(f"{s.value}: {verdicts[s].value}" for s in expected if s in verdicts and verdicts[s] is not Verdict.PASS)
    present = {verdicts[s] for s in expected if s in verdicts}
    if Verdict.BLOCKED in present:
        return Verdict.BLOCKED, reasons
    if Verdict.FAIL in present:
        return Verdict.FAIL, reasons
    if Verdict.INCONCLUSIVE in present:
        return Verdict.INCONCLUSIVE, reasons
    if missing:
        return Verdict.BLOCKED, reasons
    return Verdict.PASS, reasons
