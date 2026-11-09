"""Rule 1, the training-cutoff gate (method ``s2.llm_gate@1``).

A language model trained on data up to date D already knows what happened before D. No truncation test can catch
that, because the leak lives in the model weights. So the gate looks at the declaration instead: every evaluation
window must start strictly after the LATEST declared training cutoff plus a safety margin
(``llm_cutoff_margin_days``, default 180). An unknown cutoff, or a cutoff whose source is unknown, BLOCKS: a number
nobody knows cannot be compared with anything. The cutoff is a claim by the model's provider, not a measurement
(see docs/llm.md, limits).
"""

from __future__ import annotations

import datetime
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from qrh.llm.spec import UNKNOWN

THRESHOLD = "llm_cutoff_margin_days"


@dataclass
class GateResult:
    verdict: str  # PASS | FAIL | BLOCKED
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


def _utc_date(text: str) -> datetime.date:
    moment = datetime.datetime.fromisoformat(text)
    if moment.tzinfo is not None:
        moment = moment.astimezone(datetime.timezone.utc)
    return moment.date()


def _margin(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def evaluate_gate(llm: Mapping[str, Any], windows: Mapping[str, tuple[str, str]], margin_days: Any) -> GateResult:
    """Decide the gate for a declared ``llm`` section (the stored, normalized dict) and the hypothesis windows."""
    models = [dict(m) for m in llm["models"]]
    blocked: list[str] = []
    failed: list[str] = []
    margin = _margin(margin_days)
    if margin is None:
        blocked.append(f"threshold {THRESHOLD} is missing or not a non-negative number ({margin_days!r})")
    for m in models:
        if m["training_cutoff"] == UNKNOWN or m["cutoff_source"] == UNKNOWN:
            what = ("training cutoff is unknown" if m["training_cutoff"] == UNKNOWN
                    else "cutoff_source is unknown (nobody can say where the cutoff came from)")
            blocked.append(
                f"model {m['id']!r}: {what}; declare a real cutoff and its source, or do not backtest with this model")
    known = [m for m in models if m["training_cutoff"] != UNKNOWN]
    binding = max(known, key=lambda m: (m["training_cutoff"], m["id"])) if known else None
    binding_date = (
        datetime.date.fromisoformat(binding["training_cutoff"]) + datetime.timedelta(days=margin)
        if binding is not None and margin is not None else None
    )
    rows: list[dict[str, Any]] = []
    for name in sorted(windows, key=lambda n: (n != "development", n)):
        start, end = windows[name]
        first = _utc_date(start)
        row: dict[str, Any] = {"name": name, "start": start, "end": end, "start_date": first.isoformat(),
                               "status": "PASS", "earliest_contaminated_date": None}
        if binding_date is not None and not first > binding_date:
            row["status"], row["earliest_contaminated_date"] = "FAIL", first.isoformat()
            failed.append(
                f"window {name!r} starts {first.isoformat()}, not after {binding_date.isoformat()} "
                f"(model {binding['id']!r} training cutoff {binding['training_cutoff']} + margin {margin:g} days); "
                f"earliest contaminated date {first.isoformat()}")
        rows.append(row)
    retrieval = llm["live_retrieval"]
    used = retrieval is not False
    evidence_text = (retrieval or {}).get("point_in_time_evidence") if used else None
    if used and not (isinstance(evidence_text, str) and evidence_text.strip()):
        blocked.append("live_retrieval is declared without point_in_time_evidence text: say how the data a tool "
                       "returns is limited to what existed at each decision time")
        evidence_text = None
    verdict = "BLOCKED" if blocked else "FAIL" if failed else "PASS"
    reasons = blocked + failed
    evidence: dict[str, Any] = {
        "verdict": verdict,
        "margin_days": None if margin is None else (int(margin) if margin == int(margin) else margin),
        "models": [{k: m[k] for k in ("id", "provider", "training_cutoff", "cutoff_source")} for m in models],
        "binding_model": binding["id"] if binding else None,
        "binding_cutoff": binding["training_cutoff"] if binding else None,
        "binding_date": binding_date.isoformat() if binding_date else None,
        "windows": rows,
        "live_retrieval": {"used": used, "evidence": evidence_text.strip() if evidence_text else None},
        "reasons": reasons,
        "note": ("Spans derived from a window (walk-forward folds, out-of-sample stitches) lie inside it, so a window "
                 "that starts after the binding date covers them. A training cutoff is a provider claim."),
    }
    return GateResult(verdict, reasons, evidence)
