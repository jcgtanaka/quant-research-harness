"""The shape of a readiness warning, severities, stage order and the static table of known harness limits."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

SEVERITIES = ("high", "medium", "low")
STAGE_ORDER = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "B1", "B2", "B3", "B4", "B5", "B6", "ALL")
GENERAL = "ALL"  # workflow-wide items that belong to no single stage
MAX_TEXT = 400


def warning_id(kind: str, code: str, scope: str) -> str:
    """Stable id: the first 16 hex characters of the SHA-256 of source kind, code and scope."""
    return hashlib.sha256("\0".join((kind, code, scope)).encode("utf-8", "surrogatepass")).hexdigest()[:16]


def clip(text: object, limit: int = MAX_TEXT) -> str:
    """One line of at most ``limit`` characters; a cut is marked with an ellipsis, never silent."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


@dataclass(frozen=True)
class ReadinessWarning:
    id: str
    stage: str
    severity: str
    kind: str  # interview | data | onboarding | lint | exemption | override | method | pack | limit | condition | receipt | llm
    code: str
    scope: str
    message: str
    resolution: str
    receipt_hash: str | None = None
    receipt_path: str | None = None

    def doc(self) -> dict[str, Any]:
        return asdict(self)


def make_warning(*, stage: str, severity: str, kind: str, code: str, scope: str, message: str, resolution: str,
                 receipt_hash: str | None = None, receipt_path: str | None = None) -> ReadinessWarning:
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}")
    return ReadinessWarning(warning_id(kind, code, scope), stage, severity, kind, code, scope, clip(message),
                            clip(resolution), receipt_hash, receipt_path)


# Known limits of the harness that matter to a deployment decision, one line each. The pointer says where the
# evidence is. ``when`` names the condition under which the limit applies to a trial (see collect.py).
KNOWN_LIMITS: tuple[dict[str, str], ...] = (
    {"code": "same-bar-execution", "stage": "S2", "when": "no-reconstruction",
     "text": "Same-bar execution (a fill at the bar that produced the signal) is detected by rebuilding returns from "
             "reported positions; here that rebuild did not conclude and s2.pnl_reconstruction is off.",
     "pointer": "docs/methods.md, 'Closing the same-bar and units gaps'"},
    {"code": "percent-units", "stage": "S2", "when": "no-reconstruction",
     "text": "A percent-instead-of-fraction units bug is caught only by the sanity bounds, and not at low "
             "volatility, unless a returns reconstruction concluded (see the same-bar limit).",
     "pointer": "docs/methods.md, 'Closing the same-bar and units gaps'"},
    {"code": "relationship-corruption", "stage": "S1", "when": "data",
     "text": "A corrupted relationship between columns at or below the rule tolerance (rel_tol 0.01) cannot be "
             "detected by the data rules.",
     "pointer": "docs/methods.md, 'Reading the numbers honestly'"},
    {"code": "bench-scope", "stage": "ALL", "when": "always",
     "text": "Detection rates describe the benchmark corpus; a clean pass does not prove the absence of "
             "look-ahead or leakage the corpus does not contain.",
     "pointer": "docs/methods.md, 'Reading the numbers honestly'"},
    {"code": "llm-cutoff-claim", "stage": "S2", "when": "llm",
     "text": "A model's documented cutoff can be wrong: the cutoff gate compares dates a provider claims, it "
             "cannot measure what a model knows.",
     "pointer": "docs/methods.md, 'Rules for strategies that use an LLM' (docs/llm.md)"},
    {"code": "llm-post-cutoff-knowledge", "stage": "S2", "when": "llm",
     "text": "Models may know events after their stated cutoff (late data, fine-tuning, retrieval), and nothing in "
             "this harness can detect that.",
     "pointer": "docs/methods.md, 'Rules for strategies that use an LLM' (docs/llm.md)"},
)
