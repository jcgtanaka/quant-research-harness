"""The packs that ship with the repository: valid, public-safe where public, and complete."""

from pathlib import Path

import pytest

from qrh.packs.model import load_pack
from qrh.packs.safety import public_safety_findings

PACKS = Path(__file__).resolve().parents[2] / "packs"


@pytest.fixture(scope="module")
def retail():
    return load_pack(PACKS / "retail-core")


def test_retail_core_is_public_and_passes_the_safety_lint(retail):
    assert retail.visibility == "public" and retail.manifest.license == "Apache-2.0"
    assert public_safety_findings(retail) == []


def test_retail_core_lessons_cover_the_generic_set(retail):
    ids = {x.id for x in retail.lessons}
    assert ids == {"base-rate-check", "count-every-variant", "attribute-worst-days", "prefer-point-in-time",
                   "verify-derived-columns", "daily-bar-sample-size", "match-reoptimization-cadence",
                   "capital-sweep", "check-mark-staleness", "llm-memorization-is-lookahead",
                   "count-every-prompt-model-variant", "record-and-replay-llm-calls"}
    assert retail.draft_count == 0 and retail.attempts == [] and retail.account is None


def test_retail_core_critical_questions(retail):
    crit = {q.id: q for q in retail.questions if q.critical}
    assert set(crit) >= {"rf-rate-source", "timestamp-semantics", "expired-contracts", "futures-back-adjusted",
                         "fundamentals-keying", "macro-vintage", "signal-vs-fill", "variants-outside-harness",
                         "reopt-cadence-live"}
    assert crit["rf-rate-source"].stage == "S1" and crit["signal-vs-fill"].stage == "S2"
    assert crit["variants-outside-harness"].stage == "S3" and crit["reopt-cadence-live"].stage == "S4"
    risks = {c.id: c.risk for c in crit["rf-rate-source"].choices}
    assert risks["current_curve"] == "lookahead" and risks["pit_daily"] is None
    assert crit["variants-outside-harness"].maps_to == {"kind": "imported_trials"}
    rec = {q.id: q for q in retail.questions if not q.critical}
    assert {rec["falsification-criterion"].stage, rec["real-fee-schedule"].stage, rec["deployment-intent"].stage} == {"S0", "S6", "S7"}
    assert all(q.unknown_risk and q.why_it_matters for q in retail.questions)


def test_example_trader_is_a_valid_synthetic_demo():
    p = load_pack(PACKS / "example-trader")
    assert p.visibility == "private" and p.account is not None and len(p.attempts) == 2
    assert p.account.caps["max_gross_leverage"] == 1.0
