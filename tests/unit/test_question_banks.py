"""retail-core question banks by data type: shape, critical set, and applicability through the real item selection."""

from pathlib import Path

import pytest

from qrh.interview.engine import items_for_trial
from qrh.memory import db
from qrh.packs.model import load_pack
from qrh.packs.runtime import PackView
from qrh.packs.safety import public_safety_findings

PACKS = Path(__file__).resolve().parents[2] / "packs"
TRIAL = {"trial_id": "t1", "family_id": "f", "text": "x", "difference_statement": "y", "datasets": [{"name": "prices"}],
         "asset_class": "synthetic", "strategy_family": "trend", "frequency": "1D"}

BANKS = {
    "options": ["option-dividend-yield-source", "option-greeks-origin", "option-greeks-underlying-price",
                "option-price-type", "option-quote-underlying-sync"],
    "futures": ["futures-continuous-construction", "futures-roll-rule", "futures-settlement-vs-last"],
    "equity": ["equity-survivorship-free-universe", "equity-delisted-included", "equity-adjustment-timing",
               "equity-corporate-action-source"],
    "fx": ["fx-quote-side", "fx-rollover-cutoff-timezone", "fx-swap-carry-source"],
    "crypto": ["crypto-venue", "crypto-perp-funding", "crypto-index-vs-last", "crypto-fee-tier"],
}
COMMON = ["data-history-revised", "data-timezone", "data-licensing"]
NEW = [q for ids in BANKS.values() for q in ids] + COMMON
CRITICAL_NEW = {"option-dividend-yield-source", "option-greeks-origin", "option-greeks-underlying-price",
                "option-quote-underlying-sync", "futures-roll-rule", "futures-settlement-vs-last",
                "equity-survivorship-free-universe", "equity-delisted-included", "equity-adjustment-timing",
                "fx-rollover-cutoff-timezone", "data-history-revised", "data-timezone"}
# every spelling a hypothesis or the onboarding vocabulary may use for the class
CLASSES = {
    "options": ["option", "options"],
    "futures": ["future", "futures"],
    "equity": ["equity", "equities", "etf"],
    "fx": ["fx", "fx_spot"],
    "crypto": ["crypto", "crypto_spot", "crypto_perp"],
}


@pytest.fixture(scope="module")
def retail():
    return load_pack(PACKS / "retail-core")


def selected(tmp_path, asset_class):
    """Question ids the real selection offers for one dataset of a hypothesis on this asset class."""
    conn = db.connect(tmp_path / "r.sqlite3")
    view = PackView([load_pack(PACKS / "retail-core")])
    out = [i for i in items_for_trial(conn, view, {**TRIAL, "asset_class": asset_class}) if i.scope_key == "dataset:prices"]
    conn.close()
    return [i.question.id for i in out]


def test_the_new_banks_exist_with_unique_ids_and_the_old_ones_are_kept(retail):
    ids = [q.id for q in retail.questions]
    assert len(ids) == len(set(ids)) == 15 + len(NEW) + 3  # +3 provenance questions: bar-type-declared, data-source-declared, time-bar-label-confirmed
    assert set(NEW) <= set(ids)
    assert {"rf-rate-source", "expired-contracts", "futures-back-adjusted", "timestamp-semantics"} <= set(ids)


def test_each_new_question_is_a_complete_dataset_choice_question(retail):
    by_id = {q.id: q for q in retail.questions}
    for qid in NEW:
        q = by_id[qid]
        assert q.scope == "dataset" and q.stage == "S1" and q.answer_type == "choice", qid
        assert q.text.endswith("?") and q.why_it_matters and q.unknown_risk, qid
        ids = [c.id for c in q.choices]
        assert len(ids) >= 3 and len(ids) == len(set(ids)) and all(c.label for c in q.choices), qid


def test_the_critical_set_is_exactly_where_look_ahead_or_a_unit_error_is_possible(retail):
    crit = {q.id for q in retail.questions if q.critical} & set(NEW)
    assert crit == CRITICAL_NEW
    by_id = {q.id: q for q in retail.questions}
    for qid in CRITICAL_NEW:  # a critical question must name at least one risky answer
        assert any(c.risk for c in by_id[qid].choices), qid


def test_bank_questions_are_scoped_to_their_asset_class_and_common_ones_to_none(retail):
    by_id = {q.id: q for q in retail.questions}
    for cls, ids in BANKS.items():
        for qid in ids:
            assert set(by_id[qid].applies_to.asset_class) == set(CLASSES[cls]), qid
    for qid in COMMON:
        assert by_id[qid].applies_to.asset_class == [], qid


@pytest.mark.parametrize("cls", list(BANKS))
def test_selection_offers_a_banks_questions_only_for_its_own_asset_class(tmp_path, cls):
    for spelling in CLASSES[cls]:
        got = set(selected(tmp_path, spelling))
        assert set(BANKS[cls]) <= got, spelling
        for other, ids in BANKS.items():
            if other != cls:
                assert not (set(ids) & got), (spelling, other)


@pytest.mark.parametrize("asset_class", ["option", "options", "future", "futures", "equity", "etf", "fx", "fx_spot",
                                         "crypto_perp", "synthetic", "index"])
def test_common_questions_are_offered_for_every_asset_class(tmp_path, asset_class):
    assert set(COMMON) <= set(selected(tmp_path, asset_class))


def test_an_unrelated_asset_class_gets_no_bank_question(tmp_path):
    got = set(selected(tmp_path, "synthetic"))
    assert not (got & {q for ids in BANKS.values() for q in ids})


def test_retail_core_with_the_new_banks_still_passes_the_public_safety_lint(retail):
    assert public_safety_findings(retail) == []
