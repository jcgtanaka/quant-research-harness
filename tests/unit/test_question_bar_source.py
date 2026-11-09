"""retail-core provenance questions: bar type, data source and time-bar label."""

from pathlib import Path

import pytest

from qrh.packs.model import load_pack

PACKS = Path(__file__).resolve().parents[2] / "packs"
IDS = ["bar-type-declared", "data-source-declared", "time-bar-label-confirmed"]


@pytest.fixture(scope="module")
def by_id():
    return {q.id: q for q in load_pack(PACKS / "retail-core").questions}


@pytest.mark.parametrize("qid", IDS)
def test_the_question_is_a_dataset_s1_choice_for_every_asset_class(by_id, qid):
    q = by_id[qid]
    assert q.scope == "dataset" and q.stage == "S1" and q.answer_type == "choice"
    assert q.applies_to.asset_class == []
    assert q.text.endswith("?") and q.why_it_matters and q.unknown_risk


def test_only_the_bar_type_question_is_critical_and_it_names_a_risky_answer(by_id):
    assert by_id["bar-type-declared"].critical is True
    assert by_id["data-source-declared"].critical is False
    assert by_id["time-bar-label-confirmed"].critical is False
    risky = {c.id: c.risk for c in by_id["bar-type-declared"].choices if c.risk}
    assert risky == {"unknown": "bar-type-unknown"}


def test_choice_ids_match_the_agreed_vocabulary(by_id):
    assert [c.id for c in by_id["bar-type-declared"].choices] == ["time", "dollar", "volume", "mixed", "unknown"]
    assert [c.id for c in by_id["data-source-declared"].choices] == ["broker_export", "data_vendor", "own_file", "unknown"]
    assert [c.id for c in by_id["time-bar-label-confirmed"].choices] == ["yes_checked", "not_checked", "not_time_bars"]
