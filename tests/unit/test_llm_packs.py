"""retail-core: the LLM lessons and critical questions, and the applicability field that scopes them."""

from pathlib import Path

import pytest
from packs_helpers import lesson, make_pack, question

from qrh.interview.engine import items_for_trial
from qrh.packs.model import AppliesTo, load_pack
from qrh.packs.runtime import PackView
from qrh.packs.safety import public_safety_findings

PACKS = Path(__file__).resolve().parents[2] / "packs"
TRIAL = {"trial_id": "t1", "family_id": "f", "text": "x", "difference_statement": "y", "datasets": [],
         "asset_class": "synthetic", "strategy_family": "trend", "frequency": "1D"}
LLM = {"models": [{"id": "m"}]}


def test_applies_to_uses_llm_matches_only_trials_with_an_llm_section():
    a = AppliesTo(uses_llm=True)
    assert a.matches({**TRIAL, "llm": LLM}) and not a.matches({**TRIAL, "llm": None}) and not a.matches(TRIAL)
    assert AppliesTo().matches(TRIAL) and AppliesTo().matches({**TRIAL, "llm": LLM})  # the default is unchanged


def test_uses_llm_combines_with_the_other_dimensions():
    a = AppliesTo(uses_llm=True, asset_class=["synthetic"])
    assert a.matches({**TRIAL, "llm": LLM}) and not a.matches({**TRIAL, "llm": LLM, "asset_class": "equity"})


def test_an_unknown_applicability_key_is_still_an_error():
    with pytest.raises(Exception):
        AppliesTo(uses_gpu=True)


@pytest.fixture(scope="module")
def retail():
    return load_pack(PACKS / "retail-core")


def test_retail_core_has_the_three_generic_llm_lessons_that_apply_only_to_llm_strategies(retail):
    by_id = {x.id: x for x in retail.lessons}
    for lid in ("llm-memorization-is-lookahead", "count-every-prompt-model-variant", "record-and-replay-llm-calls"):
        assert by_id[lid].applies_to.uses_llm is True, lid
    assert "look-ahead" in by_id["llm-memorization-is-lookahead"].statement
    assert "trial" in by_id["count-every-prompt-model-variant"].statement
    assert "record" in by_id["record-and-replay-llm-calls"].statement and "replay" in by_id["record-and-replay-llm-calls"].statement
    assert public_safety_findings(retail) == []


def test_retail_core_has_the_llm_critical_questions_scoped_to_llm_strategies(retail):
    crit = {q.id: q for q in retail.questions if q.critical}
    for qid in ("llm-cutoff-source", "llm-live-retrieval", "llm-finetune-or-test-data"):
        q = crit[qid]
        assert q.applies_to.uses_llm is True and q.stage == "S2" and q.scope == "hypothesis", qid
        assert q.unknown_risk and q.why_it_matters
    risks = {c.id: c.risk for c in crit["llm-live-retrieval"].choices}
    assert risks["yes_current"] == "lookahead" and risks["no"] is None
    assert {c.id for c in crit["llm-finetune-or-test-data"].choices} >= {"no", "fine_tuned", "prompted_with_test_period"}


def test_the_outside_variants_question_already_covers_every_strategy_and_now_names_prompts_and_models(retail):
    q = next(x for x in retail.questions if x.id == "variants-outside-harness")
    assert q.applies_to.uses_llm is False and q.maps_to == {"kind": "imported_trials"} and q.critical
    assert "How many other prompt or model variants did you try for this idea outside the harness?" in q.text
    assert "models, prompts, features, parameters" in q.text  # the general wording is kept


def items(tmp_path, trial):
    from qrh.memory import db

    conn = db.connect(tmp_path / "r.sqlite3")
    view = PackView([load_pack(PACKS / "retail-core")])
    return [i.question.id for i in items_for_trial(conn, view, trial)]


def test_llm_questions_are_asked_only_when_the_hypothesis_has_llm(tmp_path):
    (tmp_path / "b").mkdir()
    plain = items(tmp_path, TRIAL)
    with_llm = items(tmp_path / "b", {**TRIAL, "llm": LLM})
    assert not [q for q in plain if q.startswith("llm-")] and "variants-outside-harness" in plain
    assert {"llm-cutoff-source", "llm-live-retrieval", "llm-finetune-or-test-data"} <= set(with_llm)


def test_llm_lessons_fire_only_for_llm_trials(tmp_path):
    view = PackView([load_pack(PACKS / "retail-core")])
    plain = {les.id for _p, les in view.matched_lessons(TRIAL)}
    llm = {les.id for _p, les in view.matched_lessons({**TRIAL, "llm": LLM})}
    assert not {i for i in plain if "llm" in i or "prompt" in i}
    assert {"llm-memorization-is-lookahead", "count-every-prompt-model-variant", "record-and-replay-llm-calls"} <= llm


def test_a_pack_lesson_can_use_the_field_too(tmp_path):
    d = make_pack(tmp_path, lessons=[lesson("only-llm", applies_to={"uses_llm": True})],
                  questions=[question("q-llm", scope="hypothesis", applies_to={"uses_llm": True})])
    pack = load_pack(d)
    assert pack.lessons[0].applies_to.uses_llm and pack.questions[0].applies_to.uses_llm
