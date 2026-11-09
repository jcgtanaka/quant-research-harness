"""Onboarding workflow: profile, propose, verify, confirm; versions, state, questions."""

import pytest

from onboarding_helpers import make_catalog, proposal_file, register
from qrh.onboarding import questions, store, workflow
from qrh.onboarding.model import OnboardingError, load_proposal
from qrh.synthetic import instruments as inst


@pytest.fixture
def cat(tmp_path):
    c = make_catalog(tmp_path)
    yield c
    c.close()


def propose(cat, tmp_path, proposal, **kw):
    return workflow.propose(cat, "ds", load_proposal(proposal_file(tmp_path, proposal)), **kw)


def test_profile_is_stored_once_per_content_hash_and_mode(cat, tmp_path):
    register(cat, tmp_path)
    p1, h1 = workflow.profile_dataset(cat, "ds")
    p2, h2 = workflow.profile_dataset(cat, "ds")
    assert h1 == h2 and p1 == p2 and p1["no_values"] is False
    nv, hn = workflow.profile_dataset(cat, "ds", no_values=True)
    assert nv["no_values"] and hn != h1
    rows = cat.conn.execute("SELECT mode FROM dataset_profiles ORDER BY mode").fetchall()
    assert [r["mode"] for r in rows] == ["no_values", "values"]


def test_profile_refuses_an_unknown_dataset_and_a_changed_file(cat, tmp_path):
    with pytest.raises(OnboardingError, match="unknown dataset"):
        workflow.profile_dataset(cat, "nope")
    _, path = register(cat, tmp_path)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(OnboardingError, match="changed since registration"):
        workflow.profile_dataset(cat, "ds")


def test_no_onboarding_means_not_onboarded(cat, tmp_path):
    register(cat, tmp_path)
    assert workflow.describe(cat, "ds")["state"] == "NOT_ONBOARDED"


def test_propose_creates_versions_only_when_claims_change(cat, tmp_path):
    proposal, _ = register(cat, tmp_path)
    first = propose(cat, tmp_path, proposal)
    assert first == {"version": 1, "created": True}
    assert propose(cat, tmp_path, proposal) == {"version": 1, "created": False}
    second = propose(cat, tmp_path, inst.with_claim(proposal, "quote_currency", "EUR"))
    assert second == {"version": 2, "created": True}
    d = workflow.describe(cat, "ds")
    assert d["state"] == "PROPOSED" and d["version"] == 2
    assert {r["status"] for r in d["fields"]} == {"unverified"} and {r["source"] for r in d["fields"]} == {"llm"}


def test_user_source_needs_both_the_file_and_the_flag(cat, tmp_path):
    proposal, _ = register(cat, tmp_path)
    user = {**proposal, "source": "user"}
    with pytest.raises(OnboardingError, match="--as-user"):
        propose(cat, tmp_path, user)
    with pytest.raises(OnboardingError, match="--as-user"):
        propose(cat, tmp_path, proposal, as_user=True)
    propose(cat, tmp_path, user, as_user=True)
    assert {r["source"] for r in workflow.describe(cat, "ds")["fields"]} == {"user"}


def test_propose_refuses_a_dataset_that_is_unknown(cat, tmp_path):
    _, proposal = inst.equity_daily()
    with pytest.raises(OnboardingError, match="unknown dataset"):
        propose(cat, tmp_path, proposal)


def test_verify_a_correct_proposal_leaves_gaps_and_asks_the_critical_questions(cat, tmp_path):
    proposal, _ = register(cat, tmp_path, "futures")
    propose(cat, tmp_path, proposal)
    out = workflow.verify(cat, "ds")
    assert out["state"] == "VERIFIED_WITH_GAPS" and out["contradictions"] == []
    qs = {q.id: (crit, scope) for q, scope, crit, _ident in questions.dataset_questions(cat.conn)}
    assert qs["onb-adjustments"] == (True, "dataset:ds") and qs["onb-timestamp_semantics"][0] is True
    assert qs["onb-multiplier"][0] is True
    assert "onb-price_units" not in qs  # verified by code: nothing to ask
    assert qs["onb-quote_currency"][0] is False and qs["onb-roll_rule"][0] is False
    assert "onb-asset_class" not in qs  # verified by code: nothing to ask


def test_confirming_every_critical_field_reaches_onboarded(cat, tmp_path):
    proposal, _ = register(cat, tmp_path, "futures")
    propose(cat, tmp_path, proposal)
    workflow.verify(cat, "ds")
    for field in ("adjustments", "timestamp_semantics", "multiplier"):
        workflow.apply_answer(cat, "ds", field, "confirm", "human-cli")
    d = workflow.describe(cat, "ds")
    assert d["state"] == "ONBOARDED"
    by = {r["field"]: r for r in d["fields"]}
    assert by["multiplier"]["status"] == "confirmed" and by["multiplier"]["checked_by"].startswith("human:")
    assert by["quote_currency"]["status"] == "unverified"  # recommended gaps do not prevent ONBOARDED


def test_a_missing_critical_field_is_a_gap_not_a_pass(cat, tmp_path):
    proposal, _ = register(cat, tmp_path, "equity")
    propose(cat, tmp_path, inst.without_claim(proposal, "timestamp_semantics"))
    d = workflow.verify(cat, "ds")
    assert d["state"] == "VERIFIED_WITH_GAPS"
    row = {r["field"]: r for r in workflow.describe(cat, "ds")["fields"]}["timestamp_semantics"]
    assert row["status"] == "unverified" and row["claim"] == "(not proposed)"
    assert any(q["id"] == "onb-timestamp_semantics" for q in workflow.describe(cat, "ds")["pending_questions"])
    workflow.apply_answer(cat, "ds", "timestamp_semantics", "bar_close", "human-cli")  # a correction supplies the value
    row = {r["field"]: r for r in workflow.describe(cat, "ds")["fields"]}["timestamp_semantics"]
    assert row["claim"] == "bar_close" and row["status"] == "confirmed"


def test_spot_classes_do_not_require_a_multiplier(cat, tmp_path):
    proposal, _ = register(cat, tmp_path, "equity")
    propose(cat, tmp_path, proposal)
    workflow.verify(cat, "ds")
    assert "multiplier" not in {r["field"] for r in workflow.describe(cat, "ds")["fields"]}


def test_a_contradiction_blocks_the_dataset_and_fixing_it_unblocks(cat, tmp_path):
    proposal, _ = register(cat, tmp_path, "equity")
    propose(cat, tmp_path, inst.with_claim(proposal, "price_units", "fraction"))
    out = workflow.verify(cat, "ds")
    assert out["state"] == "CONTRADICTED"
    status, reasons = cat.get_status("ds")
    assert status == "BLOCKED" and any(r.startswith("onboarding: price_units") for r in reasons)
    propose(cat, tmp_path, proposal)
    workflow.verify(cat, "ds")
    assert cat.get_status("ds")[0] != "BLOCKED"


def test_check_dataset_keeps_a_contradicted_dataset_blocked(cat, tmp_path):
    from qrh.checks.data_quality import check_dataset

    proposal, _ = register(cat, tmp_path, "equity")
    rules = tmp_path / "rules.yaml"
    rules.write_text("rules:\n  - type: range\n    column: close\n    positive: true\n")
    cat.register_dataset(tmp_path / "ds.csv", name="ds", rules=rules)
    assert check_dataset(cat, "ds")[0] == "PASS"
    propose(cat, tmp_path, inst.with_claim(proposal, "price_units", "fraction"))
    workflow.verify(cat, "ds")
    status, reasons = check_dataset(cat, "ds")
    assert status == "BLOCKED" and any("onboarding" in r for r in reasons)


def test_a_changed_file_starts_a_new_onboarding_version_and_resets_confirmations(cat, tmp_path):
    proposal, path = register(cat, tmp_path, "equity")
    propose(cat, tmp_path, proposal)
    workflow.verify(cat, "ds")
    workflow.apply_answer(cat, "ds", "adjustments", "confirm", "human-cli")
    old = store.latest_version(cat.conn, "ds")
    frame, _ = inst.equity_daily(seed=5)
    inst.write_csv(frame, path)
    cat.register_dataset(path, name="ds")
    assert workflow.describe(cat, "ds")["state"] == "NOT_ONBOARDED"
    workflow.verify(cat, "ds")  # carries the claims over to the new file, nothing is confirmed any more
    new = store.latest_version(cat.conn, "ds")
    assert new["version"] == old["version"] + 1 and new["content_hash"] != old["content_hash"]
    assert {r["field"]: r["status"] for r in workflow.describe(cat, "ds")["fields"]}["adjustments"] == "unverified"
    assert store.get_version(cat.conn, "ds", old["version"])["model"].adjustments.status == "confirmed"  # old verdicts stay


def test_a_correction_makes_a_new_version_and_reverifies(cat, tmp_path):
    proposal, _ = register(cat, tmp_path, "equity")
    propose(cat, tmp_path, inst.with_claim(proposal, "price_units", "fraction"))
    workflow.verify(cat, "ds")
    assert workflow.describe(cat, "ds")["state"] == "CONTRADICTED"
    workflow.apply_answer(cat, "ds", "price_units", "currency", "human-cli")
    d = workflow.describe(cat, "ds")
    assert d["version"] == 2 and d["state"] != "CONTRADICTED"
    assert {r["field"]: r for r in d["fields"]}["price_units"]["status"] == "verified"
    assert cat.get_status("ds")[0] != "BLOCKED"
    workflow.apply_answer(cat, "ds", "price_units", "fraction", "human-cli")  # a wrong correction is contradicted again
    assert workflow.describe(cat, "ds")["state"] == "CONTRADICTED"


def test_answers_are_validated_before_anything_changes(cat, tmp_path):
    proposal, _ = register(cat, tmp_path, "equity")
    propose(cat, tmp_path, proposal)
    workflow.verify(cat, "ds")
    for field, text in (("price_units", "dollars"), ("adjustments", ""), ("multiplier", "-3"), ("quote_currency", "$$")):
        with pytest.raises(OnboardingError):
            workflow.check_answer(cat.conn, "ds", field, text)
    with pytest.raises(OnboardingError, match="nothing to confirm"):
        workflow.check_answer(cat.conn, "ds", "tick_size", "confirm")  # no such claim
    assert store.latest_version(cat.conn, "ds")["version"] == 1


def test_unknown_changes_nothing(cat, tmp_path):
    proposal, _ = register(cat, tmp_path, "equity")
    propose(cat, tmp_path, proposal)
    workflow.verify(cat, "ds")
    before = store.latest_version(cat.conn, "ds")["model_hash"]
    workflow.apply_answer(cat, "ds", "adjustments", "unknown", "human-cli")
    assert store.latest_version(cat.conn, "ds")["model_hash"] == before


def test_question_identity_follows_the_claim(cat, tmp_path):
    proposal, _ = register(cat, tmp_path, "equity")
    propose(cat, tmp_path, proposal)
    workflow.verify(cat, "ds")
    ident = {q.id: i for q, _s, _c, i in questions.dataset_questions(cat.conn)}["onb-adjustments"]
    propose(cat, tmp_path, inst.with_claim(proposal, "adjustments", "split_only"))
    workflow.verify(cat, "ds")
    assert {q.id: i for q, _s, _c, i in questions.dataset_questions(cat.conn)}["onb-adjustments"] != ident


def test_unchanged_confirmations_survive_a_new_proposal_version(cat, tmp_path):
    proposal, _ = register(cat, tmp_path, "equity")
    propose(cat, tmp_path, proposal)
    workflow.verify(cat, "ds")
    workflow.apply_answer(cat, "ds", "adjustments", "confirm", "human-cli")
    propose(cat, tmp_path, inst.with_claim(proposal, "quote_currency", "EUR"))
    workflow.verify(cat, "ds")
    assert {r["field"]: r["status"] for r in workflow.describe(cat, "ds")["fields"]}["adjustments"] == "confirmed"
