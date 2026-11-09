"""The instrument model and the (untrusted) proposal file."""

import json

import pytest
from pydantic import ValidationError

from qrh.onboarding.model import (CRITICAL, FIELDS, InstrumentModel, OnboardingError, claim_text, load_proposal,
                                  model_from_proposal)
from qrh.synthetic import instruments as inst


def write(tmp_path, doc, name="p.json"):
    p = tmp_path / name
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return p


def test_every_synthetic_proposal_is_a_valid_model(tmp_path):
    for fn in (inst.equity_daily, inst.futures_daily, inst.fx_quotes, inst.option_snapshots, inst.crypto_perp):
        _, proposal = fn()
        m = model_from_proposal(load_proposal(write(tmp_path, proposal)))
        assert all(c.status == "unverified" and c.source == "llm" for c in m.claims().values())
        assert set(m.claims()) <= set(FIELDS)


def test_critical_fields_are_the_four_named_ones():
    assert set(CRITICAL) == {"multiplier", "adjustments", "timestamp_semantics", "price_units"}


@pytest.mark.parametrize("field,value", [
    ("asset_class", "stock"), ("price_units", "dollars"), ("adjustments", "maybe"), ("timestamp_semantics", "open"),
    ("price_columns", []), ("price_columns", "close"), ("price_columns", ["a"] * 2), ("price_columns", [1]),
    ("multiplier", True), ("multiplier", -1), ("multiplier", "fifty"), ("tick_size", 0), ("tick_size", float("inf")),
    ("quote_currency", "US D"), ("session_hint", "x" * 201), ("frequency", "x" * 17),
    ("roll_rule", {"text": "r", "extra": "x"}), ("option_mapping", {"gamma": "g"}), ("option_mapping", {}),
    ("timezone", "UTC\n"), ("timezone", "a‮b"), ("session_hint", "price is $x$"),
])
def test_bad_values_are_refused(tmp_path, field, value):
    _, proposal = inst.equity_daily()
    proposal["fields"][field] = {"value": value, "evidence": ""}
    with pytest.raises(OnboardingError):
        load_proposal(write(tmp_path, proposal))


def test_numbers_may_be_unknown_and_a_roll_rule_may_be_text(tmp_path):
    _, proposal = inst.futures_daily()
    proposal["fields"]["multiplier"] = {"value": "unknown"}
    proposal["fields"]["roll_rule"] = {"value": "quarterly"}
    m = model_from_proposal(load_proposal(write(tmp_path, proposal)))
    assert m.multiplier.value == "unknown" and m.roll_rule.value == {"text": "quarterly"}


def test_unknown_keys_and_status_smuggling_are_rejected(tmp_path):
    _, proposal = inst.fx_quotes()
    for doc in (
        {**proposal, "extra_top": 1},
        {**proposal, "fields": {**proposal["fields"], "leverage": {"value": 3}}},
        {**proposal, "fields": {**proposal["fields"], "asset_class": {"value": "fx_spot", "status": "verified"}}},
        {**proposal, "fields": {**proposal["fields"], "asset_class": {"value": "fx_spot", "source": "user"}}},
        {**proposal, "source": "profile"},
    ):
        with pytest.raises(OnboardingError):
            load_proposal(write(tmp_path, doc))


def test_the_file_itself_is_untrusted(tmp_path):
    with pytest.raises(OnboardingError, match="JSON"):
        load_proposal(write(tmp_path, "not json"))
    with pytest.raises(OnboardingError, match="too large"):
        load_proposal(write(tmp_path, "{" + " " * 70_000 + "}"))
    with pytest.raises(OnboardingError, match="duplicate"):
        load_proposal(write(tmp_path, '{"fields": {"asset_class": {"value": "equity"}, "asset_class": {"value": "etf"}}}'))
    with pytest.raises(OnboardingError, match="NaN|finite|constant"):
        load_proposal(write(tmp_path, '{"fields": {"multiplier": {"value": NaN}}}'))
    with pytest.raises(OnboardingError):
        load_proposal(write(tmp_path, "[]"))
    with pytest.raises(OnboardingError, match="cannot read"):
        load_proposal(tmp_path / "missing.json")


def test_evidence_is_plain_text_with_a_limit(tmp_path):
    _, proposal = inst.fx_quotes()
    proposal["fields"]["asset_class"]["evidence"] = "<script>alert(1)</script> [link](http://x) **bold**"
    m = model_from_proposal(load_proposal(write(tmp_path, proposal)))
    assert m.asset_class.evidence.startswith("<script>")  # stored as text, never interpreted
    proposal["fields"]["asset_class"]["evidence"] = "e" * 301
    with pytest.raises(OnboardingError):
        load_proposal(write(tmp_path, proposal))


def test_model_roundtrip_and_hashes():
    _, proposal = inst.equity_daily()
    m = InstrumentModel.model_validate(model_from_proposal_dict(proposal).model_dump(mode="json", exclude_none=True))
    assert m.claims_hash() == model_from_proposal_dict(proposal).claims_hash()
    changed = model_from_proposal_dict(inst.with_claim(proposal, "price_units", "percent"))
    assert changed.claims_hash() != m.claims_hash()
    # status changes do not change the claims hash, but they change the model hash
    m.adjustments.status = "confirmed"
    assert m.claims_hash() == model_from_proposal_dict(proposal).claims_hash()
    assert m.model_hash() != model_from_proposal_dict(proposal).model_hash()


def test_strict_model_rejects_unknown_keys_and_coercion():
    with pytest.raises(ValidationError):
        InstrumentModel.model_validate({"asset_class": {"value": "equity", "source": "llm"}, "leverage": 2})
    with pytest.raises(ValidationError):
        InstrumentModel.model_validate({"asset_class": {"value": "equity", "source": "llm", "status": "verified ", }})


def test_claim_text():
    assert claim_text("price_columns", ["a", "b"]) == "a, b"
    assert claim_text("roll_rule", {"text": "q", "method": "cal"}) == "q (cal)"
    assert claim_text("option_mapping", {"strike": "k", "right": "cp"}) == "right=cp; strike=k"
    assert claim_text("multiplier", 50) == "50"


def model_from_proposal_dict(proposal):
    from qrh.onboarding.model import parse_proposal
    return model_from_proposal(parse_proposal(proposal))
