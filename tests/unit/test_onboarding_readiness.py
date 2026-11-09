"""Readiness warnings about dataset onboarding."""

import pytest

from readiness_helpers import TID, clean_run, codes, make_registry
from qrh.onboarding import store
from qrh.onboarding.model import model_from_proposal, parse_proposal
from qrh.onboarding.profile import build_profile
from qrh.readiness import collect as rc
from qrh.synthetic import instruments as inst


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    return h


def onboarding_codes(c):
    return sorted(codes(c, "onboarding"))


def put_model(reg, statuses, verified=True, proposal=None):
    _, base = inst.equity_daily()
    model = model_from_proposal(parse_proposal(proposal or base))
    for field, claim in model.claims().items():
        setattr(model, field, claim.model_copy(update={"status": statuses.get(field, "verified")}))
    row = reg.conn.execute("SELECT content_hash FROM datasets WHERE name='prices'").fetchone()
    version = store.add_version(reg.conn, "prices", row["content_hash"], model)
    store.update_version(reg.conn, "prices", version, model, verified=verified)


def test_a_dataset_that_was_never_onboarded_is_a_high_warning(home):
    reg = make_registry(home, onboarded=False)
    clean_run(home, reg)
    c = rc.collect(home, TID, reg)
    assert ("not-onboarded", "high") in onboarding_codes(c)
    w = next(w for w in c.warnings if w.code == "not-onboarded")
    assert w.stage == "S1" and "qrh data propose prices" in w.resolution
    reg.close()


def test_a_settled_dataset_has_no_onboarding_warning(home):
    reg = make_registry(home)
    clean_run(home, reg)
    assert onboarding_codes(rc.collect(home, TID, reg)) == []
    reg.close()


def test_a_contradicted_field_is_a_high_warning(home):
    reg = make_registry(home, onboarded=False)
    clean_run(home, reg)
    put_model(reg, {"price_units": "contradicted"})
    assert ("contradicted", "high") in onboarding_codes(rc.collect(home, TID, reg))
    reg.close()


def test_unverified_fields_are_high_when_critical_and_medium_otherwise_and_link_to_the_question(home):
    reg = make_registry(home, onboarded=False)
    clean_run(home, reg)
    put_model(reg, {"adjustments": "unverified", "quote_currency": "unverified"})
    c = rc.collect(home, TID, reg)
    assert onboarding_codes(c) == [("unverified-critical", "high"), ("unverified-recommended", "medium")]
    crit = next(w for w in c.warnings if w.code == "unverified-critical")
    assert crit.scope == "prices:adjustments" and "onb-adjustments" in crit.resolution
    assert "qrh interview next --stage S1 --scope dataset:prices" in crit.resolution
    reg.close()


def test_a_proposal_that_was_not_verified_yet_says_to_verify_it(home):
    reg = make_registry(home, onboarded=False)
    clean_run(home, reg)
    put_model(reg, {k: "unverified" for k in ("asset_class", "price_columns", "price_units", "adjustments",
                                              "timestamp_semantics", "quote_currency", "timezone", "frequency")}, verified=False)
    c = rc.collect(home, TID, reg)
    assert all("qrh data verify prices" in w.resolution for w in c.warnings if w.kind == "onboarding")
    assert ("unverified-critical", "high") in onboarding_codes(c)
    reg.close()


def test_the_adjustments_name_hint_is_a_medium_warning_until_the_field_is_settled(home):
    import pandas as pd

    reg = make_registry(home, onboarded=False)
    clean_run(home, reg)
    frame = pd.DataFrame({"adj_close": [1.0 + i for i in range(20)]}, index=pd.date_range("2021-01-01", periods=20, tz="UTC"))
    row = reg.conn.execute("SELECT content_hash FROM datasets WHERE name='prices'").fetchone()
    store.save_profile(reg.conn, row["content_hash"], build_profile(frame), "h")
    put_model(reg, {"adjustments": "unverified"})
    assert ("adjustments-hint", "medium") in onboarding_codes(rc.collect(home, TID, reg))
    put_model(reg, {"adjustments": "confirmed"}, proposal=inst.with_claim(inst.equity_daily()[1], "quote_currency", "EUR"))
    assert "adjustments-hint" not in dict(onboarding_codes(rc.collect(home, TID, reg)))
    reg.close()


def test_onboarding_questions_are_not_reported_twice(home, tmp_path):
    from typer.testing import CliRunner

    from packs_helpers import make_pack, question
    from qrh.cli import app

    reg = make_registry(home, onboarded=False)
    clean_run(home, reg)
    put_model(reg, {"adjustments": "unverified"})
    d = make_pack(tmp_path, "bank", questions=[[question("rf", critical=False)]])
    assert CliRunner().invoke(app, ["pack", "add", str(d)]).exit_code == 0
    c = rc.collect(home, TID, reg)
    assert not [w for w in c.warnings if w.kind == "interview" and "onb-" in w.scope]
    assert [w for w in c.warnings if w.kind == "interview" and "rf" in w.scope]  # the pack's own question still shows
    assert ("unverified-critical", "high") in onboarding_codes(c)
    reg.close()
