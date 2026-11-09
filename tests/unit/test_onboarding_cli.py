"""`qrh data profile|propose|verify|onboarding|confirm|status` and the interview flow that settles the gaps."""

import json

import pytest
from typer.testing import CliRunner

from onboarding_helpers import MAKERS, proposal_file
from qrh.cli import app
from qrh.synthetic import instruments as inst

runner = CliRunner()


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    return h


def cli(*args, ok=True):
    res = runner.invoke(app, list(args))
    if ok:
        assert res.exit_code == 0, res.output
    return res


def setup(tmp_path, kind="futures", privacy="public", **kw):
    frame, proposal = MAKERS[kind](**kw)
    path = tmp_path / "ds.csv"
    inst.write_csv(frame, path)
    args = ["data", "register", str(path), "--name", "ds"] + (["--privacy", privacy] if privacy else [])
    cli(*args)
    return proposal, path


def propose(tmp_path, proposal, *extra, ok=True):
    return cli("data", "propose", "ds", "--file", str(proposal_file(tmp_path, proposal)), *extra, ok=ok)


def onboarding():
    return json.loads(cli("data", "onboarding", "ds", "--json").stdout)


def test_register_privacy_defaults_to_private_and_shows_in_status(tmp_path):
    frame, _ = inst.equity_daily()
    path = tmp_path / "a.csv"
    inst.write_csv(frame, path)
    assert "privacy=private" in cli("data", "register", str(path), "--name", "a").output
    assert "privacy=public" in cli("data", "register", str(path), "--name", "a", "--privacy", "public").output
    out = cli("data", "status", "a").output
    assert "privacy=public" in out and "onboarding: NOT_ONBOARDED" in out
    assert runner.invoke(app, ["data", "register", str(path), "--privacy", "secret"]).exit_code == 2


def test_profile_prints_without_rows_and_stores_json(tmp_path):
    setup(tmp_path)
    out = cli("data", "profile", "ds").output
    assert "frequency 1B" in out and "contract: string" in out and "AAA9" in out
    doc = json.loads(cli("data", "profile", "ds", "--json", "--no-values").stdout)
    assert doc["profile"]["no_values"] is True and len(doc["profile_hash"]) == 64
    assert runner.invoke(app, ["data", "profile", "nope"]).exit_code == 2


def test_whole_flow_from_proposal_to_onboarded_through_the_interview(tmp_path):
    proposal, _ = setup(tmp_path)
    assert onboarding()["state"] == "NOT_ONBOARDED"
    assert "stored" in propose(tmp_path, proposal).output
    assert "unchanged" in propose(tmp_path, proposal).output
    assert onboarding()["state"] == "PROPOSED"
    cli("data", "verify", "ds")
    doc = onboarding()
    assert doc["state"] == "VERIFIED_WITH_GAPS"
    open_ids = {q["id"]: q["critical"] for q in doc["pending_questions"]}
    assert open_ids["onb-adjustments"] and open_ids["onb-multiplier"] and not open_ids["onb-quote_currency"]
    # the interview asks the critical ones first, each with its choices
    nxt = json.loads(cli("interview", "next", "--stage", "S1", "--scope", "dataset:ds", "--json").stdout)
    q = nxt["question"]
    assert q["critical"] and q["id"].startswith("onb-") and nxt["remaining"] == len(open_ids)
    for qid in ("onb-adjustments", "onb-timestamp_semantics", "onb-multiplier"):
        cli("interview", "answer", qid, "--scope", "dataset:ds", "--value", "confirm", "--human")
    assert onboarding()["state"] == "ONBOARDED"
    assert "onboarding: ONBOARDED" in cli("data", "status", "ds").output
    # the remaining recommended questions are still open and never block
    assert all(not q["critical"] for q in onboarding()["pending_questions"])


def test_a_correction_through_the_interview_makes_a_new_version(tmp_path):
    proposal, _ = setup(tmp_path, "equity")
    propose(tmp_path, inst.with_claim(proposal, "adjustments", "none"))
    cli("data", "verify", "ds")
    cli("interview", "answer", "onb-adjustments", "--scope", "dataset:ds", "--value", "split_and_dividend", "--human")
    doc = onboarding()
    row = {r["field"]: r for r in doc["fields"]}["adjustments"]
    assert doc["version"] == 2 and row["claim"] == "split_and_dividend" and row["status"] == "confirmed" and row["source"] == "user"
    status = json.loads(cli("interview", "status", "--json").stdout)
    assert any(a["qid"] == "onb-adjustments" for a in status["answers"])


def test_an_invalid_answer_is_refused_and_changes_nothing(tmp_path):
    proposal, _ = setup(tmp_path, "equity")
    propose(tmp_path, proposal)
    cli("data", "verify", "ds")
    res = runner.invoke(app, ["interview", "answer", "onb-quote_currency", "--scope", "dataset:ds", "--value", "$$", "--human"])
    assert res.exit_code == 2
    assert onboarding()["version"] == 1


def test_data_confirm_is_the_same_as_a_human_interview_answer(tmp_path):
    proposal, _ = setup(tmp_path, "equity")
    propose(tmp_path, proposal)
    cli("data", "verify", "ds")
    assert "confirmed" in cli("data", "confirm", "ds", "adjustments").output
    assert "corrected" in cli("data", "confirm", "ds", "quote_currency", "--value", "EUR").output
    rows = {r["field"]: r for r in onboarding()["fields"]}
    assert rows["adjustments"]["status"] == "confirmed" and rows["quote_currency"]["claim"] == "EUR"
    assert runner.invoke(app, ["data", "confirm", "ds", "price_units"]).exit_code == 2  # verified by code: no open question
    answers = json.loads(cli("interview", "status", "--json").stdout)["answers"]
    assert {a["qid"] for a in answers} >= {"onb-adjustments", "onb-quote_currency"} and {a["provenance"] for a in answers} == {"human-cli"}


def test_a_contradicted_proposal_blocks_the_dataset(tmp_path):
    proposal, _ = setup(tmp_path, "equity")
    propose(tmp_path, inst.with_claim(proposal, "price_units", "fraction"))
    res = cli("data", "verify", "ds", ok=False)
    assert res.exit_code == 2 and "CONTRADICTED" in res.output
    status = runner.invoke(app, ["data", "status", "ds"])
    assert status.exit_code == 2 and "onboarding: price_units" in status.output


def test_bad_proposals_are_refused(tmp_path):
    proposal, _ = setup(tmp_path, "equity")
    bad = inst.with_claim(proposal, "asset_class", "stock")
    assert propose(tmp_path, bad, ok=False).exit_code == 2
    assert propose(tmp_path, {**proposal, "source": "user"}, ok=False).exit_code == 2  # needs --as-user too
    assert propose(tmp_path, proposal, "--as-user", ok=False).exit_code == 2
    assert propose(tmp_path, {**proposal, "source": "user"}, "--as-user").exit_code == 0
    assert runner.invoke(app, ["data", "propose", "ds", "--file", str(tmp_path / "missing.json")]).exit_code == 2
    assert runner.invoke(app, ["data", "verify", "nope"]).exit_code == 2
    assert runner.invoke(app, ["data", "onboarding", "nope"]).exit_code == 2


def test_interview_next_with_a_trial_lists_the_dataset_questions(tmp_path):
    proposal, _ = setup(tmp_path, "equity")
    propose(tmp_path, proposal)
    cli("data", "verify", "ds")
    spec = tmp_path / "h.yaml"
    spec.write_text("text: t\nstrategy_family: trend\nasset_class: equity\nfrequency: 1D\nparameter_space_size: 1\n"
                    "thresholds: {k: 1}\ndifference_statement: d\ndatasets: [ds]\n")
    tid = cli("new", str(spec)).stdout.strip().splitlines()[-1]
    out = json.loads(cli("interview", "status", "--trial", tid, "--json").stdout)
    rows = out["stages"]["S1"]["rows"]
    assert {r["qid"] for r in rows if r["origin"] == "onboarding" and r["critical"]} == {"onb-adjustments", "onb-timestamp_semantics"}
    assert {c["question_id"] for c in out["conditions"]} == {"onb-adjustments", "onb-timestamp_semantics"}
