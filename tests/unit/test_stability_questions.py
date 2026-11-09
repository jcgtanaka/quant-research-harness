"""The four stability thresholds are always asked of the trader, frozen at registration, and flagged when unconfirmed."""

import itertools
import json

import pytest
from typer.testing import CliRunner

from packs_helpers import make_pack
from qrh.cli import app
from qrh.interview import engine
from qrh.interview.stability import (
    STABILITY_NAMES, confirmation, stability_items, stability_questions, trader_thresholds,
)
from qrh.memory.registry import Registry
from qrh.packs.runtime import PackView
from qrh.pipeline.receipt import Stage, load_receipt
from qrh.pipeline.stages import register_new_trial
from qrh.validation.defaults import DEFAULT_THRESHOLDS

runner = CliRunner()
SPEC = {
    "text": "t", "strategy_family": "trend", "asset_class": "equity", "frequency": "1D",
    "parameter_space_size": 1, "thresholds": {"k": 1}, "difference_statement": "d",
}
FAMILY = "strategy:trend|equity|1D"
QIDS = ["stability-closeness", "stability-plateau-share", "stability-neighbour-ratio", "stability-steps"]


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    return h


@pytest.fixture
def reg(home):
    home.mkdir(parents=True, exist_ok=True)
    r = Registry(home / "registry.sqlite3", id_factory=lambda c=itertools.count(1): f"t{next(c)}")
    yield r
    r.close()


def cli(*args):
    return runner.invoke(app, ["interview", *args])


def human_answer(qid, value, scope=FAMILY):
    res = cli("answer", qid, "--scope", scope, "--value", value, "--human")
    assert res.exit_code == 0, res.output
    return res


# --- the questions ----------------------------------------------------------------------------------------------


def test_there_are_four_plain_language_questions_with_the_default_as_the_recommended_choice():
    qs = stability_questions()
    assert [q.id for q in qs] == QIDS
    for q, name in zip(qs, STABILITY_NAMES):
        assert q.stage == "S0" and q.scope == "strategy" and q.answer_type == "choice" and not q.critical
        keep = next(c for c in q.choices if c.id == "keep_default")
        assert "recommended" in keep.label.lower() and keep.label.lower().startswith("keep the default")
        assert q.choices[0].id == "keep_default"  # the recommended option comes first
        assert len(q.choices) >= 3 and q.why_it_matters and "?" in q.text
        rules = {c.id: c.maps_to for c in q.choices}
        assert all(r["kind"] == "threshold" and r["name"] == name for r in rules.values())
        assert rules["keep_default"]["value"] is None  # whatever the default is (core plus packs) stays
    texts = " ".join(q.text for q in qs)
    assert "70%" in texts and "50%" in texts and "10% and 20%" in texts
    by = {q.id: q.text for q in qs}
    assert "chosen setting" in by["stability-closeness"] and "nearby" in by["stability-plateau-share"]
    assert "each parameter" in by["stability-neighbour-ratio"] and "worst" in by["stability-neighbour-ratio"]
    assert "grid" in by["stability-steps"] and "not extra trials" in " ".join(q.why_it_matters for q in qs if q.id == "stability-steps")
    assert "best setting" not in texts  # the plateau is local: nothing is compared with the grid's best


def test_the_questions_do_not_need_any_pack(reg):
    register_new_trial(SPEC, reg, reg.path.parent)
    trial = reg.get_trial("t1")
    items = [i for i in engine.items_for_trial(reg.conn, PackView.empty(), trial) if i.origin == "stability"]
    assert [i.question.id for i in items] == QIDS and {i.scope_key for i in items} == {FAMILY}
    assert not any(i.critical for i in items)


def test_the_questions_are_also_offered_by_a_registered_pack_and_not_duplicated(reg, tmp_path):
    d = make_pack(tmp_path, "bank")
    assert runner.invoke(app, ["pack", "add", str(d)]).exit_code == 0
    register_new_trial(SPEC, reg, reg.path.parent)
    view = PackView.load(reg.path.parent)
    ids = [i.question.id for i in engine.items_for_trial(reg.conn, view, reg.get_trial("t1"))]
    assert ids.count("stability-closeness") == 1


def test_next_asks_them_before_registration_one_at_a_time_without_packs(home):
    out = cli("next", "--stage", "S0", "--scope", FAMILY, "--json")
    assert out.exit_code == 0, out.output
    doc = json.loads(out.stdout)
    assert doc["question"]["id"] == "stability-closeness" and doc["remaining"] == 4
    assert [c["id"] for c in doc["question"]["choices"]][0] == "keep_default"
    text = cli("next", "--stage", "S0", "--scope", FAMILY).stdout
    assert "unknown: I don't know" in text and "Keep the default" in text


def test_an_answered_question_is_not_asked_again_and_the_next_one_comes(home):
    cli("next", "--stage", "S0", "--scope", FAMILY)
    human_answer("stability-closeness", "keep_default")
    doc = json.loads(cli("next", "--stage", "S0", "--scope", FAMILY, "--json").stdout)
    assert doc["question"]["id"] == "stability-plateau-share" and doc["remaining"] == 3


# --- provenance: nobody answers for the trader ----------------------------------------------------------------------


def test_an_agent_cannot_answer_without_the_traders_words(home):
    cli("next", "--stage", "S0", "--scope", FAMILY)
    res = cli("answer", "stability-closeness", "--scope", FAMILY, "--value", "keep_default")
    assert res.exit_code == 2 and "--transcript" in res.output


def test_an_answer_not_found_in_a_user_message_is_refused(home, tmp_path):
    from datetime import datetime, timedelta, timezone

    from qrh.hooks import sessions

    cli("next", "--stage", "S0", "--scope", FAMILY)
    p = tmp_path / "user" / ".claude" / "projects" / "proj" / "sess.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
    p.write_text(json.dumps({"type": "user", "timestamp": ts, "sessionId": "sess",
                             "message": {"role": "user", "content": "tell me about the weather"}}) + "\n")
    sessions.record(home, "sess", str(p), datetime.now(timezone.utc))
    res = cli("answer", "stability-closeness", "--scope", FAMILY, "--value", "keep_default",
              "--transcript", str(p), "--session", "sess")
    assert res.exit_code == 2 and "answer not found in a user message" in res.output


def test_the_answer_is_recorded_with_its_provenance_and_maps_to_the_threshold(home, reg):
    human_answer("stability-closeness", "stricter")
    row = reg.conn.execute("SELECT * FROM interview_answers").fetchone()
    assert row["provenance"] == "human-cli" and row["answered_by"] == "human"
    assert json.loads(row["mapped"])["threshold"] == {"name": "stability_closeness", "value": 0.85}
    human_answer("stability-plateau-share", "keep_default")
    last = reg.conn.execute("SELECT mapped FROM interview_answers ORDER BY id DESC").fetchone()
    assert json.loads(last["mapped"])["threshold"] == {"name": "stability_plateau_share", "value": None}


# --- frozen at registration ---------------------------------------------------------------------------------------


def test_unanswered_questions_leave_the_defaults_and_the_receipt_says_so(reg, home):
    res = register_new_trial(SPEC, reg, home)
    th = reg.get_trial(res.trial_id)["thresholds"]
    assert {n: th[n] for n in STABILITY_NAMES} == {n: DEFAULT_THRESHOLDS[n] for n in STABILITY_NAMES}
    src = res.receipt.params["stability_thresholds"]
    assert {v["source"] for v in src.values()} == {"default"}


def test_the_traders_answers_are_frozen_into_the_hypothesis_thresholds(reg, home):
    human_answer("stability-closeness", "stricter")
    human_answer("stability-plateau-share", "keep_default")
    human_answer("stability-neighbour-ratio", "looser")
    human_answer("stability-steps", "wider")
    res = register_new_trial(SPEC, reg, home)
    th = reg.get_trial(res.trial_id)["thresholds"]
    assert th["stability_closeness"] == 0.85 and th["stability_plateau_share"] == 0.5
    assert th["stability_neighbour_ratio"] == 0.3 and th["stability_steps"] == [0.1, 0.2, 0.3]
    src = res.receipt.params["stability_thresholds"]
    assert src["stability_closeness"]["source"] == "trader answer (stability-closeness v1)"
    assert src["stability_plateau_share"]["source"] == "trader answer: keep the default"
    receipt = load_receipt(home / "receipts" / res.trial_id / f"{Stage.S0.value}.json")
    assert receipt.params["thresholds"] == th


def spec_with(**values):
    return {**SPEC, "thresholds": {"k": 1, **values}}


def test_a_spec_value_that_differs_from_the_traders_answer_is_refused(reg, home):
    human_answer("stability-closeness", "stricter")  # 0.85
    res = register_new_trial(spec_with(stability_closeness=0.6), reg, home)
    assert res.trial_id is None and reg.list_trials() == []
    assert "stability_closeness" in res.reasons[0] and "trader answered" in res.reasons[0]
    assert "remove" in res.reasons[0]


def test_a_spec_value_equal_to_the_traders_answer_is_accepted_and_labelled_hypothesis(reg, home):
    human_answer("stability-closeness", "stricter")
    res = register_new_trial(spec_with(stability_closeness=0.85), reg, home)
    assert res.trial_id and reg.get_trial(res.trial_id)["thresholds"]["stability_closeness"] == 0.85
    assert res.receipt.params["stability_thresholds"]["stability_closeness"]["source"] == "hypothesis"


def test_a_spec_value_looser_than_the_default_with_no_answer_is_refused(reg, home):
    res = register_new_trial(spec_with(stability_closeness=0.5), reg, home)
    assert res.trial_id is None and "looser" in res.reasons[0] and "not answered" in res.reasons[0]
    human_answer("stability-closeness", "unknown")  # "I don't know" is no answer
    assert register_new_trial(spec_with(stability_closeness=0.5), reg, home).trial_id is None


def test_a_tighter_spec_value_with_no_answer_is_accepted_but_stays_unconfirmed(reg, home):
    res = register_new_trial(spec_with(stability_closeness=0.8), reg, home)
    assert res.trial_id and any("stability thresholds not confirmed by the trader" in n for n in res.notes)


def test_keep_the_default_counts_only_for_the_default_in_force(reg, home):
    human_answer("stability-closeness", "keep_default")
    assert register_new_trial(spec_with(stability_closeness=0.8), reg, home).trial_id is None
    ok = register_new_trial(spec_with(stability_closeness=0.7), reg, home)
    assert ok.trial_id and ok.receipt.params["stability_thresholds"]["stability_closeness"]["default"] == 0.7


def test_the_receipt_records_the_default_in_force_including_a_packs_tighter_value(reg, home, tmp_path):
    d = make_pack(tmp_path, "strict", thresholds={"all": {"stability_closeness": 0.8}})
    assert runner.invoke(app, ["pack", "add", str(d)]).exit_code == 0
    human_answer("stability-closeness", "keep_default")
    res = register_new_trial(SPEC, reg, home)
    assert res.receipt.params["stability_thresholds"]["stability_closeness"]["default"] == 0.8
    assert reg.get_trial(res.trial_id)["thresholds"]["stability_closeness"] == 0.8


def test_an_unknown_answer_changes_nothing(reg, home):
    human_answer("stability-closeness", "unknown")
    res = register_new_trial(SPEC, reg, home)
    assert reg.get_trial(res.trial_id)["thresholds"]["stability_closeness"] == 0.7
    assert trader_thresholds(reg.conn, "trend|equity|1D") == {}


def test_a_pack_that_tightens_a_stability_threshold_is_not_undone_by_keep_the_default(reg, home, tmp_path):
    d = make_pack(tmp_path, "strict", thresholds={"all": {"stability_closeness": 0.8}})
    assert runner.invoke(app, ["pack", "add", str(d)]).exit_code == 0
    human_answer("stability-closeness", "keep_default")
    res = register_new_trial(SPEC, reg, home)
    assert reg.get_trial(res.trial_id)["thresholds"]["stability_closeness"] == 0.8


# --- confirmation, as the readiness report reads it ----------------------------------------------------------------


def frozen(extra=None):
    return {**{n: DEFAULT_THRESHOLDS[n] for n in STABILITY_NAMES}, **(extra or {})}


def test_confirmation_lists_each_question_with_its_state(reg):
    states = {c["name"]: c["status"] for c in confirmation(reg.conn, "trend|equity|1D", frozen())}
    assert set(states.values()) == {"pending"} and list(states) == list(STABILITY_NAMES)
    human_answer("stability-closeness", "keep_default")
    human_answer("stability-plateau-share", "unknown")
    got = {c["name"]: c for c in confirmation(reg.conn, "trend|equity|1D", frozen())}
    assert got["stability_closeness"]["status"] == "answered" and got["stability_closeness"]["confirmed"] is True
    assert got["stability_plateau_share"]["status"] == "unknown" and got["stability_plateau_share"]["confirmed"] is False
    assert got["stability_neighbour_ratio"]["confirmed"] is False


def test_an_answer_that_differs_from_the_frozen_value_is_not_a_confirmation(reg):
    human_answer("stability-closeness", "stricter")
    got = {c["name"]: c for c in confirmation(reg.conn, "trend|equity|1D", frozen())}
    assert got["stability_closeness"]["status"] == "answered" and got["stability_closeness"]["confirmed"] is False
    assert got["stability_closeness"]["answered_value"] == 0.85 and got["stability_closeness"]["frozen_value"] == 0.7
    ok = {c["name"]: c for c in confirmation(reg.conn, "trend|equity|1D", frozen({"stability_closeness": 0.85}))}
    assert ok["stability_closeness"]["confirmed"] is True


def test_stability_items_are_only_for_strategy_scopes(reg):
    assert stability_items(reg.conn, "dataset:prices") == [] and len(stability_items(reg.conn, FAMILY)) == 4


# --- the notice at registration -----------------------------------------------------------------------------------


def test_registration_says_which_thresholds_the_trader_did_not_confirm(reg, home):
    res = register_new_trial(SPEC, reg, home)
    note = next(n for n in res.notes if "stability thresholds not confirmed by the trader" in n)
    assert "stability_closeness" in note and "qrh interview next --stage S0 --scope strategy:trend|equity|1D" in note


def test_registration_is_quiet_when_every_threshold_came_from_the_trader(reg, home):
    for qid, value in zip(QIDS, ("stricter", "keep_default", "looser", "wider")):
        human_answer(qid, value)
    res = register_new_trial(SPEC, reg, home)
    assert not any("stability thresholds" in n for n in res.notes)


def test_the_cli_prints_the_notice_without_blocking(home, tmp_path):
    spec = tmp_path / "h.json"
    spec.write_text(json.dumps(SPEC))
    res = runner.invoke(app, ["new", str(spec)])
    assert res.exit_code == 0 and "NOTE: stability thresholds not confirmed by the trader" in res.output


def test_keep_the_default_is_not_a_confirmation_when_the_frozen_value_is_not_the_default_in_force(reg):
    human_answer("stability-closeness", "keep_default")
    got = {c["name"]: c for c in confirmation(reg.conn, "trend|equity|1D", frozen({"stability_closeness": 0.8}))}
    assert got["stability_closeness"]["confirmed"] is False  # 0.8 is not the core default 0.7
    again = {c["name"]: c for c in confirmation(reg.conn, "trend|equity|1D", frozen({"stability_closeness": 0.8}),
                                                defaults={"stability_closeness": 0.8})}
    assert again["stability_closeness"]["confirmed"] is True
