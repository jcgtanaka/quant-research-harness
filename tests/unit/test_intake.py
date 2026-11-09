"""Intake: ten approved questions, validation, and a small JSON store."""
from __future__ import annotations

import json
import os

import pytest

from qrh.intake import (
    QUESTIONS,
    UNKNOWN,
    AnswerError,
    answers_path,
    load_answers,
    run_intake,
    save_answer,
    validate_answer,
)

IDS = [
    "country", "markets", "broker", "account_currency", "initial_equity",
    "trading_style", "idea", "own_file", "data_source", "experience",
]


def q(qid):
    return next(x for x in QUESTIONS if x.id == qid)


def test_ten_questions_in_order_with_explanations():
    assert [x.id for x in QUESTIONS] == IDS
    for x in QUESTIONS:
        assert x.text.strip() and x.explanation.strip()
    assert q("country").text == "In which country do you live?"


def test_choice_outside_options_is_refused():
    with pytest.raises(AnswerError) as e:
        validate_answer(q("own_file"), "maybe")
    assert "have_file" in str(e.value)
    assert validate_answer(q("own_file"), "have_file") == "have_file"
    assert validate_answer(q("experience"), "Beginner") == "beginner"


@pytest.mark.parametrize("bad", ["-5", -5, "0", "abc", "nan", "inf", True])
def test_bad_equity_is_refused(bad):
    with pytest.raises(AnswerError):
        validate_answer(q("initial_equity"), bad)


def test_good_equity_and_currency_and_country():
    assert validate_answer(q("initial_equity"), "12000.5") == 12000.5
    assert validate_answer(q("account_currency"), "usd") == "USD"
    with pytest.raises(AnswerError):
        validate_answer(q("account_currency"), "US")
    with pytest.raises(AnswerError):
        validate_answer(q("country"), "   ")


def test_markets_is_a_list():
    assert validate_answer(q("markets"), "US stocks, US options") == ["US stocks", "US options"]
    assert validate_answer(q("markets"), ["futures"]) == ["futures"]
    with pytest.raises(AnswerError):
        validate_answer(q("markets"), " , ")


@pytest.mark.parametrize("qid", IDS)
def test_unknown_accepted_for_any_question(qid):
    assert validate_answer(q(qid), "I don't know") == UNKNOWN
    assert validate_answer(q(qid), "unknown") == UNKNOWN


def test_round_trip(tmp_path):
    assert load_answers(tmp_path) == {}
    save_answer(tmp_path, "country", "Example Country")
    save_answer(tmp_path, "initial_equity", 5000.0)
    assert load_answers(tmp_path) == {"country": "Example Country", "initial_equity": 5000.0}
    assert answers_path(tmp_path) == tmp_path / "intake" / "answers.json"
    assert isinstance(json.loads(answers_path(tmp_path).read_text()), dict)


def test_unknown_question_id_refused(tmp_path):
    with pytest.raises(AnswerError):
        save_answer(tmp_path, "shoe_size", "9")


def test_store_written_atomically(tmp_path, monkeypatch):
    seen = []
    real = os.replace

    def spy(src, dst):
        seen.append((os.path.exists(src), os.path.exists(dst), str(src) != str(dst)))
        return real(src, dst)

    monkeypatch.setattr("qrh.intake.store.os.replace", spy)
    save_answer(tmp_path, "country", "Example Country")
    assert seen == [(True, False, True)]
    save_answer(tmp_path, "broker", "undecided")
    assert seen[1] == (True, True, True)
    leftovers = [p.name for p in (tmp_path / "intake").iterdir()]
    assert leftovers == ["answers.json"]


def test_full_sequence_stores_only_given_answers(tmp_path):
    script = {
        "country": "Example Country",
        "markets": "US stocks",
        "broker": "undecided",
        "account_currency": "usd",
        "initial_equity": "-1",  # refused first, then corrected below
        "trading_style": "daily",
        "idea": "Buy dips.",
        "own_file": "build_for_me",
        "data_source": "unknown",
        "experience": "beginner",
    }
    asked, refusals = [], []
    retry = {"initial_equity": "3000"}

    def ask(question):
        asked.append(question.id)
        if question.id in retry and asked.count(question.id) == 2:
            return retry[question.id]
        return script[question.id]

    stored = run_intake(ask, tmp_path, notify=refusals.append)
    assert asked.count("initial_equity") == 2 and len(asked) == 11
    assert len(refusals) == 1
    assert stored == load_answers(tmp_path)
    assert set(stored) == set(IDS)
    assert stored["initial_equity"] == 3000.0
    assert stored["account_currency"] == "USD"
    assert stored["data_source"] == UNKNOWN


def test_existing_answers_not_overwritten_unless_asked_again(tmp_path):
    save_answer(tmp_path, "country", "Example Country")
    asked = []

    def ask(question):
        asked.append(question.id)
        return "unknown"

    run_intake(ask, tmp_path)
    assert "country" not in asked
    assert load_answers(tmp_path)["country"] == "Example Country"
    run_intake(ask, tmp_path, redo=["country"])
    assert load_answers(tmp_path)["country"] == UNKNOWN


def test_corrupt_store_is_reported(tmp_path):
    p = answers_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text("[1, 2]")
    with pytest.raises(AnswerError):
        load_answers(tmp_path)
