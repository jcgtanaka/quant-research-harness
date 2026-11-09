"""Tests for the Track 2 question list and answer rules (pure module)."""

from __future__ import annotations

import pytest

from qrh.checks.two_track import (
    FAIL,
    PASS,
    QUESTIONS,
    WARNING,
    evaluate_two_track,
)

EXPECTED_OPTIONS = {
    "price_source": ("broker", "data_vendor", "own_file", "unknown"),
    "price_adjustment": ("none", "split_only", "total_return", "unknown"),
    "price_timing": ("bar_close", "next_open", "other", "unknown"),
    "bar_type": ("time", "dollar", "volume", "mixed", "unknown"),
    "trade_size_rounding": ("whole_units", "fractional", "unknown"),
    "cost_basis": ("per_trade", "per_unit", "percentage", "none", "unknown"),
    "cash_interest": ("included", "not_included", "not_applicable", "unknown"),
    "borrow_cost": ("included", "not_included", "not_applicable", "unknown"),
    "dividends": ("included", "not_included", "not_applicable", "unknown"),
    "halts_and_stale": ("handled_declared", "not_handled", "not_applicable", "unknown"),
}

SETTLED = {
    "price_source": "broker",
    "price_adjustment": "total_return",
    "price_timing": "next_open",
    "bar_type": "time",
    "trade_size_rounding": "whole_units",
    "cost_basis": "per_trade",
    "cash_interest": "included",
    "borrow_cost": "included",
    "dividends": "included",
    "halts_and_stale": "handled_declared",
}


def _status(result, qid):
    return next(r for r in result.results if r.question_id == qid)


def test_question_ids_unique_and_ordered():
    ids = [q.id for q in QUESTIONS]
    assert ids == list(EXPECTED_OPTIONS)
    assert len(set(ids)) == 10


def test_question_options_match_design():
    for q in QUESTIONS:
        assert q.options == EXPECTED_OPTIONS[q.id]
        assert q.text


def test_questions_are_frozen():
    with pytest.raises(Exception):
        QUESTIONS[0].id = "x"  # type: ignore[misc]


def test_fully_settled_is_pass():
    result = evaluate_two_track(SETTLED, {}, {})
    assert result.overall == PASS
    assert all(r.status == PASS for r in result.results)


def test_unknown_answer_warns_naming_question():
    answers = {**SETTLED, "bar_type": "unknown"}
    result = evaluate_two_track(answers, {}, {})
    r = _status(result, "bar_type")
    assert r.status == WARNING
    assert "bar_type" in r.reason
    assert next(q.text for q in QUESTIONS if q.id == "bar_type") in r.reason
    assert result.overall == WARNING


def test_not_applicable_without_confirmation_warns():
    answers = {**SETTLED, "borrow_cost": "not_applicable"}
    result = evaluate_two_track(answers, {}, {})
    assert _status(result, "borrow_cost").status == WARNING
    result = evaluate_two_track(answers, {"borrow_cost": False}, {})
    assert _status(result, "borrow_cost").status == WARNING


def test_not_applicable_with_confirmation_passes():
    answers = {**SETTLED, "borrow_cost": "not_applicable"}
    result = evaluate_two_track(answers, {"borrow_cost": True}, {})
    assert _status(result, "borrow_cost").status == PASS
    assert result.overall == PASS


def test_data_contradiction_fails_naming_both():
    for answer in ("none", "split_only"):
        answers = {**SETTLED, "price_adjustment": answer}
        result = evaluate_two_track(
            answers, {}, {"price_adjustment_consistent": False}
        )
        r = _status(result, "price_adjustment")
        assert r.status == FAIL
        assert "price_adjustment_consistent" in r.reason
        assert answer in r.reason
        assert result.overall == FAIL


def test_true_fact_or_nonmatching_answer_does_not_fail():
    answers = {**SETTLED, "price_adjustment": "none"}
    result = evaluate_two_track(answers, {}, {"price_adjustment_consistent": True})
    assert _status(result, "price_adjustment").status == PASS
    # total_return does not match the fact's answers
    result = evaluate_two_track(SETTLED, {}, {"price_adjustment_consistent": False})
    assert _status(result, "price_adjustment").status == PASS


def test_invalid_option_raises_value_error():
    with pytest.raises(ValueError):
        evaluate_two_track({**SETTLED, "bar_type": "tick"}, {}, {})


def test_unknown_question_id_raises_value_error():
    with pytest.raises(ValueError):
        evaluate_two_track({**SETTLED, "nonsense": "broker"}, {}, {})


def test_missing_answer_warns_not_raises():
    answers = dict(SETTLED)
    del answers["dividends"]
    result = evaluate_two_track(answers, {}, {})
    r = _status(result, "dividends")
    assert r.status == WARNING
    assert "dividends" in r.reason
    assert len(result.results) == 10


def test_empty_answers_all_warning():
    result = evaluate_two_track({}, {}, {})
    assert result.overall == WARNING
    assert all(r.status == WARNING for r in result.results)


def test_overall_reduction_fail_over_warning_over_pass():
    answers = {**SETTLED, "bar_type": "unknown", "price_adjustment": "none"}
    result = evaluate_two_track(answers, {}, {"price_adjustment_consistent": False})
    assert result.overall == FAIL
    result = evaluate_two_track(answers, {}, {})
    assert result.overall == WARNING
    result = evaluate_two_track(SETTLED, {}, {})
    assert result.overall == PASS
