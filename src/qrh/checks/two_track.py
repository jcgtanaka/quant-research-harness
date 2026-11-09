"""Track 2 (the user brought the backtest): fixed questions and answer rules.

Pure module: no I/O and no wiring into the interview or pipeline. Answers are
never coerced; an answer outside the allowed options raises ``ValueError``.

Hook (documented, not implemented): ``explain_unexplained_remainder`` is the
name reserved for the later step that accounts for the part of the result the
answers do not explain. It is intentionally absent from this module.
"""

from __future__ import annotations

from dataclasses import dataclass

PASS = "PASS"
WARNING = "WARNING"
FAIL = "FAIL"

UNKNOWN = "unknown"
NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    status: str
    reason: str


@dataclass(frozen=True)
class TwoTrackResult:
    results: tuple[QuestionResult, ...]
    overall: str


QUESTIONS: tuple[Question, ...] = (
    Question("price_source", "Where do the prices come from?",
             ("broker", "data_vendor", "own_file", "unknown")),
    Question("price_adjustment", "How are prices adjusted for corporate actions?",
             ("none", "split_only", "total_return", "unknown")),
    Question("price_timing", "At which price is a trade assumed to fill?",
             ("bar_close", "next_open", "other", "unknown")),
    Question("bar_type", "What kind of bars does the backtest use?",
             ("time", "dollar", "volume", "mixed", "unknown")),
    Question("trade_size_rounding", "How are trade sizes rounded?",
             ("whole_units", "fractional", "unknown")),
    Question("cost_basis", "How are trading costs charged?",
             ("per_trade", "per_unit", "percentage", "none", "unknown")),
    Question("cash_interest", "Is interest on cash included?",
             ("included", "not_included", "not_applicable", "unknown")),
    Question("borrow_cost", "Is the cost of borrowing for shorts included?",
             ("included", "not_included", "not_applicable", "unknown")),
    Question("dividends", "Are dividends included?",
             ("included", "not_included", "not_applicable", "unknown")),
    Question("halts_and_stale", "Are trading halts and stale prices handled?",
             ("handled_declared", "not_handled", "not_applicable", "unknown")),
)

_BY_ID = {q.id: q for q in QUESTIONS}

# fact name -> (question id, answers that the fact must support)
_CONTRADICTION_RULES: dict[str, tuple[str, tuple[str, ...]]] = {
    "price_adjustment_consistent": ("price_adjustment", ("none", "split_only")),
}


def _validate(answers: dict[str, str]) -> None:
    for qid, answer in answers.items():
        question = _BY_ID.get(qid)
        if question is None:
            raise ValueError(f"unknown question id: {qid!r}")
        if answer not in question.options:
            raise ValueError(
                f"invalid answer {answer!r} for {qid}; allowed: {question.options}"
            )


def _contradiction(qid: str, answer: str, data_facts: dict[str, bool]) -> str | None:
    for fact, (rule_qid, matching) in _CONTRADICTION_RULES.items():
        if rule_qid == qid and answer in matching and data_facts.get(fact) is False:
            return (
                f"{qid}: answer {answer!r} contradicts data fact "
                f"{fact!r}=False"
            )
    return None


def _evaluate_one(
    question: Question,
    answers: dict[str, str],
    confirmed_by_data: dict[str, bool],
    data_facts: dict[str, bool],
) -> QuestionResult:
    qid = question.id
    answer = answers.get(qid)
    if answer is None or answer == UNKNOWN:
        return QuestionResult(
            qid, WARNING, f"{qid}: {question.text} Not answered or unknown."
        )
    contradiction = _contradiction(qid, answer, data_facts)
    if contradiction:
        return QuestionResult(qid, FAIL, contradiction)
    if answer == NOT_APPLICABLE and confirmed_by_data.get(qid) is not True:
        return QuestionResult(
            qid, WARNING,
            f"{qid}: {question.text} Answered not_applicable but not confirmed by data.",
        )
    return QuestionResult(qid, PASS, f"{qid}: settled as {answer!r}.")


def evaluate_two_track(
    answers: dict[str, str],
    confirmed_by_data: dict[str, bool],
    data_facts: dict[str, bool],
) -> TwoTrackResult:
    """Evaluate answers to the fixed question list.

    Raises ``ValueError`` for an unknown question id or a disallowed option.
    Missing answers give WARNING. Overall is FAIL > WARNING > PASS.
    """
    _validate(answers)
    results = tuple(
        _evaluate_one(q, answers, confirmed_by_data, data_facts) for q in QUESTIONS
    )
    statuses = {r.status for r in results}
    overall = FAIL if FAIL in statuses else WARNING if WARNING in statuses else PASS
    return TwoTrackResult(results, overall)
