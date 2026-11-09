"""The ten approved intake questions and answer validation. Pure: no I/O, no guessing."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

UNKNOWN = "unknown"
_UNKNOWN_WORDS = {"unknown", "i don't know", "i dont know", "idk", "don't know", "dont know"}


class AnswerError(ValueError):
    """An answer was refused; the message is plain language for the user."""


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    explanation: str
    kind: str  # text | text_list | currency | positive_number | choice
    options: tuple[str, ...] = ()


QUESTIONS: tuple[Question, ...] = (
    Question("country", "In which country do you live?",
             "This decides taxes, which brokers can serve you, and the currency you pay fees in.",
             "text"),
    Question("markets",
             "Which markets do you want to trade (for example US stocks, US options, futures)?",
             "This decides which exchanges and fee schedules apply.", "text_list"),
    Question("broker", "Which broker do you plan to use, or 'undecided'?",
             "The harness looks up that broker's official commission and tick size. "
             "'undecided' means it will compare options later.", "text"),
    Question("account_currency", "In which currency is your account?",
             "All money in the harness is converted to this currency, and currency "
             "mismatches block results. Use a 3-letter code such as USD.", "currency"),
    Question("initial_equity", "How much money do you plan to start with?",
             "Many brokers charge less per trade with higher volume, so this sets the fee tier. "
             "Give a positive number in your account currency.", "positive_number"),
    Question("trading_style",
             "How often would you trade, and with which bars: daily, hourly, or other?",
             "The length of each price bar decides how fresh prices must be.", "text"),
    Question("idea", "Describe your trading idea in one or two sentences.",
             "The harness uses this to set up the test.", "text"),
    Question("own_file",
             "Do you already have a backtest file, or should the harness build one from your idea?",
             "If you have one, the harness checks it. If not, it builds one and explains every number.",
             "choice", ("have_file", "build_for_me")),
    Question("data_source",
             "Where does your price data come from: a file you have, or a data vendor?",
             "The harness must know the source to check timing and data quality.",
             "choice", ("own_file", "vendor", "unknown")),
    Question("experience",
             "How would you describe your trading experience: beginner, intermediate, or advanced?",
             "The harness adjusts how much detail it explains.",
             "choice", ("beginner", "intermediate", "advanced")),
)

QUESTION_IDS = tuple(q.id for q in QUESTIONS)


def get_question(qid: str) -> Question:
    for q in QUESTIONS:
        if q.id == qid:
            return q
    raise AnswerError(f"There is no intake question called '{qid}'.")


def _is_unknown(raw: Any) -> bool:
    return isinstance(raw, str) and raw.strip().lower() in _UNKNOWN_WORDS


def validate_answer(question: Question, raw: Any) -> Any:
    """Return the cleaned answer, or raise AnswerError. 'I don't know' becomes 'unknown'."""
    if _is_unknown(raw):
        return UNKNOWN
    kind = question.kind
    if kind == "positive_number":
        if isinstance(raw, bool):
            raise AnswerError("Please give the amount as a number, for example 5000.")
        try:
            value = float(str(raw).replace(",", "").strip()) if isinstance(raw, str) else float(raw)
        except (TypeError, ValueError):
            raise AnswerError("Please give the amount as a number, for example 5000.") from None
        if not math.isfinite(value) or value <= 0:
            raise AnswerError("The amount must be a number greater than zero.")
        return value
    if kind == "text_list":
        items = raw.split(",") if isinstance(raw, str) else list(raw) if isinstance(raw, (list, tuple)) else None
        if items is None:
            raise AnswerError("Please list at least one item, separated by commas.")
        cleaned = [str(i).strip() for i in items if str(i).strip()]
        if not cleaned:
            raise AnswerError("Please list at least one item, separated by commas.")
        return cleaned
    if not isinstance(raw, str) or not raw.strip():
        raise AnswerError("This answer cannot be empty. You can also answer 'I don't know'.")
    text = raw.strip()
    if kind == "currency":
        if len(text) != 3 or not text.isalpha() or not text.isascii():
            raise AnswerError("Please use a 3-letter currency code, for example USD.")
        return text.upper()
    if kind == "choice":
        value = text.lower()
        if value not in question.options:
            raise AnswerError("Please choose one of: " + ", ".join(question.options) + ".")
        return value
    return text
