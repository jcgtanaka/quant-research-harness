"""Interview questions generated from an onboarding model: one per field a human still has to settle.

Ids are stable (``onb-<field>``), the scope is ``dataset:<name>`` and the stage is S1. The question is critical for
the four fields whose error silently corrupts every result (multiplier, adjustments, timestamp semantics, price
units) and recommended for the rest. Its identity is the dataset content hash together with the current claim, so an
answer is stale as soon as the file or the claim it was about changes.
"""

from __future__ import annotations

import sqlite3

from qrh.onboarding import store
from qrh.onboarding.model import CRITICAL, ENUMS, FIELDS, InstrumentModel, claim_text
from qrh.packs.model import Choice, Question

QID_PREFIX = "onb-"
HELP = {
    "asset_class": "what kind of instrument the rows describe",
    "price_columns": "the columns that hold prices (comma separated)",
    "price_units": "how the price columns are scaled (fraction, percent, currency, index points, rate in percent)",
    "quote_currency": "the currency the prices are quoted in",
    "timestamp_semantics": "whether a timestamp marks the bar open, the bar close, an event or a publication time",
    "timezone": "the time zone of the timestamps",
    "session_hint": "the trading session the bars cover",
    "adjustments": "whether prices are adjusted for splits and dividends, or back/forward adjusted for rolls",
    "multiplier": "the contract multiplier (value of one price point per contract)",
    "tick_size": "the minimum price increment",
    "frequency": "the bar size",
    "roll_rule": "how the continuous futures series is rolled between contracts",
    "option_mapping": "which columns hold the option fields",
}
WHY = {
    "multiplier": "A wrong multiplier scales every profit and loss by the same factor, and nothing in the prices reveals it.",
    "adjustments": "Unadjusted splits look like crashes and back-adjusted prices are not real traded prices; either one changes every return.",
    "timestamp_semantics": "Whether a bar's timestamp is its open or its close decides whether a signal can have used its own data.",
    "price_units": "A series in percent read as a fraction is a hundred times too large, with no error raised.",
}


def qid_for(field: str) -> str:
    return QID_PREFIX + field


def field_of(qid: str) -> str | None:
    if qid.startswith(QID_PREFIX) and qid[len(QID_PREFIX):] in FIELDS:
        return qid[len(QID_PREFIX):]
    return None


def build_question(dataset: str, model: InstrumentModel, field: str) -> Question:
    claim = model.claims().get(field)
    shown = claim_text(field, claim.value) if claim else None
    if shown is None:
        text = f"Dataset {dataset}: what is the {field.replace('_', ' ')}? ({HELP[field]})"
    else:
        text = f"Dataset {dataset}: the proposed {field.replace('_', ' ')} is \"{shown}\". Is that right? ({HELP[field]})"
    choices: list[Choice] = []
    answer_type = "text"
    if field in ENUMS:
        answer_type = "choice"
        if claim is not None:
            choices.append(Choice(id="confirm", label=f"Yes, {claim.value} is correct"))
        choices += [Choice(id=v, label=f"No, it is {v}") for v in ENUMS[field] if claim is None or v != claim.value]
    else:
        text += " Answer \"confirm\" if it is right, or give the correct value."
    return Question(
        id=qid_for(field), stage="S1", scope="dataset", text=text,
        why_it_matters=WHY.get(field, f"The harness cannot prove this from the data: {HELP[field]}."),
        answer_type=answer_type, choices=choices, critical=field in CRITICAL,  # type: ignore[arg-type]
        unknown_risk=f"The {field.replace('_', ' ')} of dataset {dataset} stays unverified, so results built on it are conditional.",
    )


def _fields_to_ask(model: InstrumentModel) -> list[str]:
    rows = store.field_rows(model)
    return [r["field"] for r in rows if r["status"] in ("unverified", "confirmed")]


def dataset_questions(conn: sqlite3.Connection, scope_keys: set[str] | None = None
                      ) -> list[tuple[Question, str, bool, str]]:
    """(question, scope key, critical, identity) for every dataset with a verified onboarding of its current file."""
    out: list[tuple[Question, str, bool, str]] = []
    rows = conn.execute("SELECT name, content_hash FROM datasets WHERE content_hash IS NOT NULL ORDER BY name").fetchall()
    for r in rows:
        scope = f"dataset:{r['name']}"
        if scope_keys is not None and scope not in scope_keys:
            continue
        v = store.latest_version(conn, r["name"], r["content_hash"])
        if v is None or v["verified_at"] is None:
            continue
        model: InstrumentModel = v["model"]
        for field in _fields_to_ask(model):
            q = build_question(r["name"], model, field)
            out.append((q, scope, q.critical, f"{r['content_hash']}:{store.claim_digest(model, field)}"))
    return out


def find(conn: sqlite3.Connection, qid: str, scope_key: str) -> tuple[Question, bool, str] | None:
    for q, scope, crit, ident in dataset_questions(conn, {scope_key}):
        if q.id == qid and scope == scope_key:
            return q, crit, ident
    return None


def answer_identity(conn: sqlite3.Connection, qid: str, scope_key: str) -> str | None:
    """Identity an answer to ``qid`` is stored with, or None when the question is not an onboarding one."""
    field = field_of(qid)
    if field is None or not scope_key.startswith("dataset:"):
        return None
    name = scope_key.split(":", 1)[1]
    v = store.current_version(conn, name)
    if v is None:
        return None
    return f"{v['content_hash']}:{store.claim_digest(v['model'], field)}"


def asked_in(model: InstrumentModel) -> list[str]:
    return _fields_to_ask(model)


__all__ = ["build_question", "dataset_questions", "find", "answer_identity", "field_of", "qid_for", "asked_in"]
