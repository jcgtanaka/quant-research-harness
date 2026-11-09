"""Questions, scope keys, answer state and answer mapping. Pure functions over a registry connection.

Scope keys: ``hypothesis:<trial_id>``, ``dataset:<name>``, ``strategy:<family_id>``, ``account:<name>``,
``session:<id>``. An answer is reused until the *identity* of its scope changes (hypothesis hash, dataset content
hash, account hash); a changed answer is a new version, and receipts reference ``qid@scope@version``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from qrh.hashing import hash_json
from qrh.packs.model import Question
from qrh.packs.runtime import PackView

UNKNOWN = "unknown"
DEFAULT_ACCOUNT = "default"


class InterviewError(ValueError):
    """An interview operation was refused (the message says why)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scope_kind(scope_key: str) -> str:
    kind, sep, rest = scope_key.partition(":")
    if not sep or not rest or kind not in ("hypothesis", "dataset", "strategy", "account", "session"):
        raise InterviewError(f"bad scope {scope_key!r}; use hypothesis:<trial>, dataset:<name>, strategy:<family>, "
                             f"account:<name> or session:<id>")
    return kind


def identity_for(conn: sqlite3.Connection, view: PackView, scope_key: str) -> str:
    kind, _, rest = scope_key.partition(":")
    if kind == "hypothesis":
        row = conn.execute("SELECT hypothesis_hash FROM hypotheses WHERE trial_id=?", (rest,)).fetchone()
        return row[0] if row else ""
    if kind == "dataset":
        row = conn.execute("SELECT content_hash FROM datasets WHERE name=?", (rest,)).fetchone()
        return (row[0] or "") if row else ""
    if kind == "account":
        return view.account_hash()
    return rest if kind == "strategy" else ""


@dataclass(frozen=True)
class Item:
    question: Question
    scope_key: str
    identity: str
    critical: bool
    origin: str = "pack"

    @property
    def key(self) -> str:
        return f"{self.question.id}@{self.scope_key}"


def llm_questions(conn: sqlite3.Connection) -> list[tuple[Question, str]]:
    out = []
    for r in conn.execute("SELECT * FROM interview_questions WHERE origin='llm' ORDER BY created_at, qid"):
        out.append((Question(id=r["qid"], stage=r["stage"], scope=scope_kind(r["scope_key"]),  # type: ignore[arg-type]
                             text=r["text"], why_it_matters=r["why"], answer_type="text", critical=False),
                    r["scope_key"]))
    return out


def onboarding_items(conn: sqlite3.Connection, scope_keys: set[str]) -> list[Item]:
    """Questions about the claims of a dataset's instrument model that a human still has to settle (see
    ``qrh.onboarding``); critical for the multiplier, adjustments, timestamp semantics and price units."""
    from qrh.onboarding.questions import dataset_questions

    return [Item(q, scope, ident, crit, "onboarding") for q, scope, crit, ident in dataset_questions(conn, scope_keys)]


def find_question(conn: sqlite3.Connection, view: PackView, qid: str, scope_key: str) -> tuple[Question, str] | None:
    """(question, origin) for an id usable in this scope, or None."""
    kind = scope_kind(scope_key)
    for q in view.questions():
        if q.id == qid and q.scope == kind:
            return q, "pack"
    for q, key in llm_questions(conn):
        if q.id == qid and key == scope_key:
            return q, "llm"
    for it in onboarding_items(conn, {scope_key}):
        if it.question.id == qid:
            return it.question, "onboarding"
    for it in track2_items(conn, view, scope_key):
        if it.question.id == qid:
            return it.question, "track2"
    from qrh.interview.stability import stability_items

    for it in stability_items(conn, scope_key):
        if it.question.id == qid:
            return it.question, "stability"
    return None


T2_PREFIX = "t2-"
T2_STAGE = "S1"


def track2_items(conn: sqlite3.Connection, view: PackView, scope_key: str) -> list[Item]:
    """The ten fixed Track 2 questions (``qrh.checks.two_track``) for a hypothesis the user brought (origin user).

    Empty for any other scope or origin. The answer rules stay in ``two_track``; this only schedules the questions."""
    kind, _, trial_id = scope_key.partition(":")
    if kind != "hypothesis":
        return []
    row = conn.execute("SELECT binding FROM hypotheses WHERE trial_id=?", (trial_id,)).fetchone()
    if row is None or json.loads(row[0]).get("origin") != "user":
        return []
    from qrh.checks.two_track import QUESTIONS, UNKNOWN as T2_UNKNOWN
    from qrh.packs.model import Choice

    ident = identity_for(conn, view, scope_key)
    return [Item(Question(id=T2_PREFIX + q.id, stage=T2_STAGE, scope="hypothesis", text=q.text, answer_type="choice",
                          choices=[Choice(id=o, label=o.replace("_", " ")) for o in q.options if o != T2_UNKNOWN],
                          unknown_risk=f"{q.id} is unknown: the result cannot be checked against it."),
                 scope_key, ident, False, "track2") for q in QUESTIONS]


def track2_answers(conn: sqlite3.Connection, scope_key: str) -> dict[str, str]:
    """Current Track 2 answers for a hypothesis, keyed by two_track question id, ready for ``evaluate_two_track``."""
    from qrh.checks.two_track import QUESTIONS

    out: dict[str, str] = {}
    for q in QUESTIONS:
        ans = latest_answer(conn, T2_PREFIX + q.id, scope_key)
        if ans is not None:
            out[q.id] = ans["answer"]
    return out


def trial_scope_keys(trial: dict[str, Any], view: PackView) -> dict[str, list[str]]:
    acct = view.account()
    return {
        "hypothesis": [f"hypothesis:{trial['trial_id']}"],
        "dataset": [f"dataset:{d['name']}" for d in trial.get("datasets", [])],
        "strategy": [f"strategy:{trial['family_id']}"],
        "account": [f"account:{acct[0] if acct else DEFAULT_ACCOUNT}"],
        "session": [],
    }


def items_for_trial(conn: sqlite3.Connection, view: PackView, trial: dict[str, Any]) -> list[Item]:
    """Every question that applies to this trial, one item per scope key, in bank order."""
    keys = trial_scope_keys(trial, view)
    required = view.required_questions(trial)
    items: list[Item] = []
    for q in view.questions():
        if not q.applies_to.matches(trial):
            continue
        for key in keys[q.scope]:
            items.append(Item(q, key, identity_for(conn, view, key), q.critical or q.id in required))
    wanted = {k for ks in keys.values() for k in ks}
    for q, key in llm_questions(conn):
        if key in wanted:
            items.append(Item(q, key, identity_for(conn, view, key), False, "llm"))
    items += onboarding_items(conn, wanted)
    for key in keys["hypothesis"]:
        items += track2_items(conn, view, key)
    from qrh.interview.stability import stability_items

    for key in keys["strategy"]:  # the four stability thresholds: always asked, with or without a pack
        items += stability_items(conn, key)
    return items


def latest_answer(conn: sqlite3.Connection, qid: str, scope_key: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM interview_answers WHERE qid=? AND scope_key=? ORDER BY version DESC LIMIT 1",
                       (qid, scope_key)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["mapped"] = json.loads(d["mapped"])
    return d


def state_of(conn: sqlite3.Connection, item: Item) -> dict[str, Any]:
    """status is answered, unknown or pending (no answer, or the scope identity changed since it was given)."""
    ans = latest_answer(conn, item.question.id, item.scope_key)
    base: dict[str, Any] = {"qid": item.question.id, "scope": item.scope_key, "stage": item.question.stage,
                            "text": item.question.text, "why": item.question.why_it_matters,
                            "critical": item.critical, "origin": item.origin}
    if ans is None:
        return {**base, "status": "pending", "stale": False}
    if ans["identity"] != item.identity:
        return {**base, "status": "pending", "stale": True, "previous_version": ans["version"]}
    status = "unknown" if ans["answer"] == UNKNOWN else "answered"
    label = ans["answer"]
    for c in item.question.choices:
        if c.id == ans["answer"]:
            label = c.label
    risk = ans["mapped"].get("risk") or ""
    return {**base, "status": status, "stale": False, "answer": ans["answer"], "label": label, "version": ans["version"],
            "risk": risk, "unknown_risk": item.question.unknown_risk if status == "unknown" else "",
            "mapped": ans["mapped"], "answered_by": ans["answered_by"], "provenance": ans["provenance"],
            "answered_at": ans["answered_at"]}


def next_item(conn: sqlite3.Connection, items: list[Item], include_unknown: bool = False) -> tuple[Item | None, int]:
    """The next unanswered item (critical first, then bank order) and how many remain pending.

    ``include_unknown`` also offers questions the human answered "don't know", to revisit them."""
    wanted = ("pending", "unknown") if include_unknown else ("pending",)
    pending = [it for it in items if state_of(conn, it)["status"] in wanted]
    ordered = sorted(pending, key=lambda it: not it.critical)  # stable: bank order inside each group
    return (ordered[0] if ordered else None), len(pending)


def issue(conn: sqlite3.Connection, qid: str, scope_key: str) -> str:
    when = _now()
    conn.execute("INSERT INTO interview_issued(qid, scope_key, issued_at) VALUES (?,?,?)", (qid, scope_key, when))
    return when


def last_issued(conn: sqlite3.Connection, qid: str, scope_key: str) -> str | None:
    row = conn.execute("SELECT issued_at FROM interview_issued WHERE qid=? AND scope_key=? ORDER BY id DESC LIMIT 1",
                       (qid, scope_key)).fetchone()
    return row[0] if row else None


def canonical_answer(q: Question, value: str) -> tuple[str, Any]:
    """(stored answer, matching choice or None). Raises InterviewError for a value the question cannot take."""
    v = value.strip()
    if not v:
        raise InterviewError("the answer is empty")
    if v.casefold() == UNKNOWN:
        return UNKNOWN, None
    if q.answer_type == "choice":
        for c in q.choices:
            if v.casefold() in (c.id.casefold(), c.label.casefold()):
                return c.id, c
        raise InterviewError(f"{v!r} is not one of the choices: " + ", ".join(f"{c.id} ({c.label})" for c in q.choices)
                             + ", or unknown")
    if q.answer_type == "number":
        try:
            n = float(v.replace(",", ""))
        except ValueError:
            raise InterviewError(f"{v!r} is not a number") from None
        return (str(int(n)) if n == int(n) else str(n)), None
    if q.answer_type == "bool":
        if v.casefold() in ("yes", "y", "true", "1"):
            return "yes", None
        if v.casefold() in ("no", "n", "false", "0"):
            return "no", None
        raise InterviewError(f"{v!r} is not yes or no")
    return v, None


def apply_mapping(q: Question, scope_key: str, answer: str, choice: Any) -> tuple[dict[str, Any], int]:
    """What the answer changes: a risk flag on the dataset, an extra trial count, a named parameter/threshold/flag."""
    mapped: dict[str, Any] = {}
    extra = 0
    if answer == UNKNOWN:
        if q.unknown_risk:
            mapped["unknown_risk"] = q.unknown_risk
        return mapped, 0
    if choice is not None and choice.risk:
        mapped["risk"] = choice.risk
        if scope_key.startswith("dataset:"):
            mapped["dataset_flag"] = {"dataset": scope_key.split(":", 1)[1], "flag": choice.risk}
    rule = (choice.maps_to if choice is not None and choice.maps_to else None) or q.maps_to
    if rule:
        kind = rule.get("kind")
        if kind == "imported_trials":
            extra = max(0, int(float(answer)))
            mapped["imported_trials"] = extra
        elif kind in ("param", "threshold", "flag", "rule"):
            # a rule may carry its own value (a choice that stands for a number or a list, or None for "the default")
            mapped[kind] = {"name": rule.get("name"), "value": rule["value"] if "value" in rule else answer}
        else:
            raise InterviewError(f"unknown maps_to kind {kind!r}")
    return mapped, extra


def record_answer(
    conn: sqlite3.Connection, *, q: Question, scope_key: str, identity: str, value: str, answered_by: str,
    provenance: str, session_id: str | None,
) -> dict[str, Any]:
    answer, choice = canonical_answer(q, value)
    mapped, extra = apply_mapping(q, scope_key, answer, choice)
    prev = latest_answer(conn, q.id, scope_key)
    version = (prev["version"] + 1) if prev else 1
    conn.execute(
        "INSERT INTO interview_answers(qid, scope_key, version, identity, answer, answered_at, answered_by, provenance,"
        " session_id, mapped, extra_trials) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (q.id, scope_key, version, identity, answer, _now(), answered_by, provenance, session_id,
         json.dumps(mapped, sort_keys=True), extra))
    return {"qid": q.id, "scope": scope_key, "version": version, "answer": answer, "mapped": mapped}


def add_llm_question(conn: sqlite3.Connection, *, stage: str, scope_key: str, text: str, why: str) -> str:
    from qrh.packs.model import STAGES

    scope_kind(scope_key)
    if stage not in STAGES:
        raise InterviewError(f"unknown stage {stage!r}")
    if not text.strip():
        raise InterviewError("the question text is empty")
    qid = "llm-" + hash_json({"stage": stage, "scope": scope_key, "text": " ".join(text.split())})[:10]
    conn.execute("INSERT OR IGNORE INTO interview_questions(qid, scope_key, stage, text, why, origin, created_at)"
                 " VALUES (?,?,?,?,?,'llm',?)", (qid, scope_key, stage, text.strip(), why.strip(), _now()))
    return qid


def dataset_flags(conn: sqlite3.Connection, view: PackView, dataset: str) -> list[str]:
    """Risk flags a dataset carries from its current interview answers (sorted, unique)."""
    key = f"dataset:{dataset}"
    ident = identity_for(conn, view, key)
    flags = set()
    for r in conn.execute("SELECT a.* FROM interview_answers a WHERE a.scope_key=? AND a.version="
                          "(SELECT MAX(version) FROM interview_answers b WHERE b.qid=a.qid AND b.scope_key=a.scope_key)",
                          (key,)):
        if r["identity"] == ident:
            f = json.loads(r["mapped"]).get("dataset_flag")
            if f:
                flags.add(f["flag"])
    return sorted(flags)
