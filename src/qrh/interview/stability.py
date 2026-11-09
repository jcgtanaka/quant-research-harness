"""The four stability thresholds, always asked of the trader.

Built into the harness (no pack needed, so no pack can leave them out); they use the same question model, scope keys,
answer storage and answer provenance as every pack question. Scope is the strategy family (``strategy:<family>``),
which is known before a hypothesis is registered, so an answer exists in time to be frozen into the hypothesis
thresholds at registration (``qrh.pipeline.stages.RegisterHypothesis``). Stage S0, never critical.

Each question offers "Keep the default" (recommended) and two alternatives; "I don't know" is the engine's own
always-valid answer. "Keep the default" maps to no value, so whatever the default is (the core value, or a pack that
tightened it) stays. An unanswered or "I don't know" question leaves the default and the readiness report warns
("stability thresholds not confirmed by the trader").
"""

from __future__ import annotations

import sqlite3
from typing import Any, Mapping

from qrh.interview.engine import UNKNOWN, Item, identity_for, latest_answer
from qrh.packs.model import Choice, Question
from qrh.packs.runtime import PackView

STAGE = "S0"
KEEP = "keep_default"
STABILITY_NAMES = ("stability_closeness", "stability_plateau_share", "stability_neighbour_ratio", "stability_steps")
UNKNOWN_RISK = "A stability threshold was not confirmed by the trader; the default applies and the check may be too loose or too strict."


def _q(qid: str, name: str, text: str, why: str, options: list[tuple[str, str, Any]]) -> Question:
    return Question(
        id=qid, stage=STAGE, scope="strategy", text=text, why_it_matters=why, answer_type="choice", critical=False,
        unknown_risk=UNKNOWN_RISK,
        choices=[Choice(id=cid, label=label, maps_to={"kind": "threshold", "name": name, "value": value})
                 for cid, label, value in options],
    )


_QUESTIONS: tuple[Question, ...] = (
    _q("stability-closeness", "stability_closeness",
       "The harness looks at settings near your chosen one. A nearby setting counts as 'about as good' when its "
       "out-of-sample Sharpe ratio is at least this share of your chosen setting's. How much of it should it keep? "
       "The harness suggests at least 70%.",
       "This decides how forgiving the plateau check is. Too loose and a lucky peak can pass as stable; too strict and a "
       "smooth strategy can fail on noise.",
       [(KEEP, "Keep the default: at least 70% of your chosen setting's (recommended)", None),
        ("stricter", "Stricter: at least 85%", 0.85),
        ("looser", "Looser: at least 55%", 0.55)]),
    _q("stability-plateau-share", "stability_plateau_share",
       "The harness nudges each number of your chosen setting up and down by the step sizes you pick below, one "
       "parameter at a time. What share of those nearby settings must be 'about as good' for the result to count as a "
       "plateau and not a lucky peak? The harness suggests at least 50% of the nearby settings.",
       "A strategy that works at one setting and nowhere near it is usually an overfit. This is judged around your "
       "chosen setting, so it does not depend on how widely your grid is spaced.",
       [(KEEP, "Keep the default: at least 50% of the nearby settings (recommended)", None),
        ("stricter", "Stricter: at least 65% of the nearby settings", 0.65),
        ("looser", "Looser: at least 35% of the nearby settings", 0.35)]),
    _q("stability-neighbour-ratio", "stability_neighbour_ratio",
       "For each parameter separately, the harness nudges it up and down and averages the out-of-sample Sharpe ratios of "
       "the nearby settings. The worst parameter must keep what share of your chosen setting's ratio? The harness "
       "suggests at least 50%.",
       "If a small change to even one parameter destroys the result, the result probably came from noise, not from the "
       "idea. Parameters that change nothing cannot hide this.",
       [(KEEP, "Keep the default: at least 50% for every parameter (recommended)", None),
        ("stricter", "Stricter: at least 70%", 0.70),
        ("looser", "Looser: at least 30%", 0.30)]),
    _q("stability-steps", "stability_steps",
       "How far should the harness nudge each number of your chosen setting, up and down, to build the nearby settings? "
       "This is used for a single registered setting and for a grid alike. The harness suggests 10% and 20% of each value.",
       "These nearby runs only check your fixed choice; they are not extra trials. Wider nudges look further from your "
       "choice and are harder to pass; no nearby setting is ever closer than the smallest nudge.",
       [(KEEP, "Keep the default: 10% and 20% (recommended)", None),
        ("smaller", "Smaller nudges: 5% and 10%", [0.05, 0.10]),
        ("wider", "Wider nudges: 10%, 20% and 30%", [0.10, 0.20, 0.30])]),
)
_NAME_OF = {q.id: n for q, n in zip(_QUESTIONS, STABILITY_NAMES)}
_ID_OF = {n: q for q, n in _NAME_OF.items()}


def stability_questions() -> list[Question]:
    return list(_QUESTIONS)


def family_scope(family_id: str) -> str:
    return f"strategy:{family_id}"


def stability_items(conn: sqlite3.Connection, scope_key: str) -> list[Item]:
    """The four questions for a ``strategy:<family>`` scope; nothing for any other scope."""
    if not scope_key.startswith("strategy:"):
        return []
    ident = identity_for(conn, PackView.empty(), scope_key)
    return [Item(q, scope_key, ident, False, "stability") for q in _QUESTIONS]


def trader_answers(conn: sqlite3.Connection, family_id: str) -> dict[str, dict[str, Any]]:
    """Per threshold name: the trader's current answer ``{qid, answer, version, provenance, value}`` (value None means
    "keep the default"). Only questions with a current answer other than "I don't know" appear."""
    scope = family_scope(family_id)
    ident = identity_for(conn, PackView.empty(), scope)
    out: dict[str, dict[str, Any]] = {}
    for q in _QUESTIONS:
        ans = latest_answer(conn, q.id, scope)
        if ans is None or ans["identity"] != ident or ans["answer"] == UNKNOWN:
            continue
        rule = ans["mapped"].get("threshold")
        if rule is None:
            continue
        out[rule["name"]] = {"qid": q.id, "answer": ans["answer"], "version": ans["version"],
                             "provenance": ans["provenance"], "value": rule["value"]}
    return out


def trader_thresholds(conn: sqlite3.Connection, family_id: str) -> dict[str, Any]:
    """Threshold values the trader chose explicitly (everything but "keep the default" and "I don't know")."""
    return {n: a["value"] for n, a in trader_answers(conn, family_id).items() if a["value"] is not None}


def spec_value_problems(spec_values: Mapping[str, Any], answers: Mapping[str, Mapping[str, Any]],
                        defaults: Mapping[str, Any]) -> list[str]:
    """Reasons a hypothesis may not set these stability thresholds itself. A value written in the spec must equal what the
    trader answered ("keep the default" means the default in force: the core value, or a pack's); with no answer (or "I
    don't know") it may not be looser than the default in force. The trader owns these four numbers."""
    from qrh.packs.layering import is_looser

    out = []
    for name in STABILITY_NAMES:
        if name not in spec_values:
            continue
        v = spec_values[name]
        a = answers.get(name)
        if a is not None:
            want = a["value"] if a["value"] is not None else defaults[name]
            if v != want:
                out.append(
                    f"the hypothesis sets {name} to {v!r}, but the trader answered {a['qid']} "
                    f"({'keep the default' if a['value'] is None else repr(a['value'])}), which means {want!r}; the trader "
                    f"must answer the question with that value, or remove {name} from the spec")
        elif is_looser(name, defaults[name], v):
            out.append(
                f"the hypothesis sets {name} to {v!r}, looser than the default in force {defaults[name]!r}, and the trader "
                f"has not answered {_ID_OF[name]}; ask the trader (`qrh interview next --stage S0`) or remove {name} from the spec")
    return out


def confirmation(conn: sqlite3.Connection, family_id: str, frozen: Mapping[str, Any],
                 defaults: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """One row per threshold: ``status`` answered, unknown or pending, and ``confirmed`` (an answer that agrees with the
    frozen threshold; "keep the default" agrees only when the frozen value IS the default in force at registration,
    ``defaults``, else the core default)."""
    from qrh.validation.defaults import DEFAULT_THRESHOLDS

    defaults = defaults or DEFAULT_THRESHOLDS
    scope = family_scope(family_id)
    ident = identity_for(conn, PackView.empty(), scope)
    answers = trader_answers(conn, family_id)
    rows = []
    for q in _QUESTIONS:
        name = _NAME_OF[q.id]
        ans = latest_answer(conn, q.id, scope)
        current = ans is not None and ans["identity"] == ident
        status = "pending" if not current else ("unknown" if ans["answer"] == UNKNOWN else "answered")
        row: dict[str, Any] = {"name": name, "qid": q.id, "status": status, "confirmed": False,
                               "answered_value": None, "frozen_value": frozen.get(name)}
        if status == "answered":
            value = answers[name]["value"] if name in answers else None
            row["answered_value"] = value if value is not None else defaults.get(name)
            row["confirmed"] = row["answered_value"] == frozen.get(name)
        rows.append(row)
    return rows
