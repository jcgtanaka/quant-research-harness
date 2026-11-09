"""``qrh interview ...``: ask-only-what-is-missing questions and answers with provenance.

Warn-only: nothing here blocks a stage. Exit code 2 means a refused operation (bad scope, answer not found in a
user message, unknown question), like the rest of the CLI.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import typer

from qrh.config import QrhHomeError, ensure_home
from qrh.interview import engine
from qrh.interview.engine import InterviewError, Item
from qrh.onboarding.model import OnboardingError

interview_app = typer.Typer(help="Interview: questions the harness asks the human, with answer provenance.",
                            no_args_is_help=True, add_completion=False)
BLOCKED = 2


def _fail(message: str) -> typer.Exit:
    typer.echo(f"BLOCKED: {message}", err=True)
    return typer.Exit(code=BLOCKED)


@contextmanager
def _open() -> Iterator[tuple[Path, Any, Any]]:
    from qrh.memory.registry import Registry
    from qrh.packs.model import PackError
    from qrh.packs.runtime import PackView

    try:
        home = ensure_home()
        reg = Registry(home / "registry.sqlite3")
    except QrhHomeError as exc:
        raise _fail(str(exc)) from exc
    try:
        try:
            view = PackView.load(home)
        except PackError as exc:
            raise _fail(str(exc)) from exc
        yield home, reg, view
    except typer.Exit:
        raise
    except InterviewError as exc:
        raise _fail(str(exc)) from exc
    except OnboardingError as exc:
        raise _fail(str(exc)) from exc
    finally:
        reg.close()


def _qdict(item: Item) -> dict[str, Any]:
    q = item.question
    return {"id": q.id, "stage": q.stage, "scope": item.scope_key, "text": q.text, "why_it_matters": q.why_it_matters,
            "answer_type": q.answer_type, "choices": [{"id": c.id, "label": c.label} for c in q.choices],
            "critical": item.critical, "origin": item.origin}


def _scope_items(reg: Any, view: Any, stage: str, scope: str, trial_id: str | None) -> list[Item]:
    kind = engine.scope_kind(scope)
    if trial_id:
        trial = reg.get_trial(trial_id)
        if trial is None:
            raise InterviewError(f"unknown trial {trial_id!r}")
        return [it for it in engine.items_for_trial(reg.conn, view, trial)
                if it.question.stage == stage and it.scope_key == scope]
    ident = engine.identity_for(reg.conn, view, scope)
    items = [Item(q, scope, ident, q.critical) for q in view.questions() if q.stage == stage and q.scope == kind]
    items += [Item(q, scope, ident, False, "llm") for q, key in engine.llm_questions(reg.conn)
              if key == scope and q.stage == stage]
    items += [it for it in engine.onboarding_items(reg.conn, {scope}) if it.question.stage == stage]
    items += [it for it in engine.track2_items(reg.conn, view, scope) if it.question.stage == stage]
    from qrh.interview.stability import stability_items

    items += [it for it in stability_items(reg.conn, scope) if it.question.stage == stage]
    return items


@interview_app.command("next")
def interview_next(
    stage: str = typer.Option(..., "--stage", help="S0..S7 or B1..B6."),
    scope: str = typer.Option(..., "--scope", help="dataset:<name>, hypothesis:<trial>, strategy:<family>, ..."),
    trial: str | None = typer.Option(None, "--trial", help="Apply the questions' applies_to filters for this trial."),
    include_unknown: bool = typer.Option(False, "--include-unknown", help="Also offer questions answered \"don't know\"."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Print the next unanswered question (critical first) and record when it was issued."""
    with _open() as (_home, reg, view):
        item, remaining = engine.next_item(reg.conn, _scope_items(reg, view, stage, scope, trial), include_unknown)
        if item is not None:
            engine.issue(reg.conn, item.question.id, item.scope_key)
    if as_json:
        typer.echo(json.dumps({"scope": scope, "stage": stage, "remaining": remaining,
                               "question": _qdict(item) if item else None}, indent=1, sort_keys=True))
        return
    if item is None:
        typer.echo(f"nothing to ask for {stage} {scope}")
        return
    q = _qdict(item)
    typer.echo(f"[{q['id']}] {'CRITICAL ' if q['critical'] else ''}{q['text']}")
    typer.echo(f"  why it matters: {q['why_it_matters']}")
    for c in q["choices"]:
        typer.echo(f"  - {c['id']}: {c['label']}")
    typer.echo("  - unknown: I don't know")


def _check_session(home: Path, session: str, transcript: Path) -> None:
    """F5: the transcript must be the one Claude Code reported for this session, under ~/.claude/projects."""
    import os

    from qrh.hooks import sessions

    recorded = sessions.lookup(home, session)
    if recorded is None:
        raise _fail(f"session {session!r} was never reported by a Claude Code hook (SessionStart/UserPromptSubmit); "
                    "use the id of the current session")
    given = os.path.abspath(os.path.expanduser(str(transcript)))
    if os.path.abspath(recorded) != given:
        raise _fail(f"transcript {given!r} is not the transcript Claude Code reported for session {session!r}")
    projects = os.path.realpath(os.path.join(os.path.expanduser("~"), ".claude", "projects"))
    if not os.path.realpath(given).startswith(projects + os.sep):
        raise _fail(f"transcript must live under ~/.claude/projects (got {given!r})")


@interview_app.command("answer")
def interview_answer(
    qid: str = typer.Argument(..., help="Question id from `qrh interview next`."),
    scope: str = typer.Option(..., "--scope", help="Scope key the question was issued for."),
    value: str = typer.Option(..., "--value", help="The user's answer, or the chosen option (or unknown)."),
    transcript: Path | None = typer.Option(None, "--transcript", help="Claude Code transcript (.jsonl) holding the user's words."),
    session: str | None = typer.Option(None, "--session", help="Session id inside the transcript."),
    human: bool = typer.Option(False, "--human", help="A person typing in their own terminal. The agent may not use this."),
) -> None:
    """Record an answer. The value must appear in a USER message of the transcript, posted after the question."""
    from qrh.interview.provenance import verify_answer

    with _open() as (home, reg, view):
        found = engine.find_question(reg.conn, view, qid, scope)
        if found is None:
            raise _fail(f"no question {qid!r} for scope {scope!r}")
        q, origin = found
        answer, choice = engine.canonical_answer(q, value)  # refuses a value the question cannot take
        if origin == "onboarding":  # refuses a value that is not valid for the dataset field, before anything is checked
            from qrh.onboarding import workflow

            workflow.check_answer(reg.conn, scope.split(":", 1)[1], q.id.removeprefix("onb-"), answer)
        if human:
            provenance, by = "human-cli", "human"
        else:
            if transcript is None or session is None:
                raise _fail("give --transcript and --session so the answer can be checked against the user's own "
                            "words (a person at a terminal may use --human instead)")
            _check_session(home, session, transcript)
            issued = engine.last_issued(reg.conn, qid, scope)
            if issued is None:
                raise _fail(f"question {qid!r} was not issued for {scope!r}; run `qrh interview next` and ask the user first")
            if answer == engine.UNKNOWN:
                kind, candidates = "text", [engine.UNKNOWN]
            elif choice is not None:
                kind, candidates = "choice", [choice.id, choice.label, choice.label.split(" (")[0]]
            elif q.answer_type == "bool":
                kind, candidates = "bool", [value, answer]
            else:
                kind, candidates = ("number" if q.answer_type == "number" else "text"), [value, answer]
            try:
                ok = verify_answer(transcript, session, issued, kind=kind, candidates=candidates)
            except OSError as exc:
                raise _fail(f"cannot read transcript {str(transcript)!r}: {exc}") from exc
            if not ok:
                raise _fail("answer not found in a user message of that transcript (after the question was issued); "
                            "ask the user and use their own words")
            provenance, by = "transcript-verified", f"user via session {session}"
        if origin == "onboarding":  # applies the answer to the instrument model, then records it with the new identity
            from qrh.memory.catalog import Catalog
            from qrh.onboarding import workflow

            cat = Catalog(reg.path)
            try:
                workflow.answer_question(cat, scope.split(":", 1)[1], q.id.removeprefix("onb-"), answer,
                                         answered_by=by, provenance=provenance, session_id=session)
            finally:
                cat.close()
            rec = {"qid": q.id, "scope": scope, "answer": answer, "version": (engine.latest_answer(reg.conn, q.id, scope) or {}).get("version"),
                   "mapped": (engine.latest_answer(reg.conn, q.id, scope) or {}).get("mapped", {})}
        else:
            rec = engine.record_answer(reg.conn, q=q, scope_key=scope, identity=engine.identity_for(reg.conn, view, scope),
                                       value=value, answered_by=by, provenance=provenance, session_id=session)
    typer.echo(f"recorded {qid} = {rec['answer']} (version {rec['version']}, {provenance})")
    if rec["mapped"]:
        typer.echo("mapped: " + json.dumps(rec["mapped"], sort_keys=True))


@interview_app.command("add-question")
def interview_add_question(
    stage: str = typer.Option(..., "--stage"),
    scope: str = typer.Option(..., "--scope"),
    text: str = typer.Option(..., "--text"),
    why: str = typer.Option("", "--why"),
) -> None:
    """Record an ad-hoc follow-up question (origin llm; never critical). Prints its id."""
    with _open() as (_home, reg, _view):
        qid = engine.add_llm_question(reg.conn, stage=stage, scope_key=scope, text=text, why=why)
    typer.echo(qid)


@interview_app.command("status")
def interview_status(
    trial: str | None = typer.Option(None, "--trial"),
    stage: str | None = typer.Option(None, "--stage"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Answered/unknown/pending per stage for a trial, or every recorded answer without --trial."""
    from qrh.interview.snapshot import build_snapshot

    with _open() as (_home, reg, view):
        if trial:
            t = reg.get_trial(trial)
            if t is None:
                raise _fail(f"unknown trial {trial!r}")
            snap = build_snapshot(reg.conn, view, t, stage)
            doc: dict[str, Any] = snap
        else:
            rows = reg.conn.execute(
                "SELECT a.* FROM interview_answers a WHERE a.version=(SELECT MAX(version) FROM interview_answers b"
                " WHERE b.qid=a.qid AND b.scope_key=a.scope_key) ORDER BY a.id").fetchall()
            doc = {"answers": [{"qid": r["qid"], "scope": r["scope_key"], "version": r["version"], "answer": r["answer"],
                                "answered_by": r["answered_by"], "provenance": r["provenance"],
                                "answered_at": r["answered_at"]} for r in rows]}
    if as_json:
        typer.echo(json.dumps(doc, indent=1, sort_keys=True))
        return
    if "answers" in doc:
        for a in doc["answers"]:
            typer.echo(f"{a['qid']}@{a['scope']}  v{a['version']}  {a['answer']}  {a['provenance']}")
        if not doc["answers"]:
            typer.echo("no answers recorded")
        return
    for st, body in doc["stages"].items():
        c = body["counts"]
        typer.echo(f"{st}: {c['answered']} answered, {c['unknown']} unknown, {c['pending']} pending  [{body['light']}]")
        for r in body["rows"]:
            typer.echo(f"  {r['status']:8} {'*' if r['critical'] else ' '} {r['qid']}@{r['scope']}"
                       + (f"  = {r['label']}" if "label" in r else ""))
    for c in doc["conditions"]:
        typer.echo(f"condition: {c['question_id']} ({c['status']}) {c['risk']}".rstrip())
    if not doc["stages"]:
        typer.echo("no interview questions apply to this trial")


@interview_app.command("report")
def interview_report(
    trial: str = typer.Option(..., "--trial", help="Trial id."),
    stage: str | None = typer.Option(None, "--stage", help="One stage only (default: every stage with questions)."),
) -> None:
    """Render interview PDF + chart.json per stage (progress bar, traffic light, answers, don't-knows, pending)."""
    from qrh.interview.snapshot import build_snapshot
    from qrh.reports.interview import render_interview

    with _open() as (home, reg, view):
        t = reg.get_trial(trial)
        if t is None:
            raise _fail(f"unknown trial {trial!r}")
        snap = build_snapshot(reg.conn, view, t, stage)
        if not snap["stages"]:
            raise _fail("no interview questions apply to this trial" + (f" at {stage}" if stage else ""))
        files = render_interview(home, trial, snap, stage)
    for f in files:
        typer.echo(str(f.path))
