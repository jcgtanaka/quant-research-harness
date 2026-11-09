"""``qrh methods``: list and show the registry; promote, demote and deprecate behind a human approval."""

from __future__ import annotations

import json
from typing import Any

import typer

from qrh.methods.registry import STAGES, Method, MethodError, Reference, parse_ref

methods_app = typer.Typer(help="Versioned, cited methods and their status.", no_args_is_help=True, add_completion=False)


def _fail(message: str) -> typer.Exit:
    from qrh.cli import _fail as fail

    return fail(message)


def _home() -> Any:
    from qrh.config import QrhHomeError, ensure_home

    try:
        return ensure_home()
    except QrhHomeError as exc:
        raise _fail(str(exc)) from exc


def _ref_text(ref: Reference) -> str:
    ids = [f"doi:{ref.doi}" if ref.doi else "", f"arxiv:{ref.arxiv}" if ref.arxiv else "", ref.url]
    ident = ", ".join(i for i in ids if i) or "no identifier"
    return f"{ref.citation} [{ident}; {'verified' if ref.verified else 'not verified'}]"


def _row(m: Method, status: str) -> dict[str, Any]:
    d = m.describe()
    d["shipped_status"], d["status"] = m.status, status
    return d


@methods_app.command("list")
def methods_list(
    stage: str | None = typer.Option(None, "--stage", help="Only methods of one stage (S1..S6)."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Every method version with its status (a local promotion shows as such) and references."""
    from qrh.methods.selection import MethodSelector

    chosen = stage.upper() if stage else None
    if chosen is not None and chosen not in STAGES:
        raise _fail(f"unknown stage {stage!r}; use one of {', '.join(STAGES)}")
    try:
        sel = MethodSelector(_home())
        rows = [_row(m, sel.status(m)) for m in sel.registry.all(chosen)]
    except MethodError as exc:
        raise _fail(str(exc)) from exc
    if as_json:
        typer.echo(json.dumps(rows, indent=2, sort_keys=True))
        return
    for r in rows:
        local = " (local)" if r["status"] != r["shipped_status"] else ""
        refs = "; ".join(x["citation"].split(".")[0] + ("" if x["verified"] else " (unverified)") for x in r["references"])
        typer.echo(f"{r['id']:<28} {r['version']:<36} {r['status']}{local:<8} {r['stage']}  {refs}".rstrip())


@methods_app.command("show")
def methods_show(ref: str = typer.Argument(..., help="Method reference, for example s5.pbo=cscv@1 or s5.pbo@cscv@1.")) -> None:
    """Everything recorded about one method version."""
    from qrh.methods.selection import MethodSelector

    try:
        mid, version = parse_ref(ref)
        sel = MethodSelector(_home())
        m = sel.registry.get(mid, version)
        status = sel.status(m)
    except MethodError as exc:
        raise _fail(str(exc)) from exc
    local = " (local change)" if status != m.status else ""
    typer.echo(f"{m.id}={m.version}")
    typer.echo(f"stage: {m.stage}   kind: {m.kind}   status: {status}{local}   added in: {m.added_in}")
    if m.notes:
        typer.echo(f"notes: {m.notes}")
    typer.echo("parameters: " + json.dumps(m.parameters, sort_keys=True, default=str))
    typer.echo("references:" if m.references else "references: none (in-house method)")
    for r in m.references:
        typer.echo(f"  - {_ref_text(r)}")


def _act(action: str, ref: str, reason: str, **kw: Any) -> None:
    from qrh.methods.promotion import run_action

    outcome = run_action(_home(), action, ref, reason=reason, **kw)
    if outcome.status == "refused":
        for r in outcome.reasons:
            typer.echo(f"BLOCKED: {r}", err=True)
        raise typer.Exit(code=2)
    if outcome.status == "pending":
        typer.echo(outcome.message, nl=False)
        raise typer.Exit(code=2)
    typer.echo(outcome.message)


@methods_app.command("promote")
def methods_promote(
    ref: str = typer.Argument(..., help="Experimental method version, for example s4.walkforward@purged_embargo@1."),
    bench: str = typer.Option(..., "--bench", help="Bench receipt (id or path) run with --method ... --compare-to default --full."),
    reason: str = typer.Option(..., "--reason", help="Why (at least 10 characters)."),
    tie_reason: str | None = typer.Option(None, "--tie-reason", help="Required only when no gated metric is better: why promote on a tie."),
) -> None:
    """Make an experimental version the local default. Needs a full bench win (see docs/methods.md) and a human approval."""
    _act("promote", ref, reason, bench=bench, tie_reason=tie_reason)


@methods_app.command("demote")
def methods_demote(
    ref: str = typer.Argument(..., help="The current default version to demote."),
    restore: str = typer.Option(..., "--restore", help="Version of the same method that becomes the default again."),
    reason: str = typer.Option(..., "--reason", help="Why (at least 10 characters)."),
) -> None:
    """Return the default to an earlier version. Needs a human approval."""
    _act("demote", ref, reason, restore=restore)


@methods_app.command("deprecate")
def methods_deprecate(
    ref: str = typer.Argument(..., help="Experimental method version to deprecate."),
    reason: str = typer.Option(..., "--reason", help="Why (at least 10 characters)."),
) -> None:
    """Refuse a version for new hypotheses (old receipts still reproduce). Needs a human approval."""
    _act("deprecate", ref, reason)
