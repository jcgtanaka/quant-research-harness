"""``qrh readiness report | ack | status``: the deployment readiness report and its signed acknowledgement.

``ack`` is a human step (a controlling terminal and the approval passphrase, exactly like ``qrh approve``); the
approval guard blocks an agent from running it. Exit code 2 means a refused or failed operation, like the rest of the
CLI; ``status`` exits 0 when acknowledged and 1 while pending.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer

from qrh.config import QrhHomeError, ensure_home

readiness_app = typer.Typer(help="Deployment readiness: every warning of a trial in one signed-off document.",
                            no_args_is_help=True, add_completion=False)
BLOCKED = 2


def _fail(message: str) -> typer.Exit:
    typer.echo(f"BLOCKED: {message}", err=True)
    return typer.Exit(code=BLOCKED)


@contextmanager
def _open() -> Iterator[tuple[Path, Any]]:
    from qrh.memory.registry import Registry

    try:
        home = ensure_home()
        reg = Registry(home / "registry.sqlite3")
    except Exception as exc:  # QrhHomeError or a damaged registry: BLOCKED either way
        raise _fail(str(exc)) from exc
    try:
        yield home, reg
    finally:
        reg.close()


def _counts_line(totals: dict[str, int]) -> str:
    return f"{totals['high']} HIGH, {totals['medium']} MEDIUM, {totals['low']} LOW"


@readiness_app.command("report")
def report(trial_id: str = typer.Argument(..., help="Trial id from `qrh new`.")) -> None:
    """Build the readiness report: data JSON, chart JSON and a PDF under <QRH_HOME>/reports/<trial>/.

    Gathers the warnings of every stage, the interview and the registry. Never changes a verdict.
    """
    from qrh.readiness import collect, data
    from qrh.readiness.report import render_readiness

    with _open() as (home, reg):
        try:
            report_data = data.build_data(collect.collect(home, trial_id, reg))
            files = render_readiness(trial_id, home, reg)
        except collect.ReadinessError as exc:
            raise _fail(str(exc)) from exc
    typer.echo(f"readiness report for {trial_id}: verdict {report_data.doc['final_verdict_display']}, "
               f"{_counts_line(report_data.doc['totals'])}")
    typer.echo(f"report hash {report_data.hash}")
    for f in files:
        typer.echo(str(f.path))


@readiness_app.command("status")
def status(trial_id: str = typer.Argument(..., help="Trial id.")) -> None:
    """Show whether the trial's current warning set is acknowledged (exit 0) or pending (exit 1), and why."""
    from qrh.readiness import ack, collect, data

    with _open() as (home, reg):
        try:
            report_data = data.build_data(collect.collect(home, trial_id, reg))
        except collect.ReadinessError as exc:
            raise _fail(str(exc)) from exc
        ok, why = ack.evaluate(home, report_data)
    typer.echo(f"trial {trial_id}: verdict {report_data.doc['final_verdict_display']}, "
               f"{_counts_line(report_data.doc['totals'])}, report hash {report_data.hash[:12]}")
    typer.echo(f"readiness: {'ACKNOWLEDGED' if ok else 'PENDING'}: {why}")
    if not ok:
        typer.echo(f"A person acknowledges it with `qrh readiness ack {trial_id}` in their own terminal.")
        raise typer.Exit(code=1)


@readiness_app.command("ack")
def ack_command(trial_id: str = typer.Argument(..., help="Trial id.")) -> None:
    """Acknowledge the readiness report with a signature (run this in your own terminal, never through the agent).

    Shows the verdict, the counts and every HIGH warning, then asks for the approval passphrase on the terminal.
    The acknowledgement is valid only for this exact warning set.
    """
    from qrh import human

    try:
        doc = human.acknowledge_readiness(ensure_home(), trial_id)
    except (QrhHomeError, human.HumanError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"readiness acknowledged for {trial_id} with key {doc['key_id']}: report {doc['report_hash'][:12]}, "
               f"{doc['high_count']} HIGH warning(s) accepted. It becomes invalid if any warning changes.")
