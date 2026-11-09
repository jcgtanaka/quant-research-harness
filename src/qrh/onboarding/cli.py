"""`qrh data` onboarding commands: profile, propose, verify, onboarding, confirm.

Registered on the `data` group by qrh.cli through `register`; behaviour and help text are unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer


@contextmanager
def _onboarding() -> Iterator[Any]:
    """A catalog for the onboarding commands; a refused operation is BLOCKED (exit 2)."""
    from qrh.cli import _fail, _open
    from qrh.onboarding.model import OnboardingError

    with _open("catalog") as (_home, cat):
        try:
            yield cat
        except OnboardingError as exc:
            raise _fail(str(exc)) from exc


def data_profile(
    name: str = typer.Argument(..., help="Dataset name in the catalog."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
    no_values: bool = typer.Option(False, "--no-values", help="Leave out every aggregate value (min, max, quantiles, "
                                   "first and last timestamp, flag values)."),
) -> None:
    """Privacy-safe profile of a dataset: the only view of its rows meant for the agent. Never prints a row."""
    from qrh.onboarding import profile as prof
    from qrh.onboarding import workflow

    with _onboarding() as cat:
        profile, digest = workflow.profile_dataset(cat, name, no_values=no_values)
    if as_json:
        typer.echo(json.dumps({"dataset": name, "profile_hash": digest, "profile": profile}, indent=1, sort_keys=True))
        return
    typer.echo(f"profile of {name}  hash {digest[:12]}")
    for line in prof.profile_lines(profile):
        typer.echo(line)


def data_propose(
    name: str = typer.Argument(..., help="Dataset name in the catalog."),
    file: Path = typer.Option(..., "--file", help="Proposal JSON: {\"fields\": {<field>: {\"value\": ..., \"evidence\": ...}}}."),
    as_user: bool = typer.Option(False, "--as-user", help="The claims are the user's own words. The agent may not use this."),
) -> None:
    """Store an instrument-model proposal as UNVERIFIED claims (the file is untrusted input; run `qrh data verify` next)."""
    from qrh.onboarding import workflow
    from qrh.onboarding.model import load_proposal

    with _onboarding() as cat:
        result = workflow.propose(cat, name, load_proposal(file), as_user=as_user)
    typer.echo(f"{name}: onboarding version {result['version']} "
               + ("stored" if result["created"] else "unchanged (same claims)") + "; run `qrh data verify " + name + "`")


def data_verify(name: str = typer.Argument(..., help="Dataset name in the catalog.")) -> None:
    """Check every claim against the data. A contradiction BLOCKS the dataset (exit 2); the rest becomes questions."""
    from qrh.cli import BLOCKED_EXIT_CODE
    from qrh.onboarding import workflow

    with _onboarding() as cat:
        out = workflow.verify(cat, name)
    _echo_onboarding(out)
    if out["state"] == "CONTRADICTED":
        raise typer.Exit(code=BLOCKED_EXIT_CODE)


def _echo_onboarding(doc: dict[str, Any]) -> None:
    typer.echo(f"{doc['dataset']}  onboarding {doc['state']}"
               + (f"  version {doc['version']}" if doc.get("version") else "") + f"  privacy {doc.get('privacy', 'private')}")
    for r in doc["fields"]:
        typer.echo(f"  {'*' if r['critical'] else ' '} {r['field']:20} {r['status'].upper():12} {r['claim']}  [{r['source']}]")
        if r["evidence"]:
            typer.echo(f"      {r['evidence']}")
    for q in doc["pending_questions"]:
        typer.echo(f"  question {q['id']} ({'CRITICAL ' if q['critical'] else ''}{q['status']}): "
                   f"qrh interview next --stage S1 --scope {q['scope']}")


def data_onboarding(
    name: str = typer.Argument(..., help="Dataset name in the catalog."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """The instrument model of a dataset: claim, source, status and evidence per field, open questions and the state."""
    from qrh.onboarding import workflow

    with _onboarding() as cat:
        doc = workflow.describe(cat, name)
    if as_json:
        typer.echo(json.dumps(doc, indent=1, sort_keys=True))
        return
    _echo_onboarding(doc)


def data_confirm(
    name: str = typer.Argument(..., help="Dataset name in the catalog."),
    field: str = typer.Argument(..., help="Field to confirm, for example adjustments."),
    value: str = typer.Option("confirm", "--value", help="`confirm` (the claim is right) or the correct value."),
) -> None:
    """A person at a terminal confirms or corrects one claim (the same as `qrh interview answer --human`). The agent may not use this."""
    from qrh.onboarding import workflow

    with _onboarding() as cat:
        result = workflow.answer_question(cat, name, field, value, answered_by="human", provenance="human-cli")
    typer.echo(f"{name}: {field} {result['action']} (onboarding version {result['version']})")


def register(data_app: typer.Typer) -> None:
    """Add the onboarding commands to the `qrh data` group, in their original order."""
    data_app.command("profile")(data_profile)
    data_app.command("propose")(data_propose)
    data_app.command("verify")(data_verify)
    data_app.command("onboarding")(data_onboarding)
    data_app.command("confirm")(data_confirm)
