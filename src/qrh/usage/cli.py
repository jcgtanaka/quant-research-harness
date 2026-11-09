"""`qrh usage ...` commands: AI usage metrics from Claude Code transcripts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer

from qrh.config import QrhHomeError, ensure_home

usage_app = typer.Typer(help="AI usage metrics from Claude Code transcripts.", no_args_is_help=True,
                        add_completion=False)


@contextmanager
def _usage_home() -> Iterator[Path]:
    """QRH_HOME for the usage commands; known usage errors are BLOCKED (exit 2)."""
    from qrh.cli import _fail
    from qrh.usage.local import LocalLogError
    from qrh.usage.privacy import PrivacyError
    from qrh.usage.settings import SettingsError

    try:
        yield ensure_home()
    except typer.Exit:
        raise
    except (QrhHomeError, SettingsError, PrivacyError, LocalLogError, ValueError, OSError) as exc:
        raise _fail(str(exc)) from exc


@usage_app.command("init")
def usage_init(force: bool = typer.Option(False, "--force", help="Overwrite existing files.")) -> None:
    """Write example thresholds.json and prices.json under QRH_HOME/usage (to review and edit)."""
    from qrh.usage.settings import init_files

    with _usage_home() as home:
        written = init_files(home, force=force)
    for p in written:
        typer.echo(f"wrote {p}  (EXAMPLE values: review them; fill prices from the provider's pricing page)")
    if not written:
        typer.echo("nothing written: the files already exist (use --force to overwrite)")


@usage_app.command("harvest")
def usage_harvest(
    claude_dir: Path | None = typer.Option(None, "--claude-dir", help="Claude config dir (default $CLAUDE_CONFIG_DIR, else ~/.claude)."),
) -> None:
    """Read Claude Code transcripts (read-only) and merge their metrics into QRH_HOME/usage/sessions.json."""
    from qrh.usage.harvest import claude_dir_default, harvest

    with _usage_home() as home:
        s = harvest(claude_dir or claude_dir_default(), home)
    typer.echo(
        f"harvested {s.scanned - s.skipped_empty} session(s) ({s.new} new, {s.updated} updated), "
        f"{s.archived_only} archived-only, {s.skipped_empty} without usage, {s.subagent_files} subagent "
        f"transcript(s), {s.malformed_lines} malformed line(s)"
    )


@usage_app.command("report")
def usage_report(
    by: str = typer.Option("week", "--by", help="session, day, week, model, trial, trial-stage or project."),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    pdf: Path | None = typer.Option(None, "--pdf", help="Also write the report as a PDF."),
) -> None:
    """Cost and token report from the archive and logs; writes chart JSON under QRH_HOME/usage/reports."""
    import os

    from qrh.usage.report import build_charts, build_report, render_pdf, render_text, write_chart_files

    with _usage_home() as home:
        rep = build_report(home, by, env=os.environ)
        charts = build_charts(rep)
        write_chart_files(home, charts)
        if pdf is not None:
            render_pdf(charts, pdf)
    if as_json:
        typer.echo(json.dumps(rep, indent=1, sort_keys=True))
        if pdf is not None:
            typer.echo(f"wrote {pdf}", err=True)
        return
    typer.echo(render_text(rep), nl=False)
    if pdf is not None:
        typer.echo(f"wrote {pdf}")


@usage_app.command("waste")
def usage_waste(
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    pdf: Path | None = typer.Option(None, "--pdf", help="Also write the report as a PDF."),
) -> None:
    """Waste report per session and in total: re-reads, large outputs, failing loops, cold cache, compactions."""
    import os

    from qrh.usage.report import render_pdf, write_chart_files
    from qrh.usage.waste import build_waste, render_waste_text, waste_charts

    with _usage_home() as home:
        rep = build_waste(home, env=os.environ)
        charts = waste_charts(rep)
        write_chart_files(home, charts)
        if pdf is not None:
            render_pdf(charts, pdf, title="AI usage waste report")
    if as_json:
        typer.echo(json.dumps(rep, indent=1, sort_keys=True))
        return
    typer.echo(render_waste_text(rep), nl=False)
    if pdf is not None:
        typer.echo(f"wrote {pdf}")


@usage_app.command("regress")
def usage_regress(
    by: str = typer.Option("week", "--by", help="Change point: week, model or config (hooks + CLAUDE.md hash)."),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    pdf: Path | None = typer.Option(None, "--pdf", help="Also write the report as a PDF."),
) -> None:
    """Compare per-turn metrics before and after each change point (Mann-Whitney U, Fisher exact); no causal claims."""
    import os

    from qrh.usage.regress import build_regress, regress_charts, render_regress_text
    from qrh.usage.report import render_pdf, write_chart_files

    with _usage_home() as home:
        rep = build_regress(home, by, env=os.environ)
        charts = regress_charts(rep)
        write_chart_files(home, charts)
        if pdf is not None:
            render_pdf(charts, pdf, title="AI usage regression report")
    if as_json:
        typer.echo(json.dumps(rep, indent=1, sort_keys=True))
        return
    typer.echo(render_regress_text(rep), nl=False)
    if pdf is not None:
        typer.echo(f"wrote {pdf}")


@usage_app.command("verify")
def usage_verify(
    scope: str = typer.Option(
        "transcript", "--scope",
        help="transcript (default: models seen in transcripts; input, cache read, cache creation) or strict "
             "(every model and kind; a diagnostic that is expected to fail, see docs/usage.md)."),
    ack: str | None = typer.Option(None, "--ack", help="Acknowledge one unexplained mismatch by session alias (S038)."),
    reason: str | None = typer.Option(None, "--reason", help="Why (at least 10 characters); required with --ack."),
) -> None:
    """Check our token totals against the harness cost-state.

    Exit 2 if a session without subagents differs by more than 1% (transcript scope), or any session does (strict).
    Sessions with subagents are reported but do not gate: their tokens are a lower bound.
    """
    import os

    from qrh.cli import BLOCKED_EXIT_CODE, _fail
    from qrh.usage.harvest import load_archive
    from qrh.usage.privacy import label_map
    from qrh.usage.verify import AckError, load_acks, record_ack, render_verify, verify_sessions

    with _usage_home() as home:
        archive = load_archive(home)
        if not archive:
            raise _fail("the archive is empty; run `qrh usage harvest` first")
        ordered = sorted(archive.values(), key=lambda s: (s.get("start") or "", s["session_id"]))
        ranks = {s["session_id"]: i for i, s in enumerate(ordered, 1)}
        try:
            if ack is not None:
                if reason is None:
                    raise _fail("--ack needs --reason \"<at least 10 characters>\"")
                by_alias = {f"S{n:03d}": sid for sid, n in ranks.items()}
                if ack not in by_alias:
                    raise _fail(f"unknown session alias {ack!r}")
                row = next((r for r in verify_sessions(archive, scope="transcript").rows
                            if r.session_id == by_alias[ack]), None)
                if row is None or not row.gating or row.ok:
                    raise _fail(f"{ack} is not a gating mismatch in transcript scope, so there is nothing to acknowledge")
                record_ack(home, archive[by_alias[ack]], ack, row, reason)
                typer.echo(f"acknowledged {ack}: best difference {row.best_diff_pct:.2f}%; it no longer gates until its numbers change")
            res = verify_sessions(archive, scope=scope, acks=load_acks(home))
        except AckError as exc:
            raise _fail(str(exc)) from exc
        text = render_verify(res, ranks, label_map(ordered, os.environ))
    typer.echo(text, nl=False)
    if res.verified == 0:
        raise _fail("no archived session has a harness cost-state, so nothing could be verified")
    if res.gating_verified == 0:
        raise _fail("no archived session without subagents has a harness cost-state, so the gate has nothing to check")
    if not res.ok:
        raise typer.Exit(code=BLOCKED_EXIT_CODE)


@usage_app.command("log-local")
def usage_log_local(
    task: str = typer.Option("", "--task", help="Short label for what the call was for."),
    session_id: str | None = typer.Option(None, "--session-id", help="Claude Code session id, if known."),
    model: str | None = typer.Option(None, "--model", help="Model name if the response has none."),
) -> None:
    """Append a local-model usage record from an Ollama /api/generate or /api/chat JSON response on stdin."""
    import sys

    from qrh.usage.local import append_local, record_from_ollama

    text = sys.stdin.read()
    with _usage_home() as home:
        rec = record_from_ollama(text, task=task, session_id=session_id, model=model)
        append_local(home, rec)
    typer.echo(f"logged {rec['model']}: {rec['prompt_tokens']} prompt + {rec['completion_tokens']} completion tokens")
