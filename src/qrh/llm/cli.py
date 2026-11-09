"""``qrh llm status|verify <trial>``: look at a trial's replay cache (never prints a prompt)."""

from __future__ import annotations

import re

import typer

llm_app = typer.Typer(help="The LLM replay cache of a trial.", no_args_is_help=True, add_completion=False)
_TRIAL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _cache(trial_id: str):  # type: ignore[no-untyped-def]
    from qrh.cli import _fail
    from qrh.config import QrhHomeError, ensure_home
    from qrh.llm.cache import ReplayCache, cache_dir

    if not _TRIAL.match(trial_id) or ".." in trial_id:
        raise _fail(f"{trial_id!r} is not a valid trial id")
    try:
        home = ensure_home()
    except QrhHomeError as exc:
        raise _fail(str(exc)) from exc
    return ReplayCache(cache_dir(home, trial_id)), cache_dir(home, trial_id)


@llm_app.command("status")
def status(trial_id: str = typer.Argument(..., help="Trial id.")) -> None:
    """Entry count, cache content hash and the models seen in the cache."""
    from qrh.cli import _fail
    from qrh.llm.cache import LLMCacheCorrupt

    cache, folder = _cache(trial_id)
    try:
        st = cache.status()
    except LLMCacheCorrupt as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"trial {trial_id}  cache {folder}")
    typer.echo(f"entries: {st['entries']}")
    typer.echo(f"cache hash: {st['cache_hash']}")
    typer.echo("models: " + (", ".join(st["models"]) or "none"))


@llm_app.command("verify")
def verify(trial_id: str = typer.Argument(..., help="Trial id.")) -> None:
    """Recompute every entry hash; any mismatch is exit 2."""
    from qrh.cli import _fail

    cache, _folder = _cache(trial_id)
    problems = cache.verify()
    if problems:
        for p in problems:
            typer.echo(f"MISMATCH: {p}", err=True)
        raise _fail(f"the replay cache of {trial_id} does not verify ({len(problems)} problem(s))")
    typer.echo(f"ok: {len(cache)} entries verified")
