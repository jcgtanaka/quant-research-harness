"""``qrh pack ...`` commands. Exit code 2 means BLOCKED, rejected or findings (like the rest of the CLI)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import typer
import yaml

from qrh import approvals
from qrh.config import QrhHomeError, ensure_home
from qrh.hashing import sha256_file

pack_app = typer.Typer(help="Experience packs: lessons, attempts, account, questions.", no_args_is_help=True,
                       add_completion=False)
BLOCKED = 2

SCAFFOLD_LESSON = {
    "id": "example-lesson", "statement": "Replace with a general lesson, no tickers, figures or dates.",
    "applies_to": {}, "trigger": {"keywords": ["example"]}, "action": "warn", "severity": "info", "status": "draft",
}


def _fail(message: str) -> typer.Exit:
    typer.echo(f"BLOCKED: {message}", err=True)
    return typer.Exit(code=BLOCKED)


def _home() -> Path:
    try:
        return ensure_home()
    except QrhHomeError as exc:
        raise _fail(str(exc)) from exc


def _load(path: Path):  # noqa: ANN202
    from qrh.packs.model import PackError, load_pack

    try:
        return load_pack(path)
    except PackError as exc:
        raise _fail(str(exc)) from exc


def _gate(home: Path, hook: str, parts: tuple[str, ...], summary: str, reason: str) -> None:
    """Human approval gate: consume a valid approval for exactly this action, else request one and exit 2."""
    from qrh.hooks.runner import human_block_text

    now = datetime.now(timezone.utc)
    action = approvals.parts_action(hook, summary, *parts)
    try:
        got = approvals.consume(home, hook=hook, fingerprint=approvals.fingerprint_of(action), session_id="", now=now)
        if got.ok:
            return
        rid = approvals.request(home, hook=hook, action=action, reason=reason, session_id="", now=now)
    except approvals.ApprovalError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(human_block_text(
        hook, approvals.summarize(action), reason,
        "a pack that loosens a core threshold or adds a lesson changes what every later trial is judged against",
        f'qrh approve {rid} --reason "<why this is acceptable, at least {approvals.MIN_REASON} characters>"',
        f"Approval id {rid}: valid once, within {approvals.DEFAULT_TTL_S // 60} minutes; then repeat the identical "
        "command.", "\n".join(got.notes)), err=True, nl=False)
    raise typer.Exit(code=BLOCKED)


@pack_app.command("init")
def pack_init(
    directory: Path = typer.Argument(..., help="New pack folder."),
    visibility: str = typer.Option("private", "--visibility", help="private (default) or public."),
) -> None:
    """Scaffold an empty pack (manifest plus one draft example lesson)."""
    if directory.exists() and any(directory.iterdir()):
        raise _fail(f"{directory} already exists and is not empty")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "qrh-pack.yaml").write_text(yaml.safe_dump({
        "name": re.sub(r"[^A-Za-z0-9_.-]", "-", directory.name) or "my-pack", "version": "0.1.0", "schema_version": 1,
        "visibility": visibility, "description": "Describe what this pack knows.", "applies_to": {},
        "authors": [], "license": "Apache-2.0",
    }, sort_keys=True), encoding="utf-8")
    (directory / "lessons").mkdir()
    (directory / "lessons" / "example.yaml").write_text(yaml.safe_dump(SCAFFOLD_LESSON, sort_keys=True), encoding="utf-8")
    pack = _load(directory)
    typer.echo(f"created {directory} ({pack.name} {pack.manifest.version}, {pack.visibility})")


@pack_app.command("add")
def pack_add(path: Path = typer.Argument(..., help="Pack folder; it is registered by path, not copied.")) -> None:
    """Register a pack (path and content hash). A pack that loosens a core threshold needs a human approval."""
    from qrh.memory.registry import Registry
    from qrh.packs import store
    from qrh.packs.imports import sync_imported
    from qrh.packs.layering import loosenings

    home = _home()
    pack = _load(path)
    loose = loosenings([pack], None)
    approved: list[str] = []
    if loose:
        ids = sorted(lo.id for lo in loose)
        _gate(home, "pack-loosen", (pack.name, pack.hash, *ids),
              f"pack {pack.name} loosens {', '.join(ids)}",
              f"pack {pack.name!r} would loosen core threshold(s): "
              + "; ".join(f"{lo.name} {lo.core} -> {lo.value}" for lo in loose))
        approved = ids
    store.register(home, pack, approved)
    reg = Registry(home / "registry.sqlite3")
    try:
        n = sync_imported(reg.conn, pack.name, pack.hash, pack.attempts)
    finally:
        reg.close()
    typer.echo(f"added {pack.name} {pack.manifest.version} ({pack.visibility}) hash {pack.hash[:12]}; "
               f"{len(pack.lessons)} lesson(s), {len(pack.questions)} question(s), {n} imported attempt(s)"
               + (f", {pack.draft_count} draft lesson(s) not loaded" if pack.draft_count else ""))


@pack_app.command("list")
def pack_list() -> None:
    """List registered packs, flagging any whose folder changed since it was added."""
    from qrh.packs.model import PackError, load_pack
    from qrh.packs.store import load_index

    entries = load_index(_home())["packs"]
    if not entries:
        typer.echo("no packs registered")
    for e in entries:
        try:
            state = "" if load_pack(e["path"]).hash == e["hash"] else "  CHANGED since added (re-run `qrh pack add`)"
        except PackError as exc:
            state = f"  INVALID: {exc}"
        typer.echo(f"{e['name']}  {e['version']}  {e['visibility']}  {e['hash'][:12]}  {e['path']}{state}")


@pack_app.command("remove")
def pack_remove(name: str = typer.Argument(..., help="Pack name.")) -> None:
    """Unregister a pack (its folder is untouched) and drop its imported attempts."""
    from qrh.memory.registry import Registry
    from qrh.packs import store
    from qrh.packs.imports import remove_imported

    home = _home()
    if not store.unregister(home, name):
        raise _fail(f"no registered pack named {name!r}")
    reg = Registry(home / "registry.sqlite3")
    try:
        remove_imported(reg.conn, name)
    finally:
        reg.close()
    typer.echo(f"removed {name}")


@pack_app.command("validate")
def pack_validate(
    path: Path = typer.Argument(..., help="Pack folder."),
    config: Path | None = typer.Option(None, "--config", help="Safety lint config (ticker_patterns, allow)."),
) -> None:
    """Check the schema; for visibility=public also the public-safety lint. Exit 2 on findings."""
    from qrh.packs.safety import public_safety_findings

    pack = _load(path)
    findings = public_safety_findings(pack, config)
    for f in findings:
        typer.echo(str(f))
    if findings:
        typer.echo(f"{len(findings)} public-safety finding(s) in public pack {pack.name!r}", err=True)
        raise typer.Exit(code=BLOCKED)
    typer.echo(f"{pack.name} {pack.manifest.version}: valid"
               + (" (public-safety lint clean)" if pack.visibility == "public" else " (private: no safety lint)"))


@pack_app.command("draft")
def pack_draft(
    source: Path = typer.Option(..., "--from", help="Notes file or folder (.md/.txt)."),
    out: Path = typer.Option(..., "--out", help="Private output folder."),
) -> None:
    """Prepare draft lesson skeletons for the agent to refine and a human to approve."""
    from qrh.packs.draft import draft_from
    from qrh.packs.model import PackError

    try:
        written = draft_from(source, out)
    except PackError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"wrote {len(written)} draft lesson(s) under {out / 'lessons'}; "
               f"each stays excluded until `qrh pack approve-lesson <file>`")


@pack_app.command("approve-lesson")
def pack_approve_lesson(file: Path = typer.Argument(..., help="Draft lesson file.")) -> None:
    """Mark a draft lesson approved. Needs a human approval token for this exact file content."""
    from qrh.packs.model import Lesson

    home = _home()
    try:
        doc = yaml.safe_load(file.read_text(encoding="utf-8"))
        lesson = Lesson.model_validate(doc)
    except Exception as exc:  # unreadable, not YAML, or not a valid lesson
        raise _fail(f"{file}: not a valid lesson: {exc}") from exc
    if lesson.status != "draft":
        raise _fail(f"{file}: status is {lesson.status!r}, only a draft can be approved")
    _gate(home, "pack-approve-lesson", (sha256_file(file),), f"approve draft lesson {lesson.id}",
          f"approving draft lesson {lesson.id!r} makes it active in its pack")
    doc["status"] = "approved"
    file.write_text(yaml.safe_dump(doc, sort_keys=True), encoding="utf-8")
    typer.echo(f"approved {lesson.id}")

