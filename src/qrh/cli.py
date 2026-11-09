"""Command line interface. Exit code 2 means BLOCKED or not PASS; unimplemented commands fail closed."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer
import yaml

from qrh import __version__
from qrh.config import QrhHomeError, ensure_home
from qrh.memory.registry import Registry
from qrh.pipeline.receipt import Stage, Verdict, display_verdict, is_pass, load_receipt

app = typer.Typer(help="quant-research-harness", no_args_is_help=True, add_completion=False)
data_app = typer.Typer(help="Data catalog.", no_args_is_help=True, add_completion=False)
app.add_typer(data_app, name="data")
hooks_app = typer.Typer(help="Claude Code hooks: settings and helpers.", no_args_is_help=True, add_completion=False)
app.add_typer(hooks_app, name="hooks")

from qrh.bench.cli import bench_app  # noqa: E402  (lazy imports inside; keeps hooks fast)
from qrh.methods.cli import methods_app  # noqa: E402
from qrh.interview.cli import interview_app  # noqa: E402
from qrh.llm.cli import llm_app  # noqa: E402
from qrh.packs.cli import pack_app  # noqa: E402
from qrh.readiness.cli import readiness_app  # noqa: E402

app.add_typer(methods_app, name="methods")
app.add_typer(bench_app, name="bench")
app.add_typer(pack_app, name="pack")
app.add_typer(interview_app, name="interview")
app.add_typer(readiness_app, name="readiness")
app.add_typer(llm_app, name="llm")
publish_app = typer.Typer(help="Publication checks run by git itself.", no_args_is_help=True, add_completion=False)
app.add_typer(publish_app, name="publish")


@publish_app.command("pre-push")
def publish_pre_push(
    remote_name: str = typer.Argument(..., help="Remote name, as git passes it (not part of the approval)."),
    remote_url: str = typer.Argument(..., help="Remote URL, as git passes it."),
    home: str = typer.Option(..., "--home", help="Absolute QRH home that holds the keys (QRH_HOME and HOME are ignored)."),
    key_id: str = typer.Option(..., "--key-id", help="Key id the hook was installed for."),
) -> None:
    """Git pre-push hook: exit 0 only with a signed, unused human approval for exactly this push (reads stdin)."""
    from qrh import prepush

    try:
        raw = typer.get_binary_stream("stdin").read(prepush.MAX_STDIN + 1)
        allowed, text = prepush.check(home, key_id, remote_url, raw)
    except BaseException as exc:  # fail closed on anything unexpected, including a broken stdin
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        allowed, text = False, f"BLOCKED: git pre-push refused: {type(exc).__name__}\n"
    if not allowed:
        typer.echo(text, err=True, nl=False)
        raise typer.Exit(code=BLOCKED_EXIT_CODE)


@publish_app.command("print-hook")
def publish_print_hook() -> None:
    """Print the pinned pre-push hook file for the current QRH home and key. Read-only: it writes no file."""
    import os
    import sys

    from qrh import prepush, signing
    from qrh.config import qrh_home

    home = os.path.abspath(str(qrh_home()))
    try:
        pub = signing.load_public(home)
    except signing.SigningError as exc:
        raise _fail(f"the approval key is unusable: {exc}") from exc
    if pub is None:
        raise _fail("no approval key exists: create one in your own terminal with `qrh approve --init`")
    typer.echo(prepush.hook_text(home, pub.key_id, os.path.abspath(sys.executable), prepush.default_path_dirs()), nl=False)


BLOCKED_EXIT_CODE = 2
DB_NAME = "registry.sqlite3"


def _fail(message: str) -> typer.Exit:
    typer.echo(f"BLOCKED: {message}", err=True)
    return typer.Exit(code=BLOCKED_EXIT_CODE)


@contextmanager
def _open(kind: str = "registry") -> Iterator[Any]:
    """Open the registry or catalog under QRH_HOME; any failure is BLOCKED (exit 2)."""
    try:
        home = ensure_home()
        if kind == "catalog":
            from qrh.memory.catalog import Catalog  # pandas is slow to import; hooks must stay fast
        store = Catalog(home / DB_NAME) if kind == "catalog" else Registry(home / DB_NAME)
    except QrhHomeError as exc:
        raise _fail(str(exc)) from exc
    except Exception as exc:
        raise _fail(f"{type(exc).__name__}: {exc}") from exc
    try:
        yield home, store
    except typer.Exit:
        raise
    except Exception as exc:  # fail closed on anything unexpected
        raise _fail(f"{type(exc).__name__}: {exc}") from exc
    finally:
        store.close()


def _load_spec(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise _fail(f"cannot read hypothesis file {str(path)!r}: {exc}") from exc


def _blocked(command: str) -> None:
    typer.echo(f"BLOCKED: {command} is not implemented yet (phase P0)", err=True)
    raise typer.Exit(code=BLOCKED_EXIT_CODE)


@app.command()
def new(spec: Path = typer.Argument(..., help="Hypothesis file (.yaml or .json).")) -> None:
    """Validate and register a hypothesis (stage S0); prints the trial id."""
    from qrh.pipeline.stages import register_new_trial

    raw = _load_spec(spec)
    with _open() as (home, reg):
        result = register_new_trial(raw, reg, home, base_dir=spec.parent)
    if result.trial_id is None or result.reasons:
        for reason in result.reasons:
            typer.echo(f"INVALID: {reason}", err=True)
        raise typer.Exit(code=BLOCKED_EXIT_CODE)
    for note in result.notes:  # a notice, never a block
        typer.echo(f"NOTE: {note}", err=True)
    typer.echo(result.trial_id)


@app.command()
def run(
    trial_id: str = typer.Argument(..., help="Trial id from `qrh new`."),
    llm_mode: str = typer.Option(
        "replay", "--llm-mode", help="Strategies that use a language model: replay (default; a cache miss is BLOCKED) "
        "or record (a miss calls the registered backend and stores the answer)."),
) -> None:
    """Run the pipeline for a trial; exit 2 unless the final verdict is PASS."""
    from qrh.llm import MODES
    from qrh.pipeline.stages import Pipeline

    if llm_mode not in MODES:
        raise _fail(f"--llm-mode must be record or replay, got {llm_mode!r}")
    with _open() as (home, reg):
        if reg.get_trial(trial_id) is None:
            raise _fail(f"unknown trial {trial_id!r}")
        result = Pipeline(reg, home, llm_mode=llm_mode).run(trial_id)
    for stage, receipt in sorted(result.stage_receipts.items(), key=lambda kv: kv[0].value):
        typer.echo(f"{stage.value}: {receipt.verdict.value}  {'; '.join(receipt.reasons)}".rstrip())
    typer.echo(f"{result.final.stage.value}: {display_verdict(result.final.verdict)}")
    for c in result.final.params.get("conditions") or []:
        typer.echo(f"condition: {c['question_id']} ({c['status']}, {c['stage']}) {c['text']}"
                   + (f" Risk: {c['risk']}" if c.get("risk") else ""))
    typer.echo(str(result.final_path))
    if not is_pass(result.final.verdict):
        raise typer.Exit(code=BLOCKED_EXIT_CODE)


@app.command()
def status(trial_id: str | None = typer.Argument(None, help="Trial id; omit to list recent trials.")) -> None:
    """List recent trials, or show each stage verdict and reasons for one trial."""
    with _open() as (_home, reg):
        if trial_id is None:
            for t in reg.list_trials(limit=20):
                typer.echo(
                    f"{t['trial_id']}  {t['verdict'] or 'NONE'}  "
                    f"{t['strategy_family']}/{t['asset_class']}/{t['frequency']}"
                )
            return
        trial = reg.get_trial(trial_id)
        if trial is None:
            raise _fail(f"unknown trial {trial_id!r}")
        typer.echo(f"{trial_id}  {trial['strategy_family']}/{trial['asset_class']}/{trial['frequency']}")
        bad = False
        for entry in reg.list_receipts(trial_id):
            try:
                r = load_receipt(entry["path"])
            except Exception as exc:
                typer.echo(f"{entry['stage']}: BLOCKED  receipt invalid: {exc}")
                bad = True
                continue
            typer.echo(f"{r.stage.value}: {display_verdict(r.verdict)}  {'; '.join(r.reasons)}".rstrip())
        _echo_reports(_home, trial_id)
        if bad:
            raise _fail("one or more receipts failed verification")


def _echo_reports(home: Path, trial_id: str) -> None:
    """One line on the rendered reports, plus every rendering error (these never change a verdict)."""
    from qrh.reports.manifest import manifest_path, read_manifest

    if not manifest_path(home, trial_id).is_file():
        return
    doc = read_manifest(home, trial_id)
    typer.echo(f"reports: {len(doc['files'])} file(s), {len(doc['errors'])} error(s)")
    for e in doc["errors"]:
        typer.echo(f"report error {e['stage']}: {e['error']}")


@app.command()
def report(
    trial_id: str = typer.Argument(..., help="Trial id from `qrh new`."),
    stage: str | None = typer.Option(None, "--stage", help="Render one stage only (S0..S7)."),
    open_: bool = typer.Option(False, "--open", help="Open the rendered PDF."),
) -> None:
    """Re-render PDF reports and chart data from persisted receipts and artifacts.

    Exit 2 if a receipt or artifact is missing or does not match its recorded hash.
    """
    from qrh.reports.inputs import ReportError
    from qrh.reports.render import render_stage, render_trial

    try:
        chosen = Stage(stage.upper()) if stage else None
    except ValueError:
        raise _fail(f"unknown stage {stage!r}; use S0..S7") from None
    with _open() as (home, reg):
        if reg.get_trial(trial_id) is None:
            raise _fail(f"unknown trial {trial_id!r}")
        try:
            files = render_stage(trial_id, chosen, home, reg) if chosen else render_trial(trial_id, home, reg)
        except ReportError as exc:
            raise _fail(str(exc)) from exc
    for f in files:
        typer.echo(str(f.path))
    pdfs = [f.path for f in files if f.kind == "pdf"]
    if open_ and pdfs:
        target = pdfs[0] if chosen else next((p for p in pdfs if p.stem == trial_id), pdfs[-1])
        typer.launch(str(target))


@app.command()
def search(
    query: str | None = typer.Argument(None, help="Keywords to look for in registered hypotheses."),
    limit: int = typer.Option(5, "--limit", min=1, help="Maximum number of hits."),
    reindex: bool = typer.Option(False, "--reindex", help="Rebuild the search index first."),
) -> None:
    """Keyword search (BM25) over registered hypotheses and their latest verdicts."""
    from qrh.memory.search import FtsSearcher

    if query is None and not reindex:
        raise _fail("give a query, or --reindex to rebuild the index")
    with _open() as (_home, reg):
        searcher = FtsSearcher(reg.conn)
        if reindex:
            typer.echo(f"reindexed {searcher.reindex()} hypothesis(es)")
        for hit in searcher.search(query or "", limit=limit):
            first_line = " ".join(hit.text.split())
            typer.echo(f"{hit.trial_id}  {hit.verdict or 'NONE'}  {hit.family}  {first_line[:100]}")


@app.command(
    "test-receipt", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def test_receipt(ctx: typer.Context) -> None:
    """Run a test command (after `--`) and record its result against the current source hash.

    PASS only on exit code 0; exit 2 otherwise. The pre-restart guard reads these receipts.
    """
    from qrh.hooks.common import ConfigError
    from qrh.hooks.pre_restart_guard import SourceHashError
    from qrh.hooks.testreceipt import record_test_receipt

    command = list(ctx.args)
    if not command:
        raise _fail("give the test command after --, for example: qrh test-receipt -- pytest -q")
    with _open() as (home, reg):
        try:
            result = record_test_receipt(command, home, reg, Path.cwd())
        except (ConfigError, SourceHashError) as exc:
            raise _fail(str(exc)) from exc
    verdict = "PASS" if result.passed else "FAIL"
    typer.echo(
        f"test receipt {verdict}: exit code {result.exit_code}, {result.duration_s}s, "
        f"source hash {result.source_hash[:12]}"
    )
    if not result.passed:
        raise typer.Exit(code=BLOCKED_EXIT_CODE)


@app.command()
def approve(
    approval_id: str | None = typer.Argument(None, help="Approval id printed by the blocked hook."),
    reason: str | None = typer.Option(None, "--reason", help="Why this is acceptable (at least 10 characters)."),
    init: bool = typer.Option(False, "--init", help="Create the approval key pair (passphrase read twice from the terminal)."),
    rotate: bool = typer.Option(False, "--rotate", help="With --init: replace the key pair; needs the old passphrase."),
) -> None:
    """Approve ONE blocked agent action (run this in your own terminal, never through the agent).

    Shows the request, asks for your passphrase on the terminal and signs the request with your private key.
    The hook that blocked the action lets the identical retried action, in the same session, through once
    before it expires (30 minutes), and records it. `--init` creates the key pair once; `--init --rotate`
    replaces it (old passphrase required) and drops every open approval.
    """
    from qrh import human

    if rotate and not init:
        raise _fail("--rotate is only valid together with --init")
    if init:
        if approval_id is not None or reason is not None:
            raise _fail("--init takes no approval id and no --reason")
        try:
            pub = human.init_keys(ensure_home(), rotate=rotate)
        except (QrhHomeError, human.HumanError) as exc:
            raise _fail(str(exc)) from exc
        typer.echo(f"approval key {'rotated' if rotate else 'created'}: key id {pub.key_id}. "
                   "Keep the passphrase to yourself; never type it into an agent session.")
        typer.echo(f"Public key fingerprint (short id): {pub.key_id}\n"
                   "Note it now somewhere the agent cannot write (paper, a password manager, another machine). "
                   "Every session start prints the fingerprint of the key in use; a different value means the key was "
                   "replaced and approvals since then are suspect.")
        return
    if approval_id is None or reason is None:
        raise _fail('usage: qrh approve <id> --reason "<why, at least 10 characters>"  (or qrh approve --init)')
    try:
        rec = human.approve(ensure_home(), approval_id, reason)
    except (QrhHomeError, human.HumanError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"approved {rec['id']} for hook {rec['hook']}: {rec.get('summary', '')}")
    typer.echo(f"valid once, in session {rec.get('session_id') or '(none)'}, until {rec['expires_at']}. "
               "Ask the agent to retry the identical action.")


config_app = typer.Typer(help="Config integrity: a signed baseline of config.yaml and hook registrations.",
                          no_args_is_help=True, add_completion=False)
app.add_typer(config_app, name="config")


@config_app.command("sign")
def config_sign(
    settings: list[Path] = typer.Option([], "--settings", help="Claude Code settings file whose hooks section is covered (repeatable)."),
) -> None:
    """Sign the current config.yaml and hook registrations (run this in your own terminal, never through the agent).

    Enforcement hooks then block whenever config.yaml or the `hooks` section of a given settings file changes,
    until you sign again. Needs the approval passphrase.
    """
    from qrh import human

    try:
        doc = human.sign_config(ensure_home(), [str(p) for p in settings])
    except (QrhHomeError, human.HumanError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"config baseline signed with key {doc['key_id']}: config.yaml and {len(doc['settings'])} settings "
               "file(s). Enforcement hooks now block if they change.")


@hooks_app.command("source-hash")
def hooks_source_hash() -> None:
    """Print the current hash of guard.source_globs (what the build hash file should contain)."""
    from qrh.hooks.common import ConfigError, load_config
    from qrh.hooks.pre_restart_guard import SourceHashError, compute_source_hash

    try:
        guard = load_config(ensure_home()).guard
        if guard is None:
            raise ConfigError("the 'guard:' section of config.yaml is not configured")
        typer.echo(compute_source_hash(guard.source_globs, Path.cwd()))
    except (ConfigError, SourceHashError, QrhHomeError) as exc:
        raise _fail(str(exc)) from exc


@app.command("hook")
def hook(name: str = typer.Argument(..., help="Hook name, for example lookahead-lint.")) -> None:
    """Run one Claude Code hook: the event JSON arrives on stdin; exit 2 blocks (reason on stderr)."""
    import sys

    from qrh.hooks.runner import run_hook

    stream = getattr(sys.stdin, "buffer", None)
    text = stream.read().decode("utf-8", errors="replace") if stream is not None else sys.stdin.read()
    outcome = run_hook(name, text)
    if outcome.stdout:
        typer.echo(outcome.stdout, nl=False)
    if outcome.stderr:
        typer.echo(outcome.stderr, nl=False, err=True)
    raise typer.Exit(code=outcome.exit_code)


@hooks_app.command("print-settings")
def hooks_print_settings(
    scope: str = typer.Option("user", "--scope", help="user or project: which settings file this is meant for."),
    settings: Path | None = typer.Option(None, "--settings", help="Existing settings file to merge into."),
    write: bool = typer.Option(False, "--write", help="Write the merged settings (needs --settings); backs up first."),
) -> None:
    """Print the settings JSON that registers every qrh hook, merged into --settings if given.

    Nothing is read or written unless --settings is given; --write never defaults to a real file.
    """
    from qrh.hooks.settings import SCOPE_DEFAULTS, SettingsError, merge_settings, read_settings, render, write_settings

    if scope not in SCOPE_DEFAULTS:
        raise _fail(f"unknown scope {scope!r}; use user or project")
    if write and settings is None:
        raise _fail("--write needs an explicit --settings PATH")
    try:
        current = read_settings(settings) if settings is not None else {}
        merged = merge_settings(current)
    except SettingsError as exc:
        raise _fail(str(exc)) from exc
    if not write:
        typer.echo(render(merged), nl=False)
        typer.echo(f"scope {scope}: the usual file is {SCOPE_DEFAULTS[scope]} (not touched)", err=True)
        return
    assert settings is not None
    if settings.exists() and merged == current:
        typer.echo(f"{settings} is already up to date")
        return
    backup = write_settings(settings, merged)
    typer.echo(f"wrote {settings}" + (f" (backup: {backup})" if backup else ""))


audit_app = typer.Typer(help="Audits. Only `tests` is implemented.", add_completion=False, invoke_without_command=True)
app.add_typer(audit_app, name="audit")


@audit_app.callback()
def audit_root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _blocked("audit")


@audit_app.command("tests")
def audit_tests(
    base: str = typer.Option("HEAD", "--base", help="Git ref to compare the working tree against."),
) -> None:
    """Find weakened tests in the diff (deleted tests, removed assertions, new skips); exit 2 on findings."""
    from qrh.checks.weakened_tests import AuditError, audit_repo

    try:
        findings = audit_repo(Path.cwd(), base)
    except AuditError as exc:
        raise _fail(str(exc)) from exc
    for f in findings:
        typer.echo(f"{f.file}: {f.kind}: {f.detail}")
    if findings:
        raise typer.Exit(code=BLOCKED_EXIT_CODE)
    typer.echo(f"no weakened tests found against {base}")


@audit_app.command("receipts")
def audit_receipts() -> None:
    """Audit receipts (not implemented)."""
    _blocked("audit receipts")


@audit_app.command("data")
def audit_data() -> None:
    """Audit datasets (not implemented)."""
    _blocked("audit data")


@app.command()
def lint(file: Path = typer.Argument(..., help="Python source file to lint for look-ahead.")) -> None:
    """Static look-ahead lint (early warning); exit 2 on BLOCK findings."""
    from qrh.checks.lookahead import blocking, lint_file

    findings = lint_file(file)
    for f in findings:
        tag = "ALLOWED" if f.suppressed else f.severity
        extra = f"  (reason: {f.reason})" if f.suppressed else ""
        typer.echo(f"{f.file}:{f.line}: [{tag}] {f.rule_id} {f.message}{extra}")
        if f.snippet:
            typer.echo(f"    {f.snippet}")
    if blocking(findings):
        raise typer.Exit(code=BLOCKED_EXIT_CODE)
    if not findings:
        typer.echo("no findings")


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Only 127.0.0.1 is accepted."),
    port: int = typer.Option(8765, "--port", min=0, max=65535, help="Port to listen on."),
) -> None:
    """Serve the read-only dashboard data API on 127.0.0.1 (GET only)."""
    from qrh.ui.server import make_server

    try:
        server = make_server(ensure_home(), host=host, port=port)
    except (ValueError, QrhHomeError, OSError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"serving on http://127.0.0.1:{server.server_address[1]} (read-only; Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def synth(
    out: Path = typer.Option(..., "--out", help="Output CSV path."),
    seed: int = typer.Option(0, help="Random seed."),
    periods: int = typer.Option(252, min=1, help="Number of bars."),
    freq: str = typer.Option("D", help="Any pandas frequency string."),
    start: str = typer.Option("2020-01-01", help="Start timestamp."),
    tz: str = typer.Option("UTC", help="Time zone."),
    start_price: float = typer.Option(100.0, help="Initial price."),
    drift: float = typer.Option(0.0, help="Per-bar log drift."),
    volatility: float = typer.Option(0.01, help="Per-bar log volatility."),
    gap_probability: float = typer.Option(0.0, help="Probability of dropping a bar."),
) -> None:
    """Write a seeded synthetic price series to a CSV file."""
    from qrh.synthetic.generator import Regime, SyntheticConfig, generate

    cfg = SyntheticConfig(
        seed=seed,
        start=start,
        periods=periods,
        freq=freq,
        tz=tz,
        start_price=start_price,
        regimes=[Regime(share=1.0, drift=drift, volatility=volatility)],
        gap_probability=gap_probability,
    )
    generate(cfg).to_csv(out)
    typer.echo(f"wrote {out}")


def _echo_dataset(rec: dict[str, Any]) -> None:
    typer.echo(
        f"{rec['name']}  {rec['status']}  rows={rec['row_count']}  "
        f"range={rec['start']}..{rec['end']}  hash={(rec['content_hash'] or '')[:12]}  privacy={rec.get('privacy', 'private')}"
    )
    for reason in rec["reasons"]:
        typer.echo(f"  reason: {reason}")


@data_app.command("register")
def data_register(
    path: Path = typer.Argument(..., help="CSV or parquet file."),
    name: str | None = typer.Option(None, "--name", help="Catalog name (default: file stem)."),
    rules: Path | None = typer.Option(None, "--rules", help="Data-quality rules file (YAML)."),
    privacy: str | None = typer.Option(None, "--privacy", help="private (default) or public (synthetic data only)."),
) -> None:
    """Profile a dataset and add it to the catalog."""
    if privacy is not None and privacy not in ("private", "public"):
        raise _fail(f"--privacy must be private or public, got {privacy!r}")
    with _open("catalog") as (_home, cat):
        rec = cat.register_dataset(path, name=name, rules=rules, privacy=privacy)
    _echo_dataset(rec)
    if rec["status"] == "BLOCKED":
        raise typer.Exit(code=BLOCKED_EXIT_CODE)


@data_app.command("check")
def data_check(name: str = typer.Argument(..., help="Dataset name in the catalog.")) -> None:
    """Run a dataset's quality rules; exit 2 unless it PASSes."""
    from qrh.checks.data_quality import check_dataset

    with _open("catalog") as (_home, cat):
        status_, reasons = check_dataset(cat, name)
    typer.echo(f"{name}  {status_}")
    for reason in reasons:
        typer.echo(f"  reason: {reason}")
    if status_ != "PASS":
        raise typer.Exit(code=BLOCKED_EXIT_CODE)


def _onboarding_state(cat: Any, name: str) -> str:
    from qrh.onboarding import store

    return store.dataset_state(cat.conn, name)["state"]


@data_app.command("status")
def data_status(name: str | None = typer.Argument(None, help="Dataset name; omit to list all.")) -> None:
    """Show dataset status and onboarding state (unknown datasets are BLOCKED)."""
    with _open("catalog") as (_home, cat):
        if name is None:
            for rec in cat.list_datasets():
                _echo_dataset(rec)
                typer.echo(f"  onboarding: {_onboarding_state(cat, rec['name'])}")
            return
        status_, reasons = cat.get_status(name)
        rec = cat.get_dataset(name)
        state = _onboarding_state(cat, name) if rec is not None else ""
    if rec is None:
        typer.echo(f"{name}  {status_}", err=True)
        for reason in reasons:
            typer.echo(f"  reason: {reason}", err=True)
        raise typer.Exit(code=BLOCKED_EXIT_CODE)
    _echo_dataset(rec)
    typer.echo(f"  onboarding: {state}")
    if status_ == "BLOCKED":
        raise typer.Exit(code=BLOCKED_EXIT_CODE)


# --- dataset onboarding (P3e-a1): commands live in qrh.onboarding.cli -----------------------------------------

from qrh.onboarding.cli import register as _register_onboarding  # noqa: E402

_register_onboarding(data_app)


# --- AI usage metrics (P3b): commands live in qrh.usage.cli ---------------------------------------------------

from qrh.usage.cli import usage_app  # noqa: E402

app.add_typer(usage_app, name="usage")
