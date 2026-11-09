"""PreToolUse (Bash): before a live or paper engine start, check build, tests, data and parity.

Every failed check is reported, not only the first. Checks:
 (a) the build hash file equals the current hash of ``guard.source_globs``;
 (b) a passing test receipt (``qrh test-receipt``) exists for that source hash and is recent enough;
 (c) each required dataset is PASS, unchanged on disk and fresh (its data ends within the max age);
 (d) parity: a recent parity report for this trial in ``<QRH_HOME>/parity/``. A missing, unreadable, other-trial or
     old report only WARNS (never blocks), saying what is wrong and what to do; or parity is disabled with a reason;
 (e) readiness: when ``guard.require_readiness: <trial id>`` is set, that trial's deployment readiness report is
     acknowledged with a valid human signature for its CURRENT warning set (``qrh readiness status``). Absent key:
     the check is not applied.
"""

from __future__ import annotations

import glob
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from qrh.hashing import code_hash
from qrh.hooks.common import GuardConfig
from qrh.hooks.datacheck import dataset_problems, dataset_row
from qrh.hooks.runner import HookContext, HookResult


# A parity report older than this gets a warning (the default). A warning never blocks a restart.
PARITY_MAX_AGE = timedelta(days=7)


class SourceHashError(RuntimeError):
    """The source files to hash cannot be determined."""


def source_files(globs: tuple[str, ...] | list[str], base: str | Path) -> list[Path]:
    seen: dict[str, Path] = {}
    for pattern in globs:
        for name in glob.glob(pattern, root_dir=str(base), recursive=True):
            p = Path(base) / name
            if p.is_file():
                seen[str(p.resolve())] = p
    return [seen[k] for k in sorted(seen)]


def compute_source_hash(globs: tuple[str, ...] | list[str], base: str | Path) -> str:
    if not globs:
        raise SourceHashError("guard.source_globs is empty")
    files = source_files(globs, base)
    if not files:
        raise SourceHashError(f"guard.source_globs matched no files under {base}")
    try:
        return code_hash(files)
    except OSError as exc:
        raise SourceHashError(f"cannot read source files: {exc}") from exc


def parse_offset(text: str) -> timedelta:
    """A pandas offset string such as ``3D``, ``12h`` or ``30min`` as a fixed duration."""
    import pandas as pd
    from pandas.tseries.frequencies import to_offset

    try:
        return pd.Timedelta(to_offset(text)).to_pytimedelta()
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{text!r} is not a fixed pandas offset string (examples: 3D, 12h, 30min)") from exc


def _aware(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parity_report_exists(home: Path) -> bool:
    d = home / "parity"
    return d.is_dir() and any(p.is_file() for p in d.iterdir())


def _session_trial(home: Path, session_id: str) -> str | None:
    """The trial most recently tagged on this session, or None."""
    try:
        lines = (home / "logs" / "session_trials.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    found: str | None = None
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("session_id") == session_id and rec.get("trial_id"):
            found = str(rec["trial_id"])
    return found


_PARITY_NEXT = "Next step: build a fresh parity report for this trial (backtest vs live) before you rely on this restart."


def parity_warning(home: Path, trial: str | None, now: datetime) -> str | None:
    """A plain warning when there is no recent, readable parity report for this trial; None when all is well."""
    who = f"trial {trial}" if trial else "this trial"
    d = home / "parity"
    files = sorted(d.glob("*.json")) if d.is_dir() else []
    if not files:
        return (f"parity report missing for {who}. Without it nobody has checked that the live engine trades "
                f"like the backtest. {_PARITY_NEXT}")
    good: list[tuple[Path, dict]] = []
    unreadable: list[Path] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unreadable.append(f)
            continue
        if isinstance(data, dict):
            good.append((f, data))
        else:
            unreadable.append(f)
    mine = [f for f, data in good if trial is None or data.get("trial_id") == trial]
    if not mine:
        if unreadable:
            return (f"a parity report for {who} cannot be read ({unreadable[0].name}), so its result is unknown. "
                    f"{_PARITY_NEXT}")
        return (f"parity report missing for {who}: the reports found are for other trials. "
                f"A report for another trial says nothing about this one. {_PARITY_NEXT}")
    try:
        newest = max(datetime.fromtimestamp(f.stat().st_mtime, timezone.utc) for f in mine)
    except OSError:
        return f"a parity report for {who} cannot be read, so its age is unknown. {_PARITY_NEXT}"
    if now - newest > PARITY_MAX_AGE:
        return (f"the newest parity report for {who} is older than {PARITY_MAX_AGE.days} days "
                f"(from {newest.date().isoformat()}), so it may no longer match the live engine. {_PARITY_NEXT}")
    return None


def _check_build(guard: GuardConfig, base: Path, current: str | None) -> str | None:
    if not guard.build_hash_file:
        return "build hash: guard.build_hash_file is not configured"
    path = Path(guard.build_hash_file)
    path = path if path.is_absolute() else base / path
    try:
        recorded = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return f"build hash: cannot read {path} ({type(exc).__name__})"
    if current is None:
        return "build hash: cannot compare, the current source hash is unavailable"
    if recorded != current:
        return f"build hash: {path.name} ({recorded[:12]}) does not match the current source ({current[:12]})"
    return None


def run(ctx: HookContext) -> HookResult:
    p = ctx.payload
    if (p.event and p.event != "PreToolUse") or p.tool_name != "Bash" or not p.command:
        return HookResult("allow", "not a Bash command")
    if not any(rx.search(p.command) for rx in ctx.config.engine_start_commands):
        return HookResult("allow", "not an engine start command")
    guard = ctx.config.guard
    if guard is None:
        return HookResult(
            "block", "guard is not configured",
            stderr="An engine start was recognized but the 'guard:' section of config.yaml is not configured.",
        )
    base = Path(p.cwd or ".")
    now = ctx.now()
    failures: list[str] = []

    current: str | None = None
    try:
        current = compute_source_hash(guard.source_globs, base)
    except SourceHashError as exc:
        failures.append(f"source hash: {exc}")
    if (msg := _check_build(guard, base, current)) is not None:
        failures.append(msg)

    try:
        conn = ctx.registry().conn
        max_receipt_age = parse_offset(guard.test_receipt_max_age) if guard.test_receipt_max_age else None
        if current is not None:
            receipt = ctx.registry().latest_passing_test_receipt(current)
            if receipt is None:
                failures.append(
                    "test receipt: no passing test receipt for the current source "
                    "(run: qrh test-receipt -- <test command>)")
            elif max_receipt_age is not None and now - _aware(receipt["created_at"]) > max_receipt_age:
                failures.append(f"test receipt: the latest pass is older than {guard.test_receipt_max_age}")
        for name in guard.required_datasets:
            for why in dataset_problems(conn, name):
                failures.append(f"dataset {name!r}: {why}")
            age = guard.max_dataset_age.get(name)
            rec = dataset_row(conn, name)
            if age is not None and rec is not None and not dataset_problems(conn, name):
                limit = parse_offset(age)
                if not rec["end"]:
                    failures.append(f"dataset {name!r}: no end timestamp to measure its age")
                elif now - _aware(rec["end"]) > limit:
                    failures.append(f"dataset {name!r}: data ends {rec['end']}, older than {age}")
    except ValueError as exc:
        failures.append(f"config: {exc}")
    except Exception as exc:  # the registry itself is unusable
        failures.append(f"registry: {type(exc).__name__}: {exc}")

    if guard.require_readiness:
        from qrh.readiness import ack

        try:
            acknowledged, why = ack.is_acknowledged(guard.require_readiness, ctx.home, ctx.registry())
        except Exception as exc:  # the registry itself is unusable: fail closed
            acknowledged, why = False, f"cannot be checked ({type(exc).__name__}: {exc})"
        if not acknowledged:
            failures.append(f"readiness: trial {guard.require_readiness} is not acknowledged: {why}. "
                            f"The human acknowledges with `qrh readiness ack {guard.require_readiness}` in their own terminal")

    extra: dict[str, object] = {}
    parity_warn: str | None = None
    if guard.parity_required:
        trial = _session_trial(ctx.home, p.session_id) or guard.require_readiness
        parity_warn = parity_warning(ctx.home, trial, now)
    else:
        extra["parity_disabled_reason"] = guard.parity_disabled_reason

    note = f"WARNING (does not block the restart): {parity_warn}\n" if parity_warn else ""
    if parity_warn:
        extra["parity_warning"] = parity_warn
    if failures:
        return HookResult(
            "block", f"{len(failures)} check(s) failed", extra=extra,
            stderr="Engine start refused; failed checks:\n" + "\n".join(f"  - {f}" for f in failures) + "\n" + note,
        )
    if parity_warn:
        return HookResult("warn", "pre-restart checks passed; parity report warning", stdout=note, extra=extra)
    return HookResult("allow", "all pre-restart checks passed", extra=extra)
