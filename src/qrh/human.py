"""The human steps: create or rotate the approval key, approve one request, sign the config baseline, acknowledge a
readiness report.

Every function here needs a controlling terminal and reads the passphrase from it (``qrh.tty``). There is no
other input path: no option, no stdin fallback, no variable. A process without a terminal (an agent's shell,
a pipe, CI) gets HumanError, which the CLI turns into exit code 2.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qrh import approvals, signing, tty


class HumanError(RuntimeError):
    """The human step cannot run (no terminal, wrong passphrase, invalid request, ...)."""


def _require_terminal(what: str) -> None:
    if not tty.has_terminal():
        raise HumanError(f"{what} needs a person at a terminal: no controlling terminal is available. Run it "
                         "yourself in your own terminal, not through an agent, a pipe or CI.")


def _read_new_passphrase() -> str:
    first = tty.read_secret("New passphrase (at least 12 characters, not echoed): ")
    second = tty.read_secret("Repeat the new passphrase: ")
    if first != second:
        raise HumanError("the two passphrases do not match; nothing was changed")
    if len(first) < signing.MIN_PASSPHRASE:
        raise HumanError(f"the passphrase must be at least {signing.MIN_PASSPHRASE} characters")
    return first


def _unlock(home: Path, now: datetime, prompt: str = "Approval passphrase (not echoed): ") -> Any:
    try:
        return signing.unlock(home, tty.read_secret(prompt), now)
    except signing.SigningError as exc:
        raise HumanError(str(exc)) from exc


def init_keys(home: Path, rotate: bool = False, now: datetime | None = None) -> signing.PublicKey:
    """Create the key pair, or with ``rotate`` replace it after the old passphrase (dropping open approvals)."""
    now = now or datetime.now(timezone.utc)
    _require_terminal("qrh approve --init")
    try:
        if not rotate:
            if signing.has_keypair(home):
                raise HumanError("an approval key already exists; to replace it use `qrh approve --init --rotate` "
                                 "(needs the old passphrase). Lost passphrase: see docs/security.md")
            return signing.init_keys(home, _read_new_passphrase(), now)
        if not signing.has_keypair(home):
            raise HumanError("there is no approval key to rotate; run `qrh approve --init`")
        tty.write("Rotating the approval key. Every pending and approved request will be dropped.\n")
        old = tty.read_secret("Current passphrase (not echoed): ")
        signing.unlock(home, old, now)  # fail fast on a wrong passphrase, before asking for the new one
        pub = signing.rotate_keys(home, old, _read_new_passphrase(), now)
        approvals.invalidate_all(home)
        return pub
    except signing.SigningError as exc:
        raise HumanError(str(exc)) from exc


def approve(home: Path, approval_id: str, reason: str, now: datetime | None = None) -> dict[str, Any]:
    """Show the request on the terminal, read the passphrase, sign. Validation happens before any prompt."""
    now = now or datetime.now(timezone.utc)
    _require_terminal("qrh approve")
    try:
        approvals.check_reason(reason)
        rec = approvals.load_pending(home, approval_id, now)
        if signing.load_public(home) is None:
            raise HumanError("no approval key yet: run `qrh approve --init` in your own terminal first")
    except (approvals.ApprovalError, signing.SigningError) as exc:
        raise HumanError(str(exc)) from exc
    tty.write(approvals.describe(rec) + f"Your reason: {reason.strip()}\n")
    private = _unlock(home, now)
    try:
        return approvals.sign_approval(home, approval_id, reason, private, now=now)
    except approvals.ApprovalError as exc:
        raise HumanError(str(exc)) from exc


def sign_config(home: Path, settings: list[str], now: datetime | None = None) -> dict[str, Any]:
    """Show the hashes about to be signed, read the passphrase, sign the config baseline."""
    from qrh import integrity

    now = now or datetime.now(timezone.utc)
    _require_terminal("qrh config sign")
    try:
        if signing.load_public(home) is None:
            raise HumanError("no approval key yet: run `qrh approve --init` in your own terminal first")
    except signing.SigningError as exc:
        raise HumanError(str(exc)) from exc
    cur = integrity.current(home, settings)
    lines = ["Config baseline to sign:", f"  {home / 'config.yaml'}  sha256 {cur['config_sha256']}"]
    lines += [f"  {s['path']}  hooks section sha256 {s['hooks_sha256']}" for s in cur["settings"]]
    if not settings:
        lines.append("  (no --settings given: Claude Code hook registrations are NOT covered)")
    tty.write("\n".join(lines) + "\nSign only if these are the files and the hook setup you intend.\n")
    private = _unlock(home, now)
    return integrity.sign_baseline(home, private, settings, now)


def _readiness_text(data: Any, home: Path) -> str:
    """What the human reads before typing the passphrase: verdict, counts and EVERY high warning."""
    from qrh.banner import BANNER

    d = data.doc
    t = d["totals"]
    lines = [BANNER.rstrip("\n"),
             f"Deployment readiness acknowledgement for trial {d['trial_id']}",
             f"Final verdict: {d['final_verdict_display']}",
             f"Warnings: {t['high']} HIGH, {t['medium']} MEDIUM, {t['low']} LOW ({sum(t.values())} in total)",
             f"Report: {home / 'reports' / d['trial_id'] / 'readiness.pdf'} (report hash {data.hash[:12]}; "
             "run `qrh readiness report` to refresh it)"]
    if d["final_verdict"] not in ("PASS", "PASS_CONDITIONAL"):
        lines.append(f"The final verdict is {d['final_verdict_display']}, not a pass: you would be accepting a trial "
                     "that did not pass.")
    high = [w for w in d["warnings"] if w["severity"] == "high"]
    lines.append(f"HIGH warnings ({len(high)}), every one listed:" if high else "No HIGH warnings.")
    for n, w in enumerate(high, 1):
        lines += [f"  {n}. ▲ [{w['stage']} {w['kind']} {w['code']}] id {w['id'][:8]}: {w['message']}",
                  f"     Resolve: {w['resolution']}"]
    lines.append("By signing you confirm you read the full report and accept this exact warning set. Any later "
                 "change in the warnings invalidates the acknowledgement.")
    return "\n".join(lines) + "\n"


def acknowledge_readiness(home: Path, trial_id: str, now: datetime | None = None,
                          registry: Any = None) -> dict[str, Any]:
    """Show the readiness report summary on the terminal, read the passphrase, sign the acknowledgement.

    Everything is validated before any prompt. The report is recomputed here, so what is signed is what was shown.
    """
    from qrh.readiness import ack, collect, data

    now = now or datetime.now(timezone.utc)
    _require_terminal("qrh readiness ack")
    try:
        if signing.load_public(home) is None:
            raise HumanError("no approval key yet: run `qrh approve --init` in your own terminal first")
    except signing.SigningError as exc:
        raise HumanError(str(exc)) from exc
    own = registry is None
    if own:
        from qrh.memory.registry import Registry

        registry = Registry(home / "registry.sqlite3")
    try:
        report = data.build_data(collect.collect(home, trial_id, registry))
    except collect.ReadinessError as exc:
        raise HumanError(str(exc)) from exc
    finally:
        if own:
            registry.close()
    if report.doc["final_verdict"] == collect.NOT_RUN:
        raise HumanError("there is no final verdict (S7) for this trial yet, so there is nothing to acknowledge: "
                         f"run `qrh run {trial_id}` first")
    tty.write(_readiness_text(report, home))
    private = _unlock(home, now, "Approval passphrase to acknowledge this readiness report (not echoed): ")
    try:
        return ack.sign_ack(home, report, private, now)
    except ack.AckError as exc:
        raise HumanError(str(exc)) from exc
