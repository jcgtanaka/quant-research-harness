"""SessionStart: a short deterministic summary (<= 25 lines) of what needs attention.

Open trials (final S7 verdict not PASS, newest 10), datasets that are BLOCKED, overrides used in the last
7 days, and report render errors. Prints nothing when there is nothing to say. Informational: never blocks.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from qrh.hooks.runner import HookContext, HookResult

MAX_LINES = 25
MAX_TRIALS = 10
MAX_DATASETS = 3
MAX_CONDITIONAL = 3
MAX_OVERRIDES = 3
MAX_REPORT_ERRORS = 2
OVERRIDE_DAYS = 7


def _short(text: str, n: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= n else flat[: n - 1] + "…"


def _open_trials(conn: Any) -> list[str]:
    lines: list[str] = []
    rows = conn.execute(
        "SELECT h.trial_id, h.strategy_family, h.asset_class, h.frequency,"
        " (SELECT r.verdict FROM receipts r WHERE r.trial_id=h.trial_id AND r.stage='S7') AS final,"
        " (SELECT r.stage || ' ' || r.verdict FROM receipts r WHERE r.trial_id=h.trial_id"
        "  ORDER BY r.stage DESC LIMIT 1) AS latest"
        " FROM hypotheses h ORDER BY h.rowid DESC"
    )
    for r in rows:
        if r["final"] in ("PASS", "PASS_CONDITIONAL"):
            continue
        lines.append(
            f"- {r['trial_id'][:8]}  {r['latest'] or 'not run'}  "
            f"{r['strategy_family']}/{r['asset_class']}/{r['frequency']}"
        )
        if len(lines) == MAX_TRIALS:
            break
    return lines


def _conditional_passes(conn: Any) -> list[str]:
    """Trials whose final verdict is PASS · CONDITIONAL, with the open conditions to quote alongside any result."""
    from qrh.pipeline.receipt import load_receipt

    lines: list[str] = []
    rows = conn.execute("SELECT trial_id, path FROM receipts WHERE stage='S7' AND verdict='PASS_CONDITIONAL'"
                        " ORDER BY rowid DESC LIMIT ?", (MAX_CONDITIONAL,))
    for r in rows:
        try:
            conditions = load_receipt(r["path"]).params.get("conditions") or []
        except Exception:
            conditions = []
        shown = ", ".join(f"{c.get('question_id')} ({c.get('status')})" for c in conditions[:4])
        more = f" +{len(conditions) - 4} more" if len(conditions) > 4 else ""
        lines.append(f"- {r['trial_id'][:8]}  PASS · CONDITIONAL  conditions: {shown or 'see the S7 receipt'}{more}")
    return lines


def _blocked_datasets(conn: Any) -> tuple[int, list[str]]:
    rows = conn.execute("SELECT name, reasons FROM datasets WHERE status='BLOCKED' ORDER BY name").fetchall()
    lines = []
    for r in rows[:MAX_DATASETS]:
        try:
            why = "; ".join(json.loads(r["reasons"]))
        except (ValueError, TypeError):
            why = ""
        lines.append(f"- {r['name']}: BLOCKED" + (f" ({_short(why, 60)})" if why else ""))
    return len(rows), lines


def _overrides(conn: Any, since: str) -> tuple[int, list[str]]:
    rows = conn.execute("SELECT * FROM overrides WHERE created_at>=? ORDER BY id DESC", (since,)).fetchall()
    lines = []
    for r in rows[:MAX_OVERRIDES]:
        what = r["hook"] or f"{r['stage']} trial {str(r['trial_id'])[:8]}"
        lines.append(f"- {r['created_at'][:10]} {what}: {_short(r['reason'], 70)}")
    return len(rows), lines


def _report_errors(home: Path) -> tuple[int, list[str]]:
    found: list[str] = []
    for manifest in sorted((home / "reports").glob("*/manifest.json")):
        try:
            errors = json.loads(manifest.read_text(encoding="utf-8")).get("errors", [])
        except (OSError, ValueError, AttributeError):
            errors = [{"stage": "manifest", "error": "unreadable"}]
        for e in errors:
            found.append(f"- {manifest.parent.name[:8]} {e.get('stage', '?')}: {_short(e.get('error', ''), 70)}")
    return len(found), found[:MAX_REPORT_ERRORS]


LOG_TAIL_BYTES = 1 << 20
KEY_RECENT_DAYS = 7


def _loop_guard_passes(home: Path, since: datetime) -> list[str]:
    """F7: Stop answers the loop guard let through without any check, from the tail of the hook log."""
    path = home / "logs" / "hooks.jsonl"
    try:
        with open(path, "rb") as fh:
            size = fh.seek(0, 2)
            fh.seek(max(0, size - LOG_TAIL_BYTES))
            tail = fh.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []
    hits: list[dict[str, Any]] = []
    for line in tail:
        if '"loop_guard"' not in line:
            continue
        try:
            rec = json.loads(line)
            when = datetime.fromisoformat(str(rec.get("ts")))
        except (ValueError, TypeError):
            continue
        if rec.get("loop_guard") is True and when >= since:
            hits.append(rec)
    if not hits:
        return []
    last = hits[-1]
    return [f"- LOOP GUARD: {len(hits)} final answer(s) in the last {OVERRIDE_DAYS} days went through UNCHECKED on a "
            f"second Stop attempt (latest {str(last.get('ts'))[:16]}, hook {last.get('hook')}). Re-check those answers."]


def code_exposure(package_dir: Path | None = None) -> str | None:
    """A line when the code the hooks run is writable by the current user (an editable install from a source
    checkout, or any install directory this user owns); None when it is not. The hooks then run code an agent in
    this account could edit."""
    if package_dir is None:
        import qrh

        package_dir = Path(qrh.__file__).resolve().parent
    if not (os.access(package_dir, os.W_OK) or os.access(package_dir / "signing.py", os.W_OK)):
        return None
    installed = bool({"site-packages", "dist-packages"} & set(package_dir.parts))
    kind = "an install directory" if installed else "a source checkout (editable install)"
    return (f"- qrh is imported from {kind} this user can write ({package_dir}): the hooks then run code an agent "
            "in this account could edit. See 'Deployment hardening' in docs/security.md")


def _exposure_line() -> str | None:
    return code_exposure()


def _security(home: Path, now: datetime) -> list[str]:
    """Approval key and config baseline status: what the human still has to set up, or what changed."""
    from qrh import integrity, signing

    lines: list[str] = []
    exposure = _exposure_line()
    if exposure:
        lines.append(exposure)
    try:
        pub = signing.load_public(home)
    except signing.SigningError as exc:
        pub = None
        lines.append(f"- approval key unusable: {exc} (see docs/security.md)")
    else:
        if pub is None:
            lines.append("- no approval key: blocks cannot be approved until the user runs `qrh approve --init` in "
                         "their own terminal")
        else:
            lines.append(f"- approval key fingerprint {pub.key_id}: compare it with the one you noted when you ran "
                         "`qrh approve --init` (kept where the agent cannot write); a different value means the key "
                         "was replaced")
            try:  # ctime cannot be set back by an unprivileged process: a recent change is worth a look
                changed = datetime.fromtimestamp(signing.pub_path(home).stat().st_ctime, tz=timezone.utc)
            except OSError:
                changed = None
            if changed is not None and timedelta(0) <= now - changed <= timedelta(days=KEY_RECENT_DAYS):
                lines.append(f"- approval key {pub.key_id} was created or changed on {changed.isoformat()[:16]}; if the "
                             "user did not run `qrh approve --init` or `--rotate`, treat approvals since then as suspect")
    st = integrity.check(home)
    if st.state == "missing":
        lines.append("- no signed config baseline: the user should run `qrh config sign --settings "
                     "~/.claude/settings.json` in their own terminal so changed config or hooks are detected")
    elif st.state in ("changed", "invalid"):
        lines.append(f"- {integrity.MESSAGE} ({st.detail}); hooks that enforce are blocking until then")
    return lines


def run(ctx: HookContext) -> HookResult:
    p = ctx.payload
    if p.event and p.event != "SessionStart":
        return HookResult("info", "not a SessionStart event")
    sections: list[tuple[str, list[str]]] = []
    since = (ctx.now() - timedelta(days=OVERRIDE_DAYS)).isoformat()
    security = _security(ctx.home, ctx.now())
    if security:
        sections.append(("Security setup:", security))
    loop = _loop_guard_passes(ctx.home, ctx.now() - timedelta(days=OVERRIDE_DAYS))
    if loop:
        sections.append(("Stop loop guard:", loop))
    try:
        conn = ctx.registry().conn
        _registry_sections(ctx, conn, since, sections)
    except Exception as exc:  # informational: still print the security status
        return HookResult("error", f"{type(exc).__name__}: {exc}", stdout=_render(sections))
    if ctx.config.publish_guard is None:
        sections.append(("Publish guard:", ["- disabled: no `publish_guard` section in config.yaml (no protected repos: "
                                            "pushes, hook changes and publication commands are not gated)"]))
    if ctx.config.runaway is None:
        sections.append(("Runaway guard:", ["- disabled: no `runaway` section in config.yaml (no loop, failure or spend limits)"]))
    if not sections:
        return HookResult("info", "nothing to report")
    text = _render(sections)
    return HookResult("info", f"{text.count(chr(10))} line(s) of context", stdout=text)


def _render(sections: list[tuple[str, list[str]]]) -> str:
    if not sections:
        return ""
    lines = ["qrh context:"]
    for title, body in sections:
        lines.append(title)
        lines.extend(body)
    if len(lines) > MAX_LINES:
        omitted = len(lines) - (MAX_LINES - 1)
        lines = lines[: MAX_LINES - 1] + [f"... {omitted} more line(s) omitted"]
    return "\n".join(lines) + "\n"


def _registry_sections(ctx: HookContext, conn: Any, since: str, sections: list[tuple[str, list[str]]]) -> None:
    n_ds, ds_lines = _blocked_datasets(conn)
    if ds_lines:
        sections.append((f"Datasets BLOCKED ({n_ds}):", ds_lines))
    n_err, err_lines = _report_errors(ctx.home)
    if err_lines:
        sections.append((f"Report render errors ({n_err}):", err_lines))
    n_ov, ov_lines = _overrides(conn, since)
    if ov_lines:
        sections.append((f"Overrides used in the last {OVERRIDE_DAYS} days ({n_ov}):", ov_lines))
    conditional = _conditional_passes(conn)
    if conditional:
        sections.append(("Conditional passes (quote the conditions whenever you quote a result):", conditional))
    trials = _open_trials(conn)
    if trials:
        sections.append(("Open trials, latest verdict (newest first):", trials))
