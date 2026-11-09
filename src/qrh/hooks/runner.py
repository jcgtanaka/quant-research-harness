"""Hook dispatch: parse the payload, load config, run the hook, apply the fail-closed policy.

Enforcement hooks fail closed: anything that prevents the check from running blocks (exit 2).
Informational hooks never break a session: they exit 0 whatever happens.
A block can be turned into one allow only by a human approval token (``qrh approve``, see ``qrh.approvals``)
for the identical action; the override is recorded in the registry and the log. There is no environment
variable or other switch an agent could set for itself.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qrh import approvals
from qrh.banner import decision_banner
from qrh.hooks import sessions
from qrh.config import QrhHomeError, ensure_home
from qrh.hooks.common import (
    Config,
    ConfigError,
    Payload,
    PayloadError,
    load_config,
    log_invocation,
    parse_payload,
    redact,
    redact_command,
)

if TYPE_CHECKING:
    from qrh.memory.registry import Registry

BLOCK_EXIT_CODE = 2
DB_NAME = "registry.sqlite3"
REASON_LIMIT = 200

STOP_BLOCK_CAP = 3  # consecutive blocked Stops in one session before the block says the user must decide

CodegraphRunner = Callable[[str, Path], "tuple[int, str] | None"]


@dataclass
class HookResult:
    decision: str  # allow | block | warn | info | error
    reason: str = ""  # one short line, goes to the log
    stdout: str = ""  # added to the agent's context (SessionStart, UserPromptSubmit)
    stderr: str = ""  # shown to the agent on a block; defaults to the reason
    trial_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # additional log fields
    # identifies the blocked action for approval when there is no tool call (Stop): fixed strings the human sees
    action_parts: tuple[str, ...] | None = None
    on_override: Callable[[], None] | None = None  # runs when a human approval lets this block through
    approvable: bool | None = None  # None: the hook spec decides; True/False overrides it for this block
    human_command: str = ""  # a non-approval human decision (config sign): shown in the banner instead


@dataclass(frozen=True)
class HookSpec:
    name: str
    event: str
    matcher: str | None
    timeout: int
    enforcement: bool
    module: str
    run: Callable[["HookContext"], HookResult] | None = None
    approvable: bool = True  # False: a block is final, no approval is offered (approval-guard)
    needs_config: bool = True  # False: runs with the defaults even if config.yaml is broken
    risk: str = "the check exists to stop exactly this; passing it means acting on unverified ground"

    def resolve(self) -> Callable[["HookContext"], HookResult]:
        return self.run or importlib.import_module(self.module).run


@dataclass(frozen=True)
class HookOutcome:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class HookContext:
    """Everything a hook may use; the registry is opened lazily and closed by the runner."""

    def __init__(
        self, home: Path, payload: Payload, config: Config, env: Mapping[str, str],
        now: Callable[[], datetime], codegraph: CodegraphRunner | None,
    ):
        self.home, self.payload, self.config, self.env = home, payload, config, env
        self.now, self.codegraph = now, codegraph
        self._registry: Registry | None = None

    def registry(self) -> "Registry":
        if self._registry is None:
            from qrh.memory.registry import Registry

            try:
                self._registry = Registry(self.home / DB_NAME)
            except QrhHomeError as exc:
                raise RuntimeError(f"registry unreadable: {exc}") from exc
        return self._registry

    def close(self) -> None:
        if self._registry is not None:
            self._registry.close()
            self._registry = None


def _hooks() -> Mapping[str, HookSpec]:
    from qrh.hooks.specs import HOOKS

    return HOOKS


def _block_text(name: str, result: HookResult, note: str = "") -> str:
    body = result.stderr or result.reason
    return "\n".join(x for x in (f"BLOCKED by qrh hook {name}: {body}".rstrip(), note) if x) + "\n"


def _home_prefix(home: Path) -> str:
    return "" if home == Path.home() / ".qrh" else f"QRH_HOME={home} "


def _approve_command(home: Path, rid: str) -> str:
    prefix = _home_prefix(home)
    return f'{prefix}qrh approve {rid} --reason "<why this is acceptable, at least {approvals.MIN_REASON} characters>"'


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.strip("\n").splitlines())


def human_block_text(
    name: str, summary: str, why: str, risk: str, command: str, after: str, notes: str = "",
) -> str:
    """The banner form of a block that needs a human decision."""
    details = (
        f"Hook: {name} (BLOCKED by qrh hook {name})\n"
        f"Blocked action: {summary}\n"
        f"Why it was blocked:\n{_indent(why)}\n"
        f"Risk: {risk}\n"
        "A human must decide. To allow it, run in your OWN terminal (not through this agent):\n"
        f"    {command}\n"
        f"{after}\n"
        + (f"Notes:\n{_indent(notes)}\n" if notes.strip() else "")
    )
    return decision_banner(details)


def _approval_text(name: str, spec: HookSpec, result: HookResult, home: Path, rid: str, summary: str,
                   notes: str) -> str:
    return human_block_text(
        name, summary, result.stderr or result.reason, spec.risk, _approve_command(home, rid),
        f"Approval id {rid}: valid once, for this exact action in this session, within "
        f"{approvals.DEFAULT_TTL_S // 60} minutes. After approving, the user asks the agent to retry the identical "
        "action once. Otherwise fix the cause instead.", notes)


def _action(result: HookResult, payload: Payload) -> dict[str, Any] | None:
    if result.action_parts:
        return approvals.parts_action(*result.action_parts)
    if payload.tool_name:
        return approvals.tool_action(payload.tool_name, dict(payload.tool_input), payload.cwd)
    return None


def _apply_approval(
    spec: HookSpec, ctx: HookContext | None, payload: Payload | None, result: HookResult, home: Path, now: datetime,
) -> tuple[HookResult, str | None]:
    """Returns (result, stderr). A valid, unspent human approval for this exact action in this session turns the
    block into an allow; otherwise a pending request is created and the banner tells the human what to run."""
    approvable = spec.approvable if result.approvable is None else result.approvable
    if result.human_command:
        why = result.stderr or result.reason
        return result, human_block_text(spec.name, _summary(result, payload), why, spec.risk, result.human_command,
                                        "An approval id does not clear this block; only the command above does.")
    if not approvable:
        return result, _block_text(spec.name, result, "This block cannot be overridden.")
    action = _action(result, payload) if payload is not None else None
    if action is None:
        return result, _block_text(spec.name, result)
    fp = approvals.fingerprint_of(action)
    sid = payload.session_id if payload else ""
    try:
        got = approvals.consume(home, hook=spec.name, fingerprint=fp, session_id=sid, now=now)
    except approvals.ApprovalError as exc:
        return result, human_block_text(spec.name, approvals.summarize(action), result.stderr or result.reason,
                                        spec.risk, "see docs/security.md (the approval key is unusable)",
                                        f"Approval could not be checked: {exc}")
    if not got.ok:
        try:
            rid = approvals.request(home, hook=spec.name, action=action, reason=redact(result.reason)[:REASON_LIMIT],
                                    session_id=sid, now=now)
        except (OSError, approvals.ApprovalError) as exc:
            return result, _block_text(spec.name, result, "\n".join(got.notes + [f"Approval request could not be "
                                                                                 f"written: {exc}"]))
        result.extra = {**result.extra, "approval_pending": rid}
        return result, _approval_text(spec.name, spec, result, home, rid, approvals.summarize(action),
                                      "\n".join(got.notes))
    rec = got.record or {}
    rid, reason = str(rec.get("id", "")), str(rec.get("approval_reason", ""))
    recorded = True
    try:
        reg = ctx.registry() if ctx else _open_registry(home)
        try:
            reg.log_hook_override(spec.name, reason, payload.session_id if payload else "", result.trial_id,
                                  approval_id=rid)
        finally:
            if not ctx:
                reg.close()
    except Exception:
        recorded = False  # the registry may be the very thing that is broken; the hook log still records it
    if result.on_override is not None:
        try:
            result.on_override()
        except Exception:
            pass  # bookkeeping of the hook must not undo a human decision
    out = HookResult(
        "allow", f"override: {redact(reason)}", trial_id=result.trial_id,
        extra={**result.extra, "override": True, "approval_id": rid, "overridden_reason": result.reason[:REASON_LIMIT],
               **({} if recorded else {"override_recorded_in_registry": False})},
    )
    return out, None


def _summary(result: HookResult, payload: Payload | None) -> str:
    if payload is None:
        return "(no payload)"
    action = _action(result, payload)
    return approvals.summarize(action) if action else (payload.event or "(no tool call)")


def _integrity_block(spec: HookSpec, home: Path) -> HookResult | None:
    """Enforcement hooks that read config refuse to run on a config or hook registration that changed since signed."""
    if not (spec.enforcement and spec.needs_config):
        return None
    from qrh import integrity

    st = integrity.check(home)
    if st.state == "missing":
        from qrh import signing

        if not signing.has_keypair(home):
            return None  # first use: no key to sign with yet; session-context warns
        return HookResult("block", integrity.MISSING_MESSAGE, stderr=f"{integrity.MISSING_MESSAGE}.\n{st.detail}",
                          human_command=_home_prefix(home) + st.command(), extra={"integrity": st.state})
    if st.state not in ("changed", "invalid"):
        return None
    return HookResult("block", integrity.MESSAGE, stderr=f"{integrity.MESSAGE}.\n{st.detail}",
                      human_command=_home_prefix(home) + st.command(), extra={"integrity": st.state})


def _stop_blocks_path(home: Path, session_id: str) -> Path:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
    return sessions.sessions_dir(home) / f"{digest}.stop_blocks.json"


def _count_stop_block(home: Path, session_id: str, blocked: bool) -> int:
    """Consecutive blocked Stops of this session (kept next to the session record); an unblocked Stop resets it.
    Only bookkeeping for the cap message: it never turns a block into an allow, and a failure to store it is ignored."""
    if not session_id:
        return 0
    path = _stop_blocks_path(home, session_id)
    try:
        if not blocked:
            path.unlink(missing_ok=True)
            return 0
        try:
            count = int(json.loads(path.read_text(encoding="utf-8")).get("blocks", 0))
        except (OSError, ValueError, AttributeError, TypeError):
            count = 0
        count += 1
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"session_id": session_id, "blocks": count}), encoding="utf-8")
        os.replace(tmp, path)
        return count
    except OSError:
        return 0


def _cap_note(blocks: int) -> str:
    return (f"This Stop has now been blocked {blocks} times in a row in this session (cap {STOP_BLOCK_CAP}). The claim "
            "cannot be verified from the transcript and receipts, so the user must decide. Do not repeat it: remove "
            "the claim or produce the evidence, and let the user choose.\n")


def _open_registry(home: Path) -> "Registry":
    from qrh.memory.registry import Registry

    return Registry(home / DB_NAME)


def run_hook(
    name: str,
    stdin_text: str,
    *,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
    now: Callable[[], datetime] | None = None,
    codegraph: CodegraphRunner | None = None,
    hooks: Mapping[str, HookSpec] | None = None,
) -> HookOutcome:
    """Run one hook for one invocation and return exit code, stdout and stderr."""
    started = time.perf_counter()
    spec = (hooks if hooks is not None else _hooks()).get(name)
    if spec is None:
        return HookOutcome(BLOCK_EXIT_CODE, stderr=f"BLOCKED: unknown hook {name!r}\n")
    env = os.environ if env is None else env
    clock = now or (lambda: datetime.now(timezone.utc))
    notes_out = ""
    try:
        if home is None:
            home = ensure_home()
        else:
            home.mkdir(parents=True, exist_ok=True)
    except (QrhHomeError, OSError) as exc:
        if spec.enforcement:
            return HookOutcome(BLOCK_EXIT_CODE, stderr=_block_text(name, HookResult("block", f"QRH_HOME unusable: {exc}")))
        return HookOutcome(0)

    payload: Payload | None = None
    ctx: HookContext | None = None
    try:
        try:
            payload = parse_payload(stdin_text)
            if payload.event in sessions.EVENTS:
                try:  # F5: remember the real transcript of this session (first path wins)
                    sessions.record(home, payload.session_id, payload.transcript_path, clock())
                except OSError:
                    pass
            integrity_block = _integrity_block(spec, home)
            if integrity_block is not None:
                result = integrity_block
            else:
                config = load_config(home) if spec.needs_config else Config()
                ctx = HookContext(home, payload, config, env, clock, codegraph)
                result = spec.resolve()(ctx)
        except PayloadError as exc:
            result = HookResult("block" if spec.enforcement else "error", str(exc))
        except ConfigError as exc:
            if spec.enforcement:
                result = HookResult("block", f"config.yaml is invalid: {exc}")
            else:
                result = HookResult("info", f"config.yaml invalid, hook skipped: {exc}")
                if spec.event == "SessionStart":
                    notes_out = "qrh: config.yaml is invalid; hooks that need it are skipped.\n"
        except Exception as exc:  # the check itself could not run
            text = f"{type(exc).__name__}: {exc}"
            result = HookResult("block", f"check could not run: {text}") if spec.enforcement else HookResult("error", text)
        block_text: str | None = None
        if result.decision == "block":
            result, block_text = _apply_approval(spec, ctx, payload, result, home, clock())
        if payload is not None and payload.event == "Stop":
            blocks = _count_stop_block(home, payload.session_id, result.decision == "block")
            if result.decision == "block" and blocks >= STOP_BLOCK_CAP:
                result.extra = {**result.extra, "stop_block_cap": blocks}
                block_text = (block_text or _block_text(name, result)) + _cap_note(blocks)
    finally:
        if ctx is not None:
            ctx.close()

    record: dict[str, Any] = {
        "ts": clock().isoformat(),
        "session_id": payload.session_id if payload else "",
        "hook": name,
        "event": (payload.event if payload else "") or spec.event,
        "decision": result.decision,
        "reason": redact(result.reason.replace("\n", " "))[:REASON_LIMIT],
        "trial_id": result.trial_id,
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    if payload and payload.command:
        record["command"] = redact_command(payload.command)
    record.update(result.extra)
    log_invocation(home, record)

    if result.decision == "block":
        return HookOutcome(BLOCK_EXIT_CODE, stderr=block_text or _block_text(name, result))
    return HookOutcome(0, stdout=notes_out + result.stdout)
