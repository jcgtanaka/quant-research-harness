"""Runaway guard: PreToolUse (all tools) plus PostToolUseFailure bookkeeping.

Three trip wires per session, configured under ``runaway`` in ``<QRH_HOME>/config.yaml`` (no numeric
fallbacks; without the section the guard is disabled):

* ``repeat_limit``   the same tool call (tool + normalized input) this many times in a row;
* ``failure_streak`` this many consecutive tool calls that failed (PostToolUseFailure events);
* ``spend_cap_usd``  session spend, read from the transcript's latest cost-state (exact). No cost-state
  yet means no spend trip.

A project file ``<cwd>/.qrh/config.yaml`` may lower a limit but never raise it or switch on a wire the
user level left off. A trip blocks (exit 2) through the approval-token flow. Documented fail-open
exception, like Stop: an unreadable transcript allows and is logged.

Design after the runaway-guard skill by Ryan Alberts (MIT); the code is written independently. See NOTICE.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any

import yaml

from qrh.approvals import action_fingerprint
from qrh.hooks.common import ConfigError, RunawayConfig, parse_runaway
from qrh.hooks.runner import HookContext, HookResult

# Calls whose repeats are deliberate (waiting, polling, delegating, asking): they never count and end a run.
NEVER_LOOP = frozenset({"task", "agent", "taskoutput", "bashoutput", "sleep", "monitor", "askuserquestion",
                        "exitplanmode", "schedulewakeup", "taskstop", "killshell"})
KEEP_CALLS = 50
CHUNK = 1 << 20


def state_dir(home: Path) -> Path:
    return home / "runaway"


def _state_path(home: Path, session_id: str) -> Path:
    return state_dir(home) / (hashlib.sha256(session_id.encode()).hexdigest()[:32] + ".json")


def load_state(home: Path, session_id: str) -> tuple[dict[str, Any], bool]:
    """(state, was_reset). A corrupt file is replaced by an empty state."""
    path = _state_path(home, session_id)
    if not path.exists():
        return {"last_fp": None, "repeat": 0, "calls": []}, False
    try:
        st = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(st, dict) and isinstance(st.get("calls"), list) and isinstance(st.get("repeat"), int):
            return st, False
    except (OSError, ValueError):
        pass
    return {"last_fp": None, "repeat": 0, "calls": []}, True


def save_state(home: Path, session_id: str, st: dict[str, Any]) -> None:
    d = state_dir(home)
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    path = _state_path(home, session_id)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(3)}.tmp")
    st["calls"] = st["calls"][-KEEP_CALLS:]
    tmp.write_text(json.dumps(st), encoding="utf-8")
    os.replace(tmp, path)


def project_runaway(cwd: str) -> tuple[RunawayConfig | None, str | None]:
    """The project-level section, or (None, error text) when the file is unusable. Never raises."""
    if not cwd:
        return None, None
    path = Path(cwd) / ".qrh" / "config.yaml"
    if not path.is_file():
        return None, None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "runaway" not in raw:
            return None, None
        return parse_runaway(raw["runaway"]), None
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ConfigError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def effective_limits(user: RunawayConfig, project: RunawayConfig | None) -> RunawayConfig:
    """Project values only lower a limit the user level already set."""
    if project is None:
        return user

    def lower(a: Any, b: Any) -> Any:
        return a if a is None or b is None else min(a, b)

    return RunawayConfig(lower(user.repeat_limit, project.repeat_limit),
                         lower(user.failure_streak, project.failure_streak),
                         lower(user.spend_cap_usd, project.spend_cap_usd))


def latest_cost(path: str) -> float | None:
    """totalCostUSD of the last cost-state record in the transcript (read backwards). Raises OSError."""
    if not path:
        raise OSError("no transcript_path in the payload")
    with open(path, "rb") as fh:
        size = fh.seek(0, os.SEEK_END)
        end, carry = size, b""
        while end > 0:
            start = max(0, end - CHUNK)
            fh.seek(start)
            buf = fh.read(end - start) + carry
            lines = buf.split(b"\n")
            carry = lines[0] if start > 0 else b""
            for line in reversed(lines[1:] if start > 0 else lines):
                if b'"cost-state"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                v = rec.get("totalCostUSD") if isinstance(rec, dict) and rec.get("type") == "cost-state" else None
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return float(v)
            end = start
    return None


def _fingerprint(tool: str, tool_input: dict[str, Any]) -> str:
    """Loop detection treats whitespace variants of a command as the same call (unlike approval fingerprints)."""
    norm = dict(tool_input)
    if tool == "Bash" and isinstance(norm.get("command"), str):
        norm["command"] = " ".join(norm["command"].split())
    return action_fingerprint(tool, norm, "")


def run(ctx: HookContext) -> HookResult:
    p = ctx.payload
    if p.event and p.event != "PreToolUse":
        return HookResult("allow", "not a PreToolUse event")
    user = ctx.config.runaway
    if user is None:
        return HookResult("allow", "runaway guard disabled (no runaway section in config.yaml)")
    project, perr = project_runaway(p.cwd)
    limits = effective_limits(user, project)
    extra: dict[str, Any] = {"project_config_error": perr} if perr else {}
    sid = p.session_id
    st, reset = load_state(ctx.home, sid)
    if reset:
        extra["state_reset"] = True
    problems: list[str] = []

    if p.tool_name.lower() in NEVER_LOOP:
        st["last_fp"], st["repeat"] = None, 0
    else:
        fp = _fingerprint(p.tool_name, dict(p.tool_input))
        st["repeat"] = st["repeat"] + 1 if st.get("last_fp") == fp else 1
        st["last_fp"] = fp
        if limits.repeat_limit is not None and st["repeat"] >= limits.repeat_limit:
            problems.append(f"the same {p.tool_name} call ran {st['repeat']} times in a row (limit {limits.repeat_limit})")
    if limits.failure_streak is not None:
        streak = 0
        for c in reversed(st["calls"]):
            if not c[1]:
                break
            streak += 1
        if streak >= limits.failure_streak:
            problems.append(f"{streak} tool calls in a row failed (limit {limits.failure_streak})")
    st["calls"].append([p.raw.get("tool_use_id") if isinstance(p.raw.get("tool_use_id"), str) else "", False])

    unreadable = None
    if limits.spend_cap_usd is not None:
        try:
            spent = latest_cost(p.transcript_path)
        except OSError as exc:
            spent, unreadable = None, f"transcript unreadable ({type(exc).__name__}); spend not checked"
        if spent is not None and spent >= limits.spend_cap_usd:
            problems.append(f"session spend ${spent:.2f} reached the cap ${limits.spend_cap_usd:.2f} "
                            "(from the Claude Code cost-state)")
    save_state(ctx.home, sid, st)
    if problems:
        def clear() -> None:
            fresh, _ = load_state(ctx.home, sid)
            fresh.update(last_fp=None, repeat=0, calls=[])
            save_state(ctx.home, sid, fresh)

        return HookResult("block", "; ".join(problems), stderr="Runaway guard: " + "; ".join(problems) + ".",
                          extra=extra, on_override=clear)
    if unreadable:
        return HookResult("error", unreadable, extra=extra)
    return HookResult("allow", "within runaway limits", extra=extra)
