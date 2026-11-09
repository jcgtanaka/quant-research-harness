"""Stream one Claude Code transcript into usage facts. Message text is never kept.

Claude Code writes one assistant record per content block, each repeating ``message.usage``; a message id
is therefore counted once (a record without an id counts once). Copies are identical in main transcripts,
but in subagent transcripts ``output_tokens`` grows across the copies (measured), so the largest value wins.
Unknown record types and unparsable lines are tolerated (the latter are counted).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SYNTHETIC_MODEL = "<synthetic>"
COMPACT_SUBTYPE = "compact_boundary"
KINDS = ("input", "output", "cache_read", "cache_write_5m", "cache_write_1h")


@dataclass
class TokenCounts:
    """Token totals by kind for some set of requests."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    requests: int = 0

    @property
    def cache_write(self) -> int:
        return self.cache_write_5m + self.cache_write_1h

    @property
    def context(self) -> int:
        """Prompt size of a request: everything the model read, cached or not."""
        return self.input + self.cache_read + self.cache_write

    def add(self, other: "TokenCounts") -> None:
        for k in (*KINDS, "requests"):
            setattr(self, k, getattr(self, k) + getattr(other, k))

    def as_dict(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in (*KINDS, "requests")}


@dataclass(frozen=True)
class Request:
    msg_id: str | None
    ts: str | None
    model: str
    tokens: TokenCounts
    sidechain: bool
    after_compaction: bool = False
    turn: int = 0  # index into Transcript.turns (main chain; a sidechain record belongs to the turn it ran in)
    complete: bool = True  # False: a stream that never reached a stop reason (the harness does not bill it)


@dataclass
class TurnStats:
    """One typed prompt and everything the agent did until the next one (counts only)."""

    ts: str | None
    requests: int = 0
    tools: int = 0
    failures: int = 0


HARNESS_TEXT_PREFIXES = ("<system-reminder", "<command-", "<local-command", "Caveat:", "<user-prompt-submit-hook")
EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
NOT_FAILURES = ("The user doesn't want", "Permission to use", "[Request interrupted", "BLOCKED by qrh hook",
                "The user rejected")


def alias(text: str) -> str:
    """Eight hex characters of a SHA-256: enough to tell two things apart, useless for recovering them."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8", "surrogatepass")).hexdigest()[:8]


@dataclass
class Transcript:
    requests: list[Request] = field(default_factory=list)  # deduplicated, in file order
    raw_main: dict[str, TokenCounts] = field(default_factory=dict)  # main chain, before deduplication
    tools: Counter[str] = field(default_factory=Counter)  # main chain, names only
    compactions: int = 0
    cost_state: dict[str, Any] | None = None  # the last cost-state record, normalized
    cost_state_main_requests: int | None = None  # main-chain requests seen before that last cost-state record
    malformed: int = 0
    cwds: Counter[str] = field(default_factory=Counter)
    first_ts: str | None = None
    last_ts: str | None = None
    turns: list[TurnStats] = field(default_factory=list)
    reread_runs: dict[str, int] = field(default_factory=dict)  # hashed path -> longest run of reads with no change between
    failing_commands: Counter[str] = field(default_factory=Counter)  # hashed Bash command -> failed runs
    output_tokens: list[int] = field(default_factory=list)  # estimated tokens (chars / 4) of each tool result


def parse_ts(value: str | None) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)  # naive stamps are read as UTC


def _int(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else 0


def _tokens(usage: dict[str, Any]) -> TokenCounts:
    total_w = _int(usage.get("cache_creation_input_tokens"))
    split = usage.get("cache_creation")
    if isinstance(split, dict):
        w1h = min(_int(split.get("ephemeral_1h_input_tokens")), total_w)
    else:
        w1h = 0
    return TokenCounts(
        input=_int(usage.get("input_tokens")), output=_int(usage.get("output_tokens")),
        cache_read=_int(usage.get("cache_read_input_tokens")), cache_write_5m=total_w - w1h,
        cache_write_1h=w1h, requests=1,
    )


def _fieldwise_max(a: TokenCounts, b: TokenCounts) -> TokenCounts:
    return TokenCounts(**{k: max(getattr(a, k), getattr(b, k)) for k in (*KINDS, "requests")})


def _cost_state(rec: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, dict[str, float]] = {}
    raw = rec.get("modelUsage")
    for name, mu in (raw.items() if isinstance(raw, dict) else ()):
        if not isinstance(mu, dict):
            continue
        cost = mu.get("costUSD")
        models[str(name)] = {
            "input": _int(mu.get("inputTokens")), "output": _int(mu.get("outputTokens")),
            "cache_read": _int(mu.get("cacheReadInputTokens")),
            "cache_creation": _int(mu.get("cacheCreationInputTokens")),
            "cost_usd": float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else 0.0,
        }
    total = rec.get("totalCostUSD")
    return {
        "total_cost_usd": float(total) if isinstance(total, (int, float)) and not isinstance(total, bool) else 0.0,
        "models": models,
    }


def parse_transcript(path: str | Path) -> Transcript:
    out = Transcript()
    seen_ids: dict[str, int] = {}
    seen_tools: set[str] = set()
    first: tuple[datetime, str] | None = None
    last: tuple[datetime, str] | None = None
    pending_compaction = False
    n_main = 0
    tools_by_id: dict[str, tuple[int, str, str | None]] = {}  # tool_use id -> (turn, name, hashed path or command)
    cur_runs: dict[str, int] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                out.malformed += 1
                continue
            if not isinstance(rec, dict):
                out.malformed += 1
                continue
            rtype = rec.get("type")
            if rtype == "cost-state":
                out.cost_state = _cost_state(rec)
                out.cost_state_main_requests = n_main
                continue
            ts = rec.get("timestamp")
            when = parse_ts(ts)
            if when is not None:
                if first is None or when < first[0]:
                    first = (when, ts)
                if last is None or when > last[0]:
                    last = (when, ts)
            cwd = rec.get("cwd")
            if isinstance(cwd, str) and cwd:
                out.cwds[cwd] += 1
            if rtype == "system" and rec.get("subtype") == COMPACT_SUBTYPE:
                out.compactions += 1
                pending_compaction = True
                cur_runs.clear()
                continue
            if rtype == "user" and rec.get("isSidechain") is not True:
                if _is_typed_prompt(rec):
                    out.turns.append(TurnStats(ts if isinstance(ts, str) else None))
                    cur_runs.clear()  # the user may have changed files themselves
                else:
                    _tool_results(rec, out, tools_by_id, cur_runs)
                continue
            if rtype != "assistant":
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            sidechain = rec.get("isSidechain") is True
            if not out.turns:  # records before any typed prompt
                out.turns.append(TurnStats(ts if isinstance(ts, str) else None))
            if not sidechain:
                _count_tools(msg, out.tools, seen_tools)
                _register_tool_uses(msg, out, tools_by_id)
            usage = msg.get("usage")
            model = msg.get("model")
            if not isinstance(usage, dict) or not isinstance(model, str) or model == SYNTHETIC_MODEL:
                continue
            tokens = _tokens(usage)
            done = "stop_reason" not in msg or msg.get("stop_reason") is not None
            if not sidechain:
                out.raw_main.setdefault(model, TokenCounts()).add(tokens)
            mid = msg.get("id")
            if isinstance(mid, str) and mid:
                if mid in seen_ids:
                    # Streamed copies of one message can differ in output_tokens (it grows); keep the final value.
                    i = seen_ids[mid]
                    old = out.requests[i]
                    out.requests[i] = replace(old, tokens=_fieldwise_max(old.tokens, tokens),
                                              complete=old.complete or done)
                    continue
                seen_ids[mid] = len(out.requests)
            else:
                mid = None
            after = pending_compaction and not sidechain
            if after:
                pending_compaction = False
            out.requests.append(Request(mid, ts if isinstance(ts, str) else None, model, tokens, sidechain, after,
                                        turn=len(out.turns) - 1, complete=done))
            if not sidechain:
                out.turns[-1].requests += 1
            if not sidechain:
                n_main += 1
    out.first_ts = first[1] if first else None
    out.last_ts = last[1] if last else None
    return out


def _is_typed_prompt(rec: dict[str, Any]) -> bool:
    if rec.get("isMeta") is True or rec.get("isCompactSummary") is True:
        return False
    msg = rec.get("message")
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return False
        text = next((b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"), "")
    else:
        return False
    text = text.lstrip()
    return bool(text) and not text.startswith(HARNESS_TEXT_PREFIXES)


def _register_tool_uses(msg: dict[str, Any], out: Transcript, tools: dict[str, tuple[int, str, str | None]]) -> None:
    content = msg.get("content")
    for b in (content if isinstance(content, list) else ()):
        if not isinstance(b, dict) or b.get("type") != "tool_use" or not isinstance(b.get("id"), str):
            continue
        if b["id"] in tools:
            continue
        inp = b.get("input") if isinstance(b.get("input"), dict) else {}
        name = str(b.get("name"))
        key = None
        if name == "Bash" and isinstance(inp.get("command"), str):
            key = alias(inp["command"])
        elif name in EDIT_TOOLS or name == "Read":
            path = inp.get("file_path") or inp.get("notebook_path")
            key = alias(path) if isinstance(path, str) and path else None
        tools[b["id"]] = (len(out.turns) - 1, name, key)
        out.turns[-1].tools += 1


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content if isinstance(b, dict) and isinstance(b.get("text"), str))
    return ""


def _tool_results(rec: dict[str, Any], out: Transcript, tools: dict[str, tuple[int, str, str | None]],
                  runs: dict[str, int]) -> None:
    msg = rec.get("message")
    content = msg.get("content") if isinstance(msg, dict) else None
    for b in (content if isinstance(content, list) else ()):
        if not isinstance(b, dict) or b.get("type") != "tool_result":
            continue
        known = tools.get(b.get("tool_use_id")) if isinstance(b.get("tool_use_id"), str) else None
        text = _result_text(b.get("content"))
        if known is None:
            continue
        turn, name, key = known
        if len(text) // 4 > 0:
            out.output_tokens.append(len(text) // 4)
        failed = b.get("is_error") is True and not text.lstrip().startswith(NOT_FAILURES)
        if failed:
            out.turns[turn].failures += 1
            if name == "Bash" and key:
                out.failing_commands[key] += 1
            continue
        if b.get("is_error") is True or not key:
            continue
        if name == "Read":
            runs[key] = runs.get(key, 0) + 1
            out.reread_runs[key] = max(out.reread_runs.get(key, 0), runs[key])
        elif name in EDIT_TOOLS:
            runs.pop(key, None)


def _count_tools(msg: dict[str, Any], tools: Counter[str], seen: set[str]) -> None:
    content = msg.get("content")
    for block in (content if isinstance(content, list) else ()):
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name, tid = block.get("name"), block.get("id")
        if not isinstance(name, str):
            continue
        if isinstance(tid, str):
            if tid in seen:
                continue
            seen.add(tid)
        tools[name] += 1
