"""Answer provenance: did a human say this?

``qrh interview answer --transcript`` accepts a value only if it (or the label of the chosen option) appears,
case and whitespace normalized, in a USER message of that Claude Code transcript, in the stated session, posted
after the question was issued. Assistant text and ordinary tool results never count, nor do subagent (sidechain)
or harness-injected records.

One deliberate extension of "user message": the answer returned by the ``AskUserQuestion`` tool. Claude Code
delivers a user's pick from that native question UI as a tool result; only the structured ``toolUseResult.answers``
values are used (F6). The result's text, which also repeats the question the agent wrote, is never parsed.

Matching (F4): a candidate counts only as a whole word or phrase, case and whitespace normalized. Text answers need
at least 3 characters (or the whole message); numbers match as whole numbers; a bool or choice label or id must be
the whole message, or a whole word of at least 3 characters in a short message (a long label may appear anywhere).

Limits (see docs/interviews.md and docs/security.md): the agent runs as the same OS user. The transcript must be
the one Claude Code reported for the session (``qrh.hooks.sessions``), which stops a forged file, but this remains
process discipline against shortcuts, not protection against a hostile agent that controls what the user typed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qrh.usage.parse import HARNESS_TEXT_PREFIXES, parse_ts

ASK_TOOL = "AskUserQuestion"
MIN_TEXT = 3
SHORT_MESSAGE = 60  # characters: a bool/choice word counts only in a message this short (unless it is a long label)
LONG_LABEL = 8
UNKNOWN = "unknown"
UNKNOWN_PHRASES = ("unknown", "don't know", "dont know", "do not know", "don’t know", "not sure", "no idea")


@dataclass(frozen=True)
class UserText:
    ts: datetime
    text: str


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def user_texts(path: str | Path, session: str) -> list[UserText]:
    """Every user-authored text in ``session``, in file order. Raises OSError if the file cannot be read."""
    out: list[UserText] = []
    ask_ids: set[str] = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict) or rec.get("isSidechain") or rec.get("isMeta"):
                continue
            if rec.get("sessionId") is not None and rec.get("sessionId") != session:
                continue
            msg = rec.get("message")
            content = msg.get("content") if isinstance(msg, dict) else None
            if rec.get("type") == "assistant":
                for b in content if isinstance(content, list) else []:
                    if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == ASK_TOOL:
                        ask_ids.add(str(b.get("id")))
                continue
            if rec.get("type") != "user":
                continue
            when = parse_ts(rec.get("timestamp"))
            if when is None:
                continue
            texts: list[str] = []
            if isinstance(content, str):
                texts.append(content)
            for b in content if isinstance(content, list) else []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    texts.append(str(b.get("text", "")))
                elif b.get("type") == "tool_result" and str(b.get("tool_use_id")) in ask_ids:
                    texts += _structured_answers(rec.get("toolUseResult"))
            out += [UserText(when, t) for t in texts if t.strip() and not t.lstrip().startswith(HARNESS_TEXT_PREFIXES)]
    return out


def _structured_answers(result: Any) -> list[str]:
    answers = result.get("answers") if isinstance(result, dict) else None
    if not isinstance(answers, dict):
        return []
    return [v for v in answers.values() if isinstance(v, str)]


def _strip(text: str) -> str:
    return normalize(text).strip(" .!?,;:'\"")


def _whole(body: str, cand: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(cand) + r"(?!\w)", body) is not None


def candidate_matches(message: str, candidate: str, kind: str) -> bool:
    """Whether one user message carries one candidate answer (see the module docstring for the rules)."""
    body, cand = normalize(message), _strip(candidate)
    if not cand:
        return False
    if _strip(message) == cand:
        return True
    if kind == "number":
        return re.search(r"(?<![\w.])" + re.escape(cand) + r"(?![\w]|\.\d)", body) is not None
    if kind in ("bool", "choice"):
        if len(cand) < MIN_TEXT or not _whole(body, cand):
            return False
        return len(body) <= SHORT_MESSAGE or len(cand) >= LONG_LABEL
    return len(cand) >= MIN_TEXT and _whole(body, cand)


def verify_answer(path: str | Path, session: str, issued_at: str, *, kind: str, candidates: list[str]) -> bool:
    """True if a user message posted after ``issued_at`` carries one of the candidates (kind: text, number, bool,
    choice or unknown)."""
    issued = parse_ts(issued_at) or datetime.min.replace(tzinfo=timezone.utc)
    wanted = [c for c in candidates if c and c.strip()]
    if any(normalize(c) == UNKNOWN for c in wanted):
        wanted = [c for c in wanted if normalize(c) != UNKNOWN] + list(UNKNOWN_PHRASES)
        kind = "text" if kind in ("bool", "choice", "number") else kind
    for item in user_texts(path, session):
        if item.ts <= issued:
            continue
        if any(candidate_matches(item.text, c, kind) for c in wanted):
            return True
    return False


def verify_in_transcript(path: str | Path, session: str, issued_at: str, candidates: list[str]) -> bool:
    """Text-kind verification (kept for callers that do not know the answer type)."""
    return verify_answer(path, session, issued_at, kind="text", candidates=candidates)
