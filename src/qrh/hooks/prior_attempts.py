"""UserPromptSubmit: when the prompt looks like research work, show the closest registered hypotheses.

Deterministic keyword search (SQLite FTS5, BM25) through the ``Searcher`` interface. Prints at most
three lines. The prompt text is used for the query only and is never logged.
"""

from __future__ import annotations

import re

from qrh.hooks.runner import HookContext, HookResult
from qrh.memory.search import FtsSearcher

TOP = 3
_STOPWORDS = frozenset(
    "a an the of and or to for in on at is it its this that these those with without from by as be are was were "
    "i we you me my our your please can could would should do does did how what why when which who whom "
    "let lets new run make try test add use using want need about into over under then than so if not no".split()
)
_WORD = re.compile(r"[A-Za-z0-9]+")


def triggered(prompt: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(r"(?<![A-Za-z0-9])" + re.escape(t), prompt, re.IGNORECASE) for t in terms)


def query_from_prompt(prompt: str, limit: int = 20) -> str:
    seen: list[str] = []
    for w in _WORD.findall(prompt.lower()):
        if len(w) >= 3 and w not in _STOPWORDS and w not in seen:
            seen.append(w)
    return " ".join(seen[:limit])


def run(ctx: HookContext) -> HookResult:
    p = ctx.payload
    if p.event and p.event != "UserPromptSubmit":
        return HookResult("info", "not a UserPromptSubmit event")
    if not triggered(p.prompt, ctx.config.trigger_terms):
        return HookResult("info", "no trigger term")
    hits = FtsSearcher(ctx.registry().conn).search(query_from_prompt(p.prompt), limit=TOP)
    if not hits:
        return HookResult("info", "no prior attempts matched")
    lines = ["Prior attempts in the registry (keyword match; check before repeating work):"]
    for h in hits:
        text = " ".join(h.text.split())
        lines.append(f"- {h.trial_id[:8]}  {h.verdict or 'NONE'}  {h.family}  {text[:100]}")
    return HookResult("info", f"{len(hits)} prior attempt(s)", stdout="\n".join(lines) + "\n", trial_id=hits[0].trial_id)
