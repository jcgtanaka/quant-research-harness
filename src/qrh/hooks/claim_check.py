"""Stop: block a final answer that claims tests pass / the build is clean / lint is clean / all checks are green
when the transcript shows no matching command that ran AFTER the last edit and succeeded.

Claims are found with plain regex families, one claim per kind. Sentences that are negated, hedged, planned
or about the past are left out (precision first: a sentence that might not be a claim is not one). A claim is
accepted only if the latest command of the matching kind after the last Edit/Write/MultiEdit/NotebookEdit
returned without the error flag and without failure markers in its output. Command families (pytest, npm test,
go test, cargo test, make test, ``qrh test-receipt``, ...) are built in and extendable in ``config.yaml``
(``claim_check``). Like ``receipt-required``, an unreadable transcript exits 0; a repeated Stop
(``stop_hook_active``) is checked like the first (the runner caps the repeats).

The idea of checking "tests pass" claims against the session transcript follows claim-check by Ryan Alberts (MIT,
https://github.com/RyanAlberts/best-of-Agent-Harnesses); the patterns and code here are independent. See NOTICE.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from qrh.hooks.receipt_required import last_assistant_text
from qrh.hooks.runner import HookContext, HookResult

EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
KINDS = ("tests", "build", "lint", "checks")


@dataclass(frozen=True)
class Claim:
    kind: str  # tests | build | lint | checks
    text: str
    counts: tuple[int, ...] = ()  # test counts stated with the claim ("3,072 passed" -> 3072)


# "<n> passed", "<n> tests pass", "all <n> unit tests passed"; thousands separators (, or _) are stripped.
_COUNT = re.compile(r"(?<![\d,._])(\d{1,3}(?:[,_]\d{3})+|\d+)\s+(?:(?:unit|new|existing|python)\s+)?(?:tests?\s+)?pass(?:ed|es)?\b")


def _to_int(raw: str) -> int:
    return int(re.sub(r"[,_]", "", raw))


def passed_counts(text: str) -> tuple[int, ...]:
    """Every '<n> passed' count in text, as ints, in order, without duplicates."""
    return tuple(dict.fromkeys(_to_int(m.group(1)) for m in _COUNT.finditer(text.lower())))


_PASS = r"(?:pass(?:es|ed|ing)?|green)"
_FAMILIES: dict[str, tuple[re.Pattern[str], ...]] = {
    "tests": tuple(re.compile(p) for p in (
        rf"\ball (?:\d+ )?(?:the )?(?:unit |new |existing |python )?tests?\b[^.]{{0,20}}?\b{_PASS}\b",
        rf"\b(?:test suite|tests?|pytest|suite)\s+(?:(?:is|are|now|all)\s+)*{_PASS}\b",
        r"\b\d+ passed\b",
    )),
    "checks": tuple(re.compile(p) for p in (
        rf"\ball (?:the )?checks?\b[^.]{{0,15}}?\b{_PASS}\b", rf"\beverything (?:passes|is green)\b",
        r"\bci (?:is |are )?green\b", r"\ball green\b",
    )),
    "build": tuple(re.compile(p) for p in (
        r"\bbuild\s+(?:(?:is|now)\s+)*(?:succeeds|succeeded|passes|clean|green|successful)\b",
        r"\bcompiles? (?:cleanly|without errors)\b",
    )),
    "lint": tuple(re.compile(p) for p in (
        r"\blint(?:ing)?\s+(?:(?:is|are|now)\s+)*(?:clean|passes|green|passing)\b",
        r"\bno lint (?:errors|warnings|issues)\b",
        r"\b(?:ruff|flake8|eslint|pylint|mypy)\s+(?:passes|is clean)\b",
    )),
}
# a statement that is not a plain report of a result: negation, plan, condition, question, past, wish
_NOT_A_CLAIM = re.compile(
    r"\bnot\b|n't\b|\bno tests?\b|\bfail|\bshould\b|\bwill\b|\bwould\b|\bcould\b|\bcan\b|\bif\b|\buntil\b|\bonce\b"
    r"|\bwhen\b|make sure|\bensure\b|\bhope|\bgoal\b|\bplease\b|\bneed|\bbefore\b|previously|\bwas\b|\bwere\b|\?"
    r"|\bunknown\b|\bwant|\bto pass\b|\bcheck that\b|\bverify that\b")
_ZERO_FAILURES = re.compile(r"\b0 fail\w*|\bno (?:lint |test |type )?(?:errors|failures|warnings)\b")


def extract_claims(text: str) -> list[Claim]:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    found: dict[str, Claim] = {}
    counts: list[int] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        low = " ".join(sentence.lower().split())
        probe = _ZERO_FAILURES.sub(" ", low)
        if not low or _NOT_A_CLAIM.search(probe):
            continue
        counts.extend(n for n in passed_counts(low) if n not in counts)
        for kind, patterns in _FAMILIES.items():
            if kind in found:
                continue
            for rx in patterns:
                m = rx.search(low)
                if m:
                    found[kind] = Claim(kind, m.group(0))
                    break
    if counts and "tests" in found:
        found["tests"] = Claim("tests", found["tests"].text, tuple(counts))
    return [found[k] for k in KINDS if k in found]


# --- commands -----------------------------------------------------------------------------------

_WRAPPERS = re.compile(
    r"^(?:(?:env(?:\s+-u\s+\S+|\s+-i)*|uv\s+run|poetry\s+run|pipenv\s+run|npx|time|sudo|nice|python3?\s+-m|\w+=\S*)\s+)*")
_SEGMENTS = re.compile(r"&&|;|\n|\|\||\|")
_TEST = re.compile(
    r"^(?:pytest|py\.test|tox|nox|jest|vitest|mocha|rspec|phpunit|ctest|unittest|(?:npm|pnpm|yarn|bun)(?:\s+run)?\s+test(?::\S+)?"
    r"|go\s+test|cargo\s+test|make\s+test|qrh\s+test-receipt|dotnet\s+test|mvn\s+test|gradle\s+test)\b")
_BUILD = re.compile(
    r"^(?:(?:npm|pnpm|yarn|bun)(?:\s+run)?\s+build|cargo\s+(?:build|check)|go\s+build|tsc|make(?:\s+build|\s+all)?\s*$"
    r"|mvn\s+(?:compile|package)|uv\s+build|gradle\s+build)\b")
_LINT = re.compile(
    r"^(?:ruff|flake8|pylint|eslint|mypy|pyright|biome|cargo\s+clippy|golangci-lint|pre-commit\s+run"
    r"|(?:npm|pnpm|yarn)(?:\s+run)?\s+lint|black\s+--check|isort\s+--check)\b")
_NOT_A_RUN = re.compile(r"--collect-only|--help|\s-h\b|--version|--co\b")
_IGNORES_FAILURE = re.compile(r"\|\|\s*(?:true|:|exit\s+0)\b|;\s*true\b")


def _classify(command: str, builtin: re.Pattern[str], extra: tuple[re.Pattern[str], ...]) -> bool:
    if _IGNORES_FAILURE.search(command) or _NOT_A_RUN.search(command):
        return False
    if any(x.search(command) for x in extra):
        return True
    for seg in _SEGMENTS.split(command):
        seg = seg.strip()
        if seg and builtin.match(_WRAPPERS.sub("", seg)):
            return True
    return False


def is_test(command: str, extra: tuple[re.Pattern[str], ...] = ()) -> bool:
    return _classify(command, _TEST, extra)


def is_build(command: str, extra: tuple[re.Pattern[str], ...] = ()) -> bool:
    return _classify(command, _BUILD, extra)


def is_lint(command: str, extra: tuple[re.Pattern[str], ...] = ()) -> bool:
    return _classify(command, _LINT, extra)


_FAILURE = re.compile(
    r"\b[1-9]\d*\s+(?:failed|failures?|errors?)\b|(?m:^\s*(?:FAILED|ERROR)\b)|Traceback \(most recent|command not found"
    r"|no tests ran|Exit code [1-9]")


# --- transcript ---------------------------------------------------------------------------------

@dataclass
class Run:
    command: str
    success: bool | None  # None: the result never reached the transcript
    detail: str = ""
    output: str = ""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and isinstance(b.get("text"), str))
    return ""


def read_events(path: str) -> list[Any]:
    """Ordered events: the string 'edit' for each edit tool call, or a Run for each Bash call. Raises OSError."""
    results: dict[str, tuple[bool, str]] = {}
    uses: list[tuple[str, str, Any]] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg = rec.get("message") if isinstance(rec, dict) else None
            content = msg.get("content") if isinstance(msg, dict) else None
            for b in (content if isinstance(content, list) else ()):
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use" and isinstance(b.get("id"), str):
                    inp = b.get("input") if isinstance(b.get("input"), dict) else {}
                    uses.append((b["id"], str(b.get("name")), inp))
                elif b.get("type") == "tool_result" and isinstance(b.get("tool_use_id"), str):
                    results[b["tool_use_id"]] = (b.get("is_error") is True, _content_text(b.get("content")))
    events: list[Any] = []
    for tid, name, inp in uses:
        if name in EDIT_TOOLS:
            events.append("edit")
        elif name == "Bash" and isinstance(inp.get("command"), str):
            if tid not in results:
                events.append(Run(inp["command"], None, "its result is missing"))
                continue
            is_err, text = results[tid]
            bad = _FAILURE.search(text)
            if is_err or bad:
                events.append(Run(inp["command"], False, "the command reported a failure"))
            else:
                events.append(Run(inp["command"], True, output=text))
    return events


def _latest(events: list[Any], pred: Any) -> tuple[Run | None, bool]:
    """(latest matching run after the last edit, whether any matching run exists before the last edit)."""
    last_edit = max((i for i, e in enumerate(events) if e == "edit"), default=-1)
    after = [e for e in events[last_edit + 1:] if isinstance(e, Run) and pred(e.command)]
    before = any(isinstance(e, Run) and pred(e.command) for e in events[: last_edit + 1])
    return (after[-1] if after else None), before


def run(ctx: HookContext) -> HookResult:
    p = ctx.payload
    if p.event and p.event != "Stop":
        return HookResult("allow", "not a Stop event")
    cfg = ctx.config.claim_check
    if not cfg.enabled:
        return HookResult("allow", "disabled in config")
    try:
        if not p.transcript_path:
            raise OSError("no transcript_path in the payload")
        text = last_assistant_text(p.transcript_path)
        claims = extract_claims(text) if text else []
        events = read_events(p.transcript_path) if claims else []
    except OSError as exc:
        return HookResult("error", f"transcript unreadable ({type(exc).__name__}); not blocking Stop")
    if not claims:
        return HookResult("allow", "no pass/clean claims")
    preds = {
        "tests": lambda c: is_test(c, cfg.test_commands), "build": lambda c: is_build(c, cfg.build_commands),
        "lint": lambda c: is_lint(c, cfg.lint_commands),
    }
    noun = {"tests": "test", "build": "build", "lint": "lint", "checks": "check"}
    unsupported: list[str] = []
    kinds: list[str] = []

    def verdict(kind: str) -> str | None:
        run_, before = _latest(events, preds[kind])
        if run_ is None:
            return ("last matching run was before the last edit" if before
                    else f"no matching {noun[kind]} run in the transcript")
        if run_.success is None:
            return f"latest matching run: {run_.detail}"
        if not run_.success:
            return "latest matching run failed"
        return None

    def count_verdict(counts: tuple[int, ...]) -> str | None:
        # every stated count must equal the "<n> passed" figure in the LATEST test run after the last edit
        latest, _ = _latest(events, preds["tests"])
        seen = set(passed_counts(latest.output)) if latest is not None and latest.success else set()
        wrong = [n for n in counts if n not in seen]
        if wrong:
            shown = ", ".join(str(n) for n in sorted(seen)) or "none"
            return f"stated count {', '.join(map(str, wrong))} passed does not match the latest run after the last edit (it shows: {shown})"
        return None

    for c in claims:
        if c.kind == "checks":
            why = verdict("tests")
            if why is None:
                # A "checks" claim needs a successful latest build run AND lint run, not merely no failing one.
                missing = [f"{other}: {w}" for other in ("build", "lint") if (w := verdict(other))]
                if missing:
                    why = "; ".join(missing)
        else:
            why = verdict(c.kind)
            if why is None and c.kind == "tests" and c.counts:
                why = count_verdict(c.counts)
        if why:
            unsupported.append(f'  - {c.kind}: "{c.text}" ({why})')
            kinds.append(c.kind)
    if not unsupported:
        return HookResult("allow", f"{len(claims)} claim(s) backed by later successful runs")
    return HookResult(
        "block", f"{len(unsupported)} unsupported claim(s): {', '.join(kinds)}",
        action_parts=("claim-check", "unsupported claims:", *sorted(kinds)),
        stderr="Your last message claims results the transcript does not support:\n" + "\n".join(unsupported)
        + "\nFix: run the command after your last edit and read its result, or drop the claim; then answer again.",
    )
