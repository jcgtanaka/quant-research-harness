"""PostToolUse (Edit|Write|MultiEdit): run the look-ahead lint on strategy files; BLOCK findings exit 2.

PostToolUse cannot undo the edit; it feeds stderr back to the agent. WARN findings go to the log only
(output of a successful PostToolUse is not shown).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from qrh.checks.lookahead import WARN, blocking, lint_source
from qrh.hooks.common import glob_match
from qrh.hooks.runner import HookContext, HookResult

EDIT_TOOLS = {"Edit", "Write", "MultiEdit"}
_IMPORT_TEXT = re.compile(r"^\s*(?:from\s+qrh\.adapters\b|import\s+qrh\.adapters\b)", re.MULTILINE)
_RUN_TEXT = re.compile(r"^\s*def\s+run\s*\(", re.MULTILINE)
_CLASS_TEXT = re.compile(r"^\s*class\s+\w+", re.MULTILINE)


def defines_strategy(source: str) -> bool:
    """A class with a ``run`` method in a module that imports from ``qrh.adapters``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:  # cannot tell structurally; fall back to text so a broken strategy is still caught
        return bool(_IMPORT_TEXT.search(source) and _CLASS_TEXT.search(source) and _RUN_TEXT.search(source))
    imports = any(
        (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[:2] == ["qrh", "adapters"])
        or (isinstance(n, ast.Import) and any(a.name.split(".")[:2] == ["qrh", "adapters"] for a in n.names))
        for n in ast.walk(tree)
    )
    has_run = any(
        isinstance(c, ast.ClassDef)
        and any(isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == "run" for f in c.body)
        for c in ast.walk(tree)
    )
    return imports and has_run


def run(ctx: HookContext) -> HookResult:
    p = ctx.payload
    if (p.event and p.event != "PostToolUse") or p.tool_name not in EDIT_TOOLS:
        return HookResult("allow", "not a PostToolUse edit")
    if not p.file_path.endswith(".py"):
        return HookResult("allow", "not a python file")
    cwd = p.cwd
    path = Path(p.file_path if Path(p.file_path).is_absolute() else Path(cwd or ".") / p.file_path)
    by_glob = any(glob_match(str(path), g, cwd) for g in ctx.config.strategy_globs)
    if not by_glob and not ctx.config.detect_strategies:
        return HookResult("allow", "not a strategy file")
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return HookResult("block", f"cannot read {path} to lint it: {type(exc).__name__}")
    if not by_glob and not defines_strategy(source):
        return HookResult("allow", "not a strategy file")
    findings = [f for f in lint_source(source, str(path)) if not f.suppressed]
    blockers = blocking(findings)
    if blockers:
        lines = [f"{f.file}:{f.line} {f.rule_id} {f.message}" for f in blockers]
        return HookResult(
            "block", f"{len(blockers)} look-ahead finding(s) in {path.name}",
            stderr="\n".join(lines) + "\nFix the look-ahead, or suppress with a reason: # qrh: allow <rule> <reason>",
        )
    warns = [f for f in findings if f.severity == WARN]
    if warns:
        return HookResult(
            "warn", f"{len(warns)} WARN finding(s): " + ", ".join(sorted({f.rule_id for f in warns})),
            extra={"warnings": [f"{f.rule_id}:{f.line}" for f in warns]},
        )
    return HookResult("allow", "no look-ahead findings")


