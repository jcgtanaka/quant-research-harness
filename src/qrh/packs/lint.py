"""Pack-declared lint rules (``lint/*.yaml``), run next to the built-in look-ahead lint at S2.

A rule is a regex over source lines, or an AST descriptor: ``{call: name, keyword: {k: v}, negative_first_arg: true}``
(a call to ``name`` or ``.name``) or ``{import: module}``. Findings use the same ``Finding`` record and the same
reasoned suppression comment (``# qrh: allow <rule-id> <reason>``) as the built-in rules.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

from qrh.checks.lookahead import Finding, _allowances
from qrh.packs.model import LintRule


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _negative(node: ast.AST | None) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float))
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool) and node.value < 0


def _ast_hits(tree: ast.AST, desc: dict) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if "call" in desc and isinstance(node, ast.Call) and _call_name(node) == desc["call"]:
            kw = desc.get("keyword") or {}
            ok = all(any(k.arg == name and isinstance(k.value, ast.Constant) and k.value.value == want for k in node.keywords)
                     for name, want in kw.items())
            if desc.get("negative_first_arg") and not (node.args and _negative(node.args[0])):
                ok = False
            if ok:
                lines.append(node.lineno)
        if "import" in desc:
            if isinstance(node, ast.Import) and any(a.name.split(".")[0] == desc["import"] for a in node.names):
                lines.append(node.lineno)
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == desc["import"]:
                lines.append(node.lineno)
    return lines


def lint_with_rules(files: Iterable[str | Path], rules: list[LintRule]) -> list[Finding]:
    out: list[Finding] = []
    if not rules:
        return out
    for path in files:
        p = Path(path)
        try:
            source = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        lines = source.splitlines()
        allow = _allowances(source)
        try:
            tree: ast.AST | None = ast.parse(source)
        except SyntaxError:
            tree = None  # the built-in lint already reports unparseable source
        for rule in rules:
            if rule.regex is not None:
                rx = re.compile(rule.regex)
                hits = [n for n, line in enumerate(lines, 1) if rx.search(line)]
            elif tree is not None and rule.ast is not None:
                hits = sorted(set(_ast_hits(tree, rule.ast)))
            else:
                hits = []
            for n in hits:
                reason = next((why for ln in (n, n - 1) for r, why in allow.get(ln, []) if r == rule.id and why
                               and (ln == n or lines[ln - 1].lstrip().startswith("#"))), None)
                out.append(Finding(rule.id, str(p), n, rule.severity, lines[n - 1].strip(), rule.message,
                                   reason is not None, reason))
    return sorted(out, key=lambda f: (f.file, f.line, f.rule_id))
