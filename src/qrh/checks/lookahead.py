"""Static look-ahead lint over strategy source (an early warning, not proof).

The runtime tests (``truncation.py`` and ``perturbation.py``) are the actual proof; this lint
catches common patterns cheaply and explains them. A shift amount that is not a literal number is only a warning. Suppress a finding with a
comment on the same or the previous line: ``# qrh: allow <rule-id> <reason>``.
A reason is required.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BLOCK = "BLOCK"
WARN = "WARN"

_ALLOW = re.compile(r"#\s*qrh:\s*allow\s+(?P<rule>[A-Za-z0-9_-]+)(?:\s+(?P<reason>\S.*?))?\s*$")
_STATS = {"mean", "std", "max", "min"}
_WINDOWED = {"rolling", "expanding", "ewm", "groupby"}


@dataclass(frozen=True)
class Finding:
    rule_id: str
    file: str
    line: int
    severity: str
    snippet: str
    message: str
    suppressed: bool = False
    reason: str | None = None


_NUMPY_MODULES = {"np", "numpy"}


def _const_number(node: ast.AST | None) -> float | None:
    """The value of a numeric literal (a leading minus allowed), else None."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _const_number(node.operand)
        return None if inner is None else (-inner if isinstance(node.op, ast.USub) else inner)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    return None


def _named_numbers(tree: ast.AST) -> dict[str, float]:
    """Names bound exactly once, to a numeric literal, and never rebound or used as a parameter or loop target."""
    seen: dict[str, list[float | None]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = _const_number(node.value)
            for t in node.targets:
                for name in ast.walk(t):
                    if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                        seen.setdefault(name.id, []).append(value if isinstance(t, ast.Name) else None)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and isinstance(node.target, ast.Name):
            seen.setdefault(node.target.id, []).append(None)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            for name in ast.walk(node.target):
                if isinstance(name, ast.Name):
                    seen.setdefault(name.id, []).append(None)
        elif isinstance(node, ast.arg):
            seen.setdefault(node.arg, []).append(None)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                for name in ast.walk(item.optional_vars) if item.optional_vars is not None else ():
                    if isinstance(name, ast.Name):
                        seen.setdefault(name.id, []).append(None)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            seen.setdefault(node.target.id, []).append(None)
    return {n: vals[0] for n, vals in seen.items() if len(vals) == 1 and vals[0] is not None}


def _numpy_aliases(tree: ast.AST) -> set[str]:
    out = set(_NUMPY_MODULES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.asname or a.name for a in node.names if a.name == "numpy"}
    return out


def _str_const(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _kw(call: ast.Call, name: str) -> ast.AST | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def _is_full_stat(node: ast.AST) -> bool:
    """A ``.mean()``-style call whose receiver chain has no rolling/expanding/ewm/groupby."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _STATS):
        return False
    recv: ast.AST = node.func.value
    while True:
        if isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute):
            if recv.func.attr in _WINDOWED:
                return False
            recv = recv.func.value
        elif isinstance(recv, ast.Attribute):
            recv = recv.value
        elif isinstance(recv, ast.Subscript):
            if isinstance(recv.value, ast.Attribute) and recv.value.attr in ("iloc", "loc"):
                return False  # an explicit slice of the data, assume deliberate
            recv = recv.value
        else:
            return True


_UNKNOWN = "unknown"


class _Visitor(ast.NodeVisitor):
    def __init__(self, tree: ast.AST | None = None) -> None:
        self.hits: list[tuple[str, str, ast.AST, str]] = []
        self.loop_depth = 0
        self.stat_names: set[str] = set()
        self.numbers = _named_numbers(tree) if tree is not None else {}
        self.numpy_names = _numpy_aliases(tree) if tree is not None else set(_NUMPY_MODULES)

    def amounts(self, node: ast.AST | None) -> list[float] | str:
        """The shift amount(s) of a call: numbers when they are literal or a once-assigned named number, else UNKNOWN."""
        if isinstance(node, (ast.Tuple, ast.List)):
            parts = [self.amounts(e) for e in node.elts]
            return _UNKNOWN if any(p == _UNKNOWN for p in parts) else [v for p in parts for v in p]  # type: ignore[union-attr]
        v = _const_number(node)
        if v is None and isinstance(node, ast.Name):
            v = self.numbers.get(node.id)
        return _UNKNOWN if v is None else [v]

    def add(self, rule: str, severity: str, node: ast.AST, message: str) -> None:
        self.hits.append((rule, severity, node, message))

    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_full_stat(node.value):
            self.stat_names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    visit_While = visit_For  # type: ignore[assignment]
    visit_AsyncFor = visit_For  # type: ignore[assignment]

    def visit_Subscript(self, node: ast.Subscript) -> None:
        v = node.value
        if (
            self.loop_depth
            and isinstance(v, ast.Attribute)
            and v.attr == "iloc"
            and isinstance(node.slice, ast.UnaryOp)
            and isinstance(node.slice.op, ast.USub)
            and isinstance(node.slice.operand, ast.Constant)
            and node.slice.operand.value == 1
        ):
            self.add("LA004", BLOCK, node, ".iloc[-1] inside a loop reads the latest row, which may be the future")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Sub, ast.Div)):
            r = node.right
            if _is_full_stat(r) or (isinstance(r, ast.Name) and r.id in self.stat_names):
                self.add(
                    "LA005", WARN, node,
                    "full-sample statistic used as divisor/subtractor; compute it only from data before the split",
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Attribute):
            if f.attr == "shift":
                arg = node.args[0] if node.args else _kw(node, "periods")
                if arg is not None:
                    amount = self.amounts(arg)
                    if amount != _UNKNOWN and any(v < 0 for v in amount):  # type: ignore[union-attr]
                        self.add("LA001", BLOCK, node, "negative shift moves future values into the past")
                    elif amount == _UNKNOWN:
                        self.add("LA007", WARN, node,
                                 "shift amount is not a literal number, so a negative (future) shift cannot be ruled out; "
                                 "the future-perturbation test of S2 is the proof")
            elif f.attr == "roll" and isinstance(f.value, ast.Name) and f.value.id in self.numpy_names:
                arg = node.args[1] if len(node.args) > 1 else _kw(node, "shift")
                amount = self.amounts(arg) if arg is not None else [0.0]
                if amount != _UNKNOWN and any(v < 0 for v in amount):  # type: ignore[union-attr]
                    self.add("LA006", BLOCK, node, "np.roll with a negative shift moves future values into the past")
                elif amount == _UNKNOWN:
                    self.add("LA007", WARN, node,
                             "np.roll amount is not a literal number, so a negative (future) shift cannot be ruled out; "
                             "the future-perturbation test of S2 is the proof")
            elif f.attr == "rolling":
                c = _kw(node, "center")
                if isinstance(c, ast.Constant) and c.value is True:
                    self.add("LA002", BLOCK, node, "centered rolling window uses future observations")
            elif f.attr in ("bfill", "backfill"):
                self.add("LA003", BLOCK, node, f"{f.attr}() fills from later observations")
            elif f.attr == "fillna" and _str_const(_kw(node, "method")) in ("bfill", "backfill"):
                self.add("LA003", BLOCK, node, "fillna(method=backward) fills from later observations")
        self.generic_visit(node)


def _allowances(source: str) -> dict[int, list[tuple[str, str | None]]]:
    found: dict[int, list[tuple[str, str | None]]] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT and (m := _ALLOW.search(tok.string)):
                found.setdefault(tok.start[0], []).append((m["rule"], m["reason"]))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return found


def lint_source(source: str, filename: str = "<string>") -> list[Finding]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [Finding("LA000", filename, exc.lineno or 1, BLOCK, "", f"cannot parse source: {exc.msg}")]
    v = _Visitor(tree)
    v.visit(tree)
    lines = source.splitlines()
    allow = _allowances(source)
    out: list[Finding] = []
    for rule, severity, node, message in v.hits:
        first = getattr(node, "lineno", 1)
        last = getattr(node, "end_lineno", first) or first
        reason = None
        suppressed = False
        for ln in range(first - 1, last + 1):
            if ln == first - 1 and not (0 < ln <= len(lines) and lines[ln - 1].lstrip().startswith("#")):
                continue  # the line above counts only when it is a comment line
            for r, why in allow.get(ln, []):
                if r == rule and why:
                    suppressed, reason = True, why
        snippet = lines[first - 1].strip() if 0 < first <= len(lines) else ""
        out.append(Finding(rule, filename, first, severity, snippet, message, suppressed, reason))
    return sorted(out, key=lambda f: (f.line, f.rule_id))


def lint_file(path: str | Path) -> list[Finding]:
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [Finding("LA000", str(p), 0, BLOCK, "", f"cannot read source: {exc}")]
    return lint_source(source, str(p))


def lint_files(paths: Iterable[str | Path]) -> list[Finding]:
    return [f for p in paths for f in lint_file(p)]


def blocking(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == BLOCK and not f.suppressed]
