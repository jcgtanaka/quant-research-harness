"""LLM001: an adapter appears to use a language model but the hypothesis declares none.

A deterministic static check over the adapter source files the loader tracks. It looks for imports of the common
LLM client libraries and for calls that load one by name. It is an early, cheap detector, not proof: a model reached
through a plain HTTP call is invisible to it (that is why the declaration, the cutoff gate and the replay cache
exist). Suppress a finding with ``# qrh: allow LLM001 <reason>``; the suppression is carried into the readiness
report. A pack can add libraries with a lint rule whose id is LLM001 (``ast: {import: <module>}``).
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

from qrh.checks.lookahead import BLOCK, WARN, Finding, _allowances

LLM_RULE = "LLM001"
MESSAGE = "adapter appears to use an LLM but the hypothesis declares none"
DIRECT_RULE = "LLM002"
DIRECT_MESSAGE = ("provider client used outside qrh.llm.complete; replay cache and the truncation prompt detector "
                  "cannot see these calls")

# Module prefixes: ``google.genai`` matches ``google.genai.types`` but not ``google.cloud``.
LLM_MODULES = (
    "anthropic", "openai", "google.generativeai", "google.genai", "ollama", "litellm", "mistralai", "cohere",
    "together", "groq", "llama_cpp", "vllm", "qrh.llm",
)
LLM_PREFIXES = ("langchain",)  # langchain, langchain_core, langchain_openai, ...


_SANCTIONED = "qrh.llm"  # the entry point itself: LLM001 counts it as LLM use, LLM002 never flags it


def _matches(module: str, extra: tuple[str, ...], clients_only: bool = False) -> bool:
    if clients_only and (module == _SANCTIONED or module.startswith(_SANCTIONED + ".")):
        return False
    top = module.split(".")[0]
    if any(top.startswith(p) for p in LLM_PREFIXES):
        return True
    return any(module == m or module.startswith(m + ".") for m in (*LLM_MODULES, *extra))


def _import_hits(node: ast.AST, extra: tuple[str, ...], clients_only: bool = False) -> bool:
    if isinstance(node, ast.Import):
        return any(_matches(a.name, extra, clients_only) for a in node.names)
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        if _matches(node.module, extra, clients_only):
            return True
        if any(_matches(f"{node.module}.{a.name}", extra, clients_only) for a in node.names):  # from google import genai
            return True
        return node.module == "transformers" and any(a.name == "pipeline" for a in node.names)
    return False


def _call_hits(node: ast.Call, extra: tuple[str, ...], clients_only: bool = False) -> bool:
    f = node.func
    name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
    if name in ("import_module", "__import__") and node.args:
        arg = node.args[0]
        return isinstance(arg, ast.Constant) and isinstance(arg.value, str) and _matches(arg.value, extra, clients_only)
    return (isinstance(f, ast.Attribute) and f.attr == "pipeline" and isinstance(f.value, ast.Name)
            and f.value.id == "transformers")


def _findings(source: str, filename: str, extra: Iterable[str], rule: str, severity: str, message: str,
              clients_only: bool) -> list[Finding]:
    extras = tuple(extra)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # the look-ahead lint already reports source that cannot be parsed
    lines = source.splitlines()
    allow = _allowances(source)
    out: list[Finding] = []
    for node in ast.walk(tree):
        if not ((isinstance(node, (ast.Import, ast.ImportFrom)) and _import_hits(node, extras, clients_only))
                or (isinstance(node, ast.Call) and _call_hits(node, extras, clients_only))):
            continue
        first = node.lineno
        last = getattr(node, "end_lineno", first) or first
        reason = None
        for ln in range(first - 1, last + 1):
            if ln == first - 1 and not (0 < ln <= len(lines) and lines[ln - 1].lstrip().startswith("#")):
                continue  # the line above counts only when it is a comment line
            for r, why in allow.get(ln, []):
                if r == rule and why:
                    reason = why
        snippet = lines[first - 1].strip() if 0 < first <= len(lines) else ""
        out.append(Finding(rule, filename, first, severity, snippet, message, reason is not None, reason))
    return sorted(out, key=lambda f: (f.line, f.snippet))


def lint_llm_source(source: str, filename: str = "<string>", extra: Iterable[str] = ()) -> list[Finding]:
    return _findings(source, filename, extra, LLM_RULE, BLOCK, MESSAGE, False)


def _registers_backend(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == "register_backend") or (
                    isinstance(f, ast.Attribute) and f.attr == "register_backend"):
                return True
    return False


def lint_llm_direct_source(source: str, filename: str = "<string>", extra: Iterable[str] = ()) -> list[Finding]:
    """LLM002 (WARN) for a source file of an adapter that declares an llm: a provider client outside qrh.llm.complete.

    A file that calls ``register_backend`` is the intended place for provider code and is exempt."""
    if _registers_backend(source):
        return []
    return _findings(source, filename, extra, DIRECT_RULE, WARN, DIRECT_MESSAGE, True)


def lint_llm_direct(files: Iterable[str | Path], extra: Iterable[str] = ()) -> list[Finding]:
    out: list[Finding] = []
    for p in files:
        path = Path(p)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        out.extend(lint_llm_direct_source(source, str(path), extra))
    return out


def lint_llm_use(files: Iterable[str | Path], extra: Iterable[str] = ()) -> list[Finding]:
    out: list[Finding] = []
    for p in files:
        path = Path(p)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        out.extend(lint_llm_source(source, str(path), extra))
    return out
