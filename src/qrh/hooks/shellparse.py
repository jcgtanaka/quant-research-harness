"""A small quote-aware split of a shell command into simple commands, shared by the guard hooks.

Not a shell: it does not expand variables or globs and cannot follow indirection built at run time. What it
does give the guards is a word list per simple command in which quoted text stays ONE word (so a commit message
that mentions ``git push`` is not a push), redirect targets, and the scripts handed to ``sh -c`` / ``eval`` or
piped into a shell from a heredoc. Input that cannot be tokenized (unbalanced quotes) falls back to a naive split
so the callers still see every word; they never get a "could not parse" escape hatch.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field

SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish", "ash", "csh", "tcsh"})
MAX_DEPTH = 4
_OPS = frozenset(";&|()<>\n")
_HEREDOC = re.compile(r"<<-?[ \t]*(['\"]?)([A-Za-z_][\w]*)\1")
_DASH_C = re.compile(r"^-[A-Za-z]*c[A-Za-z]*$")
_NAIVE_SPLIT = re.compile(r"\n|;|&&|\|\||\||&|\(|\)")


@dataclass
class Cmd:
    """One simple command: its words (redirects removed) and the targets it writes with ``>``."""

    words: list[str]
    redirects: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)  # targets this command reads with ``< file``
    heredoc: bool = False  # a heredoc or here-string feeds this command


def base(word: str) -> str:
    return os.path.basename(word)


def _split_heredocs(command: str) -> tuple[str, list[str]]:
    """(command without heredoc bodies, the bodies)."""
    lines = command.split("\n")
    out: list[str] = []
    bodies: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for m in _HEREDOC.finditer(line):
            delim, body = m.group(2), []
            while i < len(lines) and lines[i].strip("\t") != delim:
                body.append(lines[i])
                i += 1
            i += 1  # the terminator
            bodies.append("\n".join(body))
    return "\n".join(out), bodies


def _tokens(text: str) -> list[str] | None:
    lexer = shlex.shlex(text.replace("`", " ; "), posix=True, punctuation_chars=";&|()<>\n")
    lexer.whitespace, lexer.whitespace_split = " \t\r", True
    try:
        return list(lexer)
    except ValueError:
        return None


_INPUT_RX = re.compile(r"(?<![<&])<(?![<(&>])\s*([^\s;|&<>()]+)")


def _unquote(part: str) -> str:
    return part.replace("'", "").replace('"', "")


def _naive(text: str) -> list[Cmd]:
    cmds = [Cmd(w, inputs=[t.strip("'\"") for t in _INPUT_RX.findall(part)], heredoc="<<" in part)
            for part in _NAIVE_SPLIT.split(text.replace("`", ";")) if (w := _unquote(part).split())]
    return cmds


def _build(tokens: list[str]) -> list[Cmd]:
    cmds: list[Cmd] = []
    cur = Cmd([])
    i = 0
    while i < len(tokens):
        t = tokens[i]
        i += 1
        if t and set(t) <= _OPS:
            if "<" in t or ">" in t:
                if i >= len(tokens):
                    break
                target = tokens[i]
                i += 1
                if t in ("<", "<>"):
                    cur.inputs.append(target)
                elif t.startswith("<<"):
                    cur.heredoc = True
                if ">" in t and not t.endswith("&") and "<" not in t:
                    cur.redirects.append(target)
                continue
            if cur.words or cur.redirects or cur.inputs:
                cmds.append(cur)
            cur = Cmd([])
            continue
        cur.words.append(t)
    if cur.words or cur.redirects or cur.inputs:
        cmds.append(cur)
    return cmds


def _nested(cmd: Cmd, depth: int) -> list[Cmd]:
    """Scripts a command hands to a shell (`sh -c SCRIPT`, `eval ...`)."""
    out: list[Cmd] = []
    words = cmd.words
    for i, w in enumerate(words):
        if base(w) in SHELLS:
            for j in range(i + 1, len(words) - 1):
                if _DASH_C.match(words[j]):
                    out += parse(words[j + 1], depth + 1)
                    break
        elif w == "eval" and i + 1 < len(words):
            out += parse(" ".join(words[i + 1:]), depth + 1)
    return out


def parse(command: str, depth: int = 0) -> list[Cmd]:
    if depth > MAX_DEPTH:
        return []
    stripped, bodies = _split_heredocs(command)
    tokens = _tokens(stripped)
    cmds = _build(tokens) if tokens is not None else _naive(stripped)
    out = list(cmds)
    for c in cmds:
        out += _nested(c, depth)
    if bodies and any(base(w) in SHELLS or w in ("eval", "source", ".") for c in cmds for w in c.words):
        for body in bodies:  # a shell may run this text: judge it like any other script
            out += parse(body, depth + 1)
    return out


# --- what a simple command is, and what it writes ---------------------------------------------------------

WRAPPERS = frozenset({"sudo", "doas", "env", "nohup", "time", "command", "exec", "nice", "ionice", "setsid", "xargs",
                      "stdbuf", "timeout", "builtin"})
KEYWORDS = frozenset({"if", "then", "else", "elif", "do", "while", "until", "!", "{", "}"})
_ASSIGN = re.compile(r"^[A-Za-z_]\w*=")
_WRAPPER_VALUE_FLAGS = frozenset({"-u", "-g", "-n", "-C", "-s", "-k"})
_INPLACE = re.compile(r"^-[a-zA-Z]*i|^--in-place")
_COPY_LIKE = frozenset({"cp", "install", "rsync", "scp", "ln"})
_ALL_ARGS = frozenset({"tee", "mv", "rm", "unlink", "rmdir", "shred", "touch", "truncate", "chmod", "chattr", "chown", "chgrp", "setfacl"})
_INPLACE_EDITORS = frozenset({"sed", "perl", "awk", "gawk", "ed"})


def program_index(words: list[str]) -> int:
    """Index of the program word after assignments and wrappers (``sudo -u me env X=1 crontab -e`` -> crontab)."""
    i = 0
    while i < len(words):
        w = words[i]
        if _ASSIGN.match(w):
            i += 1
        elif w in KEYWORDS:
            i += 1
        elif base(w) in WRAPPERS:
            i += 1
            while i < len(words) and (words[i].startswith("-") or _ASSIGN.match(words[i])):
                i += 2 if words[i] in _WRAPPER_VALUE_FLAGS else 1
        else:
            return i
    return len(words)


def write_targets(cmd: Cmd) -> list[str]:
    """Paths (as written, unexpanded) a command writes, creates, replaces or deletes: ``>`` targets, tee, cp/mv/install,
    dd of=, in-place editors, rm/touch/truncate, chmod/chattr and ``git rm|mv|update-index``. Positional words only; what an interpreter's
    own code writes is not visible here."""
    out = list(cmd.redirects)
    w = cmd.words
    i = program_index(w)
    if i >= len(w):
        return out
    name, rest = base(w[i]), w[i + 1:]
    args = [a for a in rest if not a.startswith("-")]
    if name == "git":
        sub = next((a for a in rest if not a.startswith("-")), "")
        if sub in ("rm", "mv", "update-index"):
            out += args[1:]
    elif name in _COPY_LIKE:
        out += args[-1:]
        for j, a in enumerate(rest):
            if a in ("-t", "--target-directory") and j + 1 < len(rest):
                out.append(rest[j + 1])
            elif a.startswith("--target-directory="):
                out.append(a.split("=", 1)[1])
    elif name in _ALL_ARGS:
        out += args
    elif name == "dd":
        out += [a[3:] for a in rest if a.startswith("of=")]
    elif name in _INPLACE_EDITORS and any(_INPLACE.match(a) for a in rest if a.startswith("-")):
        out += args
    return out
