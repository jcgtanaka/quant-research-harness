"""PreToolUse (Read, Bash, Grep): keep licensed data away from the cloud model.

A dataset registered with ``qrh data register`` is ``privacy: private`` unless it was registered with
``--privacy public`` (synthetic data). When an agent reads such a file, the rows travel to the cloud model as tool
output. This hook stops that with an APPROVABLE block: the banner, a pending approval and a single-use human
approval, exactly like the other approvable blocks ("this sends licensed data to the cloud model"). Public
datasets and files that are not registered are never blocked.

What it detects (best effort, from the command text only, nothing is executed):

 * Read: the file itself (relative paths, ``..``, symlinks resolved);
 * Grep: the file itself in any output mode, or a directory above it in ``content`` mode (``glob`` and ``type``
   filters that cannot reach the file are honoured);
 * Bash readers that print what they read: cat, tac, head, tail, less, more, sed, awk, jq, cut, sort, uniq, nl, od,
   xxd, strings, column, paste, diff, tee, base64, zcat, bat, grep and rg on the file (and recursive grep/rg, find
   or xargs reaching it), also behind ``sudo``/``env``/``timeout``, ``sh -c``, ``eval``, pipes, ``$(...)``, ``cd``
   tracking, ``~``, ``$HOME``, brace and glob expansion, shell variables assigned in the same command, and input
   redirects (``< file``). A path built from a variable that cannot be resolved here blocks when the rest of the
   word could name a private dataset (conservative);
 * Python, pandas and other inline code (``python -c``, a heredoc, a pipe into an interpreter; also R, node, perl,
   ruby, duckdb) that names the file, its basename or a registered path.

What it does NOT catch, by design: a script that prints rows is NOT caught. ``python run.py --data FILE`` is the
normal way to compute on a private dataset, and the hook sees only the command line, not what the script then does
with the path. The same holds for a copy of the file made first (``cp``, then ``cat`` the copy), a path assembled
at run time in a way a text filter cannot follow, and the output of a command the agent was never shown. The point
is to stop the plain act of reading a private file into the conversation, not to be a sandbox (see
docs/security.md).

Fail closed: if the registry cannot be read or a tool input cannot be checked, the call is blocked.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

from qrh.hooks.paths import is_under
from qrh.hooks.runner import HookContext, HookResult
from qrh.hooks.shellparse import base, parse, program_index
from qrh.hooks.data_guard_parse import (  # noqa: F401  (re-exported: the old names stay importable from here)
    WILD,
    MAX_BRACES,
    READERS,
    GIT_READERS,
    GREP_FAMILY,
    ALWAYS_RECURSIVE,
    NONREADERS,
    INTERPRETER,
    _INLINE_FLAGS,
    _INTERP_VALUE_FLAGS,
    _NUMERIC,
    TRANSPARENT,
    RUNNERS,
    _RUN_VERBS,
    _RUNNER_VALUE_FLAGS,
    _EXEC_FLAGS,
    _STDOUT_SINKS,
    _TYPES,
    _PEEL,
    _REDIRECT_IN,
    _ASSIGN,
    _VAR,
    _BRACE,
    _DECLARE,
    _NAMES_ONLY,
    _NAMES_ONLY_LONG,
    _GLOB_FLAGS,
    _TYPE_FLAGS,
    _RECURSIVE,
    _brace_expand,
    _WILDS,
    _Glob,
    _glob_rx,
    _search_filters,
    _peel,
    _flags_say_names_only,
    _assignments,
    _clean_text,
    _family,
    _inline_code,
    _skip_flags,
    _program_words,
    _unknown_dir,
)

BUDGET_SECONDS = 3.0  # well under the 10 s hook timeout: a command that costs more to check is blocked, never passed


class CheckTooComplex(Exception):
    """The check used up its time budget: the caller blocks (fail closed)."""


_deadline: float | None = None


def _tick() -> None:
    if _deadline is not None and time.monotonic() > _deadline:
        raise CheckTooComplex


@dataclass(frozen=True)
class Private:
    """A registered dataset that is not public: every spelling of its path that a reader could use."""

    name: str
    forms: tuple[str, ...]
    basename: str
    legacy: bool = False  # registered with a relative path: where it points is unknown, so only its basename is matched


def private_datasets(ctx: HookContext, cwd: str) -> list[Private]:
    rows = ctx.registry().conn.execute("SELECT name, path FROM datasets WHERE privacy != 'public'").fetchall()
    out: list[Private] = []
    for r in rows:
        raw = os.path.expanduser(str(r["path"]))
        if not os.path.isabs(raw):  # a row from before paths were stored absolute: fail closed on the basename
            out.append(Private(str(r["name"]), (), os.path.basename(raw), legacy=True))
            continue
        forms = tuple(dict.fromkeys((os.path.normpath(raw), os.path.realpath(raw))))
        out.append(Private(str(r["name"]), forms, os.path.basename(raw)))
    return out


# --- word and path matching ------------------------------------------------------------------------------


class Resolver:
    """Expands a word the way a shell would (best effort) and says which private datasets it can name."""

    def __init__(self, datasets: list[Private], variables: dict[str, str]):
        self.datasets = datasets
        self.vars = variables
        self._user = os.path.expanduser("~")

    def expand(self, word: str) -> list[str]:
        def var(m: re.Match[str]) -> str:
            name = m[1] if m[1] is not None else m[2]
            if name in self.vars:
                return self.vars[name]
            if name == "HOME":
                return self._user
            if name in ("USER", "LOGNAME"):
                return os.path.basename(self._user)
            return WILD

        text = word
        for _ in range(3):  # a variable may hold another one
            new = _VAR.sub(var, text)
            if new == text:
                break
            text = new
        text = text.replace("$", WILD)
        if text == "~" or text.startswith("~/"):
            text = self._user + text[1:]
        return _brace_expand(text)

    def _hits(self, word: str, cwd: str | None, *, directory: bool, wild: bool) -> list[Private]:
        """``cwd`` None means the working directory is unknown (after ``cd $X`` or ``cd -``): a relative word is then
        matched as WILD + the word."""
        found: list[Private] = []
        for text in self.expand(word):
            _tick()
            if not text or text.startswith("-") and "=" not in text:
                continue
            fuzzy = any(c in text for c in ("*", "?", "[", WILD))
            if fuzzy and not wild and WILD in text:
                continue
            for d in self.datasets:
                if d.legacy and self._legacy_hit(d, text):
                    found.append(d)
            unknown = cwd is None and not text.startswith("/") and not text.startswith(WILD)
            if not fuzzy and not unknown:
                full = os.path.normpath(text if os.path.isabs(text) else os.path.join(cwd or "", text))
                forms = (full, os.path.realpath(full))
                for d in self.datasets:
                    if any(f == p or (directory and is_under(p, f)) for f in forms for p in d.forms):
                        found.append(d)
                continue
            if unknown:
                rel = os.path.normpath(text)
                while rel.startswith("../"):
                    rel = rel[3:]
                joined = WILD + "/" + ("" if rel in (".", "..") else rel)
            else:
                joined = text if text.startswith(("/", WILD)) else os.path.join(cwd or "", text)
            joined = re.sub(r"/(?:\./)+", "/", re.sub(r"/{2,}", "/", joined))
            rx = _glob_rx(joined)
            rx_dir = _glob_rx(joined.rstrip("/") + "/**") if directory else None
            for d in self.datasets:
                if any(rx.fullmatch(p) or (rx_dir and rx_dir.fullmatch(p)) for p in d.forms):
                    found.append(d)
        return found

    @staticmethod
    def _legacy_hit(d: Private, text: str) -> bool:
        """A dataset row with a relative path: any word whose last component matches its basename counts."""
        tail = text.rstrip("/").rsplit("/", 1)[-1]
        if not tail:
            return False
        if any(c in tail for c in ("*", "?", "[", WILD)):
            return _glob_rx(tail).fullmatch(d.basename)
        return tail == d.basename

    def file_hits(self, word: str, cwd: str | None, wild: bool = True) -> list[Private]:
        """Private datasets the word can name (an ``=value`` suffix such as ``if=FILE`` is checked too)."""
        hits = self._hits(word, cwd, directory=False, wild=wild)
        if "=" in word and not word.startswith("="):
            hits += self._hits(word.split("=", 1)[1], cwd, directory=False, wild=wild)
        return hits

    def dir_hits(self, word: str, cwd: str | None) -> list[Private]:
        """Private datasets in the file or directory the word names (a recursive search from there reaches them)."""
        return self._hits(word, cwd, directory=True, wild=False)


# --- filters of recursive searches (rg --glob/--type, grep --include, the Grep tool's glob/type) ----------------

def _filtered_out(d: Private, globs: list[str], types: list[str], scope: str) -> bool:
    """True when a recursive search restricted by these filters can never reach the dataset file."""
    path = d.forms[0]
    rel = os.path.relpath(path, scope) if path.startswith(scope.rstrip("/") + "/") else path

    def matches(pattern: str) -> bool:
        for alt in _brace_expand(pattern):
            rx = _glob_rx(alt)
            if any(rx.fullmatch(c) for c in (d.basename, rel, path)):
                return True
        return False

    positive = [g for g in globs if not g.startswith("!")]
    negative = [g[1:] for g in globs if g.startswith("!")]
    if positive and not any(matches(g) for g in positive):
        return True
    if any(matches(g) for g in negative):
        return True
    ext = os.path.splitext(d.basename)[1].lower()
    known = [t for t in types if t in _TYPES]
    return bool(known) and len(known) == len(types) and not any(ext in _TYPES[t] for t in known)


# --- Bash --------------------------------------------------------------------------------------------------


def _git_hits(word: str, datasets: list[Private]) -> list[Private]:
    """`git show HEAD:dir/file`, `git log -p -- dir/file`, `git blame file`: a repository path that ends a private path."""
    part = os.path.normpath(word.split(":")[-1].lstrip("/")) if word and not word.startswith("-") else ""
    if not part or part in (".", ".."):
        return []
    return [d for d in datasets if os.path.basename(part) == d.basename
            and (d.legacy or any(f.endswith("/" + part) or f == part for f in d.forms))]


def bash_hits(command: str, cwd: str, datasets: list[Private]) -> set[str]:
    """Names of the private datasets this command appears to read."""
    hits: set[str] = set()
    pieces = _peel(command)
    per_piece = [parse(p) for p in pieces]
    all_cmds = [c for group in per_piece for c in group]
    variables, aliases = _assignments(all_cmds)
    res = Resolver(datasets, variables)
    readers = READERS | aliases
    indirect = False
    cwds: set[str | None] = {cwd}

    for cmds in per_piece:
        cur: str | None = cwd  # None: unknown (after `cd $X`, `cd -`, a bare `cd`)
        for c in cmds:
            _tick()
            words = c.words
            if words and base(words[0]) in ("cd", "pushd", "popd"):
                args = [w for w in words[1:] if not (w.startswith("-") and w != "-")]
                target = res.expand(args[0])[0] if args else None
                if _unknown_dir(target):
                    cur = None
                elif cur is not None or os.path.isabs(target or ""):
                    cur = os.path.normpath(target if os.path.isabs(target or "") else os.path.join(cur or "", target or ""))
                cwds.add(cur)
                continue
            pwords = _program_words(words) if words else []
            progs = [base(words[k]) for k in pwords]
            if c.inputs and not (progs and progs[0] in NONREADERS):
                for target in (t.strip("'\"") for t in c.inputs):  # the redirect belongs to THIS command
                    for where in {cur, *cwds}:
                        for d in res.file_hits(target, where):
                            hits.add(d.name)
            if not words:
                continue
            names = [base(w) for w in words]
            reader = any(n in readers for n in progs) or (
                any(w in _STDOUT_SINKS for w in words) and any(n in ("cp", "dd", "install") for n in progs)) or (
                "xargs" in names and any(w in ("-a", "--arg-file") or w.startswith("--arg-file=") for w in words))
            git_reader = "git" in names and any(w in GIT_READERS for w in words)
            pi = program_index(words)
            opaque = pi < len(words) and (WILD in words[pi] or words[pi].startswith("$"))
            if reader:
                greps = [n for n in progs if n in GREP_FAMILY]
                names_only = bool(greps) and _flags_say_names_only(words)
                recursive = bool(greps) and (any(g in ALWAYS_RECURSIVE for g in greps)
                                             or any(_RECURSIVE.match(w) for w in words if w.startswith("-")))
                globs, types = _search_filters(words) if greps else ([], [])
                for w in words:
                    if w.startswith("-") and "=" not in w:
                        continue
                    for d in res.file_hits(w, cur):
                        hits.add(d.name)
                    for d in datasets:  # a path inside a script word: sed '1r FILE', awk 'getline < "FILE"', jq --rawfile
                        if any(f in w for f in d.forms):
                            hits.add(d.name)
                    if recursive and not names_only:
                        for d in res.dir_hits(w, cur):
                            if not _filtered_out(d, globs, types, os.path.normpath(cur or "/")):
                                hits.add(d.name)
                first = next((k for k in pwords if base(words[k]) in GREP_FAMILY), len(words))
                positional = [w for w in words[first + 1:] if not w.startswith("-")]
                if recursive and not names_only and len(positional) <= 1:
                    for d in datasets:  # `rg PATTERN` with no path searches the working directory
                        if d.forms and (cur is None or any(p.startswith(cur.rstrip("/") + "/") for p in d.forms)) and (
                                not _filtered_out(d, globs, types, cur or "/")):
                            hits.add(d.name)
                if "xargs" in names or any(w in _EXEC_FLAGS for w in words):
                    indirect = True
            if git_reader:
                for w in words:
                    for d in _git_hits(w, datasets):
                        hits.add(d.name)
            if not reader and opaque:
                for w in words[pi + 1:]:
                    for d in res.file_hits(w, cur, wild=False):
                        hits.add(d.name)

    if indirect:  # `find DIR ... | xargs cat`, `find DIR -exec head {} +`: the reader and the path are in different words
        for c in all_cmds:
            for w in c.words:
                for cur in cwds:
                    for d in res.file_hits(w, cur, wild=False) + res.dir_hits(w, cur):
                        hits.add(d.name)

    if _inline_code(all_cmds):
        text = _clean_text(command)
        globbed = [_glob_rx(w) for w in re.findall(r"[^\s'\"()\[\],;=]*[*?][^\s'\"()\[\],;=]*", text)
                   if w.strip("*?/")]
        for d in datasets:
            _tick()
            parents = [os.path.dirname(f) for f in d.forms]
            if (d.basename and d.basename in text) or any(f in text for f in d.forms) or any(
                    p in text for p in parents if p.count("/") >= 3) or any(
                    rx.fullmatch(c) for rx in globbed for c in (d.basename, *d.forms)):
                hits.add(d.name)
    return hits


# --- the file tools ------------------------------------------------------------------------------------------

def _str_field(ti: dict[str, Any], key: str) -> str:
    v = ti.get(key)
    if v is None:
        return ""
    if not isinstance(v, str):
        raise TypeError(f"cannot check {key}: not a string")
    return v


def read_hits(ti: dict[str, Any], cwd: str, datasets: list[Private]) -> set[str]:
    path = _str_field(ti, "file_path")
    return {d.name for d in Resolver(datasets, {}).file_hits(path, cwd, wild=False)} if path else set()


def grep_hits(ti: dict[str, Any], cwd: str, datasets: list[Private]) -> set[str]:
    path = _str_field(ti, "path") or cwd
    mode = _str_field(ti, "output_mode") or "files_with_matches"
    glob, ftype = _str_field(ti, "glob"), _str_field(ti, "type")
    res = Resolver(datasets, {})
    hits = {d.name for d in res.file_hits(path, cwd, wild=False)}  # an explicit file: any output mode says something
    if mode == "content":
        scope = os.path.normpath(path if os.path.isabs(path) else os.path.join(cwd, path))
        for d in res.dir_hits(path, cwd):
            if d.name in hits:
                continue
            if not _filtered_out(d, [glob] if glob else [], [ftype] if ftype else [], scope):
                hits.add(d.name)
    return hits


MESSAGE = ("this sends licensed data to the cloud model: {how} registered dataset {names} (privacy: private). "
           "Public or synthetic datasets (registered with --privacy public) are not affected")


def run(ctx: HookContext) -> HookResult:
    p = ctx.payload
    if p.event and p.event != "PreToolUse":
        return HookResult("allow", "not a PreToolUse event")
    ti = dict(p.tool_input)
    cwd = p.cwd or os.getcwd()
    if p.tool_name == "Bash":
        if not p.command:
            return HookResult("allow", "no command")
        how, check = "a command that reads", lambda d: bash_hits(p.command, cwd, d)
    elif p.tool_name == "Read":
        how, check = "reading", lambda d: read_hits(ti, cwd, d)
    elif p.tool_name == "Grep":
        how, check = "a search that returns content of", lambda d: grep_hits(ti, cwd, d)
    else:
        return HookResult("allow", "not a reading tool")
    datasets = private_datasets(ctx, cwd)
    if not datasets:
        return HookResult("allow", "no private dataset is registered")
    global _deadline
    _deadline = time.monotonic() + BUDGET_SECONDS
    extra_note = ""
    try:
        hits = sorted(check(datasets))
    except CheckTooComplex:  # cannot be checked in time: block, never let the hook timeout pass it
        hits = sorted(d.name for d in datasets)
        how = "a command too complex to check in time may read"
        extra_note = " This input is too complex to check within the time budget, so it is blocked."
    finally:
        _deadline = None
    if not hits:
        return HookResult("allow", "no private dataset is read")
    names = ", ".join(repr(n) for n in hits)
    text = MESSAGE.format(how=how, names=names)
    legacy = sorted(d.name for d in datasets if d.legacy and d.name in hits)
    if legacy:
        extra_note += (f" Dataset {', '.join(repr(n) for n in legacy)} was registered with a relative path, so the hook "
                       "can only match it by file name: re-register it with its absolute path "
                       "(qrh data register PATH --name NAME).")
    return HookResult(
        "block", f"reads private dataset {names}", approvable=True, extra={"datasets": hits},
        stderr=text + ". A human decides whether the model may see this data; a script that prints rows is not "
        "checked, so run the computation in a script and show only its summary." + extra_note,
    )

