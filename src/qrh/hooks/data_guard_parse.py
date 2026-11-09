"""Text and glob helpers of the data guard: brace expansion, a linear-time glob matcher, shell-command peeling and
interpreter/launcher detection. Pure functions of strings and parsed commands; the decisions stay in data_guard.

Everything here is re-exported from qrh.hooks.data_guard, so existing imports keep working.
"""

from __future__ import annotations

import re

from qrh.hooks.shellparse import WRAPPERS, Cmd, base, program_index

WILD = "\x00"  # an expansion this hook cannot resolve ($UNSET, $(...), <(...)): any text
MAX_BRACES = 64

READERS = frozenset({
    "cat", "tac", "head", "tail", "less", "more", "sed", "awk", "gawk", "mawk", "nawk", "jq", "yq", "cut", "sort", "uniq",
    "nl", "od", "xxd", "hexdump", "strings", "column", "paste", "diff", "comm", "join", "tee", "base64", "zcat", "gzcat",
    "bzcat", "xzcat", "bat", "batcat", "fold", "fmt", "pr", "rev", "tr", "expand", "unexpand", "shuf", "dd", "look",
    "csvlook", "csvcut", "csvstat", "xsv", "qsv", "mlr", "vd", "visidata", "datamash", "sqlite3", "grep", "egrep", "fgrep", "zgrep", "rg", "ag", "ack",
})
GIT_READERS = frozenset({"show", "diff", "log", "blame", "annotate", "cat-file", "grep", "whatchanged", "difftool"})
GREP_FAMILY = frozenset({"grep", "egrep", "fgrep", "zgrep", "rg", "ag", "ack"})
ALWAYS_RECURSIVE = frozenset({"rg", "ag", "ack"})
# commands that never show the content of a file (their input redirect is harmless)
NONREADERS = frozenset({"wc", "md5sum", "sha1sum", "sha224sum", "sha256sum", "sha384sum", "sha512sum", "b2sum", "cksum",
                        "test", "[", "true", ":", "stat", "file"})
INTERPRETER = re.compile(r"^(?:python[\d.]*|pypy[\d.]*|ipython[\d.]*|perl|ruby|node|deno|bun|php|lua|Rscript|R|julia|"
                         r"duckdb)$")
# the flags that hand an interpreter code on the command line (per interpreter; -E, -r, -x of python and the like are not)
_INLINE_FLAGS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^-[A-Za-z]*c$"), "ipython": re.compile(r"^-[A-Za-z]*c$"),
    "perl": re.compile(r"^-[A-Za-z]*[eE]$"), "ruby": re.compile(r"^-[A-Za-z]*e$"),
    "node": re.compile(r"^-[A-Za-z]*[ep]$|^--(?:eval|print)$"), "deno": re.compile(r"^-[A-Za-z]*[ep]$|^--(?:eval|print)$"),
    "bun": re.compile(r"^-[A-Za-z]*[ep]$|^--(?:eval|print)$"),
    "rscript": re.compile(r"^-[A-Za-z]*e$|^--expression"), "r": re.compile(r"^-[A-Za-z]*e$|^--expression"),
    "php": re.compile(r"^-[A-Za-z]*r$"), "lua": re.compile(r"^-[A-Za-z]*e$"),
    "julia": re.compile(r"^-[A-Za-z]*[eE]$|^--(?:eval|print)$"), "duckdb": re.compile(r"^-(?:c|s|cmd)$|^-[A-Za-z]*c$"),
}
_INTERP_VALUE_FLAGS = frozenset({"-W", "-X", "-I", "-o", "--check-hash-based-pycs"})
_NUMERIC = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")
# launchers that run the words after them as a command; every later word may be the program
TRANSPARENT = frozenset({"watch", "strace", "ltrace", "flock", "taskset", "chrt", "unbuffer", "numactl", "parallel",
                         "busybox", "script", "firejail", "bwrap", "chroot", "runuser", "su", "ssh", "setpriv", "unshare"})
# package runners (`uv run [flags] PROGRAM ...`): the program is the first word after the run verb and its flags
RUNNERS = frozenset({"uv", "poetry", "pipx", "conda", "mamba", "micromamba", "pdm", "hatch", "rye", "npx", "pnpm", "yarn", "bunx"})
_RUN_VERBS = frozenset({"run", "exec", "x", "dlx"})
_RUNNER_VALUE_FLAGS = frozenset({"--with", "--with-requirements", "--python", "-p", "--directory", "--project", "--env-file",
                                 "--group", "--extra", "--package", "--with-editable", "--index", "--index-url", "-n", "--name",
                                 "--prefix", "--channel"})
_EXEC_FLAGS = ("-exec", "-execdir", "-ok", "-okdir")
_STDOUT_SINKS = ("/dev/stdout", "/dev/fd/1", "/dev/tty", "/proc/self/fd/1")
_TYPES: dict[str, tuple[str, ...]] = {
    "py": (".py", ".pyi"), "js": (".js", ".jsx", ".mjs"), "ts": (".ts", ".tsx"), "md": (".md", ".markdown"),
    "json": (".json",), "yaml": (".yaml", ".yml"), "toml": (".toml",), "rust": (".rs",), "go": (".go",),
    "java": (".java",), "c": (".c", ".h"), "cpp": (".cpp", ".hpp", ".cc", ".h"), "sh": (".sh", ".bash"),
    "html": (".html", ".htm"), "css": (".css",), "txt": (".txt",), "csv": (".csv",), "sql": (".sql",),
    "rb": (".rb",), "php": (".php",),
}
_PEEL = re.compile(r"[$<>]?\(([^()]*)\)|`([^`]*)`")
_REDIRECT_IN = re.compile(r"(?<![<&])<(?![<(&])\s*([^\s;|&<>()]+)")
_ASSIGN = re.compile(r"^([A-Za-z_]\w*)=(.*)$", re.DOTALL)
_VAR = re.compile(r"\$\{([^}]*)\}|\$([A-Za-z_]\w*|[0-9@*#?!$-])")
_BRACE = re.compile(r"([^\s{}]*)\{([^{}\s]*,[^{}\s]*)\}([^\s{}]*)")
_DECLARE = frozenset({"export", "declare", "typeset", "local", "readonly", "set", "env"})
_NAMES_ONLY = re.compile(r"^-[A-Za-z]*[lLcq][A-Za-z]*$")
_NAMES_ONLY_LONG = frozenset({"--files", "--files-with-matches", "--files-without-match", "--count", "--quiet",
                              "--count-matches"})
_GLOB_FLAGS = ("-g", "--glob", "--include", "--iglob")
_TYPE_FLAGS = ("-t", "--type")
_RECURSIVE = re.compile(r"^-[A-Za-z]*[rR][A-Za-z]*$|^--recursive$|^--dereference-recursive$")


def _brace_expand(word: str) -> list[str]:
    words = [word]
    for _ in range(6):
        nxt: list[str] = []
        changed = False
        for w in words:
            m = _BRACE.search(w)
            if m is None:
                nxt.append(w)
                continue
            changed = True
            nxt += [w[:m.start()] + m[1] + alt + m[3] + w[m.end():] for alt in m[2].split(",")]
        words = nxt[:MAX_BRACES]
        if not changed:
            break
    return words


_WILDS = ("star", "any", "dirstar")


class _Glob:
    """A glob compiled to a small token list and matched in O(len(text) x len(tokens)): no backtracking, so adjacent
    wildcards (``$A$B$C...``) cannot make the check run into the hook timeout."""

    def __init__(self, tokens: list[tuple[str, object]]):
        self.tokens = tokens

    def _close(self, states: set[int]) -> set[int]:
        todo = list(states)
        while todo:
            i = todo.pop()
            if i < len(self.tokens) and self.tokens[i][0] in _WILDS and i + 1 not in states:
                states.add(i + 1)
                todo.append(i + 1)
        return states

    def fullmatch(self, text: str) -> bool:
        states = self._close({0})
        for ch in text:
            nxt: set[int] = set()
            for i in states:
                if i >= len(self.tokens):
                    continue
                kind, arg = self.tokens[i]
                if kind == "lit":
                    if ch == arg:
                        nxt.add(i + 1)
                elif kind == "one":
                    if ch != "/":
                        nxt.add(i + 1)
                elif kind == "cls":
                    if arg.fullmatch(ch):  # type: ignore[union-attr]
                        nxt.add(i + 1)
                elif kind == "star":
                    if ch != "/":
                        nxt.add(i)
                elif kind == "any":
                    nxt.add(i)
                elif kind == "dirstar":  # (?:.*/)?
                    nxt.add(i)
                    if ch == "/":
                        nxt.add(i + 1)
            if not nxt:
                return False
            states = self._close(nxt)
        return len(self.tokens) in states


def _glob_rx(pattern: str) -> _Glob:
    tokens: list[tuple[str, object]] = []

    def add(kind: str, arg: object = None) -> None:
        if kind in _WILDS and tokens and tokens[-1][0] in _WILDS:  # collapse a run of wildcards into the widest one
            prev = tokens[-1][0]
            if "any" in (kind, prev):
                tokens[-1] = ("any", None)
                return
            if kind == prev:
                return
        tokens.append((kind, arg))

    i = 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            add("dirstar")
            i += 3
            continue
        if pattern.startswith("**", i):
            add("any")
            i += 2
            continue
        if c == WILD:
            add("any")
        elif c == "*":
            add("star")
        elif c == "?":
            add("one")
        elif c == "[" and (j := pattern.find("]", i + 2)) > 0:
            body = pattern[i + 1:j]
            body = "^" + body[1:] if body[:1] in ("!", "^") else body
            try:
                add("cls", re.compile("[" + body.replace("\\", "\\\\") + "]", re.DOTALL))
                i = j
            except re.error:
                add("lit", c)
        else:
            add("lit", c)
        i += 1
    return _Glob(tokens)


def _search_filters(words: list[str]) -> tuple[list[str], list[str]]:
    globs: list[str] = []
    types: list[str] = []
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else ""
        for flag in _GLOB_FLAGS:
            if w == flag:
                globs.append(nxt)
            elif w.startswith(flag + "="):
                globs.append(w.split("=", 1)[1])
            elif flag == "-g" and w.startswith("-g") and len(w) > 2 and not w.startswith("--"):
                globs.append(w[2:])
        for flag in _TYPE_FLAGS:
            if w == flag:
                types.append(nxt)
            elif w.startswith(flag + "="):
                types.append(w.split("=", 1)[1])
    return globs, types


def _peel(command: str) -> list[str]:
    """The command with every ``$(...)``, backtick, ``<(...)`` and subshell body cut out (left as WILD), and the bodies
    themselves, innermost first, as pieces of their own."""
    pieces: list[str] = []
    text = command
    for _ in range(40):
        bodies: list[str] = []

        def cut(m: re.Match[str]) -> str:
            bodies.append(m[1] if m[1] is not None else m[2])
            return WILD

        new = _PEEL.sub(cut, text)
        pieces += bodies
        if new == text:
            break
        text = new
    return [text, *pieces]


def _flags_say_names_only(words: list[str]) -> bool:
    return any(_NAMES_ONLY.match(w) or w in _NAMES_ONLY_LONG for w in words if w.startswith("-"))


def _assignments(cmds: list[Cmd]) -> tuple[dict[str, str], set[str]]:
    variables: dict[str, str] = {}
    aliases: set[str] = set()
    for c in cmds:
        words = c.words
        if words and words[0] == "alias":
            for w in words[1:]:
                m = _ASSIGN.match(w)
                if m and base(m[2].split()[0] if m[2].split() else "") in READERS:
                    aliases.add(m[1])
            continue
        i = 0
        while i < len(words) and (words[i] in _DECLARE or _ASSIGN.match(words[i])):
            m = _ASSIGN.match(words[i])
            if m:
                variables[m[1]] = m[2]
            i += 1
    return variables, aliases


def _clean_text(command: str) -> str:
    return command.replace("\\", "").replace("'", "").replace('"', "")


def _family(word: str) -> str | None:
    """The interpreter family of a program word (``python3.12`` -> python), or None."""
    name = base(word)
    if not INTERPRETER.match(name):
        return None
    fam = re.sub(r"[\d.]+$", "", name).lower()
    return "python" if fam == "pypy" else fam


def _inline_code(cmds: list[Cmd]) -> bool:
    """Whether an interpreter runs code that is part of this command (its real inline flag, a heredoc that feeds it, or
    stdin) rather than a script file. Only the words before the script are the interpreter's own flags."""
    for c in cmds:
        for i, w in enumerate(c.words):
            fam = _family(w)
            if fam is None:
                continue
            if c.heredoc:
                return True
            flag = _INLINE_FLAGS.get(fam)
            rest = c.words[i + 1:]
            for k, a in enumerate(rest):
                if a == "-m" or a == "--":
                    break  # a module or the end of the options: the rest belongs to the program run
                if a.startswith("-"):
                    if a == "-" or (flag is not None and flag.match(a)):
                        return True
                    continue
                if k > 0 and rest[k - 1] in _INTERP_VALUE_FLAGS:
                    continue
                break  # the script file: later flags are the script's own
            else:
                return True  # no script file: it reads its code from stdin
    return False


def _skip_flags(words: list[str], i: int, value_flags: frozenset[str]) -> tuple[int, bool]:
    """Index of the first non-flag word from ``i`` and whether an option of unknown arity was passed on the way."""
    unsure = False
    while i < len(words):
        w = words[i]
        if w.startswith("-") and len(w) > 1:
            if w in value_flags:
                i += 2
            else:
                unsure = unsure or "=" not in w
                i += 1
        elif _NUMERIC.match(w):
            unsure = True
            i += 1
        else:
            break
    return i, unsure


def _program_words(words: list[str]) -> list[int]:
    """Indexes of the words that can be the program a simple command runs: the program itself (after assignments and
    wrappers), the word after an option of unknown arity or a duration (``timeout 5 cat``, ``ionice -c 2 cat``), what a
    launcher or package runner starts, and what ``find -exec`` / ``xargs`` run. Arguments are never candidates."""
    start = program_index(words)
    if start >= len(words):
        return []
    first = base(words[start])
    out: list[int] = [start]
    wrapper = next((k for k in range(start) if base(words[k]) in WRAPPERS), None)
    if wrapper is not None:  # an option of unknown arity (`sudo -n cat`) may have swallowed the program: look back too
        out += [k for k in range(wrapper + 1, start) if not words[k].startswith("-")]
    if first in TRANSPARENT:
        out += range(start, len(words))
    else:
        idx, unsure = start, False
        verb = next((j for j in range(start + 1, len(words)) if words[j] in _RUN_VERBS), None) if first in RUNNERS else None
        if first in RUNNERS:  # `uv run [flags] PROG`, `npx [flags] PROG`
            idx, unsure = _skip_flags(words, (verb if verb is not None else start) + 1, _RUNNER_VALUE_FLAGS)
            out.append(idx)
        elif _NUMERIC.match(words[start]):
            idx, unsure = _skip_flags(words, start, frozenset())
            out.append(idx)
        elif start > 0 and words[start - 1].startswith("-"):
            unsure = True  # the program word may be the value of a wrapper option (`exec -a NAME cat`)
        if unsure and idx + 1 < len(words):
            out.append(idx + 1)
    for k, w in enumerate(words):
        if w in _EXEC_FLAGS and k + 1 < len(words):
            out.append(k + 1)
    return sorted({k for k in out if k < len(words)})


def _unknown_dir(target: str | None) -> bool:
    return target is None or target == "-" or WILD in target
