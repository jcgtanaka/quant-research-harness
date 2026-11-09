"""PreToolUse (Bash, file tools, MCP tools): keep the agent away from the approval and integrity state.

The approval signatures (``qrh.signing``, ``qrh.approvals``) are what stops an agent from approving itself: it has
no passphrase. This hook is defense in depth on top of that, with no override of its own for path rules:

 * commands that run ``qrh approve`` (also ``--init``), ``qrh config sign``, ``qrh readiness ack``,
   ``qrh interview answer --human``, ``qrh data confirm`` or ``qrh data propose --as-user``,
   import the approval code, or set ``QRH_OVERRIDE``;
 * F1: any path at or below ``<QRH_HOME>/keys`` or ``<QRH_HOME>/approvals``, ``<QRH_HOME>`` itself, or ANY ancestor of
   it, for reads and writes alike (Bash tokens after ``$VAR``/``~``/brace/glob expansion, ``cd`` tracking and symlink
   resolution; Read/Grep/Glob paths with patterns resolved against the tool's ``path``); tree-walking commands whose
   working directory is such an ancestor;
 * F3: writes to ``<QRH_HOME>/{config.yaml,registry.sqlite3*,parity,runaway,packs.json,sessions,approvals,keys,readiness,
   llm_cache}`` (the recorded model answers: reading them stays allowed, writing a fabricated answer does not)
   and to Claude Code settings (``~/.claude/settings*.json``, ``<cwd>/.claude/settings*.json``,
   ``<cwd>/.claude/hooks``), and any sqlite3 or interpreter access to the registry; reading config.yaml and logs
   stays allowed;
 * F8: MCP tools whose input names any of those paths;
 * F2 (best-effort stopgap, approvable): file-tool content that imports the approval, signing or interview-writing
   code, names the key files, opens the registry, or builds those names dynamically;
 * state names in any spelling: an interpreter or shell command that names ``registry.sqlite3``, or names
   ``config.yaml``/``parity``/``sessions``/``keys``/``approvals``/``readiness``/``llm_cache`` next to QRH_HOME (concatenated, an f-string, an
   environment variable) is blocked for good, like a write to config.yaml;
 * ``file:`` URLs (WebFetch, MCP tools, curl): blocked for good unless they point into the working tree and not at a
   protected path;
 * approvable blocks (a human decides): writes to Python start-up code (``sitecustomize.py``, ``usercustomize.py``,
   ``*.pth``, anything under ``site-packages``) and to systemd user units; ``pip install -e``, installs from a path,
   URL or alternate index, ``uv tool install --from`` and ``PYTHONPATH=`` on a command that runs qrh; ``crontab``,
   ``at``, ``systemctl --user`` unit changes.

It is a text and path filter, not a sandbox: the agent runs as the same OS user, and a sufficiently obfuscated
program can still reach the files. What actually closes self-approval is that signing needs the human's
passphrase (see docs/security.md). Fail closed: anything this hook cannot parse or check blocks.
"""

from __future__ import annotations

import glob as globmod
import itertools
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from qrh.hooks.common import _glob_regex
from qrh.hooks.paths import is_under, under_any
from qrh.hooks.runner import HookContext, HookResult
from qrh.hooks.shellparse import base, parse, write_targets
from qrh.hooks.approval_guard_text import (  # noqa: F401  (re-exported: the old names stay importable from here)
    _QUOTES,
    _SPLIT,
    _SEGMENTS,
    _BRACES,
    _REDIRECT,
    _WORD,
    _QRH,
    _APPROVE,
    _CONFIG,
    _SIGN,
    _READINESS,
    _ACK,
    _QRH_MODULE,
    _OVERRIDE_SET,
    _INTERVIEW,
    _ANSWER,
    _HUMAN_FLAG,
    _DATA,
    _PROPOSE,
    _CONFIRM,
    _AS_USER,
    _ONBOARDING_MODULE,
    _INTERVIEW_MODULE,
    _HOOK_RUN,
    _STATE_WORD,
    _GLOB_CHARS,
    CONTENT_RULES,
    _STATE_NAME,
    _HOME_REF,
    _SQUASH,
    _STARTUP_NAMES,
    _PERSISTENCE,
    _STARTUP_WORD,
    _PIP,
    _SKIP_VALUE,
    _SOURCE_FLAGS,
    _URL_SCHEME,
    _ARCHIVE,
    _SYSTEMCTL_CHANGES,
    _expand_vars,
    _brace_expand,
    _command_problems,
    _base,
    _strings,
    state_reference,
    _VENV_LIB,
    _site_dirs,
    in_site_directory,
    startup_path,
    _risky_install_args,
    _install_risky,
    _schedules,
)

WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
PATH_KEYS = ("file_path", "notebook_path", "path")
KEY_FILES = ("approval.key", "approval.key.enc", "approval.pub", "config-baseline.json", "unlock-failures.json")
WRITE_FILES = ("config.yaml", "packs.json")
# readiness: signed acknowledgements; llm_cache: recorded model answers (a fabricated answer would be replayed as real)
WRITE_DIRS = ("parity", "runaway", "sessions", "keys", "approvals", "readiness", "llm_cache")
MAX_GLOB_HITS = 500


WALKERS = {"rg", "grep", "egrep", "fgrep", "find", "tar", "zip", "rsync", "du", "tree", "fd", "ag", "ack", "cp", "scp",
           "rm", "chmod", "chown", "ls", "7z", "cpio", "restic", "borg"}
RECURSIVE_ONLY = {"grep", "egrep", "fgrep", "cp", "scp", "rm", "chmod", "chown", "ls"}  # walk only with -r/-R/-a
WRITE_WORDS = {"rm", "rmdir", "mv", "ln", "chmod", "chown", "chgrp", "truncate", "tee", "dd", "install", "unlink",
               "shred", "touch", "mkdir", "sqlite3", "patch", "rsync", "cp", "unzip", "tar", "git", "curl", "wget",
               "setfacl", "chattr", "xattr"}
INTERPRETERS = re.compile(r"^(?:python[\d.]*|pypy[\d.]*|perl|ruby|node|deno|bun|php|lua|Rscript|julia|osascript|"
                          r"bash|sh|zsh|dash|ksh|fish|uv|pip|ipython|jupyter)$")
INPLACE = re.compile(r"(?:^|\s)-[a-zA-Z]*i|--in-place")


@dataclass
class Protected:
    """Every protected location, in normalized and symlink-resolved forms."""

    home: list[str]
    secrets: list[str]
    ancestors: set[str]
    write_files: list[str]
    write_dirs: list[str]
    registry: list[str]
    settings_dirs: list[str]
    hooks_dirs: list[str]
    candidates: dict[str, list[str]] = field(default_factory=dict)  # concrete paths to match glob patterns against


def _forms(p: str | Path) -> list[str]:
    out: list[str] = []
    for f in (os.path.normpath(str(p)), os.path.realpath(p)):
        if f not in out:
            out.append(f)
    return out


def protected(home: Path, cwd: str) -> Protected:
    user = os.path.expanduser("~")
    homes = _forms(home)
    ancestors: set[str] = set()
    for h in homes:
        parent = os.path.dirname(h)
        while True:
            ancestors.add(parent)
            nxt = os.path.dirname(parent)
            if nxt == parent:
                break
            parent = nxt
    secrets = [f for name in ("keys", "approvals") for f in _forms(home / name)]
    write_files = [f for name in WRITE_FILES for f in _forms(home / name)]
    write_dirs = [f for name in WRITE_DIRS for f in _forms(home / name)]
    registry = _forms(home / "registry.sqlite3")
    settings_dirs = _forms(os.path.join(user, ".claude"))
    hooks_dirs: list[str] = _forms(os.path.join(user, ".claude", "projects"))  # transcripts (interview provenance)
    if cwd:
        settings_dirs += [f for f in _forms(os.path.join(cwd, ".claude")) if f not in settings_dirs]
        hooks_dirs += _forms(os.path.join(cwd, ".claude", "hooks"))
    cands = {
        "secret": [os.path.join(s, n) for s in secrets for n in (*KEY_FILES, "0123456789ab.json")] + secrets,
        "home": homes, "ancestor": sorted(ancestors),
        "write": write_files + [os.path.join(d, "x") for d in write_dirs]
        + [os.path.join(d, n) for d in settings_dirs for n in ("settings.json", "settings.local.json")]
        + [os.path.join(d, "x") for d in hooks_dirs],
        "registry": registry + [r + "-wal" for r in registry],
    }
    return Protected(homes, secrets, ancestors, write_files, write_dirs, registry, settings_dirs, hooks_dirs, cands)


def _under(path: str, roots: list[str]) -> bool:
    return under_any(path, roots)


def _classify_concrete(path: str, prot: Protected) -> str | None:
    forms = _forms(path)
    for p in forms:
        if _under(p, prot.secrets):
            return "secret"
    for p in forms:
        if p in prot.home:
            return "home"
        if p in prot.ancestors:
            return "ancestor"
    for p in forms:
        if any(p.startswith(r) for r in prot.registry):
            return "registry"
        if p in prot.write_files or _under(p, prot.write_dirs) or _under(p, prot.hooks_dirs):
            return "write"
        if os.path.dirname(p) in prot.settings_dirs and re.fullmatch(r"settings.*\.json", os.path.basename(p)):
            return "write"
    return None


_RANK = {"secret": 4, "home": 3, "ancestor": 3, "registry": 2, "write": 1}


def _worst(classes: list[str | None]) -> str | None:
    found = [c for c in classes if c]
    return max(found, key=lambda c: _RANK[c]) if found else None


def _glob_class(pattern: str, prot: Protected, fs: bool = True) -> str | None:
    """A glob may name a protected path: match it against the protected candidates and (Bash) the filesystem."""
    classes: list[str | None] = []
    prefix = re.split(r"[*?\[]", pattern, maxsplit=1)[0]
    if prefix.rstrip(os.sep) and _under(os.path.normpath(prefix), prot.secrets):
        classes.append("secret")
    try:
        rx = re.compile(_glob_regex(pattern))
    except re.error:
        return "secret"  # cannot check: fail closed
    for kind, paths in prot.candidates.items():
        if any(rx.fullmatch(c) for c in paths):
            classes.append(kind)
    if fs:
        try:
            for hit in itertools.islice(globmod.iglob(pattern, include_hidden=True), MAX_GLOB_HITS):
                classes.append(_classify_concrete(os.path.normpath(hit), prot))
        except (OSError, ValueError):
            pass
    return _worst(classes)


def _full(raw: str, cwd: str, home: Path) -> str:
    text = _expand_vars(raw, home)
    base = cwd or os.getcwd()
    return os.path.normpath(text if os.path.isabs(text) else os.path.join(base, text))


def classify(raw: str, cwd: str, home: Path, prot: Protected, fs_glob: bool = True) -> str | None:
    """secret | home | ancestor | registry | write | None for a path as the agent wrote it."""
    full = _full(raw, cwd, home)
    if any(c in full for c in _GLOB_CHARS):
        return _glob_class(full, prot, fs_glob)
    return _classify_concrete(full, prot)


def _is_walker(words: list[str]) -> bool:
    names = [_base(w) for w in words]
    flags = " ".join(w for w in words if w.startswith("-"))
    recursive = re.search(r"(?:^|\s)-[a-zA-Z]*[rRa]|--recursive|--archive", flags) is not None
    if any(n in WALKERS and (n not in RECURSIVE_ONLY or recursive) for n in names):
        return True
    return "git" in names and any(x in names for x in ("grep", "add", "archive", "ls-files"))


def _is_writer(words: list[str]) -> bool:
    names = [_base(w) for w in words]
    if any(n in WRITE_WORDS or INTERPRETERS.match(n) for n in names):
        return True
    joined = " ".join(words)
    return any(n in ("sed", "perl", "awk", "gawk") for n in names) and bool(INPLACE.search(joined) or "inplace" in joined)


def _copy_dest(words: list[str], cur: str, home: Path) -> str | None:
    """For cp/install/rsync/scp: the destination (only it is written); None for other commands."""
    if not words or _base(words[0]) not in ("cp", "install", "rsync", "scp"):
        return None
    pos = [w for w in words[1:] if not w.startswith("-")]
    return _full(pos[-1], cur, home) if pos else None


def bash_problems(command: str, cwd: str, home: Path) -> list[str]:
    plain = _QUOTES.sub("", command)
    problems = _command_problems(plain)
    expanded = _brace_expand(_expand_vars(plain, home))
    reasons: set[str] = set()
    cur = cwd or os.getcwd()
    prot = protected(home, cur)
    if _classify_concrete(os.path.normpath(cur), prot) == "secret":
        reasons.add("secret")
    if any(k in expanded for k in KEY_FILES):
        reasons.add("secret")
    # an interpreter or sqlite3 anywhere in the command can write any path it names, even inside its own code
    program = any(w == "sqlite3" or INTERPRETERS.match(w) for w in (_base(x) for x in expanded.split()))
    if program and state_reference(command):
        reasons.add("state")
    for segment in _SEGMENTS.split(expanded):
        words = [w for w in segment.split() if w]
        if not words:
            continue
        if _base(words[0]) in ("cd", "pushd") and len(words) > 1:
            target = _full(words[1], cur, home)
            kind = _classify_concrete(target, prot)
            if kind in ("secret", "home", "ancestor"):
                reasons.add(kind)
            cur = target
            continue
        writer = program or _is_writer(words)
        redirects = {_full(t, cur, home) for t in _REDIRECT.findall(segment)}
        dest = _copy_dest(words, cur, home)
        db_tool = program
        for tok in (t for t in _SPLIT.split(segment) if t):
            kind = classify(tok, cur, home, prot)
            if kind in ("secret", "home", "ancestor"):
                reasons.add(kind)
            elif kind in ("write", "registry"):
                full = _full(tok, cur, home)
                if full in redirects or (kind == "registry" and db_tool):
                    reasons.add(kind)
                elif writer and (dest is None or full == dest):
                    reasons.add(kind)
        here = _classify_concrete(os.path.normpath(cur), prot)
        if here in ("home", "ancestor") and _is_walker(words):
            reasons.add("ancestor")  # a recursive walk from QRH_HOME or above reaches the keys
    if not reasons and (str(home) in expanded or "QRH_HOME" in plain or ".qrh" in plain):
        # e.g. `D=$QRH_HOME; cat $D/keys/*`: a reference we cannot resolve, next to a state directory name
        if _STATE_WORD.search(expanded):
            reasons.add("secret")
    return problems + _path_messages(reasons)


def _path_messages(reasons: set[str]) -> list[str]:
    out = []
    if reasons & {"secret", "home", "ancestor"}:
        out.append("the approval keys and requests, QRH_HOME and every directory above it are off limits to the agent "
                   "(reads included)")
    if "state" in reasons:
        out.append("an interpreter or shell command may not name the qrh registry, config, parity, sessions, keys or "
                   "approvals, in any spelling (concatenated, an f-string, an environment variable)")
    if "registry" in reasons:
        out.append("the qrh registry may only be changed through qrh commands, not sqlite3 or a script")
    if "write" in reasons:
        out.append("qrh config, state and Claude Code hook settings may only be changed by a human")
    return out


def _tool_paths(tool: str, ti: dict[str, Any], cwd: str) -> tuple[list[str], list[str]]:
    """(concrete paths, glob patterns) a file tool touches; Grep/Glob patterns resolve against its ``path``."""
    paths: list[str] = []
    patterns: list[str] = []
    for key in PATH_KEYS:
        v = ti.get(key)
        if v is None:
            continue
        if not isinstance(v, str):
            raise TypeError(f"cannot check {key}: not a string")
        if v:
            paths.append(v)
    if tool not in ("Grep", "Glob"):
        return paths, patterns
    base = ti.get("path") if isinstance(ti.get("path"), str) and ti.get("path") else (cwd or os.getcwd())
    for key in ("pattern", "glob"):
        v = ti.get(key)
        if v is None:
            continue
        if not isinstance(v, str):
            raise TypeError(f"cannot check {key}: not a string")
        if tool == "Grep" and key == "pattern":
            continue  # a regular expression over file contents, not a path
        full = v if os.path.isabs(v) else os.path.join(base, v)
        (patterns if any(c in full for c in _GLOB_CHARS) else paths).append(full)
        if not os.path.isabs(v):
            patterns.append(os.path.join(base, "**", v))  # Grep globs and Glob patterns also match deeper
    return paths, patterns


def _content(ti: dict[str, Any]) -> str:
    parts = [ti.get(k) for k in ("content", "new_string", "new_source")]
    edits = ti.get("edits")
    if isinstance(edits, list):
        parts += [e.get("new_string") for e in edits if isinstance(e, dict)]
    return "\n".join(p for p in parts if isinstance(p, str))


def content_problems(text: str, home: Path) -> list[str]:
    found = [why for rx, why in CONTENT_RULES if rx.search(text)]
    if (str(home) in text or "QRH_HOME" in text or ".qrh" in text) and _STATE_WORD.search(text):
        found.append("names QRH_HOME next to the keys or approvals")
    return sorted(set(found))


def file_tool_problems(tool: str, tool_input: dict[str, Any], cwd: str, home: Path) -> list[str]:
    here = cwd or os.getcwd()
    prot = protected(home, here)
    try:
        paths, patterns = _tool_paths(tool, tool_input, here)
    except TypeError as exc:
        return [str(exc)]
    kinds = [classify(p, here, home, prot, fs_glob=False) for p in paths]
    kinds += [_glob_class(os.path.normpath(p), prot, fs=False) for p in patterns]
    if tool in ("Grep", "Glob"):
        base = tool_input.get("path") or here
        if _classify_concrete(_full(str(base), here, home), prot) in ("home", "ancestor"):
            kinds.append("ancestor")  # a search rooted at QRH_HOME or above walks into the keys
    reasons = {k for k in kinds if k in ("secret", "home", "ancestor")}
    if tool in WRITE_TOOLS:
        reasons |= {k for k in kinds if k in ("write", "registry")}
        reasons = {"write" if k == "registry" else k for k in reasons}
    return _path_messages(reasons)


def mcp_problems(tool_input: dict[str, Any], cwd: str, home: Path) -> list[str]:
    """Any string in an MCP tool's input that names a protected path blocks (read or write is unknowable)."""
    here = cwd or os.getcwd()
    prot = protected(home, here)
    reasons: set[str] = set()
    for s in _strings(tool_input):
        text = _expand_vars(s.strip().replace("file://", ""), home)
        if any(k in text for k in KEY_FILES):
            reasons.add("secret")
        for tok in _SPLIT.split(text):
            if tok and (tok.startswith(("/", "~", ".")) or "/" in tok):
                kind = classify(tok, here, home, prot, fs_glob=False)
                if kind:
                    reasons.add("write" if kind == "registry" else kind)
    return _path_messages(reasons)


# --- file: URLs -------------------------------------------------------------------------------------------

_FILE_URL = re.compile(r"""\bfile:[^\s'"<>)]*""", re.IGNORECASE)
_URL_MESSAGE = ("file: URLs may only point inside the working directory and never at qrh state, keys or other "
                "protected paths")


def file_url_blocked(url: str, cwd: str, home: Path) -> bool:
    parts = urlsplit(url.strip())
    if parts.scheme.lower() != "file":
        return False
    if parts.netloc.lower() not in ("", "localhost"):
        return True
    path = unquote(parts.path)
    if not cwd or not path or "\x00" in path or not os.path.isabs(path):
        return True
    root = os.path.realpath(cwd)
    if any(not is_under(f, root) for f in _forms(path)):
        return True
    return _classify_concrete(path, protected(home, cwd)) is not None or any(k in path for k in KEY_FILES)


def url_problems(tool: str, ti: dict[str, Any], command: str, cwd: str, home: Path) -> list[str]:
    urls = _FILE_URL.findall(command) if tool == "Bash" else [s for s in _strings(ti) if s.lstrip()[:5].lower() == "file:"]
    return [_URL_MESSAGE] if any(file_url_blocked(u, cwd, home) for u in urls) else []


# --- approvable blocks: start-up code, installs from places the agent controls, scheduling -------------------

STARTUP_MESSAGE = ("writes Python start-up code (sitecustomize.py, usercustomize.py, a .pth file, anything under "
                   "site-packages) or a systemd user unit; it runs in every later interpreter, hooks included")
INSTALL_MESSAGE = ("installs from a path, URL, editable checkout or alternate index, or sets PYTHONPATH for a qrh "
                   "command; that can replace the code the hooks run")
SCHEDULE_MESSAGE = "schedules commands that run later, outside this session (crontab, at, systemctl --user)"


def bash_approvable(command: str, cwd: str, home: Path) -> list[str]:
    cur = cwd or os.getcwd()
    found: set[str] = set()
    cmds = parse(command)
    all_words = [w for c in cmds for w in c.words]
    for cmd in cmds:
        w = cmd.words
        if w and base(w[0]) in ("cd", "pushd") and len(w) > 1:
            cur = _full(w[1], cur, home)
            continue
        if any(startup_path(_full(t, cur, home)) for t in write_targets(cmd)):
            found.add(STARTUP_MESSAGE)
        if _install_risky(w):
            found.add(INSTALL_MESSAGE)
        if _schedules(w):
            found.add(SCHEDULE_MESSAGE)
    runs_qrh = any(x == "qrh" or x.endswith("/bin/qrh") for x in all_words)
    if runs_qrh and any(x.startswith("PYTHONPATH=") for x in all_words):
        found.add(INSTALL_MESSAGE)
    squeezed = _SQUASH.sub("", command)
    if any(INTERPRETERS.match(base(x)) for x in all_words) and (
            _STARTUP_WORD.search(command) or ("site-packages" in squeezed and ".pth" in squeezed)):
        found.add(STARTUP_MESSAGE)
    return sorted(found)


def file_tool_approvable(tool: str, ti: dict[str, Any], cwd: str, home: Path) -> list[str]:
    if tool not in WRITE_TOOLS:
        return []
    here = cwd or os.getcwd()
    try:
        paths, _ = _tool_paths(tool, ti, here)
    except TypeError:
        return []  # the final checks already report it
    return [STARTUP_MESSAGE] if any(startup_path(_full(p, here, home)) for p in paths) else []


F2_NOTE = (
    "This content check is a best-effort stopgap: a determined program can hide what it imports. What actually "
    "prevents self-approval is that signing needs the human's passphrase."
)


def run(ctx: HookContext) -> HookResult:
    p = ctx.payload
    if p.event and p.event != "PreToolUse":
        return HookResult("allow", "not a PreToolUse event")
    home = Path(os.path.abspath(ctx.home))
    ti = dict(p.tool_input)
    approvable: list[str] = []
    if p.tool_name == "Bash":
        problems = bash_problems(p.command, p.cwd, home) if p.command else []
        problems += url_problems("Bash", ti, p.command, p.cwd, home) if p.command else []
        approvable = bash_approvable(p.command, p.cwd, home) if p.command else []
    elif p.tool_name.startswith("mcp__"):
        problems = mcp_problems(ti, p.cwd, home) + url_problems(p.tool_name, ti, "", p.cwd, home)
    else:
        problems = file_tool_problems(p.tool_name, ti, p.cwd, home) + url_problems(p.tool_name, ti, "", p.cwd, home)
        approvable = file_tool_approvable(p.tool_name, ti, p.cwd, home)
    if problems:
        return HookResult(
            "block", "; ".join(problems),
            stderr="; ".join(problems) + ".\nOnly a human can approve a blocked action, sign config or change these "
            "files, in their own terminal. Do not look for another way: tell the user what you need and wait.",
        )
    if p.tool_name in WRITE_TOOLS:
        text = _content(ti)
    elif p.tool_name.startswith("mcp__"):
        text = "\n".join(_strings(ti))
    else:
        text = ""
    found = content_problems(text, home) if text else []
    if found:
        return HookResult(
            "block", "content reaches qrh approval state: " + "; ".join(found), approvable=True,
            stderr="The content you are writing " + "; ".join(found) + ". A helper that touches the approval, signing "
            "or interview-writing code, the key files or the registry needs a human decision.\n" + F2_NOTE,
        )
    if approvable:
        return HookResult(
            "block", "needs a human decision: " + "; ".join(approvable), approvable=True,
            stderr="This action " + "; ".join(approvable) + ". It needs a human decision.",
        )
    return HookResult("allow", "no approval-state access")
