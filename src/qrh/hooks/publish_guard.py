"""publish-guard: PreToolUse (Bash, Write, Edit, MultiEdit). Outward-facing and publication-weakening actions in
protected repositories need a human decision (the approval flow and banner).

Configured under ``publish_guard: {protected_repos: [absolute paths]}`` in ``<QRH_HOME>/config.yaml``. There are no
path fallbacks: without the section the hook is disabled and session-context says so. A repository is protected
when its real path equals or is under a listed path (symlinks resolved, whole path components compared).

Gated (in a protected repository):

* publishing: ``git push``, ``git remote add|set-url``, ``gh repo create|edit|fork|delete|transfer|rename``,
  ``gh release create|upload``, ``gh pr create|merge``, ``gh api`` writes against repos, ``npm|pnpm|yarn publish``,
  ``twine upload``, ``uv|cargo|poetry|hatch|flit publish``;
* weakening the checks that run before publication: ``git commit --no-verify|-n``, ``git config core.hooksPath``
  (also at ``--global``/``--system`` scope, wherever it runs), ``pre-commit uninstall``, and any file tool or shell
  write, move or delete of ``.git/hooks/*``, ``.git/config``, ``.pre-commit-config.yaml``, ``.gitleaks.toml``,
  ``scripts/denylist_check.py`` and ``.github/workflows/*``.

It is a text filter over the command as written: it follows ``cd``, ``git -C``, ``sh -c`` and heredocs piped to a
shell, but not variables, aliases or code an interpreter runs. A determined program in the agent's own session can
still publish; the point is that an ordinary or careless action cannot.
"""

from __future__ import annotations

import os
import re

from qrh.hooks.paths import is_under, real_path
from qrh.hooks.runner import HookContext, HookResult
from qrh.hooks.shellparse import base, parse, program_index, write_targets

FILE_TOOLS = ("Write", "Edit", "MultiEdit")
GUARD_FILES = (".pre-commit-config.yaml", ".gitleaks.toml", "scripts/denylist_check.py", ".git/config")
GUARD_DIRS = (".git/hooks", ".github/workflows")
GUARD_MESSAGE = "changes a file that guards what gets published ({}): git hooks, git config, pre-commit, gitleaks, " \
                "the denylist check or CI workflows"
_SHORT_NO_VERIFY = re.compile(r"^-[a-zA-Z]*n[a-zA-Z]*$")
_HOOKS_PATH = "core.hookspath"
_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_API_VALUE_FLAGS = frozenset({"-X", "--method", "-H", "--header", "-f", "--raw-field", "-F", "--field", "--input",
                              "-q", "--jq", "-t", "--template", "--hostname", "--cache", "-R", "--repo"})
_API_FIELD_FLAGS = frozenset({"-f", "--raw-field", "-F", "--field", "--input"})
_GH_VALUE_FLAGS = frozenset({"-R", "--repo", "--hostname"})
_PUBLISHERS = {"uv": "publish", "cargo": "publish", "poetry": "publish", "hatch": "publish", "flit": "publish",
               "twine": "upload"}
_GH_GATED = {"repo": {"create", "edit", "fork", "delete", "transfer", "rename"}, "release": {"create", "upload"},
             "pr": {"create", "merge"}}


_real = real_path
_under = is_under


def locate(path: str, listed: list[str]) -> str | None:
    """The repository root when ``path`` is in a protected repository, else None. The root is the nearest ancestor
    holding a ``.git`` (inside the listed path), or the listed path itself."""
    real = _real(path)
    top = next((r for r in listed if _under(real, r)), None)
    if top is None:
        return None
    d = real
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            return d
        parent = os.path.dirname(d)
        if d == top or parent == d or not _under(parent, top):
            return top
        d = parent


def _resolve(raw: str, cur: str) -> str:
    text = raw.replace("${HOME}", "~").replace("$HOME", "~")
    text = os.path.expanduser(text)
    return os.path.normpath(text if os.path.isabs(text) else os.path.join(cur, text))


def guard_file(path: str, listed: list[str]) -> bool:
    """Whether a path is one of the publication guard files of a protected repository."""
    root = locate(os.path.dirname(path) if not os.path.isdir(path) else path, listed)
    if root is None:
        return False
    rel = os.path.relpath(_real(path), root).replace(os.sep, "/")
    return rel in GUARD_FILES or any(rel == d or rel.startswith(d + "/") for d in GUARD_DIRS)


def _git_reasons(args: list[str], cur: str) -> tuple[list[str], str]:
    """(reasons, effective directory) for the words after ``git``."""
    reasons: list[str] = []
    eff, j, sub = cur, 0, ""
    while j < len(args):
        a = args[j]
        if a == "-C" and j + 1 < len(args):
            eff, j = _resolve(args[j + 1], eff), j + 2
        elif a == "-c" and j + 2 <= len(args):
            j += 2
        elif a.startswith("-"):
            j += 1
        else:
            sub = a
            break
    rest = args[j + 1:]
    if sub == "push":
        reasons.append("git push publishes commits")
    elif sub == "remote" and next((a for a in rest if not a.startswith("-")), "") in ("add", "set-url"):
        reasons.append("changing a git remote changes where code is published")
    elif sub == "commit" and any(a == "--no-verify" or (not a.startswith("--") and _SHORT_NO_VERIFY.match(a))
                                 for a in rest):
        reasons.append("git commit --no-verify skips the pre-commit checks")
    return reasons, eff


def _gh_reasons(args: list[str]) -> list[str]:
    words: list[str] = []
    skip = False
    for a in args:
        if skip:
            skip = False
        elif a in _GH_VALUE_FLAGS:
            skip = True
        elif not a.startswith("-"):
            words.append(a)
    if len(words) >= 2 and words[1] in _GH_GATED.get(words[0], ()):
        return [f"gh {words[0]} {words[1]} changes or publishes repository state"]
    if words[:1] == ["api"]:
        method, fields, endpoint, skip = "", False, "", False
        for a in args[args.index("api") + 1:] if "api" in args else []:
            if skip:
                skip = False
                continue
            if a in ("-X", "--method"):
                skip, method = True, "?"
            elif a.startswith("--method="):
                method = a.split("=", 1)[1].upper()
            elif a.startswith("-X") and len(a) > 2:
                method = a[2:].upper()
            elif a in _API_VALUE_FLAGS:
                skip = True
                fields = fields or a in _API_FIELD_FLAGS
            elif not a.startswith("-") and not endpoint:
                endpoint = a.lstrip("/")
        if method == "?":  # `-X VERB`: the verb is the next word
            i = next((k for k, a in enumerate(args) if a in ("-X", "--method")), -1)
            method = args[i + 1].upper() if 0 <= i < len(args) - 1 else ""
        writes = method in _WRITE_METHODS or (not method and fields)
        if writes and (endpoint.startswith("repos/") or "/repos/" in endpoint):
            return ["gh api writes to a repository"]
        if endpoint == "graphql" and any("mutation" in a for a in args):
            return ["gh api graphql mutation changes repository state"]
    return []


def bash_reasons(command: str, cwd: str, listed: list[str]) -> list[str]:
    cur = cwd
    found: list[str] = []

    def hit(directory: str, why: str) -> None:
        if locate(directory, listed) is not None:
            found.append(why)

    for cmd in parse(command):
        w = cmd.words
        if w and base(w[0]) in ("cd", "pushd") and len(w) > 1:
            cur = _resolve(w[1], cur)
            continue
        for t in write_targets(cmd):
            path = _resolve(t, cur)
            if guard_file(path, listed):
                found.append(GUARD_MESSAGE.format(os.path.basename(path)))
        i = program_index(w)
        if i >= len(w):
            continue
        name, args = base(w[i]), w[i + 1:]
        lowered = [a.lower() for a in args]
        if name == "git":
            reasons, eff = _git_reasons(args, cur)
            for why in reasons:
                hit(eff, why)
            if any(_HOOKS_PATH in a for a in lowered):
                if "--global" in args or "--system" in args:
                    found.append("a global git hooksPath replaces the hooks of every repository")
                else:
                    hit(eff, "core.hooksPath redirects the git hooks that check what is committed")
        elif name == "gh":
            for why in _gh_reasons(args):
                hit(cur, why)
        elif name in ("npm", "pnpm", "yarn"):
            if "publish" in args:
                hit(cur, f"{name} publish releases a package")
        elif name in _PUBLISHERS or (name.startswith("python") and "twine" in args):
            rest = args[args.index("twine") + 1:] if name.startswith("python") and "twine" in args else args
            verb = _PUBLISHERS.get(name, "upload")
            if next((a for a in rest if not a.startswith("-")), "") == verb:
                hit(cur, f"{name} {verb} releases a package")
        elif name == "pre-commit" and "uninstall" in args:
            hit(cur, "pre-commit uninstall removes the commit checks")
    return sorted(set(found))


def file_reasons(ti: dict, cwd: str, listed: list[str]) -> list[str]:
    path = ti.get("file_path")
    if not isinstance(path, str) or not path:
        return []
    full = _resolve(path, cwd)
    return [GUARD_MESSAGE.format(os.path.basename(full))] if guard_file(full, listed) else []


def run(ctx: HookContext) -> HookResult:
    p = ctx.payload
    if p.event and p.event != "PreToolUse":
        return HookResult("allow", "not a PreToolUse event")
    cfg = ctx.config.publish_guard
    if cfg is None:
        return HookResult("allow", "publish guard disabled (no publish_guard section in config.yaml)")
    listed = [_real(r) for r in cfg.protected_repos]
    cwd = p.cwd or os.getcwd()
    if p.tool_name == "Bash" and p.command:
        reasons = bash_reasons(p.command, cwd, listed)
    elif p.tool_name in FILE_TOOLS:
        reasons = file_reasons(dict(p.tool_input), cwd, listed)
    else:
        reasons = []
    if reasons:
        return HookResult(
            "block", "; ".join(reasons),
            stderr="This action in a protected repository " + "; ".join(reasons) + ". Publishing, or weakening the "
            "checks that run before publication, needs a human decision.")
    return HookResult("allow", "no publication action in a protected repository")
