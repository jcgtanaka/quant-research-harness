"""Text and command-line helpers of the approval guard: the regular expressions, expansion of ``$VAR``, ``~`` and
braces, detection of approval-related commands, state-name references, and the start-up and install checks. Pure
functions of strings; the path classification and the decisions stay in approval_guard.

Everything here is re-exported from qrh.hooks.approval_guard, so existing imports keep working.
"""

from __future__ import annotations

import os
import posixpath
import re
from pathlib import Path
from typing import Any

from qrh.hooks.shellparse import base, program_index

_QUOTES = re.compile(r"""['"\\]""")
_SPLIT = re.compile(r"""[\s;|&<>(){}=,:`]+""")
_SEGMENTS = re.compile(r"""\n|;|&&|\|\||\||&|\$\(|`|\(|\)|<\(|>\(""")
_BRACES = re.compile(r"([^\s{}]*)\{([^{}\s]*,[^{}\s]*)\}([^\s{}]*)")
_REDIRECT = re.compile(r"(?:^|[^<>])(?:\d*>>?|&>>?|>\|)\s*([^\s;|&<>()]+)")
_WORD = lambda w: re.compile(rf"(?<![\w-]){w}(?![\w-])")  # noqa: E731
_QRH = _WORD("qrh")
_APPROVE = _WORD("approve")
_CONFIG = _WORD("config")
_SIGN = _WORD("sign")
_READINESS = _WORD("readiness")
_ACK = _WORD("ack")
_QRH_MODULE = re.compile(r"qrh[./](?:approvals|signing|human|tty)\b"
                         r"|from\s+qrh\s+import\s+[^;\n]*\b(?:approvals|signing|human|tty)\b"
                         r"|qrh[./]readiness[./]ack\b|from\s+qrh\.readiness\s+import\s+[^;\n]*\back\b"
                         r"|qrh\.cli\b[^\n]*\b(?:approve|sign)\b")
_OVERRIDE_SET = re.compile(
    r"\bQRH_OVERRIDE(?=\s*=)|\b(?:export|declare|typeset|env|set|setenv|readonly)\b[^;&|\n]*\bQRH_OVERRIDE\b")
_INTERVIEW = _WORD("interview")
_ANSWER = _WORD("answer")
_HUMAN_FLAG = re.compile(r"--human\b")
_DATA = _WORD("data")
_PROPOSE = _WORD("propose")
_CONFIRM = _WORD("confirm")
_AS_USER = re.compile(r"--as-user\b")
_ONBOARDING_MODULE = re.compile(r"\bqrh\.onboarding\.(?:store|workflow)\b"
                                r"|from\s+qrh\.onboarding\s+import\s+[^;\n]*\b(?:store|workflow)\b")
_INTERVIEW_MODULE = re.compile(r"\bqrh\.interview\b|from\s+qrh\s+import\s+[^;\n]*\binterview\b")
_HOOK_RUN = re.compile(
    r"(?:^|[;&|\n(`]|\$\(|\b(?:run|exec|env|xargs|sudo|nohup|time|-c|-m))\s*(?:[\w./~-]*/)?qrh(?:\s+-{1,2}[\w-]+)*\s+hook"
    r"(?![\w-])|qrh\.cli\b[^\n]*(?<![\w-])hook(?![\w-])|qrh\.hooks\.runner|run_hook\s*\(", re.MULTILINE)
_STATE_WORD = re.compile(r"(?<![\w.-])(?:keys|approvals)(?![\w.-])")
_GLOB_CHARS = "*?["

CONTENT_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"qrh[./](?:approvals|signing|human|tty)\b"
                r"|from\s+qrh\s+import\s+[^\n]*\b(?:approvals|signing|human|tty)\b"
                r"|qrh[./]readiness[./]ack\b|from\s+qrh\.readiness\s+import\s+[^\n]*\back\b"),
     "imports the approval or signing code"),
    (re.compile(r"qrh[./]interview[./](?:engine|provenance|cli)\b|from\s+qrh\.interview\b|from\s+qrh\s+import\s+[^\n]*"
                r"\binterview\b|\brecord_answer\s*\("), "writes interview answers directly"),
    (re.compile(r"approval\.(?:key|pub)|config-baseline|unlock-failures|[/\\]approvals[/\\]|\bkeys[/\\]approval"),
     "names the approval key or request files"),
    (re.compile(r"(?:importlib|__import__|import_module)[\s\S]{0,300}\bqrh\b|\bqrh\b[\s\S]{0,300}(?:importlib|__import__|"
                r"import_module)"), "imports qrh dynamically"),
    (re.compile(r"""['"](?:appro|approv|signi|q|qr)['"]\s*\+|\+\s*['"](?:vals|rovals|ng|rh|h)['"]"""),
     "builds the name of qrh approval code from pieces"),
    (re.compile(r"registry\.sqlite3"), "opens the qrh registry"),
)

_STATE_NAME = re.compile(r"(?:^|[/\\,(\[{])(?:parity|sessions|keys|approvals|readiness|llm_cache)(?=[/\\,)\]};]|$)")
_HOME_REF = re.compile(r"QRH_HOME|\.qrh\b")
_SQUASH = re.compile(r"""['"\s+\\]""")

_STARTUP_NAMES = frozenset({"sitecustomize.py", "usercustomize.py"})
_PERSISTENCE = ("/.config/systemd/user/", "/etc/systemd/", "/etc/cron", "/var/spool/cron")
_STARTUP_WORD = re.compile(r"(?<![\w])(?:sitecustomize|usercustomize)(?:\.py)?(?![\w])")
_PIP = re.compile(r"^pip[\d.]*$")
_SKIP_VALUE = frozenset({"-r", "--requirement", "-c", "--constraint", "-t", "--target", "--prefix", "--root",
                         "--python", "-p", "--cache-dir", "--python-platform", "--only-binary", "--no-binary",
                         "--platform", "--abi", "--implementation", "-C", "--config-settings", "--group", "--extra",
                         "--optional", "--python-version", "--index-strategy", "--resolution", "--exclude-newer",
                         "--with", "--with-requirements"})
_SOURCE_FLAGS = ("-i", "--index-url", "--extra-index-url", "-f", "--find-links", "--from", "--index",
                 "--default-index", "--extra-index", "--editable")
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_ARCHIVE = (".whl", ".tar.gz", ".tgz", ".zip", ".tar.bz2", ".tar")
_SYSTEMCTL_CHANGES = frozenset({"enable", "reenable", "link", "edit", "daemon-reload", "start", "restart", "reload",
                                "set-environment", "add-wants", "add-requires", "preset", "revert", "mask"})


def _expand_vars(text: str, home: Path) -> str:
    user = os.path.expanduser("~")
    name = os.path.basename(user)
    text = re.sub(r"\$\{QRH_HOME\}|\$QRH_HOME\b", str(home), text)
    text = re.sub(r"\$\{HOME\}|\$HOME\b", user, text)
    text = re.sub(r"\$\{(?:USER|LOGNAME)\}|\$(?:USER|LOGNAME)\b", name, text)
    return re.sub(r"(?<![\w$/.~-])~(?=/|\s|$|[;|&)'\"])", user, text)


def _brace_expand(text: str) -> str:
    for _ in range(6):
        new = _BRACES.sub(lambda m: " ".join(m[1] + alt + m[3] for alt in m[2].split(",")), text)
        if new == text:
            break
        text = new
    return text


def _command_problems(plain: str) -> list[str]:
    problems: list[str] = []
    if _QRH_MODULE.search(plain) or (_QRH.search(plain) and _APPROVE.search(plain)) or \
            (_QRH.search(plain) and _CONFIG.search(plain) and _SIGN.search(plain)) or \
            (_QRH.search(plain) and _READINESS.search(plain) and _ACK.search(plain)):
        problems.append("approvals, keys, the config baseline and readiness acknowledgements can only be handled by a "
                        "human in their own terminal")
    if (_QRH.search(plain) and _INTERVIEW.search(plain) and _ANSWER.search(plain) and _HUMAN_FLAG.search(plain)) \
            or _INTERVIEW_MODULE.search(plain):
        problems.append("`qrh interview answer --human` is for a person at a terminal; the agent must record the "
                        "user's own words with --transcript and --session")
    if _ONBOARDING_MODULE.search(plain) or (_QRH.search(plain) and _DATA.search(plain) and (
            _CONFIRM.search(plain) or (_PROPOSE.search(plain) and _AS_USER.search(plain)))):
        problems.append("`qrh data confirm` and `qrh data propose --as-user` are for a person at a terminal (they record "
                        "the human's word about a dataset); the agent proposes with `qrh data propose` and asks the user "
                        "through `qrh interview`")
    if _HOOK_RUN.search(plain):
        problems.append("hooks are run by Claude Code, not by the agent (`qrh hook ...` by hand could forge session state)")
    if _OVERRIDE_SET.search(plain):
        problems.append("QRH_OVERRIDE is not an override mechanism and the agent may not set it")
    return problems


def _base(word: str) -> str:
    return os.path.basename(word)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings(v)]
    return []


def state_reference(command: str, home: Path | None = None) -> bool:
    """Whether an interpreter command names the registry, or (next to a QRH_HOME reference) config.yaml, parity,
    sessions, keys or approvals, after quotes, ``+`` joins and spaces are squeezed out. Best effort."""
    squeezed = _SQUASH.sub("", command)
    if "registry.sqlite3" in squeezed:
        return True
    if not (_HOME_REF.search(command) or (home is not None and str(home) in command)):
        return False
    return "config.yaml" in squeezed or _STATE_NAME.search(squeezed) is not None


_VENV_LIB = re.compile(r"/lib(?:64)?/python[\d.]*(?:/|$)|/Lib/")


def _site_dirs() -> list[str]:
    import site

    try:
        return [os.path.normpath(d) for d in (*site.getsitepackages(), site.getusersitepackages())]
    except (AttributeError, OSError):
        return []


def in_site_directory(p: str) -> bool:
    """site-packages/dist-packages, the user site, or a lib/pythonX directory: where Python processes .pth files."""
    return (bool({"site-packages", "dist-packages"} & set(p.split("/"))) or _VENV_LIB.search(p) is not None
            or any(p == d or p.startswith(d + "/") for d in _site_dirs()))


def startup_path(full: str) -> bool:
    """Python start-up code or a persistence location. A .pth file counts only inside a site directory (a PyTorch
    checkpoint elsewhere is inert); sitecustomize.py and usercustomize.py count everywhere."""
    p = full.replace("\\", "/")
    name = posixpath.basename(p)
    startup_name = name in _STARTUP_NAMES
    pth_in_site_dir = name.endswith(".pth") and in_site_directory(p)
    under_site_packages = in_site_directory(p) and bool({"site-packages", "dist-packages"} & set(p.split("/")))
    persistence = any(x in p + "/" for x in _PERSISTENCE)
    return startup_name or pth_in_site_dir or under_site_packages or persistence


def _risky_install_args(args: list[str]) -> bool:
    skip = False
    for a in args:
        if skip:
            skip = False
        elif a in _SKIP_VALUE:
            skip = True
        elif a.startswith("-"):
            if a.split("=", 1)[0] in _SOURCE_FLAGS or (not a.startswith("--") and "e" in a[1:] and a[1:].isalpha()):
                return True
        elif "/" in a or a.startswith((".", "~")) or a.lower().endswith(_ARCHIVE) or _URL_SCHEME.match(a):
            return True
    return False


def _install_risky(words: list[str]) -> bool:
    names = [base(w) for w in words]
    for i, n in enumerate(names):
        if _PIP.match(n) and "install" in names[i + 1:]:
            return _risky_install_args(words[names.index("install", i + 1) + 1:])
        if n == "uv" and i + 1 < len(names):
            sub = [x for x in names[i + 1:] if not x.startswith("-")][:2]
            if sub[:1] == ["add"] or sub[:2] in (["tool", "install"], ["tool", "run"]):
                return _risky_install_args(words[i + 1:])
        if n == "uvx":
            return _risky_install_args(words[i + 1:])
    return False


def _schedules(words: list[str]) -> bool:
    i = program_index(words)
    if i >= len(words):
        return False
    name, rest = base(words[i]), words[i + 1:]
    if name == "crontab":
        return rest != ["-l"]
    if name in ("at", "batch"):
        return rest != ["-l"]
    if name == "systemctl" and "--user" in rest:
        return next((a for a in rest if not a.startswith("-")), "") in _SYSTEMCTL_CHANGES
    return name == "systemd-run" and "--user" in rest
