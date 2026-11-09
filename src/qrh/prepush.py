"""Git-level pre-push check: refuse a push unless a human signed an approval for exactly this push.

``qrh publish pre-push <remote_name> <remote_url>`` speaks git's pre-push protocol (stdin lines
``<local_ref> <local_sha> <remote_ref> <remote_sha>``). It reuses ``qrh.approvals`` (action kind ``parts``, hook name
``git-pre-push``, Ed25519 signature, single-use marker); there is no second scheme.

What an approval is bound to (the signed fingerprint covers all of it): the repository real path (an extra check, not a
boundary: GIT_DIR and GIT_WORK_TREE come from the environment), the remote URL and the sorted, de-duplicated set of
``(local_ref, local_sha, remote_ref, remote_sha)`` updates. A deletion is an update whose local sha is all zeros. The
remote name is not bound. Approvals are not tied to an agent session.

The command does not trust its environment: it needs ``--home`` (absolute) and ``--key-id`` and ignores QRH_HOME and
HOME. The hook file written by the user pins both (see ``hook_text``). Every failure refuses (exit 2); nothing is
installed or written outside the approvals store.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from qrh import approvals, signing

HOOK = "git-pre-push"
MAX_STDIN = 256 * 1024
MAX_UPDATES = 500
SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
REFUSED = 2
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_HELPER = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*)::(.+)$", re.S)


class PrePushError(RuntimeError):
    """The push cannot be checked; the message is one line saying why."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def parse_updates(raw: bytes) -> list[tuple[str, str, str, str]]:
    """The sorted unique (local_ref, local_sha, remote_ref, remote_sha) updates, or PrePushError on anything unexpected."""
    if len(raw) > MAX_STDIN:
        raise PrePushError(f"the push list is larger than {MAX_STDIN} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise PrePushError("the push list on stdin is not valid UTF-8") from None
    lines = text.splitlines()
    if not any(x.strip() for x in lines):
        raise PrePushError("nothing on stdin: git passes the refs being pushed there")
    found: set[tuple[str, str, str, str]] = set()
    for n, line in enumerate(lines, 1):
        fields = line.split(" ")
        if len(fields) != 4 or not all(fields):
            raise PrePushError(f"stdin line {n} is malformed (expected 4 fields)")
        local_ref, local_sha, remote_ref, remote_sha = fields
        if not SHA_RE.match(local_sha) or not SHA_RE.match(remote_sha):
            raise PrePushError(f"stdin line {n} has an invalid object id")
        if any(ord(c) < 32 or ord(c) == 127 for c in local_ref + remote_ref):
            raise PrePushError(f"stdin line {n} has control characters in a ref name")
        found.add((local_ref, local_sha, remote_ref, remote_sha))
    if len(found) > MAX_UPDATES:
        raise PrePushError(f"more than {MAX_UPDATES} updates in one push")
    return sorted(found)


def clean_url(url: str) -> str:
    """The remote URL as bound and shown: no password, query or fragment, also behind ``<helper>::`` prefixes.

    The user name is kept (``alice@`` and ``bob@`` stay distinct); a user name that looks like a token is masked.
    """
    from qrh.hooks.common import redact

    url = (url or "").strip()
    if not url:
        raise PrePushError("the remote URL is empty")
    if _CONTROL.search(url):
        raise PrePushError("the remote URL has control characters")
    helper = _HELPER.match(url)
    if helper:
        return f"{helper.group(1)}::{clean_url(helper.group(2))}"
    if "://" not in url:
        return url
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        netloc = f"[{host}]" if ":" in host else host
        if parts.port:
            netloc += f":{parts.port}"
        if parts.username:
            netloc = f"{redact(parts.username)}@{netloc}"
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except ValueError:
        raise PrePushError("the remote URL is malformed") from None


def repo_real_path() -> str:
    """Real path of the working tree (or of the git directory of a bare repository) of the current directory."""
    for args in (["--show-toplevel"], ["--absolute-git-dir"]):
        try:
            out = subprocess.run(["git", "rev-parse", *args], capture_output=True, text=True, timeout=20, check=False)
        except (OSError, subprocess.SubprocessError):
            raise PrePushError("git could not be run to find the repository") from None
        path = out.stdout.strip()
        if out.returncode == 0 and path:
            return os.path.realpath(path)
    raise PrePushError("this is not a git repository")


def build_action(repo: str, url: str, updates: list[tuple[str, str, str, str]]) -> dict:
    parts = [HOOK, f"repo={repo}", f"remote={url}"]
    parts += [f"{local_ref} {sha} -> {remote_ref} (was {old})" for local_ref, sha, remote_ref, old in updates]
    return approvals.parts_action(*parts)


def is_action(action: object) -> bool:
    parts = action.get("parts") if isinstance(action, dict) and action.get("kind") == "parts" else None
    return isinstance(parts, list) and bool(parts) and parts[0] == HOOK


def _visible(text: str) -> str:
    return _CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}", text)


def full_text(action: dict, mask=None) -> str:  # noqa: ANN001
    """What a human reads: every update on its own line, never cut. ``mask`` (redact) is applied to repo and URL only,
    never to ref names."""
    parts = [str(p) for p in action["parts"]][1:]
    head, updates = parts[:2], parts[2:]
    head = [mask(x) if mask else x for x in head]
    lines = [f"git push, {len(updates)} update(s)"] + [f"  {_visible(x)}" for x in head]
    lines += [f"  {n}. {_visible(x)}" for n, x in enumerate(updates, 1)]
    return "\n".join(lines)


def hook_text(home: str, key_id: str, python: str, path_dirs: str = "/usr/bin:/bin") -> str:
    """The hook file the user installs: everything the agent's environment could override is pinned."""
    q = shlex.quote
    return (
        "#!/bin/sh\n"
        f"exec env -i PATH={q(path_dirs)} {q(python)} -I -m qrh publish pre-push --home {q(home)} --key-id {q(key_id)} "
        '"$@"\n')


def default_path_dirs() -> str:
    dirs = ["/usr/bin", "/bin"]
    git = shutil.which("git")
    if git and os.path.dirname(git) not in dirs:
        dirs.insert(0, os.path.dirname(git))
    return ":".join(dirs)


def _approve_command(home: Path, rid: str) -> str:
    prefix = "" if home == Path.home() / ".qrh" else f"QRH_HOME={home} "
    return f'{prefix}qrh approve {rid} --reason "<why this push is acceptable, at least {approvals.MIN_REASON} characters>"'


def _fail(text: str) -> tuple[bool, str]:
    return False, f"BLOCKED: git pre-push refused: {text}\n"


def check(home: str, key_id: str, remote_url: str, stdin: bytes) -> tuple[bool, str]:
    """(allowed, text for stderr). Never raises for an expected failure. ``home`` and ``key_id`` come from the hook file."""
    from qrh.hooks.common import redact
    from qrh.hooks.runner import human_block_text

    try:
        if not os.path.isabs(home or ""):
            raise PrePushError("--home must be an absolute path")
        hp = Path(home)
        if not hp.is_dir():
            raise PrePushError("--home is not an existing directory")
        try:
            pub = signing.load_public(hp)
        except signing.SigningError as exc:
            raise PrePushError(f"the approval key is unusable: {exc}") from exc
        if pub is None:
            raise PrePushError("no approval key exists: the user creates one in their own terminal with "
                               "`qrh approve --init`")
        if pub.key_id != key_id:
            raise PrePushError("the approval key id does not match --key-id of the installed hook")
        updates = parse_updates(stdin)
        url = clean_url(remote_url)
        repo = repo_real_path()
        if _CONTROL.search(repo):
            raise PrePushError("the repository path has control characters")
        action = build_action(repo, url, updates)
        now = _now()
        got = approvals.consume(hp, hook=HOOK, fingerprint=approvals.fingerprint_of(action), session_id="", now=now)
        if got.ok:
            return True, ""
        shown = approvals.summarize(action)
        if any(str(p) not in shown for p in action["parts"][1:]):
            raise PrePushError("this push cannot be shown in full to the user, so no approval is requested")
        reason = "git push needs a human approval bound to this repository, this remote and these exact ref updates"
        rid = approvals.request(hp, hook=HOOK, action=action, reason=reason, session_id="", now=now)
    except PrePushError as exc:
        return _fail(str(exc))
    except (approvals.ApprovalError, signing.SigningError, OSError) as exc:
        return _fail(f"{type(exc).__name__}: {redact(str(exc))}")
    return False, human_block_text(
        HOOK, full_text(action, redact), reason,
        "a push publishes commits outside this machine and cannot be undone",
        _approve_command(hp, rid),
        f"Approval id {rid}: valid once, within {approvals.DEFAULT_TTL_S // 60} minutes. After approving, "
        "retry the identical git push.", "\n".join(got.notes))
