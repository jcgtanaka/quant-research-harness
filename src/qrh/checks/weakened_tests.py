"""Deterministic check of a git diff for weakened tests (``qrh audit tests``).

Signals, all about test files that already existed (a new test file weakens nothing):

* ``deleted-test-file``  a test file was deleted and its tests do not reappear elsewhere in the diff;
* ``removed-test``       a test definition was removed and the same name is not added anywhere in the diff;
* ``removed-assertions`` the diff as a whole lost more assertion lines than it gained, and a test file lost some;
* ``added-skip``         more skip / expected-failure markers (or ``if False``) added than removed in a test file.

Marker text inside string literals is ignored. Limits: a weaker assertion that keeps its line, a test that
always passes by mocking, and skips configured outside the file (``-k`` filters) are not detected.

Signal list and the idea of moves-are-not-losses follow claim-check's weakened-test rules by Ryan Alberts (MIT,
https://github.com/RyanAlberts/best-of-Agent-Harnesses); this implementation is independent. See NOTICE.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


MAX_UNTRACKED_BYTES = 1 << 20


class AuditError(RuntimeError):
    """git could not produce the diff."""


@dataclass(frozen=True)
class Finding:
    kind: str
    file: str
    detail: str


_CODE_EXT = (".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".go", ".rb", ".rs", ".java", ".kt", ".cs", ".php")
_TEST_NAME = re.compile(
    r"(?:^|/)(?:test_[^/]+\.py|[^/]+_test\.(?:py|go|rb)|[^/]+\.(?:test|spec)\.[cm]?[jt]sx?|[^/]+_spec\.rb"
    r"|[^/]+(?:Test|Tests)\.(?:java|kt|cs|php)|conftest\.py)$")
_TEST_DIR = re.compile(r"(?:^|/)(?:tests?|__tests__|specs?)/")
_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'|`(?:[^`\\\n]|\\.)*`')
_ASSERT = re.compile(
    r"\bassert\w*\b|\bexpect\s*\(|\.should\b|\bt\.(?:Error|Errorf|Fatal|Fatalf|Fail)\b|\brequire\.\w+\(|\bpytest\.raises\b"
    r"|\.to(?:Be|Equal|Match|Throw|Have|Contain|StrictEqual)\w*\(")
_DEF = (
    re.compile(r"^\s*(?:async\s+)?def\s+(test\w*)\s*\("), re.compile(r"^\s*func\s+(Test\w+)\s*\("),
    re.compile(r"\b[xf]?(?:it|test)(?:\.(?:only|skip|todo))?\s*\(\s*(['\"`])(.+?)\1"), re.compile(r"\bfunction\s+(test\w+)\s*\("),
)
_SKIP = re.compile(
    r"@pytest\.mark\.(?:skip|skipif|xfail)\b|\bpytest\.(?:skip|xfail)\(|@unittest\.(?:skip|skipIf|skipUnless|expectedFailure)\b"
    r"|\bself\.skipTest\(|\b(?:it|test|describe)\.(?:skip|todo)\s*\(|\b(?:xit|xtest|xdescribe)\s*\(|\bt\.Skip\w*\("
    r"|#\[ignore\b|@(?:Disabled|Ignore)\b|^\s*if\s+(?:False|0)\s*:")


def is_test_file(path: str) -> bool:
    return bool(_TEST_NAME.search(path)) or (path.endswith(_CODE_EXT) and bool(_TEST_DIR.search(path)))


def _code(line: str) -> str:
    return _STRING.sub('""', line)


@dataclass
class _File:
    path: str
    old: str
    new: str
    added: list[str]
    removed: list[str]

    @property
    def status(self) -> str:
        return "D" if not self.new else ("A" if not self.old else "M")


def _parse(diff: str) -> list[_File]:
    files: list[_File] = []
    cur: _File | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            cur = None
        elif line.startswith("--- "):
            old = line[4:].split("\t")[0]
            cur = _File("", old[2:] if old.startswith("a/") else "", "", [], [])
            files.append(cur)
        elif line.startswith("+++ ") and cur is not None:
            new = line[4:].split("\t")[0]
            cur.new = new[2:] if new.startswith("b/") else ""
            cur.path = cur.new or cur.old
        elif cur is not None and line.startswith("+"):
            cur.added.append(line[1:])
        elif cur is not None and line.startswith("-"):
            cur.removed.append(line[1:])
    return files


def _names(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        for rx in _DEF:
            m = rx.search(ln)
            if m:
                out.append(m.group(m.lastindex))
                break
    return out


def analyze_diff(diff: str, untracked: dict[str, str] | None = None) -> list[Finding]:
    """``untracked`` maps new, not-yet-added paths to their text: they count as added lines (a move shows up as such)."""
    files = _parse(diff)
    for path, text in (untracked or {}).items():
        files.append(_File(path, "", path, text.splitlines(), []))
    all_added = [ln for f in files for ln in f.added]
    added_names = set(_names(all_added))
    total_assert_delta = (sum(bool(_ASSERT.search(_code(ln))) for ln in all_added)
                          - sum(bool(_ASSERT.search(_code(ln))) for f in files for ln in f.removed))
    findings: list[Finding] = []
    for f in files:
        if not is_test_file(f.path) or f.status == "A":
            continue
        gone = [n for n in _names(f.removed) if n not in added_names]
        if f.status == "D":
            if gone or not _names(f.removed):
                findings.append(Finding("deleted-test-file", f.path, f"{len(gone)} test(s) not found elsewhere in the diff"))
            continue
        for n in gone:
            findings.append(Finding("removed-test", f.path, f"test {n!r} removed and not added anywhere in the diff"))
        lost = (sum(bool(_ASSERT.search(_code(ln))) for ln in f.removed)
                - sum(bool(_ASSERT.search(_code(ln))) for ln in f.added))
        if lost > 0 and total_assert_delta < 0:
            findings.append(Finding("removed-assertions", f.path, f"{lost} more assertion line(s) removed than added"))
        skips = (sum(bool(_SKIP.search(_code(ln))) for ln in f.added)
                 - sum(bool(_SKIP.search(_code(ln))) for ln in f.removed))
        if skips > 0:
            findings.append(Finding("added-skip", f.path, f"{skips} more skip / expected-failure marker(s) added than removed"))
    return findings


def _git(repo: Path, *args: str) -> str:
    try:
        done = subprocess.run(["git", "--no-optional-locks", "-c", "core.quotePath=false", "-C", str(repo), *args],
                              capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AuditError(f"git did not run: {type(exc).__name__}") from exc
    if done.returncode != 0:
        first = (done.stderr.strip().splitlines() or ["git failed"])[0]
        raise AuditError(first)
    return done.stdout


def audit_repo(repo: str | Path, base: str = "HEAD") -> list[Finding]:
    """Findings for the working tree (staged and unstaged) against ``base``; renames are followed."""
    repo = Path(repo)
    _git(repo, "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}")
    fresh: dict[str, str] = {}
    for rel in _git(repo, "ls-files", "--others", "--exclude-standard", "-z").split("\0"):
        full = repo / rel
        try:
            if rel and full.is_file() and full.stat().st_size <= MAX_UNTRACKED_BYTES:
                fresh[rel] = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return analyze_diff(_git(repo, "diff", "--no-ext-diff", "--no-color", "-M", "-U0", base, "--"), fresh)
