#!/usr/bin/env python3
"""Fail if any given file matches a private denylist.

The denylist path comes from the QRH_DENYLIST environment variable.
Format: one literal or ``re:`` regex per line; ``#`` comments and blank lines
are ignored; literals match case-insensitively.

Exit codes: 0 clean (or denylist not configured), 1 hit or denylist unreadable.
Matched terms are never printed in full.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def mask(term: str) -> str:
    return term[:2] + "***"


def load_denylist(path: Path) -> list[tuple[str, re.Pattern[str]]]:
    rules: list[tuple[str, re.Pattern[str]]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("re:"):
            pattern = line[3:]
            rules.append((pattern, re.compile(pattern, re.IGNORECASE)))
        else:
            rules.append((line, re.compile(re.escape(line), re.IGNORECASE)))
    return rules


def scan(files: list[str], rules: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    hits: list[str] = []
    for name in files:
        try:
            text = Path(name).read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # unreadable or binary files are skipped
        for lineno, line in enumerate(text.splitlines(), start=1):
            for term, rx in rules:
                if rx.search(line):
                    hits.append(f"{name}:{lineno}: denylist hit ({mask(term)})")
    return hits


def main(argv: list[str]) -> int:
    env = os.environ.get("QRH_DENYLIST")
    if not env:
        print("warning: QRH_DENYLIST is not set; denylist check skipped", file=sys.stderr)
        return 0
    try:
        rules = load_denylist(Path(env))
    except (OSError, UnicodeDecodeError, re.error) as exc:
        print(f"error: cannot load denylist ({type(exc).__name__})", file=sys.stderr)
        return 1
    hits = scan(argv, rules)
    for h in hits:
        print(h)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
