"""The one definition of 'equal to or under a root' for the hooks: whole path components, never a string prefix
(``/a/bc`` is not under ``/a/b``), and symlinks resolved where the caller asks for it."""

from __future__ import annotations

import os
from collections.abc import Iterable


def real_path(path: str | os.PathLike[str]) -> str:
    """``~`` expanded and symlinks resolved; a missing tail is kept as written (``realpath`` is not strict)."""
    return os.path.realpath(os.path.expanduser(path))


def is_under(path: str, root: str) -> bool:
    """Lexical check on strings the caller has already normalized or resolved: ``path`` equals ``root`` or is below
    it by whole components."""
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def under_any(path: str, roots: Iterable[str]) -> bool:
    return any(is_under(path, r) for r in roots)


def under_root(path: str, root: str) -> bool:
    """``path`` resolved is equal to or under ``root`` resolved."""
    return is_under(real_path(path), real_path(root))
