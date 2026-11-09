"""Deterministic hashing helpers (sha256 hex digests)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file by streaming it in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path: str | Path) -> str:
    """Hash a directory: sorted relative POSIX paths plus file contents."""
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    h = hashlib.sha256()
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())
    for p in files:
        rel = p.relative_to(root).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(8, "big") + rel)
        h.update(bytes.fromhex(sha256_file(p)))
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Sorted keys, no whitespace, floats via repr; NaN/inf raise ValueError."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)


def hash_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def code_hash(files: Iterable[str | Path]) -> str:
    """Hash a set of source files; independent of order and of file location."""
    digests = sorted(sha256_file(f) for f in files)
    return hash_json(digests)
