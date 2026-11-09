"""Read the prompt files of a variant exactly as they were hashed.

The loader hashes the prompt files into the adapter code hash at load time; ``read_prompt`` lets the adapter read the
same files and refuses to continue if their bytes no longer match the hash recorded for the variant (an edit between
the hash and the read).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qrh.hashing import hash_json, sha256_bytes
from qrh.llm.cache import LLMError


def combine(hashes: list[str]) -> str:
    """The hash of a set of prompt files: the file hash itself for one file, else the hash of the sorted hashes."""
    return hashes[0] if len(hashes) == 1 else hash_json(sorted(hashes))


def read_prompt(variant: Mapping[str, Any], anchor: str | Path) -> str:
    """The text of the variant's prompt file(s); ``anchor`` is the adapter's ``__file__`` (paths are relative to it)."""
    root = Path(anchor).resolve().parent
    texts: list[str] = []
    hashes: list[str] = []
    for rel in variant["prompt_files"]:
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            raise LLMError(f"prompt file {rel!r} is outside the adapter folder")
        data = path.read_bytes()
        hashes.append(sha256_bytes(data))
        texts.append(data.decode("utf-8"))
    if hashes and combine(hashes) != variant["prompt_hash"]:
        raise LLMError(f"prompt file(s) {variant['prompt_files']} changed after they were hashed for this run")
    return "\n\n".join(texts)
