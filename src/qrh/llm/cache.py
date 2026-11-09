"""The LLM replay cache: append-only JSONL entries with hashes, plus an index.

One folder per trial under ``<QRH_HOME>/llm_cache/<trial_id>/``:

- ``entries.jsonl``: one line per answer, ``{"entry_hash", "entry"}``; ``entry`` holds the key, model id, params,
  prompt, response and the decision time (``as_of``, recorded for audit only; it is not part of the key);
- ``index.json``: ``{key: entry_hash}``, rewritten when a session ends.

The key is the sha256 of the canonical JSON of ``{model_id, params, prompt, dataset_hash, code_hash}``; entries also
store both hashes so the key can be re-derived. Entries contain prompts that may
include licensed data, so they stay in QRH_HOME and are never written to the repository (see docs/security.md).
The entry hashes detect corruption and edits; they are not a signature (see docs/llm.md, limits).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from qrh.hashing import hash_json

CACHE_DIR = "llm_cache"
ENTRIES = "entries.jsonl"
INDEX = "index.json"
# Placeholder hash of a session opened outside the pipeline; never valid in a pipeline or bench result.
UNBOUND_HASH = "0" * 64


class LLMError(Exception):
    """An LLM call cannot be made or served."""


class LLMCacheMiss(LLMError):
    """Replay mode asked for an answer that was never recorded."""

    def __init__(self, model_id: str, key: str, as_of: str | None):
        self.model_id, self.key, self.as_of = model_id, key, as_of
        where = f" at decision time {as_of}" if as_of else ""
        super().__init__(f"replay cache miss for model {model_id!r}{where} (key {key[:12]})")


class LLMCacheCorrupt(LLMError):
    """The cache on disk does not match its own hashes."""


def cache_dir(home: str | Path, trial_id: str) -> Path:
    return Path(home) / CACHE_DIR / trial_id


def cache_key(model_id: str, params: dict[str, Any], prompt: str, *, dataset_hash: str, code_hash: str) -> str:
    """Bind an answer to the model, params, prompt, the dataset it was built from and the code that built it.

    Both hashes are required: an empty or missing value is refused, never keyed as an empty string. ``as_of`` is
    deliberately not an input.
    """
    for name, value in (("dataset_hash", dataset_hash), ("code_hash", code_hash)):
        if not isinstance(value, str) or not value.strip():
            raise LLMError(f"cache key needs a non-empty {name}; refusing to key an answer without it")
    return hash_json({"model_id": model_id, "params": params, "prompt": prompt,
                      "dataset_hash": dataset_hash, "code_hash": code_hash})


def entry_hash(entry: dict[str, Any]) -> str:
    return hash_json(entry)


def _check_line(n: int, line: str) -> tuple[dict[str, Any] | None, str | None]:
    """(doc, problem) for one JSONL line; a doc is returned only when its hash and its key are consistent."""
    try:
        doc = json.loads(line)
        entry, claimed = doc["entry"], doc["entry_hash"]
        if not isinstance(entry, dict) or not isinstance(claimed, str):
            raise TypeError("entry must be an object and entry_hash a string")
    except (ValueError, KeyError, TypeError) as exc:
        return None, f"line {n}: unreadable ({type(exc).__name__}: {exc})"
    try:
        actual = entry_hash(entry)
        key_ok = cache_key(entry["model_id"], entry["params"], entry["prompt"], dataset_hash=entry["dataset_hash"],
                           code_hash=entry["code_hash"]) == entry["key"]
    except (KeyError, TypeError, ValueError, LLMError) as exc:
        return None, f"line {n}: malformed entry ({type(exc).__name__}: {exc})"
    if actual != claimed:
        return None, f"line {n}: entry hash does not match its content"
    if not key_ok:
        return None, f"line {n}: key does not match the model, params, prompt and hashes of the entry"
    return doc, None


class ReplayCache:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self._entries: dict[str, dict[str, Any]] | None = None
        self._dirty = False

    # reading ----------------------------------------------------------------------------------------------
    def _load(self) -> dict[str, dict[str, Any]]:
        if self._entries is not None:
            return self._entries
        entries: dict[str, dict[str, Any]] = {}
        path = self.dir / ENTRIES
        if path.is_file():
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                doc, problem = _check_line(n, line)
                if doc is None:
                    raise LLMCacheCorrupt(f"replay cache {str(self.dir)!r}: {problem}; run `qrh llm verify <trial>`")
                key = doc["entry"]["key"]
                if key in entries and entries[key]["entry_hash"] != doc["entry_hash"]:
                    raise LLMCacheCorrupt(f"replay cache {str(self.dir)!r}: line {n}: key stored twice with different "
                                          "content; run `qrh llm verify <trial>`")
                entries[key] = doc
        self._entries = entries
        return entries

    def get(self, key: str) -> dict[str, Any] | None:
        return self._load().get(key)

    def __len__(self) -> int:
        return len(self._load())

    def content_hash(self) -> str:
        """Order independent: the sorted (key, entry hash) pairs of every entry."""
        return hash_json(sorted((k, d["entry_hash"]) for k, d in self._load().items()))

    def status(self) -> dict[str, Any]:
        entries = self._load()
        return {
            "entries": len(entries),
            "cache_hash": self.content_hash(),
            "models": sorted({d["entry"]["model_id"] for d in entries.values()}),
        }

    # writing ----------------------------------------------------------------------------------------------
    def add(self, entry: dict[str, Any]) -> str:
        """Append one entry (never rewrites earlier lines); returns its hash."""
        entries = self._load()
        digest = entry_hash(entry)
        doc = {"entry_hash": digest, "entry": entry}
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / ENTRIES, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, sort_keys=True, ensure_ascii=True) + "\n")
        entries[entry["key"]] = doc
        self._dirty = True
        return digest

    def flush(self) -> None:
        """Write the index (atomically) if anything was added."""
        if not self._dirty:
            return
        index = {k: d["entry_hash"] for k, d in sorted(self._load().items())}
        tmp = self.dir / f"{INDEX}.{os.getpid()}.tmp"
        try:
            tmp.write_text(json.dumps(index, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, self.dir / INDEX)
        finally:
            tmp.unlink(missing_ok=True)
        self._dirty = False

    # verification -----------------------------------------------------------------------------------------
    def verify(self) -> list[str]:
        """Every problem found on disk (an empty list means the cache is consistent)."""
        problems: list[str] = []
        path = self.dir / ENTRIES
        seen: dict[str, str] = {}
        if path.is_file():
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                doc, problem = _check_line(n, line)
                if doc is None:
                    problems.append(str(problem))
                    continue
                entry = doc["entry"]
                if UNBOUND_HASH in (entry["dataset_hash"], entry["code_hash"]):
                    problems.append(f"line {n}: unbound entry (dataset_hash or code_hash is the placeholder "
                                    f"{UNBOUND_HASH[:8]}...); it is never served to a pipeline run")
                key = entry["key"]
                if key in seen and seen[key] != doc["entry_hash"]:
                    problems.append(f"line {n}: key stored twice with different content")
                seen[key] = doc["entry_hash"]
        index_path = self.dir / INDEX
        if index_path.is_file():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except ValueError as exc:
                index = None
                problems.append(f"index unreadable: {exc}")
            # entries that failed their own check are already reported; the index is compared only when all passed
            if index is not None and not problems and index != seen:
                problems.append("index does not match the entries")
        elif seen:
            problems.append("index missing")
        return problems
