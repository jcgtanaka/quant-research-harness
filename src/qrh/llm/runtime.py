"""``qrh.llm.complete``: the single entry point adapters use to ask a model.

The mode is set by the pipeline through a context variable (never an environment variable):

- ``replay`` (the default): an answer is served from the trial's cache; a miss raises ``LLMCacheMiss``;
- ``record``: a miss calls the backend registered for the model's provider and stores the answer.

qrh ships no network backend. Private adapter code registers one with ``register_backend(provider, fn)`` where
``fn(model_id, prompt, params) -> str``. Only models the hypothesis declares can be called.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from qrh.hashing import canonical_json, hash_json
from qrh.llm.cache import LLMCacheCorrupt, LLMCacheMiss, LLMError, ReplayCache, UNBOUND_HASH, cache_key

__all__ = ["LLMCacheCorrupt", "LLMCacheMiss", "LLMError", "LLMSession", "complete", "llm_session", "register_backend",
           "replay_only", "stub_misses_after", "MODES", "MISS_MESSAGE", "current_session"]

MODES = ("replay", "record")
MISS_MESSAGE = "replay cache miss: run `qrh run <trial> --llm-mode record` first"
# UNBOUND_HASH (defined in qrh.llm.cache) is the placeholder for a session opened outside the pipeline (unit tests,
# ad-hoc use); the pipeline and the bench refuse it (require_bound), so an answer recorded under it is never served.


def require_bound(dataset_hash: str, code_hash: str) -> None:
    """Refuse the unbound placeholder: a pipeline or bench session must key answers on real hashes."""
    for name, value in (("dataset_hash", dataset_hash), ("code_hash", code_hash)):
        if value == UNBOUND_HASH:
            raise LLMError(f"{name} is the unbound placeholder ({UNBOUND_HASH[:8]}...); a pipeline or bench run "
                           "needs a real hash, so no answer is recorded or replayed under the placeholder")


Backend = Callable[[str, str, dict[str, Any]], str]
_BACKENDS: dict[str, Backend] = {}
_SESSION: contextvars.ContextVar["LLMSession | None"] = contextvars.ContextVar("qrh_llm_session", default=None)


def register_backend(provider: str, fn: Backend) -> None:
    """Register ``fn(model_id, prompt, params) -> str`` for a provider name (as written in the hypothesis)."""
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("provider must be a non-empty string")
    if not callable(fn):
        raise TypeError("a backend must be callable")
    _BACKENDS[provider] = fn


class LLMSession:
    def __init__(self, mode: str, directory: str | Path, providers: Mapping[str, str], *,
                 dataset_hash: str = UNBOUND_HASH, code_hash: str = UNBOUND_HASH):
        if mode not in MODES:
            raise ValueError(f"llm mode must be record or replay, got {mode!r}")
        self.mode = mode
        self.cache = ReplayCache(directory)
        self.providers = dict(providers)
        self.dataset_hash = dataset_hash
        self.code_hash = code_hash
        self.used: dict[str, str] = {}
        self.hits = self.recorded = self.misses = 0
        self.stub_after: Any = None  # see stub_misses_after
        self.stubbed = 0
        self.last_response: str | None = None

    def used_hash(self) -> str:
        """Hash of the (key, entry hash) pairs this session used: the same calls give the same hash in either mode."""
        return hash_json(sorted(self.used.items()))

    def stats(self) -> dict[str, int]:
        return {"calls": self.hits + self.recorded + self.misses, "hits": self.hits, "recorded": self.recorded,
                "misses": self.misses}

    def __enter__(self) -> "LLMSession":
        self._token = _SESSION.set(self)
        return self

    def __exit__(self, *exc: object) -> None:
        _SESSION.reset(self._token)
        self.cache.flush()


def llm_session(mode: str, directory: str | Path, providers: Mapping[str, str], *,
                dataset_hash: str = UNBOUND_HASH, code_hash: str = UNBOUND_HASH) -> LLMSession:
    """Context manager: the mode, cache folder and the dataset and code hashes ``complete`` keys answers on."""
    return LLMSession(mode, directory, providers, dataset_hash=dataset_hash, code_hash=code_hash)


def current_session() -> LLMSession | None:
    return _SESSION.get()


@contextlib.contextmanager
def replay_only() -> Iterator[None]:
    """Force replay for a block (the truncated runs of S2); a no-op when no session is active."""
    session = _SESSION.get()
    if session is None:
        yield
        return
    previous, session.mode = session.mode, "replay"
    try:
        yield
    finally:
        session.mode = previous


@contextlib.contextmanager
def stub_misses_after(cutoff: Any) -> Iterator[None]:
    """In replay, a miss whose ``as_of`` is strictly after ``cutoff`` is answered with the last served answer
    (deterministic) instead of raising. A miss at or before the cutoff, or with no ``as_of``, still raises: that
    prompt should have been identical in the perturbed run. Used by the S2 future-perturbation test; no-op without
    an active session."""
    session = _SESSION.get()
    if session is None:
        yield
        return
    previous, session.stub_after = session.stub_after, cutoff
    try:
        yield
    finally:
        session.stub_after = previous


def _after(when: str | None, cutoff: Any) -> bool:
    import pandas as pd

    stamp = pd.to_datetime(when, errors="coerce", utc=True) if when is not None else pd.NaT
    limit = pd.Timestamp(cutoff)
    limit = limit.tz_localize("UTC") if limit.tzinfo is None else limit
    return bool(not pd.isna(stamp) and stamp > limit)


def _plain(obj: Any) -> Any:
    return obj.item() if hasattr(obj, "item") else str(obj)


def _as_of(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def complete(model_id: str, prompt: str, *, params: Mapping[str, Any] | None = None, as_of: Any = None) -> str:
    """Ask ``model_id``; the answer comes from the replay cache (or, in record mode, from the registered backend).

    ``as_of`` is the decision time the prompt was built for. It is stored for audit and reported on a miss, but it is
    not part of the cache key.
    """
    session = _SESSION.get()
    if session is None:
        raise LLMError("qrh.llm.complete was called outside a qrh run (the pipeline sets the cache and the mode)")
    if not isinstance(model_id, str) or not isinstance(prompt, str):
        raise TypeError("model_id and prompt must be strings")
    if session.providers and model_id not in session.providers:
        raise LLMError(f"model {model_id!r} is not declared in the hypothesis llm.models")
    clean = json.loads(json.dumps(dict(params or {}), default=_plain))
    canonical_json(clean)  # NaN and inf are refused here, before they reach a key
    key = cache_key(model_id, clean, prompt, dataset_hash=session.dataset_hash, code_hash=session.code_hash)
    found = session.cache.get(key)
    if found is not None:
        session.hits += 1
        session.used[key] = found["entry_hash"]
        session.last_response = str(found["entry"]["response"])
        return session.last_response
    when = _as_of(as_of)
    if session.mode == "replay":
        if session.stub_after is not None and _after(when, session.stub_after):
            session.stubbed += 1
            return session.last_response if session.last_response is not None else ""
        session.misses += 1
        raise LLMCacheMiss(model_id, key, when)
    provider = session.providers.get(model_id)
    backend = _BACKENDS.get(provider) if provider is not None else None
    if backend is None:
        raise LLMError(f"no backend registered for provider {provider!r} (register_backend, private adapter code)")
    response = backend(model_id, prompt, dict(clean))
    if not isinstance(response, str):
        raise LLMError(f"a backend must return a string, got {type(response).__name__}")
    digest = session.cache.add({
        "key": key, "model_id": model_id, "params": clean, "prompt": prompt, "response": response, "as_of": when,
        "dataset_hash": session.dataset_hash, "code_hash": session.code_hash,
    })
    session.recorded += 1
    session.used[key] = digest
    return response
