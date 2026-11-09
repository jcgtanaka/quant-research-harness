"""LLM support: the single call entry point and the replay cache (see docs/llm.md).

This package stays light (standard library only) so that hypothesis validation and hooks can import it.
"""

from qrh.llm import runtime
from qrh.llm.cache import LLMCacheCorrupt, LLMCacheMiss, LLMError
from qrh.llm.runtime import (MISS_MESSAGE, MODES, LLMSession, complete, current_session, llm_session, register_backend,
                             replay_only)
from qrh.llm.prompts import read_prompt
from qrh.llm.stub import StubBackend

__all__ = ["LLMCacheCorrupt", "LLMCacheMiss", "LLMError", "LLMSession", "MISS_MESSAGE", "MODES", "StubBackend",
           "complete", "current_session", "llm_session", "read_prompt", "register_backend", "replay_only", "runtime"]
