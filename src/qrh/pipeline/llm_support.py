"""What the pipeline stages need to know about a strategy that uses a language model (rules 1 to 3)."""

from __future__ import annotations

from typing import Any

from qrh.llm.runtime import current_session
from qrh.llm.variants import VARIANT_KEY


def receipt_bits(trial: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(params, diagnostics) to add to a stage receipt: ``({}, {})`` for a strategy that declares no ``llm``.

    The hashed params carry the hash of the cache entries this stage used (the same calls give the same hash in record
    and in replay mode); the mode and the call counts are facts about HOW the run went, so they are diagnostics.
    """
    session = current_session()
    if not trial.get("llm") or session is None:
        return {}, {}
    return ({"llm": {"cache_hash": session.used_hash()}},
            {"llm": {"mode": session.mode, **session.stats()}})


def display_params(params: dict[str, Any]) -> dict[str, Any]:
    """Parameters as shown in reports and artifacts: the variant is named, not dumped."""
    out = {k: v for k, v in params.items() if k != VARIANT_KEY}
    variant = params.get(VARIANT_KEY)
    if isinstance(variant, dict) and not variant.get("implicit"):  # the implicit single variant adds nothing to show
        out["variant"] = variant["name"]
    return out
