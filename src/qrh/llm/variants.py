"""Rule 2: prompts, models and variants count as trials.

``resolve_variants`` turns the declared ``llm`` section into the list of variants the S3 search multiplies the
adapter's parameter grid by. With no ``variants`` declared there is exactly one, formed by all the declared models,
prompt files and params. Each variant carries the hash of its prompt files, which is logged with every evaluated
combination. ``llm_fingerprint`` is what the S0 note compares to tell that a prompt or model changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qrh.adapters.loader import read_dependencies
from qrh.hashing import hash_json, sha256_bytes
from qrh.llm.prompts import combine as _combined

VARIANT_KEY = "llm_variant"  # the adapter receives the variant in params under this key
DEFAULT_NAME = "default"


def resolve_variants(llm: Mapping[str, Any], base_dir: str | Path | None) -> list[dict[str, Any]]:
    """The variants of a declared ``llm`` section, each with its prompt hash (raises AdapterLoadError)."""
    files = read_dependencies(base_dir, list(llm.get("prompt_files") or []))
    sha = {rel: sha256_bytes(data) for rel, (_p, data) in files.items()}
    base_params = dict(llm.get("params") or {})
    declared = list(llm.get("variants") or [])
    if not declared:
        names = list(sha)
        return [{
            "name": DEFAULT_NAME, "models": [m["id"] for m in llm["models"]], "prompt_files": names,
            "params": base_params, "prompt_hash": _combined([sha[n] for n in names]), "implicit": True,
        }]
    return [{
        "name": v["name"], "models": [v["model"]], "prompt_files": [v["prompt_file"]],
        "params": {**base_params, **(v.get("params") or {})}, "prompt_hash": sha[v["prompt_file"]],
    } for v in declared]


def llm_fingerprint(llm: Mapping[str, Any], base_dir: str | Path | None) -> dict[str, str]:
    """{"prompt_hash", "fingerprint"}: the hash of every prompt file, and of prompts, models, params and variants."""
    files = read_dependencies(base_dir, list(llm.get("prompt_files") or []))
    prompt_hash = hash_json(sorted(sha256_bytes(data) for _p, data in files.values()))
    fingerprint = hash_json({
        "prompt_hash": prompt_hash, "models": sorted(m["id"] for m in llm["models"]),
        "params": llm.get("params") or {}, "variants": llm.get("variants") or [],
    })
    return {"prompt_hash": prompt_hash, "fingerprint": fingerprint}
