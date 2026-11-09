"""``<QRH_HOME>/packs.json``: which packs are registered (by path, never copied) and their content hash.

The active set is every registered pack, public packs before private ones, then by name. Packs are read from
their path each time, so editing a pack changes the hash recorded in later receipts and never an old one.
A loosening of a core threshold is loaded only if a human approval for exactly that loosening is recorded here.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qrh.packs.layering import loosenings
from qrh.packs.model import Pack, PackError, load_pack

INDEX_NAME = "packs.json"


def index_path(home: str | Path) -> Path:
    return Path(home) / INDEX_NAME


def load_index(home: str | Path) -> dict[str, Any]:
    p = index_path(home)
    if not p.is_file():
        return {"packs": []}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or not isinstance(doc.get("packs"), list):
            raise ValueError("unexpected structure")
    except (OSError, ValueError) as exc:
        raise PackError(f"{INDEX_NAME} is unreadable: {exc}") from exc
    return doc


def _save(home: str | Path, doc: dict[str, Any]) -> None:
    p = index_path(home)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def register(home: str | Path, pack: Pack, approved_loosening: list[str]) -> None:
    doc = load_index(home)
    entry = {
        "name": pack.name, "path": str(pack.path.resolve()), "hash": pack.hash, "version": pack.manifest.version,
        "visibility": pack.visibility, "added_at": datetime.now(timezone.utc).isoformat(),
        **({"approved_loosening": sorted(approved_loosening)} if approved_loosening else {}),
    }
    doc["packs"] = [e for e in doc["packs"] if e.get("name") != pack.name] + [entry]
    doc["packs"].sort(key=lambda e: e["name"])
    _save(home, doc)


def unregister(home: str | Path, name: str) -> bool:
    doc = load_index(home)
    kept = [e for e in doc["packs"] if e.get("name") != name]
    if len(kept) == len(doc["packs"]):
        return False
    doc["packs"] = kept
    _save(home, doc)
    return True


def approved_ids(home: str | Path) -> set[str]:
    return {i for e in load_index(home)["packs"] for i in e.get("approved_loosening", [])}


def active_packs(home: str | Path) -> list[Pack]:
    """Load every registered pack from its path. Raises PackError (fail closed) if one cannot be used."""
    entries = load_index(home)["packs"]
    packs: list[Pack] = []
    for e in entries:
        pack = load_pack(e["path"])
        if pack.name != e["name"]:
            raise PackError(f"pack at {e['path']!r} is now named {pack.name!r}, registered as {e['name']!r}")
        packs.append(pack)
    packs.sort(key=lambda p: (p.visibility != "public", p.name))
    approved = approved_ids(home)
    bad = [lo for lo in loosenings(packs, None) if lo.id not in approved]
    if bad:
        raise PackError("unapproved loosening of a core threshold: " + ", ".join(lo.id for lo in bad)
                        + " (re-run `qrh pack add` to request a human approval)")
    return packs


def pack_record(packs: list[Pack], home: str | Path) -> dict[str, dict[str, Any]]:
    """What a receipt stores: ``{name: {version, hash}}`` plus any approved loosening it relies on."""
    approved = {e["name"]: e.get("approved_loosening", []) for e in load_index(home)["packs"]}
    out: dict[str, dict[str, Any]] = {}
    for p in packs:
        rec: dict[str, Any] = dict(p.record())
        if approved.get(p.name):
            rec["approved_loosening"] = list(approved[p.name])
        out[p.name] = rec
    return out
