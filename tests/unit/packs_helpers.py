"""Builders for synthetic experience packs used by the pack and interview tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def write_yaml(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=True), encoding="utf-8")
    return path


def make_pack(root: Path, name: str = "demo", visibility: str = "public", version: str = "1.0.0",
              applies_to: dict[str, Any] | None = None, **content: Any) -> Path:
    """Create a pack folder. ``content`` keys: lessons, attempts, account, thresholds, questions, lint, data_rules
    (lists of documents, except account/thresholds which are one document)."""
    d = root / name
    write_yaml(d / "qrh-pack.yaml", {
        "name": name, "version": version, "schema_version": 1, "visibility": visibility,
        "description": "synthetic test pack", "applies_to": applies_to or {}, "authors": ["tester"],
        "license": "Apache-2.0",
    })
    for kind, folder in (("lessons", "lessons"), ("attempts", "attempts"), ("questions", "questions"),
                         ("lint", "lint"), ("data_rules", "data_rules")):
        for i, doc in enumerate(content.get(kind, [])):
            write_yaml(d / folder / f"{kind[:4]}{i:02d}.yaml", doc)
    for kind in ("account", "thresholds"):
        if kind in content:
            write_yaml(d / f"{kind}.yaml", content[kind])
    return d


def lesson(id_: str = "L1", **kw: Any) -> dict[str, Any]:
    doc = {"id": id_, "statement": "Count every variant as a trial.", "applies_to": {}, "trigger": {},
           "action": "warn", "severity": "warning"}
    doc.update(kw)
    return doc


def question(id_: str = "q1", **kw: Any) -> dict[str, Any]:
    doc = {"id": id_, "stage": "S1", "scope": "dataset", "applies_to": {}, "text": "How was it produced?",
           "why_it_matters": "Because timing matters.", "answer_type": "choice",
           "choices": [{"id": "pit", "label": "Point-in-time daily published"},
                       {"id": "curve", "label": "Current curve applied to history", "risk": "lookahead"}],
           "critical": True, "unknown_risk": "Possible look-ahead in the input."}
    doc.update(kw)
    return doc
