"""Deterministic drafting helper. It prepares skeletons only: the agent refines them, a human approves them.

``draft_from`` cuts a notes file (or a folder of .md/.txt files) into blocks (a heading section's bullet run or
paragraph) and writes one lesson skeleton per block with ``status: draft``. Drafts are never loaded
(``packs.model``); ``approve_lesson`` flips one to ``approved`` and is gated by the human approval token.
The output pack is always ``visibility: private``.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from qrh.packs.model import MANIFEST, PackError

EXCERPT = 300
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$")


def _blocks(text: str) -> list[str]:
    blocks: list[str] = []
    heading = ""
    cur: list[str] = []

    def flush() -> None:
        body = " ".join(" ".join(cur).split())
        if re.search(r"\w", body):
            blocks.append(f"{heading}: {body}" if heading else body)
        cur.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            heading = m.group(1).strip()
        elif not line.strip():
            flush()
        else:
            cur.append(re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line))
    flush()
    return blocks


def _sources(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    if src.is_dir():
        return sorted(p for p in src.rglob("*") if p.is_file() and p.suffix in (".md", ".txt"))
    raise PackError(f"source {str(src)!r} does not exist")


def draft_from(src: Path, out: Path) -> list[Path]:
    manifest = out / MANIFEST
    if manifest.is_file():
        doc = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        if doc.get("visibility") == "public":
            raise PackError("refusing to draft into a public pack; drafts go to a private folder")
    else:
        out.mkdir(parents=True, exist_ok=True)
        manifest.write_text(yaml.safe_dump({
            "name": re.sub(r"[^A-Za-z0-9_.-]", "-", out.name) or "drafts", "version": "0.1.0", "schema_version": 1,
            "visibility": "private", "description": "Draft lessons awaiting human approval.", "applies_to": {},
            "authors": [], "license": "",
        }, sort_keys=True), encoding="utf-8")
    written: list[Path] = []
    for path in _sources(src):
        for block in _blocks(path.read_text(encoding="utf-8", errors="replace")):
            digest = hashlib.sha256(block.encode("utf-8")).hexdigest()
            target = out / "lessons" / f"draft-{digest[:12]}.yaml"
            doc = {
                "id": f"draft-{digest[:12]}", "statement": block[:EXCERPT], "applies_to": {}, "trigger": {},
                "action": "warn", "severity": "info", "status": "draft", "source_ref": digest,
            }
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(yaml.safe_dump(doc, sort_keys=True), encoding="utf-8")
            written.append(target)
    return written
