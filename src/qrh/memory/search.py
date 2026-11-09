"""Keyword search over registered hypotheses (SQLite FTS5, BM25 ranking).

``Searcher`` is the interface; a later phase can add an embedding searcher behind it.
The FTS table is created by a migration and kept in sync by triggers on ``hypotheses``.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_MAX_TOKENS = 24


@dataclass(frozen=True)
class SearchHit:
    trial_id: str
    verdict: str | None  # verdict of the highest-numbered stage receipt, if any
    family: str
    text: str
    score: float  # higher is better


@runtime_checkable
class Searcher(Protocol):
    def search(self, query: str, limit: int = 3) -> list[SearchHit]: ...


def match_expression(query: str) -> str | None:
    """OR of quoted alphanumeric tokens; punctuation and FTS operators in the query are inert."""
    seen: list[str] = []
    for tok in _TOKEN.findall(query.lower()):
        if tok not in seen:
            seen.append(tok)
    if not seen:
        return None
    return " OR ".join(f'"{t}"' for t in seen[:_MAX_TOKENS])


class FtsSearcher:
    """Deterministic BM25 keyword search on hypothesis text and strategy family."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        expr = match_expression(query)
        if expr is None or limit < 1:
            return []
        rows = self.conn.execute(
            "SELECT f.trial_id AS trial_id, h.text AS text, h.strategy_family AS family,"
            " bm25(hypotheses_fts) AS rank,"
            " (SELECT r.verdict FROM receipts r WHERE r.trial_id=h.trial_id ORDER BY r.stage DESC LIMIT 1)"
            " AS verdict"
            " FROM hypotheses_fts f JOIN hypotheses h ON h.trial_id=f.trial_id"
            " WHERE hypotheses_fts MATCH ? ORDER BY rank, f.trial_id LIMIT ?",
            (expr, limit),
        ).fetchall()
        hits = [SearchHit(r["trial_id"], r["verdict"], r["family"], r["text"], -float(r["rank"])) for r in rows]
        hits += self._imported(query)
        hits.sort(key=lambda h: (-h.score, h.trial_id))
        return hits[:limit]

    def _imported(self, query: str) -> list[SearchHit]:
        """Pre-harness attempts imported from packs: ranked by how many query words they contain."""
        words = [t for t in dict.fromkeys(_TOKEN.findall(query.lower()))][:_MAX_TOKENS]
        out: list[SearchHit] = []
        for r in self.conn.execute(
            "SELECT pack, attempt_hash, text, strategy_family, verdict FROM imported_attempts ORDER BY id"
        ):
            hay = f"{r['text']} {r['strategy_family']}".lower()
            score = sum(1 for w in words if w in hay)
            if score:
                out.append(SearchHit(f"imported:{r['pack']}:{r['attempt_hash'][:8]}", r["verdict"],
                                     r["strategy_family"], r["text"], float(score)))
        return out

    def reindex(self) -> int:
        """Rebuild the index from the hypotheses table; returns the number of rows indexed."""
        self.conn.execute("BEGIN")
        try:
            self.conn.execute("DELETE FROM hypotheses_fts")
            self.conn.execute(
                "INSERT INTO hypotheses_fts(trial_id, text, family)"
                " SELECT trial_id, text, strategy_family FROM hypotheses"
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return int(self.conn.execute("SELECT COUNT(*) FROM hypotheses_fts").fetchone()[0])
