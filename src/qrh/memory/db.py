"""SQLite connection and migrations shared by the registry and the catalog."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from qrh.config import QrhHomeError

# Each migration is a script; its index + 1 is the schema version.
MIGRATIONS: list[str] = [
    # 1: registry
    """
    CREATE TABLE hypotheses (
        trial_id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        strategy_family TEXT NOT NULL,
        asset_class TEXT NOT NULL,
        frequency TEXT NOT NULL,
        parameter_space_size INTEGER NOT NULL,
        thresholds TEXT NOT NULL,
        difference_statement TEXT NOT NULL,
        family_id TEXT NOT NULL,
        hypothesis_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TRIGGER hypotheses_immutable
    BEFORE UPDATE OF thresholds, difference_statement, hypothesis_hash, text,
        strategy_family, asset_class, frequency, parameter_space_size, family_id
    ON hypotheses
    BEGIN
        SELECT RAISE(ABORT, 'hypothesis fields are immutable; register a new trial');
    END;
    CREATE TABLE trials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trial_id TEXT NOT NULL REFERENCES hypotheses(trial_id),
        combo_hash TEXT NOT NULL,
        params TEXT NOT NULL,
        metrics TEXT NOT NULL,
        is_oos INTEGER NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX trials_by_trial ON trials(trial_id);
    CREATE TABLE receipts (
        trial_id TEXT NOT NULL REFERENCES hypotheses(trial_id),
        stage TEXT NOT NULL,
        receipt_hash TEXT NOT NULL,
        verdict TEXT NOT NULL,
        path TEXT NOT NULL,
        PRIMARY KEY (trial_id, stage)
    );
    CREATE TABLE overrides (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trial_id TEXT NOT NULL REFERENCES hypotheses(trial_id),
        stage TEXT NOT NULL,
        reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
        who TEXT,
        created_at TEXT NOT NULL
    );
    """,
    # 2: data catalog
    """
    CREATE TABLE datasets (
        name TEXT PRIMARY KEY,
        path TEXT NOT NULL,
        content_hash TEXT,
        schema TEXT NOT NULL,
        start TEXT,
        end TEXT,
        row_count INTEGER,
        null_rates TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('UNCHECKED', 'PASS', 'BLOCKED')),
        last_checked TEXT,
        reasons TEXT NOT NULL
    );
    """,
    # 3: strategy binding, dataset rules, check cache
    """
    ALTER TABLE hypotheses ADD COLUMN binding TEXT NOT NULL DEFAULT '{}';
    CREATE TRIGGER hypotheses_binding_immutable
    BEFORE UPDATE OF binding ON hypotheses
    BEGIN
        SELECT RAISE(ABORT, 'hypothesis fields are immutable; register a new trial');
    END;
    ALTER TABLE datasets ADD COLUMN rules_path TEXT;
    CREATE TABLE check_cache (
        key TEXT PRIMARY KEY,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    # 4: keyword search over hypotheses (FTS5, kept in sync by triggers)
    """
    CREATE VIRTUAL TABLE hypotheses_fts USING fts5(
        trial_id UNINDEXED, text, family, tokenize = 'porter unicode61'
    );
    INSERT INTO hypotheses_fts(trial_id, text, family)
        SELECT trial_id, text, strategy_family FROM hypotheses;
    CREATE TRIGGER hypotheses_fts_insert AFTER INSERT ON hypotheses
    BEGIN
        INSERT INTO hypotheses_fts(trial_id, text, family)
        VALUES (new.trial_id, new.text, new.strategy_family);
    END;
    CREATE TRIGGER hypotheses_fts_delete AFTER DELETE ON hypotheses
    BEGIN
        DELETE FROM hypotheses_fts WHERE trial_id = old.trial_id;
    END;
    """,
    # 5: overrides may come from a hook (no trial or stage), with the hook name and session
    """
    CREATE TABLE overrides_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trial_id TEXT REFERENCES hypotheses(trial_id),
        stage TEXT,
        reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
        who TEXT,
        created_at TEXT NOT NULL,
        hook TEXT,
        session_id TEXT
    );
    INSERT INTO overrides_new(id, trial_id, stage, reason, who, created_at)
        SELECT id, trial_id, stage, reason, who, created_at FROM overrides;
    DROP TABLE overrides;
    ALTER TABLE overrides_new RENAME TO overrides;
    """,
    # 6: test receipts (recorded by `qrh test-receipt`, read by the pre-restart guard)
    """
    CREATE TABLE test_receipts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_hash TEXT NOT NULL,
        command TEXT NOT NULL,
        exit_code INTEGER NOT NULL,
        passed INTEGER NOT NULL,
        duration_s REAL NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX test_receipts_by_hash ON test_receipts(source_hash);
    """,
    # 7: overrides granted by a human approval token record which token (older rows keep a NULL id)
    """
    ALTER TABLE overrides ADD COLUMN approval_id TEXT;
    """,
    # 8: experience packs (imported pre-harness attempts) and the interview framework
    """
    CREATE TABLE imported_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pack TEXT NOT NULL,
        pack_hash TEXT NOT NULL,
        attempt_hash TEXT NOT NULL,
        text TEXT NOT NULL,
        strategy_family TEXT NOT NULL,
        asset_class TEXT NOT NULL,
        frequency TEXT NOT NULL,
        family_id TEXT NOT NULL,
        variants_tried INTEGER NOT NULL,
        verdict TEXT NOT NULL,
        attempt_date TEXT NOT NULL,
        notes TEXT NOT NULL,
        imported INTEGER NOT NULL DEFAULT 1,
        UNIQUE (pack, attempt_hash)
    );
    CREATE INDEX imported_by_family ON imported_attempts(family_id);
    CREATE TABLE interview_questions (
        qid TEXT NOT NULL,
        scope_key TEXT NOT NULL,
        stage TEXT NOT NULL,
        text TEXT NOT NULL,
        why TEXT NOT NULL,
        origin TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (qid, scope_key)
    );
    CREATE TABLE interview_issued (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        qid TEXT NOT NULL,
        scope_key TEXT NOT NULL,
        issued_at TEXT NOT NULL
    );
    CREATE INDEX interview_issued_by_q ON interview_issued(qid, scope_key);
    CREATE TABLE interview_answers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        qid TEXT NOT NULL,
        scope_key TEXT NOT NULL,
        version INTEGER NOT NULL,
        identity TEXT NOT NULL,
        answer TEXT NOT NULL,
        answered_at TEXT NOT NULL,
        answered_by TEXT NOT NULL,
        provenance TEXT NOT NULL CHECK (provenance IN ('transcript-verified', 'human-cli')),
        session_id TEXT,
        mapped TEXT NOT NULL,
        extra_trials INTEGER NOT NULL DEFAULT 0,
        UNIQUE (qid, scope_key, version)
    );
    """,
    # 9: language-model strategies: the variant and prompt hash of every evaluated combination, and the
    # fingerprint (prompt files, models, params) of each trial so a later change is noticed at registration
    """
    ALTER TABLE trials ADD COLUMN variant TEXT;
    ALTER TABLE trials ADD COLUMN prompt_hash TEXT;
    CREATE TABLE llm_fingerprints (
        trial_id TEXT PRIMARY KEY REFERENCES hypotheses(trial_id),
        family_id TEXT NOT NULL,
        prompt_hash TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        notes TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX llm_fingerprints_by_family ON llm_fingerprints(family_id);
    """,
    # 10: dataset onboarding (P3e-a1): privacy flag, profiles per content hash, versioned instrument models
    """
    ALTER TABLE datasets ADD COLUMN privacy TEXT NOT NULL DEFAULT 'private' CHECK (privacy IN ('private', 'public'));
    CREATE TABLE dataset_profiles (
        content_hash TEXT NOT NULL,
        mode TEXT NOT NULL CHECK (mode IN ('values', 'no_values')),
        profile TEXT NOT NULL,
        profile_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (content_hash, mode)
    );
    CREATE TABLE onboarding_versions (
        name TEXT NOT NULL,
        version INTEGER NOT NULL,
        content_hash TEXT NOT NULL,
        model TEXT NOT NULL,
        model_hash TEXT NOT NULL,
        claims_hash TEXT NOT NULL,
        verified_at TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (name, version)
    );
    """,
]


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the database (WAL, foreign keys) and apply pending migrations."""
    try:
        conn = sqlite3.connect(str(path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current = row[0] or 0
        if current > len(MIGRATIONS):
            raise QrhHomeError(f"database schema {current} is newer than this qrh ({len(MIGRATIONS)})")
        for version in range(current + 1, len(MIGRATIONS) + 1):
            conn.execute("BEGIN")
            try:
                for stmt in _split(MIGRATIONS[version - 1]):
                    conn.execute(stmt)
                conn.execute("INSERT INTO schema_version(version) VALUES (?)", (version,))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
    except sqlite3.Error as exc:
        raise QrhHomeError(f"cannot open registry database {str(path)!r}: {exc}") from exc
    return conn


def _split(script: str) -> list[str]:
    """Split a migration into statements (triggers contain inner semicolons)."""
    out: list[str] = []
    buf: list[str] = []
    for line in script.splitlines():
        buf.append(line)
        candidate = "\n".join(buf).strip()
        if candidate and sqlite3.complete_statement(candidate):
            out.append(candidate)
            buf = []
    if "\n".join(buf).strip():
        raise ValueError("incomplete migration statement")
    return out
