"""Trial registry: hypotheses, evaluated combinations, receipt index, overrides."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qrh.hashing import hash_json
from qrh.llm.spec import LLMSpec
from qrh.memory import db


class RegistryError(ValueError):
    """Invalid registry operation."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_id() -> str:
    return uuid.uuid4().hex


BOUND_KEYS = ("max_abs_period_return", "max_gross_leverage", "max_position_weight")


class DatasetSpec(BaseModel):
    """A dataset the strategy uses. ``slow`` datasets must declare a publication column."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    slow: bool = False
    publication_column: str | None = None


class HypothesisRecord(BaseModel):
    """What is registered before any data is touched. Thresholds are frozen once stored."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    strategy_family: str = Field(min_length=1)
    asset_class: str = Field(min_length=1)
    frequency: str = Field(min_length=1)
    parameter_space_size: int = Field(ge=1)
    thresholds: dict[str, Any]
    difference_statement: str = Field(min_length=1)
    family_id: str | None = None
    # Strategy binding (optional, so older specs stay valid).
    adapter: str | None = None
    datasets: list[DatasetSpec] = Field(default_factory=list)
    windows: dict[str, tuple[str, str]] = Field(default_factory=dict)
    bounds: dict[str, float] = Field(default_factory=dict)
    s2_params: dict[str, Any] | None = None
    periods_per_year: float | None = Field(default=None, gt=0)  # else inferred from timestamps
    walk_forward: dict[str, Any] | None = None  # {train, test, step, anchored}, checked at S4
    execution: dict[str, Any] | None = None  # {lag_bars, price_column(s)}: read by s2.pnl_reconstruction
    # Settings block of the opt-in pnl.linear method. Only its shape is checked here; the method validates the
    # contents at S2 and answers BLOCKED, so a defect is a verdict and not a rejected registration.
    reconstruction: dict[str, Any] | None = None
    methods: dict[str, str] = Field(default_factory=dict)  # pinned method versions, {"s5.pbo": "cscv@1"}
    # Optional help for the S5 parameter stability check of one pre-registered combination: explicit neighbour values per
    # parameter, and [low, high] bounds that generated neighbours are clipped to. Absent means generated, unbounded.
    stability: dict[str, Any] | None = None
    llm: LLMSpec | None = None  # models, prompts and parameters of a strategy that uses a language model
    # Track 2: "user" when the person brought the backtest (the interview then asks the fixed Track 2 questions).
    # Absent or "harness" means the harness built it; neither is stored or hashed, so those hypotheses are unchanged.
    origin: Literal["user", "harness"] | None = None
    base_dir: str | None = None  # where the spec was loaded from; location only, never hashed

    @field_validator("datasets", mode="before")
    @classmethod
    def _dataset_entries(cls, v: Any) -> Any:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("must be a list")
        return [{"name": x} if isinstance(x, str) else x for x in v]

    @field_validator("adapter")
    @classmethod
    def _adapter_ref(cls, v: str | None) -> str | None:
        if v is None:
            return v
        module, sep, cls_name = v.partition(":")
        if not sep or not module.strip() or not cls_name.strip() or ":" in cls_name:
            raise ValueError("must look like 'module.path:ClassName'")
        return v

    @field_validator("execution")
    @classmethod
    def _execution(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is not None:
            from qrh.checks.pnl_reconstruction import validate_execution

            validate_execution(v)
        return v

    @field_validator("methods", mode="before")
    @classmethod
    def _methods(cls, v: Any) -> Any:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("must be a mapping of method id to version, for example {s5.pbo: cscv@1}")
        from qrh.methods.registry import MethodError, parse_ref

        for mid, version in v.items():
            try:
                parse_ref(f"{mid}={version}")
            except MethodError as exc:
                raise ValueError(str(exc)) from exc
        return dict(v)

    @field_validator("stability")
    @classmethod
    def _stability(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is None:
            return v
        from qrh.validation.param_stability import is_number

        if not isinstance(v, dict) or set(v) - {"neighbours", "bounds"}:
            raise ValueError("must be a mapping with only `neighbours` and/or `bounds`")
        nb = v.get("neighbours", {})
        if not isinstance(nb, dict) or any(
                not isinstance(vals, list) or not vals or not all(is_number(x) for x in vals) for vals in nb.values()):
            raise ValueError("neighbours must map each parameter to a non-empty list of numbers")
        bd = v.get("bounds", {})
        if not isinstance(bd, dict) or any(
                not isinstance(b, list) or len(b) != 2 or not all(is_number(x) for x in b) or not b[0] <= b[1]
                for b in bd.values()):
            raise ValueError("bounds must map each parameter to [low, high] with low <= high")
        return v

    @field_validator("bounds")
    @classmethod
    def _bounds(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(v) - set(BOUND_KEYS))
        if unknown:
            raise ValueError(f"unknown bound(s) {unknown}; allowed: {list(BOUND_KEYS)}")
        if any(not x > 0 for x in v.values()):
            raise ValueError("bounds must be positive")
        return v

    @model_validator(mode="after")
    def _windows(self) -> "HypothesisRecord":
        for name, (start, end) in self.windows.items():
            try:
                a, b = datetime.fromisoformat(start), datetime.fromisoformat(end)
            except ValueError as exc:
                raise ValueError(f"windows.{name}: not ISO timestamps") from exc
            if (a.tzinfo is None) != (b.tzinfo is None):
                raise ValueError(f"windows.{name}: mix of naive and aware timestamps")
            if not a < b:
                raise ValueError(f"windows.{name}: start must be before end")
        names = [d.name for d in self.datasets]
        if len(set(names)) != len(names):
            raise ValueError("datasets: duplicate names")
        return self

    def binding(self) -> dict[str, Any]:
        """Binding fields in storable form; empty ones are omitted."""
        out: dict[str, Any] = {
            "adapter": self.adapter,
            "datasets": [d.model_dump() for d in self.datasets],
            "windows": {k: list(v) for k, v in self.windows.items()},
            "bounds": dict(self.bounds),
            "s2_params": self.s2_params,
            "periods_per_year": self.periods_per_year,
            "walk_forward": self.walk_forward,
            "methods": dict(self.methods),
            "execution": self.execution,
            "reconstruction": self.reconstruction,
            "llm": self.llm.model_dump(mode="json") if self.llm is not None else None,
            "origin": "user" if self.origin == "user" else None,
            "stability": self.stability,
        }
        return {k: v for k, v in out.items() if v not in (None, [], {})}

    def resolved_family_id(self) -> str:
        return self.family_id or f"{self.strategy_family}|{self.asset_class}|{self.frequency}"

    def hypothesis_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"family_id"})
        payload["family_id"] = self.resolved_family_id()
        for k in (
            "adapter", "datasets", "windows", "bounds", "s2_params", "periods_per_year",
            "walk_forward", "methods", "execution", "reconstruction", "llm", "origin", "stability", "base_dir",
        ):
            payload.pop(k, None)
        # Binding is hashed only when present, so older hypotheses keep their hash.
        payload.update(self.binding())
        return hash_json(payload)


class Registry:
    def __init__(self, path: str | Path, id_factory: Callable[[], str] | None = None):
        self.path = Path(path)
        self.conn = db.connect(self.path)
        self._id_factory = id_factory or default_id

    def close(self) -> None:
        self.conn.close()

    def new_id(self) -> str:
        return self._id_factory()

    # hypotheses ---------------------------------------------------------
    def register_hypothesis(self, rec: HypothesisRecord, trial_id: str | None = None) -> str:
        tid = trial_id or self.new_id()
        self.conn.execute(
            "INSERT INTO hypotheses(trial_id, text, strategy_family, asset_class, frequency,"
            " parameter_space_size, thresholds, difference_statement, family_id, hypothesis_hash,"
            " created_at, binding) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                tid,
                rec.text,
                rec.strategy_family,
                rec.asset_class,
                rec.frequency,
                rec.parameter_space_size,
                json.dumps(rec.thresholds, sort_keys=True),
                rec.difference_statement,
                rec.resolved_family_id(),
                rec.hypothesis_hash(),
                _now(),
                json.dumps({**rec.binding(), "base_dir": rec.base_dir}, sort_keys=True),
            ),
        )
        return tid

    def get_trial(self, trial_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM hypotheses WHERE trial_id=?", (trial_id,)).fetchone()
        return _hyp(row) if row else None

    def list_trials(
        self,
        *,
        asset_class: str | None = None,
        strategy_family: str | None = None,
        frequency: str | None = None,
        verdict: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Newest first. ``verdict`` is that of the highest-numbered stage receipt."""
        sql = (
            "SELECT h.*, (SELECT r.verdict FROM receipts r WHERE r.trial_id=h.trial_id"
            " ORDER BY r.stage DESC LIMIT 1) AS verdict FROM hypotheses h"
        )
        where, args = [], []
        for col, val in (
            ("asset_class", asset_class),
            ("strategy_family", strategy_family),
            ("frequency", frequency),
        ):
            if val is not None:
                where.append(f"h.{col}=?")
                args.append(val)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY h.rowid DESC"
        rows = [_hyp(r) | {"verdict": r["verdict"]} for r in self.conn.execute(sql, args)]
        if verdict is not None:
            rows = [r for r in rows if r["verdict"] == verdict]
        return rows[:limit] if limit is not None else rows

    # evaluated combinations --------------------------------------------
    def record_trial(
        self, trial_id: str, params: dict[str, Any], metrics: dict[str, float], is_oos: bool = False
    ) -> str:
        combo = hash_json(params)
        variant = params.get("llm_variant") if isinstance(params.get("llm_variant"), dict) else {}
        self.conn.execute(
            "INSERT INTO trials(trial_id, combo_hash, params, metrics, is_oos, created_at, variant, prompt_hash)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (trial_id, combo, json.dumps(params, sort_keys=True), json.dumps(metrics, sort_keys=True),
             int(is_oos), _now(), variant.get("name"), variant.get("prompt_hash")),
        )
        return combo

    def clear_trial_combos(self, trial_id: str, is_oos: bool = False) -> None:
        """Drop the logged combinations of one trial (a stage re-run replaces its own log)."""
        self.conn.execute("DELETE FROM trials WHERE trial_id=? AND is_oos=?", (trial_id, int(is_oos)))

    def trial_count(self, trial_id: str) -> int:
        """Distinct parameter combinations evaluated under a trial."""
        return self.conn.execute(
            "SELECT COUNT(DISTINCT combo_hash) FROM trials WHERE trial_id=?", (trial_id,)
        ).fetchone()[0]

    def _outside_trials(self, family_id: str) -> int:
        """Variants of imported pre-harness attempts plus the outside-the-harness variants a human reported."""
        imported = self.conn.execute(
            "SELECT COALESCE(SUM(variants_tried), 0) FROM imported_attempts WHERE family_id=?", (family_id,)
        ).fetchone()[0]
        answered = self.conn.execute(
            "SELECT COALESCE(SUM(a.extra_trials), 0) FROM interview_answers a WHERE a.scope_key=?"
            " AND a.version=(SELECT MAX(b.version) FROM interview_answers b WHERE b.qid=a.qid AND b.scope_key=a.scope_key)",
            (f"strategy:{family_id}",),
        ).fetchone()[0]
        return int(imported + answered)

    def family_trial_count(self, family_id: str) -> int:
        """Distinct (trial, combination) pairs evaluated across a family, plus the variants of imported
        pre-harness attempts and the outside-the-harness variants a human reported in an interview."""
        logged = self.conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT t.trial_id, t.combo_hash FROM trials t"
            " JOIN hypotheses h ON h.trial_id=t.trial_id WHERE h.family_id=?)",
            (family_id,),
        ).fetchone()[0]
        return int(logged) + self._outside_trials(family_id)

    def family_declared_count(self, family_id: str) -> int:
        """What the family count becomes once every registered trial has run: the declared parameter space of each
        (S3 enforces that the evaluated grid equals it), plus the imported and outside-the-harness variants."""
        declared = self.conn.execute(
            "SELECT COALESCE(SUM(parameter_space_size), 0) FROM hypotheses WHERE family_id=?", (family_id,)
        ).fetchone()[0]
        return int(declared) + self._outside_trials(family_id)

    # language-model fingerprints ---------------------------------------------------------------------
    def record_llm_fingerprint(
        self, trial_id: str, family_id: str, prompt_hash: str, fingerprint: str, notes: list[str]
    ) -> None:
        """Written once, when the trial is registered (a re-run reads it back, so the S0 receipt cannot drift)."""
        self.conn.execute(
            "INSERT INTO llm_fingerprints(trial_id, family_id, prompt_hash, fingerprint, notes, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (trial_id, family_id, prompt_hash, fingerprint, json.dumps(notes), _now()),
        )

    def get_llm_fingerprint(self, trial_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM llm_fingerprints WHERE trial_id=?", (trial_id,)).fetchone()
        if row is None:
            return None
        return {"trial_id": row["trial_id"], "family_id": row["family_id"], "prompt_hash": row["prompt_hash"],
                "fingerprint": row["fingerprint"], "notes": json.loads(row["notes"])}

    def llm_trials_differing(self, family_id: str, trial_id: str, fingerprint: str) -> list[str]:
        """Other trials of the family that declared an llm with a different fingerprint, newest first."""
        rows = self.conn.execute(
            "SELECT trial_id FROM llm_fingerprints WHERE family_id=? AND trial_id<>? AND fingerprint<>?"
            " ORDER BY rowid DESC", (family_id, trial_id, fingerprint),
        ).fetchall()
        return [r["trial_id"] for r in rows]

    # receipts index -----------------------------------------------------
    def index_receipt(self, trial_id: str, stage: str, receipt_hash: str, verdict: str, path: str) -> None:
        self.conn.execute(
            "INSERT INTO receipts(trial_id, stage, receipt_hash, verdict, path) VALUES (?,?,?,?,?)"
            " ON CONFLICT(trial_id, stage) DO UPDATE SET receipt_hash=excluded.receipt_hash,"
            " verdict=excluded.verdict, path=excluded.path",
            (trial_id, stage, receipt_hash, verdict, path),
        )

    def get_receipt_entry(self, trial_id: str, stage: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM receipts WHERE trial_id=? AND stage=?", (trial_id, stage)
        ).fetchone()
        return dict(row) if row else None

    def list_receipts(self, trial_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM receipts WHERE trial_id=? ORDER BY stage", (trial_id,))
        return [dict(r) for r in rows]

    def clear_receipts(self, trial_id: str, from_stage: str) -> list[str]:
        """Drop index rows for ``from_stage`` and later; return their file paths."""
        rows = self.conn.execute(
            "SELECT path FROM receipts WHERE trial_id=? AND stage>=?", (trial_id, from_stage)
        ).fetchall()
        self.conn.execute("DELETE FROM receipts WHERE trial_id=? AND stage>=?", (trial_id, from_stage))
        return [r["path"] for r in rows]

    # check cache ---------------------------------------------------------
    def cache_get(self, key: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT payload FROM check_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def cache_put(self, key: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO check_cache(key, payload, created_at) VALUES (?,?,?)"
            " ON CONFLICT(key) DO UPDATE SET payload=excluded.payload, created_at=excluded.created_at",
            (key, json.dumps(payload, sort_keys=True), _now()),
        )

    # overrides ----------------------------------------------------------
    def log_override(self, trial_id: str, stage: str, reason: str, who: str | None = None) -> None:
        if not reason or not reason.strip():
            raise RegistryError("an override requires a non-empty reason")
        self.conn.execute(
            "INSERT INTO overrides(trial_id, stage, reason, who, created_at) VALUES (?,?,?,?,?)",
            (trial_id, stage, reason, who, _now()),
        )

    def list_overrides(self, trial_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM overrides WHERE trial_id=? ORDER BY id", (trial_id,))
        return [dict(r) for r in rows]

    def log_hook_override(
        self, hook: str, reason: str, session_id: str | None, trial_id: str | None = None,
        approval_id: str | None = None,
    ) -> None:
        """Record an override granted to a hook by a human approval token; never silent, reason mandatory."""
        if not reason or not reason.strip():
            raise RegistryError("an override requires a non-empty reason")
        self.conn.execute(
            "INSERT INTO overrides(trial_id, stage, reason, who, created_at, hook, session_id, approval_id)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (trial_id, None, reason, "hook:" + hook, _now(), hook, session_id or "", approval_id),
        )

    def list_hook_overrides(self, since: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM overrides WHERE hook IS NOT NULL AND created_at>=? ORDER BY id", (since,)
        )
        return [dict(r) for r in rows]

    # test receipts ------------------------------------------------------
    def record_test_receipt(self, source_hash: str, command: str, exit_code: int, duration_s: float) -> None:
        self.conn.execute(
            "INSERT INTO test_receipts(source_hash, command, exit_code, passed, duration_s, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (source_hash, command, exit_code, int(exit_code == 0), duration_s, _now()),
        )

    def latest_passing_test_receipt(self, source_hash: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM test_receipts WHERE source_hash=? AND passed=1 ORDER BY id DESC LIMIT 1",
            (source_hash,),
        ).fetchone()
        return dict(row) if row else None


def _hyp(row: sqlite3.Row) -> dict[str, Any]:
    d = {k: row[k] for k in (
        "trial_id", "text", "strategy_family", "asset_class", "frequency", "parameter_space_size",
        "difference_statement", "family_id", "hypothesis_hash", "created_at")}
    d["thresholds"] = json.loads(row["thresholds"])
    binding = json.loads(row["binding"])
    d["adapter"] = binding.get("adapter")
    d["datasets"] = binding.get("datasets", [])
    d["windows"] = {k: tuple(v) for k, v in binding.get("windows", {}).items()}
    d["bounds"] = binding.get("bounds", {})
    d["s2_params"] = binding.get("s2_params")
    d["periods_per_year"] = binding.get("periods_per_year")
    d["walk_forward"] = binding.get("walk_forward")
    d["methods"] = binding.get("methods", {})
    d["execution"] = binding.get("execution")
    d["reconstruction"] = binding.get("reconstruction")
    d["llm"] = binding.get("llm")
    d["origin"] = binding.get("origin")
    d["stability"] = binding.get("stability")
    d["base_dir"] = binding.get("base_dir")
    return d
