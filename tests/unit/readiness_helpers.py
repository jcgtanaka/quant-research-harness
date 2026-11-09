"""Builders for synthetic trials, receipts and artifacts used by the readiness tests (all data is synthetic)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from qrh.memory.registry import HypothesisRecord, Registry
from qrh.pipeline.artifacts import Artifacts
from qrh.pipeline.receipt import Override, Receipt, Stage, Verdict, write_receipt

TID = "trial-synthetic"


def onboard(reg: Registry, name: str, status: str = "confirmed") -> None:
    """Give a registered dataset a complete, settled onboarding (every claim at ``status``), as a clean run has."""
    from qrh.onboarding.model import model_from_proposal, parse_proposal
    from qrh.onboarding import store
    from qrh.synthetic import instruments as inst

    _, proposal = inst.equity_daily()
    model = model_from_proposal(parse_proposal(proposal))
    for field, claim in model.claims().items():
        setattr(model, field, claim.model_copy(update={"status": status, "checked_by": "test"}))
    row = reg.conn.execute("SELECT content_hash FROM datasets WHERE name=?", (name,)).fetchone()
    version = store.add_version(reg.conn, name, row["content_hash"], model)
    store.update_version(reg.conn, name, version, model, verified=True)


def confirm_stability(reg: Registry, family: str = "trend|synthetic|1D", answer: str = "keep_default") -> None:
    """The trader's own answers to the four stability questions (a person typing them), as the intake records them."""
    from qrh.interview import engine
    from qrh.interview.stability import stability_questions
    from qrh.packs.runtime import PackView

    scope = f"strategy:{family}"
    for q in stability_questions():
        engine.record_answer(reg.conn, q=q, scope_key=scope, identity=engine.identity_for(reg.conn, PackView.empty(), scope),
                             value=answer, answered_by="human", provenance="human-cli", session_id=None)


def make_registry(home: Path, tid: str = TID, datasets: tuple[str, ...] = ("prices",), onboarded: bool = True,
                  stability: bool = True, thresholds: dict[str, Any] | None = None) -> Registry:
    """A registry with one synthetic trial. ``stability`` records the trader's confirmation of the four stability
    thresholds (their defaults), so that a clean run has no warning about them; pass False to leave them unanswered."""
    home.mkdir(parents=True, exist_ok=True)
    reg = Registry(home / "registry.sqlite3")
    reg.register_hypothesis(HypothesisRecord(
        text="A synthetic rule carries information.", strategy_family="trend", asset_class="synthetic",
        frequency="1D", parameter_space_size=9, thresholds=thresholds or {"max_pbo": 0.2},
        difference_statement="Baseline.", datasets=[{"name": d} for d in datasets],
    ), trial_id=tid)
    for d in datasets:
        reg.conn.execute(
            "INSERT OR REPLACE INTO datasets(name, path, content_hash, schema, null_rates, status, reasons)"
            " VALUES (?,?,?,?,?,?,?)", (d, f"/synthetic/{d}.parquet", f"hash-{d}", "{}", "{}", "PASS", "[]"))
        if onboarded:
            onboard(reg, d)
    if stability:
        confirm_stability(reg)
    return reg


def put_receipt(home: Path, reg: Registry | None, stage: str, verdict: Verdict = Verdict.PASS,
                params: dict[str, Any] | None = None, artifacts: dict[str, Any] | None = None,
                methods: dict[str, str] | None = None, override: Override | None = None,
                data_hashes: dict[str, str] | None = None, tid: str = TID, reasons: list[str] | None = None) -> Receipt:
    params = dict(params or {})
    if artifacts:
        arts = Artifacts(SimpleNamespace(home=home, trial_id=tid), Stage(stage))
        for name, obj in artifacts.items():
            arts.json(name, obj)
        params["artifacts"] = arts.as_params()
    receipt = Receipt(
        trial_id=tid, stage=Stage(stage), hypothesis_hash="h" * 64, code_hash="c" * 64, verdict=verdict,
        reasons=reasons or [], params=params, methods=methods or {}, override=override,
        data_hashes=data_hashes or {},
    )
    write_receipt(receipt, home, reg)
    return receipt


def clean_run(home: Path, reg: Registry, *, s2_methods: dict[str, str] | None = None,
              s2_params: dict[str, Any] | None = None) -> None:
    """S0..S7 all PASS, a lint-clean S2 that used the reconstruction, nothing unusual."""
    put_receipt(home, reg, "S0", params={"thresholds": {"max_pbo": 0.2}})
    put_receipt(home, reg, "S1", artifacts={"coverage": {"datasets": [
        {"name": "prices", "status": "PASS", "reasons": [], "gap_count": 0, "gaps": [],
         "rules": [{"rule": "range#1", "type": "range", "passed": True, "offenders": [], "reasons": []}]}]}},
        data_hashes={"prices": "hash-prices"})
    put_receipt(home, reg, "S2",
                methods={"s2.pnl_reconstruction": "pnl_reconstruction@1"} if s2_methods is None else s2_methods,
                params=s2_params if s2_params is not None else {"lint": {"verdict": "PASS", "warnings": [], "suppressed": []}})
    for s in ("S3", "S4", "S5", "S6"):
        put_receipt(home, reg, s)
    put_receipt(home, reg, "S7", params={"stage_verdicts": {}, "failing_stage": None})


def codes(collected: Any, kind: str | None = None) -> list[tuple[str, str]]:
    """(code, severity) of the warnings, optionally of one source kind."""
    return [(w.code, w.severity) for w in collected.warnings if kind is None or w.kind == kind]
