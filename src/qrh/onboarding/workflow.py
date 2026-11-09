"""The onboarding steps: profile, propose, verify, answer. Each one fails closed with an OnboardingError.

The profile is the only view of a registered file meant for the agent. A proposal is only ever a set of unverified
claims. Verification is deterministic code. What code cannot prove is settled by a human, through the interview
(``qrh interview answer``) or ``qrh data confirm``.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

from qrh.hashing import sha256_file
from qrh.onboarding import profile as prof
from qrh.onboarding import questions, store
from qrh.onboarding.model import (ENUMS, FIELDS, NUMBER_FIELDS, OPTION_KEYS, FieldClaim, InstrumentModel,
                                  OnboardingError, Proposal, model_from_proposal, normalize_value)
from qrh.onboarding.verify import verify_model

if TYPE_CHECKING:
    from qrh.memory.catalog import Catalog

CONFIRM = "confirm"
UNKNOWN = "unknown"


def _dataset(cat: "Catalog", name: str) -> dict[str, Any]:
    rec = cat.get_dataset(name)
    if rec is None:
        raise OnboardingError(f"unknown dataset {name!r}; register it first (qrh data register)")
    if not rec.get("content_hash"):
        raise OnboardingError(f"dataset {name!r} could not be read when it was registered; fix the file and register it again")
    return rec


def _unchanged_file(rec: dict[str, Any]) -> None:
    try:
        actual = sha256_file(rec["path"])
    except OSError as exc:
        raise OnboardingError(f"dataset file cannot be read: {exc}") from exc
    if actual != rec["content_hash"]:
        raise OnboardingError(f"dataset {rec['name']!r} changed since registration (hash mismatch); register it again")


# --- profile --------------------------------------------------------------------------------------------------

def profile_dataset(cat: "Catalog", name: str, no_values: bool = False) -> tuple[dict[str, Any], str]:
    """Profile a registered dataset (its file must be unchanged) and store the result under its content hash."""
    rec = _dataset(cat, name)
    _unchanged_file(rec)
    try:
        frame, tcol = prof.load_frame(rec["path"])
        profile = prof.build_profile(frame, no_values=no_values, time_column=tcol)
    except Exception as exc:
        raise OnboardingError(f"cannot profile {name!r}: {type(exc).__name__}: {exc}") from exc
    digest = prof.profile_hash(profile)
    store.save_profile(cat.conn, rec["content_hash"], profile, digest)
    return profile, digest


# --- propose --------------------------------------------------------------------------------------------------

def propose(cat: "Catalog", name: str, proposal: Proposal, *, as_user: bool = False) -> dict[str, Any]:
    """Store a validated proposal as unverified claims. The source is ``llm`` unless the file says ``user`` AND
    ``as_user`` is given (a human-only path: the approval guard refuses it to the agent)."""
    rec = _dataset(cat, name)
    if (proposal.source == "user") != bool(as_user):
        raise OnboardingError("a proposal from the user needs both \"source\": \"user\" in the file and --as-user; "
                              "a proposal from the agent has neither")
    model = model_from_proposal(proposal)
    current = store.latest_version(cat.conn, name, rec["content_hash"])
    if current is not None:
        for field, old in current["model"].claims().items():  # an unchanged, human-confirmed claim stays confirmed
            new = model.claims().get(field)
            if new is not None and old.status == "confirmed" and old.value == new.value and old.source == new.source:
                setattr(model, field, old.model_copy())
        if current["claims_hash"] == model.claims_hash():
            return {"version": current["version"], "created": False}
    version = store.add_version(cat.conn, name, rec["content_hash"], model)
    return {"version": version, "created": True}


# --- verify ---------------------------------------------------------------------------------------------------

def _reset(model: InstrumentModel) -> InstrumentModel:
    """The same claims with nothing verified or confirmed (the file they were made for has changed)."""
    fresh = model.model_copy(deep=True)
    for field, claim in fresh.claims().items():
        setattr(fresh, field, FieldClaim(value=claim.value, source=claim.source))
    return fresh


def _sync_dataset_status(cat: "Catalog", name: str) -> None:
    """Contradictions make the dataset BLOCKED (like a failed quality rule); clearing them re-runs its rules."""
    from qrh.checks.data_quality import check_dataset

    reasons = store.contradictions(cat.conn, name)
    status, old = cat.get_status(name)
    if reasons:
        keep = [r for r in old if not r.startswith("onboarding:")] if status == "BLOCKED" else []
        cat.set_status(name, "BLOCKED", keep + reasons)
    elif status == "BLOCKED" and any(r.startswith("onboarding:") for r in old):
        check_dataset(cat, name)


def verify(cat: "Catalog", name: str) -> dict[str, Any]:
    """Run the deterministic checks on the current onboarding version, record the results, and block the dataset on
    a contradiction. Creates a new version first when the file changed since the last one."""
    rec = _dataset(cat, name)
    _unchanged_file(rec)
    v = store.latest_version(cat.conn, name, rec["content_hash"])
    if v is None:
        stale = store.latest_version(cat.conn, name)
        if stale is None:
            raise OnboardingError(f"no proposal for {name!r}; run `qrh data propose {name} --file proposal.json`")
        store.add_version(cat.conn, name, rec["content_hash"], _reset(stale["model"]))
        v = store.latest_version(cat.conn, name, rec["content_hash"])
    assert v is not None
    model: InstrumentModel = v["model"]
    try:
        frame, tcol = prof.load_frame(rec["path"])
        profile = prof.build_profile(frame, no_values=False, time_column=tcol)
        offsets = prof.raw_offsets_minutes(rec["path"])
        checks = verify_model(model, frame, profile, offsets=offsets,
                              tz_from_file=not str(rec["path"]).lower().endswith(".csv"))
    except Exception as exc:
        raise OnboardingError(f"cannot verify {name!r}: {type(exc).__name__}: {exc}") from exc
    for field, claim in model.claims().items():
        check = checks[field]
        if check.status == "unverified" and claim.status == "confirmed":
            continue  # the human's confirmation stands where code has nothing to say
        setattr(model, field, claim.model_copy(update={
            "status": check.status, "evidence": check.evidence, "checked_by": check.checked_by}))
    store.update_version(cat.conn, name, v["version"], model, verified=True)
    _sync_dataset_status(cat, name)
    return describe(cat, name) | {"contradictions": store.contradictions(cat.conn, name)}


# --- answers --------------------------------------------------------------------------------------------------

def parse_value(field: str, text: str) -> Any:
    """The value a human typed for a field, normalized; raises OnboardingError."""
    t = text.strip()
    if not t:
        raise OnboardingError("the answer is empty")
    if field == "price_columns":
        return normalize_value(field, [c.strip() for c in t.split(",") if c.strip()])
    if field in NUMBER_FIELDS:
        try:
            n = float(t.replace(",", ""))
        except ValueError:
            raise OnboardingError(f"{t!r} is not a number") from None
        return normalize_value(field, int(n) if n == int(n) else n)
    if field == "option_mapping":
        pairs = [p.split("=", 1) for p in t.split(",") if p.strip()]
        if not pairs or any(len(p) != 2 for p in pairs):
            raise OnboardingError("give the option mapping as key=column pairs, for example strike=K,expiry=E "
                                  f"(keys: {', '.join(OPTION_KEYS)})")
        return normalize_value(field, {k.strip(): v.strip() for k, v in pairs})
    if field in ENUMS:
        return normalize_value(field, t)
    return normalize_value(field, t)


def check_answer(conn: sqlite3.Connection, name: str, field: str, text: str) -> tuple[str, Any]:
    """Validate an answer before anything is recorded: ("confirm"|"unknown"|"value", normalized value)."""
    if field not in FIELDS:
        raise OnboardingError(f"unknown field {field!r}")
    v = store.current_version(conn, name)
    if v is None:
        raise OnboardingError(f"dataset {name!r} has no onboarding for its current file")
    t = text.strip()
    if t.casefold() == UNKNOWN:
        return UNKNOWN, None
    if t.casefold() == CONFIRM:
        if field not in v["model"].claims():
            raise OnboardingError(f"nothing to confirm: no {field} was proposed; give the value instead")
        return CONFIRM, None
    value = parse_value(field, t)
    claim = v["model"].claims().get(field)
    if claim is not None and claim.value == value:
        return CONFIRM, None
    return "value", value


def apply_answer(cat: "Catalog", name: str, field: str, text: str, provenance: str) -> dict[str, Any]:
    """Apply a human's answer to a field: ``confirm`` confirms the claim, a value corrects it (a new version, then
    re-verified, and the correction stands as confirmed where code cannot check it), ``unknown`` changes nothing."""
    kind, value = check_answer(cat.conn, name, field, text)
    v = store.current_version(cat.conn, name)
    assert v is not None
    who = f"human:{provenance}"
    if kind == UNKNOWN:
        return {"action": UNKNOWN, "version": v["version"]}
    model: InstrumentModel = v["model"].model_copy(deep=True)
    if kind == CONFIRM:
        claim = model.claims()[field]
        if claim.status == "contradicted":
            raise OnboardingError(f"{field} is contradicted by the data; a confirmation cannot override that "
                                  "(correct the claim instead)")
        setattr(model, field, claim.model_copy(update={
            "status": "confirmed", "checked_by": who, "evidence": f"confirmed by the user ({provenance})"}))
        store.update_version(cat.conn, name, v["version"], model)
        return {"action": "confirmed", "version": v["version"]}
    corrected = FieldClaim(value=value, source="user", evidence=f"given by the user ({provenance})")
    setattr(model, field, corrected)
    model = InstrumentModel.model_validate(model.to_doc())
    carried = store.latest_version(cat.conn, name, v["content_hash"])
    assert carried is not None
    new_version = store.add_version(cat.conn, name, v["content_hash"], model)
    out = verify(cat, name)
    latest = store.get_version(cat.conn, name, new_version)
    assert latest is not None
    final: InstrumentModel = latest["model"]
    claim = final.claims()[field]
    if claim.status == "unverified":  # code has nothing to say about it: the user's word stands
        setattr(final, field, claim.model_copy(update={"status": "confirmed", "checked_by": who,
                                                       "evidence": f"given by the user ({provenance}); not provable from data"}))
        store.update_version(cat.conn, name, new_version, final)
    return {"action": "corrected", "version": new_version, "state": out["state"]}


def answer_question(cat: "Catalog", name: str, field: str, value: str, *, answered_by: str, provenance: str,
                    session_id: str | None = None) -> dict[str, Any]:
    """Apply a human's answer to the open question about ``field`` and record it in the interview (with the identity
    of the claim after the change, so it stays valid until the file or the claim changes again)."""
    from qrh.interview import engine

    scope, qid = f"dataset:{name}", questions.qid_for(field)
    found = questions.find(cat.conn, qid, scope)
    if found is None:
        raise OnboardingError(f"no open question {qid} for {scope}; run `qrh data verify {name}` first, or the field "
                              "is already verified by code")
    q = found[0]
    try:
        answer, _choice = engine.canonical_answer(q, value)
    except engine.InterviewError as exc:
        raise OnboardingError(str(exc)) from exc
    check_answer(cat.conn, name, field, answer)
    result = apply_answer(cat, name, field, answer, provenance)
    identity = questions.answer_identity(cat.conn, qid, scope) or ""
    engine.record_answer(cat.conn, q=q, scope_key=scope, identity=identity, value=answer, answered_by=answered_by,
                         provenance=provenance, session_id=session_id)
    return result


# --- description ----------------------------------------------------------------------------------------------

def describe(cat: "Catalog", name: str) -> dict[str, Any]:
    """Everything `qrh data onboarding` shows: the state, a row per field, and the questions still open."""
    from qrh.interview import engine

    rec = cat.get_dataset(name)
    if rec is None:
        raise OnboardingError(f"unknown dataset {name!r}")
    v = store.current_version(cat.conn, name)
    base: dict[str, Any] = {"dataset": name, "privacy": rec.get("privacy", "private"), "content_hash": rec["content_hash"]}
    if v is None:
        stale = store.latest_version(cat.conn, name)
        return {**base, "state": "NOT_ONBOARDED", "version": None, "model_hash": None, "fields": [],
                "pending_questions": [], "stale_version": stale["version"] if stale else None}
    pending = []
    for q, scope, crit, ident in questions.dataset_questions(cat.conn, {f"dataset:{name}"}):
        ans = engine.latest_answer(cat.conn, q.id, scope)
        if ans is None or ans["identity"] != ident:
            status = "pending"
        elif ans["answer"] == engine.UNKNOWN:
            status = "unknown"
        else:
            continue
        pending.append({"id": q.id, "scope": scope, "critical": crit, "status": status, "text": q.text})
    return {**base, "state": store.state_of(v["model"], v["verified_at"]), "version": v["version"],
            "model_hash": v["model_hash"], "fields": store.field_rows(v["model"]), "pending_questions": pending,
            "verified_at": v["verified_at"]}
