"""The instrument model: what a dataset is, as claims that code verifies and a human confirms.

An LLM (the coding agent) may INTERPRET a dataset and write a proposal; deterministic code VERIFIES every claim it
can; the human CONFIRMS what code cannot prove. A proposal file is untrusted input: the schema is strict, unknown
keys are rejected, every string is plain text with a length limit, and nothing in it is ever interpreted as markup.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from qrh.hashing import hash_json

SCHEMA_VERSION = 1
MAX_PROPOSAL_BYTES = 64_000

ASSET_CLASSES = ("equity", "etf", "fx_spot", "crypto_spot", "future", "option", "crypto_perp", "index", "rate",
                 "macro", "other")
PRICE_UNITS = ("fraction", "percent", "currency", "index_points", "rate_percent", "other")
TIMESTAMP_SEMANTICS = ("bar_open", "bar_close", "event_time", "publication_time", "unknown")
ADJUSTMENTS = ("none", "split_only", "split_and_dividend", "back_adjusted", "forward_adjusted", "unknown")
OPTION_KEYS = ("underlying", "strike", "expiry", "right", "mid", "bid", "ask", "rate")
SOURCES = ("profile", "llm", "user")
STATUSES = ("unverified", "verified", "contradicted", "confirmed")
FIELDS = ("asset_class", "price_columns", "price_units", "quote_currency", "timestamp_semantics", "timezone",
          "session_hint", "adjustments", "multiplier", "tick_size", "frequency", "roll_rule", "option_mapping")
CRITICAL = ("multiplier", "adjustments", "timestamp_semantics", "price_units")
ENUMS = {"asset_class": ASSET_CLASSES, "price_units": PRICE_UNITS, "timestamp_semantics": TIMESTAMP_SEMANTICS,
         "adjustments": ADJUSTMENTS}
NUMBER_FIELDS = ("multiplier", "tick_size")
_CURRENCY = re.compile(r"^[A-Za-z0-9]{2,10}$")
_EXTRA_KEY = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
_BAD_CATEGORIES = {"Cc", "Cf", "Co", "Cs", "Zl", "Zp"}


class OnboardingError(ValueError):
    """An onboarding input was refused (the message says why)."""


def plain(text: Any, limit: int, what: str, allow_empty: bool = False) -> str:
    """A string that is shown as plain text: no control or format characters (bidi overrides, zero-width), no ``$``
    (the PDF renderer would read it as math), bounded length."""
    if not isinstance(text, str):
        raise OnboardingError(f"{what}: must be a string")
    if not text and not allow_empty:
        raise OnboardingError(f"{what}: must not be empty")
    if len(text) > limit:
        raise OnboardingError(f"{what}: {len(text)} characters, the limit is {limit}")
    if "$" in text:
        raise OnboardingError(f"{what}: the character $ is not allowed")
    if any(unicodedata.category(ch) in _BAD_CATEGORIES for ch in text):
        raise OnboardingError(f"{what}: control or invisible characters are not allowed")
    return text


def _number(value: Any, what: str) -> float | int | str:
    if value == "unknown" and isinstance(value, str):
        return "unknown"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OnboardingError(f"{what}: must be a positive number or \"unknown\"")
    if not math.isfinite(value) or value <= 0:
        raise OnboardingError(f"{what}: must be a positive finite number or \"unknown\"")
    return value


def normalize_value(field: str, value: Any) -> Any:
    """Validate (and normalize) the value of one field; raises OnboardingError."""
    what = field
    if field in ENUMS:
        if not isinstance(value, str) or value not in ENUMS[field]:
            raise OnboardingError(f"{what}: must be one of {', '.join(ENUMS[field])}")
        return value
    if field == "price_columns":
        if not isinstance(value, list) or not 1 <= len(value) <= 12:
            raise OnboardingError(f"{what}: must be a list of 1 to 12 column names")
        names = [plain(v, 64, f"{what} entry") for v in value]
        if len(set(names)) != len(names):
            raise OnboardingError(f"{what}: duplicate column names")
        return names
    if field == "quote_currency":
        text = plain(value, 10, what)
        if not _CURRENCY.match(text):
            raise OnboardingError(f"{what}: must be 2 to 10 letters or digits")
        return text
    if field == "timezone":
        return plain(value, 64, what)
    if field == "session_hint":
        return plain(value, 200, what)
    if field == "frequency":
        return plain(value, 16, what)
    if field in NUMBER_FIELDS:
        return _number(value, what)
    if field == "roll_rule":
        if isinstance(value, str):
            value = {"text": value}
        if not isinstance(value, dict) or not set(value) <= {"text", "method"} or "text" not in value:
            raise OnboardingError(f"{what}: must be text, or an object with text and an optional method id")
        out = {"text": plain(value["text"], 200, f"{what} text")}
        if value.get("method") is not None:
            out["method"] = plain(value["method"], 40, f"{what} method")
        return out
    if field == "option_mapping":
        if not isinstance(value, dict) or not value or not set(value) <= set(OPTION_KEYS):
            raise OnboardingError(f"{what}: must be an object with keys from {', '.join(OPTION_KEYS)}")
        return {k: plain(v, 64, f"{what}.{k}") for k, v in value.items()}
    raise OnboardingError(f"unknown field {field!r}")


def claim_text(field: str, value: Any) -> str:
    """The value of a claim as one line of plain text."""
    if field == "price_columns":
        return ", ".join(value)
    if field == "roll_rule":
        return value["text"] + (f" ({value['method']})" if value.get("method") else "")
    if field == "option_mapping":
        return "; ".join(f"{k}={value[k]}" for k in sorted(value))
    return str(value)


class FieldClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: Any
    source: Literal["profile", "llm", "user"]
    evidence: str = Field(default="", max_length=300)
    status: Literal["unverified", "verified", "contradicted", "confirmed"] = "unverified"
    checked_by: str = Field(default="", max_length=80)

    @model_validator(mode="after")
    def _text(self) -> "FieldClaim":
        plain(self.evidence, 300, "evidence", allow_empty=True)
        plain(self.checked_by, 80, "checked_by", allow_empty=True)
        return self


class InstrumentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    asset_class: FieldClaim | None = None
    price_columns: FieldClaim | None = None
    price_units: FieldClaim | None = None
    quote_currency: FieldClaim | None = None
    timestamp_semantics: FieldClaim | None = None
    timezone: FieldClaim | None = None
    session_hint: FieldClaim | None = None
    adjustments: FieldClaim | None = None
    multiplier: FieldClaim | None = None
    tick_size: FieldClaim | None = None
    frequency: FieldClaim | None = None
    roll_rule: FieldClaim | None = None
    option_mapping: FieldClaim | None = None
    extras: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _values(self) -> "InstrumentModel":
        for name, claim in self.claims().items():
            try:
                claim.value = normalize_value(name, claim.value)
            except OnboardingError as exc:
                raise ValueError(str(exc)) from exc
        if len(self.extras) > 8:
            raise ValueError("extras: at most 8 entries")
        for k, v in self.extras.items():
            if not _EXTRA_KEY.match(k):
                raise ValueError(f"extras: bad key {k!r}")
            try:
                plain(v, 120, f"extras.{k}")
            except OnboardingError as exc:
                raise ValueError(str(exc)) from exc
        return self

    def claims(self) -> dict[str, FieldClaim]:
        return {f: getattr(self, f) for f in FIELDS if getattr(self, f) is not None}

    def to_doc(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def claims_hash(self) -> str:
        """Identity of what is claimed (values, sources, extras), not of how far it is verified."""
        return hash_json({"v": SCHEMA_VERSION, "claims": {f: [c.value, c.source] for f, c in self.claims().items()},
                          "extras": self.extras})

    def model_hash(self) -> str:
        """Identity of the claims together with their verification state."""
        return hash_json(self.to_doc())


class ProposedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: Any
    evidence: str = ""


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source: Literal["llm", "user"] = "llm"
    fields: dict[str, ProposedClaim]
    extras: dict[str, str] = Field(default_factory=dict)


def parse_proposal(doc: Any) -> Proposal:
    """Validate an already parsed proposal document; raises OnboardingError with every problem found."""
    if not isinstance(doc, dict):
        raise OnboardingError("the proposal must be a JSON object")
    try:
        proposal = Proposal.model_validate(doc)
    except ValidationError as exc:
        raise OnboardingError("; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())) from exc
    unknown = sorted(set(proposal.fields) - set(FIELDS))
    if unknown:
        raise OnboardingError(f"unknown field(s) {unknown}; allowed: {', '.join(FIELDS)}")
    if not proposal.fields:
        raise OnboardingError("the proposal claims no field")
    for name, claim in proposal.fields.items():
        claim.value = normalize_value(name, claim.value)
        plain(claim.evidence, 300, f"{name} evidence", allow_empty=True)
    if len(proposal.extras) > 8:
        raise OnboardingError("extras: at most 8 entries")
    for k, v in proposal.extras.items():
        if not _EXTRA_KEY.match(k):
            raise OnboardingError(f"extras: bad key {k!r}")
        plain(v, 120, f"extras.{k}")
    return proposal


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise OnboardingError(f"duplicate key {k!r} in the proposal")
        out[k] = v
    return out


def _constant(name: str) -> Any:
    raise OnboardingError(f"the constant {name} is not allowed in the proposal (numbers must be finite)")


def load_proposal(path: str | Path) -> Proposal:
    """Read and validate a proposal file (JSON only, size limited, duplicate keys and NaN refused)."""
    p = Path(path)
    try:
        size = p.stat().st_size
        if size > MAX_PROPOSAL_BYTES:
            raise OnboardingError(f"proposal file too large ({size} bytes, the limit is {MAX_PROPOSAL_BYTES})")
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise OnboardingError(f"cannot read the proposal file {str(path)!r}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise OnboardingError(f"the proposal is not valid UTF-8 text: {exc}") from exc
    try:
        doc = json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_constant)
    except json.JSONDecodeError as exc:
        raise OnboardingError(f"the proposal is not valid JSON: {exc}") from exc
    return parse_proposal(doc)


def model_from_proposal(proposal: Proposal, source: str | None = None) -> InstrumentModel:
    """A fresh model (every claim unverified) from a validated proposal."""
    src = source or proposal.source
    return InstrumentModel(
        extras=dict(proposal.extras),
        **{f: FieldClaim(value=c.value, source=src, evidence=c.evidence) for f, c in proposal.fields.items()})
