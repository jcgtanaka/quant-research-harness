"""The ``llm`` field of a hypothesis: which models, prompts and parameters a strategy uses.

Validated and stored at S0, immutable like the thresholds, and hashed only when present (so a hypothesis written
before this field existed keeps its hash). ``training_cutoff`` and ``cutoff_source`` have no default: a cutoff
nobody wrote down is ``unknown`` on purpose, and S2 then BLOCKS (see ``qrh.llm.gate``).
"""

from __future__ import annotations

import datetime
import re
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qrh.hashing import canonical_json

CUTOFF_SOURCES = ("provider_documented", "estimated", "unknown")
UNKNOWN = "unknown"
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _json_safe(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must hold only JSON numbers, strings, booleans, lists and mappings ({exc})") from exc
    return value


class LLMModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    training_cutoff: str
    cutoff_source: Literal["provider_documented", "estimated", "unknown"]

    @field_validator("id", "provider")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @field_validator("training_cutoff", mode="before")
    @classmethod
    def _cutoff(cls, v: Any) -> Any:
        if isinstance(v, datetime.datetime):
            raise ValueError("must be a date YYYY-MM-DD or 'unknown', not a timestamp")
        if isinstance(v, datetime.date):  # an unquoted YAML date
            return v.isoformat()
        if not isinstance(v, str):
            raise ValueError("must be a date YYYY-MM-DD or 'unknown'")
        if v == UNKNOWN:
            return v
        if not _DATE.match(v):
            raise ValueError(f"{v!r} is not a date YYYY-MM-DD (use 'unknown' when the cutoff is not known)")
        try:
            datetime.date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"{v!r} is not a real date") from exc
        return v


class LLMVariant(BaseModel):
    """One explicit variant: a model, a prompt file and parameters. Each is a trial (rule 2)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    model: str
    prompt_file: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "model", "prompt_file")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @field_validator("params")
    @classmethod
    def _params(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _json_safe(v, "params")


def check_relative_path(path: str) -> str:
    """A path inside the hypothesis folder: relative, no '..', no drive or backslash."""
    p = PurePosixPath(path)
    if (not path.strip() or p.is_absolute() or "\\" in path or ".." in p.parts or re.match(r"^[A-Za-z]:", path)
            or path.startswith("~")):
        raise ValueError(f"{path!r} must be a relative path inside the hypothesis folder (no '..', no drive letter)")
    return path


class LLMSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[LLMModel] = Field(min_length=1)
    prompt_files: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    live_retrieval: bool | dict[str, str]
    variants: list[LLMVariant] = Field(default_factory=list)

    @field_validator("prompt_files")
    @classmethod
    def _prompt_files(cls, v: list[str]) -> list[str]:
        for p in v:
            check_relative_path(p)
        if len(set(v)) != len(v):
            raise ValueError("duplicate prompt files")
        return v

    @field_validator("params")
    @classmethod
    def _params(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _json_safe(v, "params")

    @field_validator("live_retrieval", mode="before")
    @classmethod
    def _live(cls, v: Any) -> Any:
        if v is True:
            return {}  # declared, but with no evidence text: S2 BLOCKS it
        if v is False:
            return False
        if isinstance(v, dict):
            extra = sorted(set(v) - {"point_in_time_evidence"})
            if extra:
                raise ValueError(f"unknown key(s) {extra}; the only key is point_in_time_evidence")
            return v
        raise ValueError("must be false, or a mapping {point_in_time_evidence: <text>} when retrieval or tools are used")

    @model_validator(mode="after")
    def _consistent(self) -> "LLMSpec":
        ids = [m.id for m in self.models]
        if len(set(ids)) != len(ids):
            raise ValueError("models: duplicate ids")
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError("variants: duplicate names")
        for v in self.variants:
            if v.model not in ids:
                raise ValueError(f"variants.{v.name}: model {v.model!r} is not one of models")
            if v.prompt_file not in self.prompt_files:
                raise ValueError(f"variants.{v.name}: prompt_file {v.prompt_file!r} is not one of prompt_files")
        return self

    def providers(self) -> dict[str, str]:
        return {m.id: m.provider for m in self.models}
