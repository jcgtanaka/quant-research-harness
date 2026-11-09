"""Pack schema and loader. A pack is a folder with ``qrh-pack.yaml`` and optional content folders.

Everything is validated up front and every problem is reported at once (``PackError``). The pack hash is the
``sha256_dir`` of the folder, so any edit, addition or removal changes it. Lessons with ``status: draft`` are
counted but never loaded (see ``qrh pack draft`` and ``qrh pack approve-lesson``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from qrh.hashing import sha256_dir

MANIFEST = "qrh-pack.yaml"
SCHEMA_VERSION = 1
STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "B1", "B2", "B3", "B4", "B5", "B6")
SCOPES = ("hypothesis", "dataset", "strategy", "account", "session")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


class PackError(ValueError):
    """A pack cannot be loaded or used; the message lists every problem found."""


class AppliesTo(BaseModel):
    """Which hypotheses a pack, lesson or question concerns. An empty list means any value.

    ``uses_llm: true`` narrows it to hypotheses that declare an ``llm`` section (the LLM lessons and questions)."""

    model_config = ConfigDict(extra="forbid")

    asset_class: list[str] = Field(default_factory=list)
    strategy_family: list[str] = Field(default_factory=list)
    frequency: list[str] = Field(default_factory=list)
    uses_llm: bool = False

    @field_validator("asset_class", "strategy_family", "frequency", mode="before")
    @classmethod
    def _listify(cls, v: Any) -> Any:
        return [v] if isinstance(v, str) else (v or [])

    def matches(self, attrs: dict[str, Any]) -> bool:
        if self.uses_llm and not attrs.get("llm"):
            return False
        for key in ("asset_class", "strategy_family", "frequency"):
            wanted = getattr(self, key)
            if wanted and attrs.get(key) not in wanted:
                return False
        return True


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    version: str
    schema_version: int = SCHEMA_VERSION
    visibility: Literal["public", "private"]
    description: str = ""
    applies_to: AppliesTo = Field(default_factory=AppliesTo)
    authors: list[str] = Field(default_factory=list)
    license: str = ""

    @field_validator("version")
    @classmethod
    def _semver(cls, v: str) -> str:
        if not _SEMVER.match(v):
            raise ValueError("must be a semantic version such as 1.2.0")
        return v

    @field_validator("schema_version")
    @classmethod
    def _schema(cls, v: int) -> int:
        if v != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {v} (this qrh reads {SCHEMA_VERSION})")
        return v


class Trigger(BaseModel):
    """When a lesson fires: any keyword in the hypothesis text, or a field matcher (substring or ``re:`` regex)."""

    model_config = ConfigDict(extra="forbid")

    keywords: list[str] = Field(default_factory=list)
    fields: dict[str, str] = Field(default_factory=dict)

    def matches(self, trial: dict[str, Any]) -> bool:
        if not self.keywords and not self.fields:
            return True
        text = " ".join(str(trial.get(k, "")) for k in ("text", "difference_statement")).lower()
        if any(k.lower() in text for k in self.keywords):
            return True
        for name, pattern in self.fields.items():
            value = str(trial.get(name, ""))
            if pattern.startswith("re:"):
                if re.search(pattern[3:], value, re.IGNORECASE):
                    return True
            elif pattern.lower() in value.lower():
                return True
        return False


REQUIREMENT_KINDS = ("interview_question", "threshold_floor", "method")


class Lesson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    applies_to: AppliesTo = Field(default_factory=AppliesTo)
    trigger: Trigger = Field(default_factory=Trigger)
    action: Literal["warn", "require"] = "warn"
    requirement: dict[str, Any] | None = None
    severity: Literal["info", "warning", "critical"] = "warning"
    status: Literal["draft", "approved"] = "approved"
    source_ref: str | None = None

    @model_validator(mode="after")
    def _requirement(self) -> "Lesson":
        if self.action == "require":
            kinds = [k for k in REQUIREMENT_KINDS if self.requirement and k in self.requirement]
            if len(kinds) != 1:
                raise ValueError(f"requirement must contain exactly one of {list(REQUIREMENT_KINDS)} when action is require")
        return self


class Attempt(BaseModel):
    """A pre-harness attempt, imported as a trial marked ``imported``."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    strategy_family: str = Field(min_length=1)
    asset_class: str = Field(min_length=1)
    frequency: str = Field(min_length=1)
    variants_tried: int = Field(ge=1)
    verdict: str = Field(min_length=1)
    date: str = ""
    notes: str = ""

    def family_id(self) -> str:
        return f"{self.strategy_family}|{self.asset_class}|{self.frequency}"


class Account(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capital: float = Field(gt=0)
    base_currency: str = Field(min_length=1)
    min_order_size: dict[str, float] = Field(default_factory=dict)
    fee_schedule: dict[str, Any] = Field(default_factory=dict)
    margin_rules: dict[str, Any] = Field(default_factory=dict)
    caps: dict[str, float] = Field(default_factory=dict)

    @field_validator("caps")
    @classmethod
    def _caps(cls, v: dict[str, float]) -> dict[str, float]:
        from qrh.memory.registry import BOUND_KEYS

        unknown = sorted(set(v) - set(BOUND_KEYS))
        if unknown:
            raise ValueError(f"unknown cap(s) {unknown}; allowed: {list(BOUND_KEYS)}")
        if any(not x > 0 for x in v.values()):
            raise ValueError("caps must be positive")
        return v


class Choice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    risk: str | None = None
    maps_to: dict[str, Any] | None = None


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    stage: str
    scope: Literal["hypothesis", "dataset", "strategy", "account", "session"]
    applies_to: AppliesTo = Field(default_factory=AppliesTo)
    text: str = Field(min_length=1)
    why_it_matters: str = ""
    answer_type: Literal["choice", "text", "number", "bool"] = "text"
    choices: list[Choice] = Field(default_factory=list)
    critical: bool = False
    maps_to: dict[str, Any] | None = None
    unknown_risk: str = ""

    @field_validator("stage")
    @classmethod
    def _stage(cls, v: str) -> str:
        if v not in STAGES:
            raise ValueError(f"stage must be one of {list(STAGES)}")
        return v

    @model_validator(mode="after")
    def _choices(self) -> "Question":
        if self.answer_type == "choice" and not self.choices:
            raise ValueError("a choice question needs choices")
        ids = [c.id for c in self.choices]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate choice ids")
        return self


class LintRule(BaseModel):
    """A declarative extra lint rule: a regex over source lines, or an AST descriptor."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    severity: Literal["BLOCK", "WARN"] = "WARN"
    message: str = Field(min_length=1)
    regex: str | None = None
    ast: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _one(self) -> "LintRule":
        if (self.regex is None) == (self.ast is None):
            raise ValueError("give exactly one of regex or ast")
        if self.regex is not None:
            try:
                re.compile(self.regex)
            except re.error as exc:
                raise ValueError(f"invalid regex: {exc}") from exc
        if self.ast is not None and not ({"call", "import"} & set(self.ast)):
            raise ValueError("ast descriptor needs 'call' or 'import'")
        return self


class Pack:
    """A loaded, validated pack."""

    def __init__(self, path: Path, manifest: Manifest, digest: str) -> None:
        self.path = path
        self.manifest = manifest
        self.hash = digest
        self.lessons: list[Lesson] = []
        self.draft_count = 0
        self.attempts: list[Attempt] = []
        self.account: Account | None = None
        self.thresholds: dict[str, Any] = {}
        self.data_rules: dict[str, Any] = {}
        self.questions: list[Question] = []
        self.lint_rules: list[LintRule] = []

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def visibility(self) -> str:
        return self.manifest.visibility

    def record(self) -> dict[str, str]:
        return {"version": self.manifest.version, "hash": self.hash}


def _yaml(path: Path, problems: list[str]) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as exc:
        problems.append(f"{path.name}: cannot read: {exc}")
        return None


def _validate(model: type[BaseModel], doc: Any, where: str, problems: list[str]) -> Any:
    try:
        return model.model_validate(doc)
    except ValidationError as exc:
        for e in exc.errors():
            loc = ".".join(str(p) for p in e["loc"])
            problems.append(f"{where}: {loc + ': ' if loc else ''}{e['msg']}")
        return None


def _documents(folder: Path, key: str | None, problems: list[str]) -> list[tuple[str, Any]]:
    """(label, document) for every YAML file; a file may hold one mapping or a list (or ``{key: [...]}``)."""
    out: list[tuple[str, Any]] = []
    if not folder.is_dir():
        return out
    for p in sorted(folder.glob("*.yaml")):
        doc = _yaml(p, problems)
        if doc is None:
            continue
        if isinstance(doc, dict) and key and isinstance(doc.get(key), list):
            doc = doc[key]
        for i, item in enumerate(doc if isinstance(doc, list) else [doc]):
            out.append((f"{folder.name}/{p.name}" + (f"[{i}]" if isinstance(doc, list) else ""), item))
    return out


def load_pack(path: str | Path) -> Pack:
    root = Path(path)
    problems: list[str] = []
    manifest_file = root / MANIFEST
    if not manifest_file.is_file():
        raise PackError(f"{str(root)!r} is not a pack: {MANIFEST} not found")
    manifest = _validate(Manifest, _yaml(manifest_file, problems), MANIFEST, problems)
    if manifest is None:
        raise PackError("; ".join(problems))
    pack = Pack(root, manifest, sha256_dir(root))
    seen: set[str] = set()
    for where, doc in _documents(root / "lessons", None, problems):
        lesson = _validate(Lesson, doc, where, problems)
        if lesson is None:
            continue
        if lesson.id in seen:
            problems.append(f"{where}: duplicate lesson id {lesson.id!r}")
        seen.add(lesson.id)
        if lesson.status == "draft":
            pack.draft_count += 1
        else:
            pack.lessons.append(lesson)
    for where, doc in _documents(root / "attempts", None, problems):
        attempt = _validate(Attempt, doc, where, problems)
        if attempt is not None:
            pack.attempts.append(attempt)
    for where, doc in _documents(root / "questions", "questions", problems):
        q = _validate(Question, doc, where, problems)
        if q is not None:
            if q.id in {x.id for x in pack.questions}:
                problems.append(f"{where}: duplicate question id {q.id!r}")
            pack.questions.append(q)
    for where, doc in _documents(root / "lint", "rules", problems):
        rule = _validate(LintRule, doc, where, problems)
        if rule is not None:
            pack.lint_rules.append(rule)
    for where, doc in _documents(root / "data_rules", None, problems):
        if isinstance(doc, dict):
            pack.data_rules[where] = doc
        else:
            problems.append(f"{where}: must be a mapping")
    if (root / "account.yaml").is_file():
        pack.account = _validate(Account, _yaml(root / "account.yaml", problems), "account.yaml", problems)
    if (root / "thresholds.yaml").is_file():
        doc = _yaml(root / "thresholds.yaml", problems)
        if isinstance(doc, dict):
            pack.thresholds = _check_thresholds(doc, problems)
        elif doc is not None:
            problems.append("thresholds.yaml: must be a mapping")
    if problems:
        raise PackError("; ".join(problems))
    return pack


DIMENSIONS = ("asset_class", "strategy_family", "frequency")


def _check_thresholds(doc: dict[str, Any], problems: list[str]) -> dict[str, Any]:
    ok: dict[str, Any] = {}
    for dim, body in doc.items():
        if dim == "all":
            if not isinstance(body, dict):
                problems.append("thresholds.yaml: 'all' must map threshold names to values")
            else:
                ok["all"] = body
        elif dim in DIMENSIONS:
            if not isinstance(body, dict) or not all(isinstance(v, dict) for v in body.values()):
                problems.append(f"thresholds.yaml: '{dim}' must map each value to a mapping of thresholds")
            else:
                ok[dim] = body
        else:
            problems.append(f"thresholds.yaml: unknown dimension {dim!r}; use all, {', '.join(DIMENSIONS)}")
    return ok
