"""Shared hook plumbing: payload parsing, configuration, redaction, globbing, logging."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

MIN_OVERRIDE_REASON = 10  # also the minimum length of a stated reason in config (guard.parity.disabled)
COMMAND_LOG_LIMIT = 160
DEFAULT_TRIGGER_TERMS = (
    "strategy", "backtest", "hypothesis", "signal", "alpha", "regime", "walk-forward", "overfit",
)


class PayloadError(ValueError):
    """The hook input is not a JSON object."""


class ConfigError(ValueError):
    """<QRH_HOME>/config.yaml is unreadable or invalid."""


# --- payload --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Payload:
    """The JSON a Claude Code hook receives on stdin (wrong-typed fields read as empty)."""

    raw: Mapping[str, Any]

    def _str(self, key: str) -> str:
        v = self.raw.get(key)
        return v if isinstance(v, str) else ""

    @property
    def session_id(self) -> str:
        return self._str("session_id")

    @property
    def event(self) -> str:
        return self._str("hook_event_name")

    @property
    def cwd(self) -> str:
        return self._str("cwd")

    @property
    def tool_name(self) -> str:
        return self._str("tool_name")

    @property
    def tool_input(self) -> Mapping[str, Any]:
        v = self.raw.get("tool_input")
        return v if isinstance(v, dict) else {}

    @property
    def prompt(self) -> str:
        return self._str("prompt")

    @property
    def transcript_path(self) -> str:
        return self._str("transcript_path")

    @property
    def stop_hook_active(self) -> bool:
        return self.raw.get("stop_hook_active") is True

    @property
    def command(self) -> str:
        v = self.tool_input.get("command")
        return v if isinstance(v, str) else ""

    @property
    def file_path(self) -> str:
        v = self.tool_input.get("file_path")
        return v if isinstance(v, str) else ""


def parse_payload(text: str) -> Payload:
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise PayloadError(f"hook input is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PayloadError("hook input must be a JSON object")
    return Payload(data)


# --- redaction ------------------------------------------------------------------------------------

_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"), "sk-***"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{8,}"), "gh_***"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{8,}"), "github_pat_***"),
    (re.compile(r"\bxox[A-Za-z]-[A-Za-z0-9\-]{6,}"), "xox-***"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{12,}"), "AWS-KEY-***"),
    (re.compile(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=\-]{4,}", re.IGNORECASE), r"\1 ***"),
    (re.compile(
        r"(?i)(\b[\w.\-]*(?:token|password|passwd|secret|api[_\-]?key|authorization)[\w.\-]*)(\s*[=:]\s*)"
        r"(\"[^\"]*\"|'[^']*'|[^\s'\"]+)"), r"\1\2***"),
    (re.compile(r"(?i)(--(?:password|passwd|token|secret|api-key)(?:=|\s+))(\S+)"), r"\1***"),
)


def redact(text: str) -> str:
    """Mask provider keys, bearer/basic tokens, secret-looking key=value pairs and secret flags."""
    for rx, repl in _REDACTIONS:
        text = rx.sub(repl, text)
    return text


def redact_command(command: str, limit: int = COMMAND_LOG_LIMIT) -> str:
    """Redact first, then truncate, so a secret cut by the limit cannot leave its head behind."""
    flat = redact(" ".join(command.split()))
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# --- globs ----------------------------------------------------------------------------------------

def _glob_regex(pattern: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**", i):
            i += 2
            if pattern.startswith("/", i):
                i += 1
                out.append("(?:.*/)?")
            else:
                out.append(".*")
            continue
        out.append("[^/]*" if c == "*" else "[^/]" if c == "?" else re.escape(c))
        i += 1
    return "".join(out)


def glob_match(path: str, pattern: str, cwd: str = "") -> bool:
    """Glob with ``**``. Absolute patterns match the whole path; relative ones match at any depth."""
    p = os.path.normpath(path if os.path.isabs(path) else os.path.join(cwd or os.getcwd(), path))
    p = p.replace(os.sep, "/")
    rx = _glob_regex(pattern)
    if not pattern.startswith("/"):
        rx = "(?:.*/)?" + rx
    return re.fullmatch(rx, p) is not None


# --- configuration --------------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestCommand:
    pattern: re.Pattern[str]
    datasets: tuple[str, ...] = ()
    trial_from_args: bool = False


@dataclass(frozen=True)
class GuardConfig:
    source_globs: tuple[str, ...] = ()
    build_hash_file: str | None = None
    required_datasets: tuple[str, ...] = ()
    max_dataset_age: dict[str, str] = field(default_factory=dict)
    test_receipt_max_age: str | None = None
    parity_required: bool = True
    parity_disabled_reason: str | None = None
    require_readiness: str | None = None  # trial id whose readiness report must be acknowledged before an engine start


@dataclass(frozen=True)
class ReceiptRequiredConfig:
    enabled: bool = True
    metric_patterns: tuple[str, ...] = ()
    metric_map: dict[str, tuple[str, ...]] = field(default_factory=dict)
    abs_tol: float = 0.01
    rel_tol: float = 0.005
    recent_days: int = 30


@dataclass(frozen=True)
class RunawayConfig:
    """Trip wires of the runaway guard. A limit that is None is off; there are no numeric fallbacks."""

    repeat_limit: int | None = None
    failure_streak: int | None = None
    spend_cap_usd: float | None = None


@dataclass(frozen=True)
class PublishGuardConfig:
    """Repositories (absolute real paths) whose publication the publish guard protects. No fallbacks."""

    protected_repos: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClaimCheckConfig:
    enabled: bool = True
    test_commands: tuple[re.Pattern[str], ...] = ()
    build_commands: tuple[re.Pattern[str], ...] = ()
    lint_commands: tuple[re.Pattern[str], ...] = ()


@dataclass(frozen=True)
class Config:
    strategy_globs: tuple[str, ...] = ()
    detect_strategies: bool = True
    trigger_terms: tuple[str, ...] = DEFAULT_TRIGGER_TERMS
    shared_module_globs: tuple[str, ...] = ()
    backtest_commands: tuple[BacktestCommand, ...] = ()
    engine_start_commands: tuple[re.Pattern[str], ...] = ()
    guard: GuardConfig | None = None
    receipt_required: ReceiptRequiredConfig = field(default_factory=ReceiptRequiredConfig)
    claim_check: ClaimCheckConfig = field(default_factory=ClaimCheckConfig)
    runaway: RunawayConfig | None = None  # None: the section is absent, the guard is disabled
    publish_guard: PublishGuardConfig | None = None  # None: the section is absent, the guard is disabled


_KNOWN_KEYS = {
    "strategy_globs", "detect_strategies", "trigger_terms", "shared_module_globs", "backtest_commands",
    "engine_start_commands", "guard", "receipt_required",
    "runaway", "claim_check", "publish_guard",
}


def config_path(home: str | Path) -> Path:
    return Path(home) / "config.yaml"


def _str_list(raw: Any, key: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(x, str) and x for x in raw):
        raise ConfigError(f"{key} must be a list of non-empty strings")
    return tuple(raw)


def _regex(raw: Any, key: str) -> re.Pattern[str]:
    if not isinstance(raw, str) or not raw:
        raise ConfigError(f"{key} must be a non-empty regular expression string")
    try:
        return re.compile(raw)
    except re.error as exc:
        raise ConfigError(f"{key}: invalid regular expression {raw!r}: {exc}") from exc


def _mapping(raw: Any, key: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{key} must be a mapping")
    return raw


def _only(raw: dict[str, Any], allowed: set[str], key: str) -> None:
    extra = sorted(set(raw) - allowed)
    if extra:
        raise ConfigError(f"{key}: unknown key(s) {extra}")


def _parse_backtest(raw: Any) -> tuple[BacktestCommand, ...]:
    if not isinstance(raw, list):
        raise ConfigError("backtest_commands must be a list")
    out = []
    for i, item in enumerate(raw):
        key = f"backtest_commands[{i}]"
        item = _mapping(item, key)
        _only(item, {"pattern", "datasets", "trial_id"}, key)
        pattern = _regex(item.get("pattern"), f"{key}.pattern")
        has_ds, has_tid = "datasets" in item, "trial_id" in item
        if has_ds == has_tid:
            raise ConfigError(f"{key}: give exactly one of 'datasets' or 'trial_id: from_args'")
        if has_tid:
            if item["trial_id"] != "from_args":
                raise ConfigError(f"{key}.trial_id must be 'from_args'")
            out.append(BacktestCommand(pattern, trial_from_args=True))
        else:
            out.append(BacktestCommand(pattern, datasets=_str_list(item["datasets"], f"{key}.datasets")))
    return tuple(out)


def _parse_guard(raw: Any) -> GuardConfig:
    raw = _mapping(raw, "guard")
    _only(raw, {"source_globs", "build_hash_file", "required_datasets", "max_dataset_age",
                "test_receipt_max_age", "parity", "require_readiness"}, "guard")
    ages = raw.get("max_dataset_age", {})
    if not isinstance(ages, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in ages.items()):
        raise ConfigError("guard.max_dataset_age must map dataset names to offset strings such as 3D")
    bhf = raw.get("build_hash_file")
    if bhf is not None and not isinstance(bhf, str):
        raise ConfigError("guard.build_hash_file must be a path string")
    tra = raw.get("test_receipt_max_age")
    if tra is not None and not isinstance(tra, str):
        raise ConfigError("guard.test_receipt_max_age must be an offset string such as 1D")
    parity = raw.get("parity", "required")
    required, why = True, None
    if parity == "required":
        pass
    elif isinstance(parity, dict) and set(parity) == {"disabled"}:
        why = parity["disabled"]
        if not isinstance(why, str) or len(why.strip()) < MIN_OVERRIDE_REASON:
            raise ConfigError(f"guard.parity.disabled needs a reason of at least {MIN_OVERRIDE_REASON} characters")
        required, why = False, why.strip()
    else:
        raise ConfigError("guard.parity must be 'required' or {disabled: <reason>}")
    readiness = raw.get("require_readiness")
    if readiness is not None:
        from qrh.readiness.collect import TRIAL_ID_RE

        if not isinstance(readiness, str) or not TRIAL_ID_RE.match(readiness) or ".." in readiness:
            raise ConfigError("guard.require_readiness must be a trial id (letters, digits, '_', '.', '-')")
    return GuardConfig(
        source_globs=_str_list(raw.get("source_globs", []), "guard.source_globs"),
        build_hash_file=bhf,
        required_datasets=_str_list(raw.get("required_datasets", []), "guard.required_datasets"),
        max_dataset_age=dict(ages),
        test_receipt_max_age=tra,
        parity_required=required,
        parity_disabled_reason=why,
        require_readiness=readiness,
    )


def _parse_receipt_required(raw: Any) -> ReceiptRequiredConfig:
    raw = _mapping(raw, "receipt_required")
    _only(raw, {"enabled", "metric_patterns", "metric_map", "tolerance", "recent_days"}, "receipt_required")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("receipt_required.enabled must be true or false")
    tol = _mapping(raw.get("tolerance", {}), "receipt_required.tolerance")
    _only(tol, {"abs", "rel"}, "receipt_required.tolerance")
    days = raw.get("recent_days", 30)
    if isinstance(days, bool) or not isinstance(days, int) or days < 0:
        raise ConfigError("receipt_required.recent_days must be a non-negative integer")
    vals = {}
    for k, default in (("abs", 0.01), ("rel", 0.005)):
        v = tol.get(k, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ConfigError(f"receipt_required.tolerance.{k} must be a non-negative number")
        vals[k] = float(v)
    mm = _mapping(raw.get("metric_map", {}), "receipt_required.metric_map")
    metric_map = {str(k).lower(): _str_list(v, f"receipt_required.metric_map.{k}") for k, v in mm.items()}
    return ReceiptRequiredConfig(
        enabled=enabled,
        metric_map=metric_map,
        metric_patterns=_str_list(raw.get("metric_patterns", []), "receipt_required.metric_patterns"),
        abs_tol=vals["abs"], rel_tol=vals["rel"], recent_days=days,
    )


def _parse_claim_check(raw: Any) -> ClaimCheckConfig:
    raw = _mapping(raw, "claim_check")
    _only(raw, {"enabled", "test_commands", "build_commands", "lint_commands"}, "claim_check")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("claim_check.enabled must be true or false")

    def rx(key: str) -> tuple[re.Pattern[str], ...]:
        items = raw.get(key, [])
        if not isinstance(items, list):
            raise ConfigError(f"claim_check.{key} must be a list of regular expressions")
        return tuple(_regex(x, f"claim_check.{key}[{i}]") for i, x in enumerate(items))

    return ClaimCheckConfig(enabled, rx("test_commands"), rx("build_commands"), rx("lint_commands"))


def parse_runaway(raw: Any) -> RunawayConfig:
    raw = _mapping(raw, "runaway")
    _only(raw, {"repeat_limit", "failure_streak", "spend_cap_usd"}, "runaway")
    if not raw:
        raise ConfigError("runaway needs at least one of repeat_limit, failure_streak, spend_cap_usd")
    out: dict[str, Any] = {}
    for key, minimum in (("repeat_limit", 2), ("failure_streak", 1)):
        if key in raw:
            v = raw[key]
            if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
                raise ConfigError(f"runaway.{key} must be an integer of at least {minimum}")
            out[key] = v
    if "spend_cap_usd" in raw:
        v = raw["spend_cap_usd"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0 or v == float("inf"):
            raise ConfigError("runaway.spend_cap_usd must be a positive finite number")
        out["spend_cap_usd"] = float(v)
    return RunawayConfig(**out)


def parse_publish_guard(raw: Any) -> PublishGuardConfig:
    raw = _mapping(raw, "publish_guard")
    _only(raw, {"protected_repos"}, "publish_guard")
    repos = _str_list(raw.get("protected_repos"), "publish_guard.protected_repos")
    if not repos:
        raise ConfigError("publish_guard.protected_repos needs at least one repository path")
    out = []
    for r in repos:
        full = os.path.expanduser(r)
        if not os.path.isabs(full):
            raise ConfigError(f"publish_guard.protected_repos: {r!r} is not an absolute path (no path fallbacks)")
        out.append(os.path.normpath(full))
    return PublishGuardConfig(tuple(out))


def load_config(home: str | Path) -> Config:
    """Read <home>/config.yaml. Missing file gives the documented defaults; invalid raises ConfigError."""
    path = config_path(home)
    if not path.exists():
        return Config()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if raw is None:
        return Config()
    raw = _mapping(raw, "config.yaml")
    unknown = sorted(set(raw) - _KNOWN_KEYS)
    if unknown:
        raise ConfigError(f"unknown config key(s) {unknown}")
    detect = raw.get("detect_strategies", True)
    if not isinstance(detect, bool):
        raise ConfigError("detect_strategies must be true or false")
    terms = _str_list(raw["trigger_terms"], "trigger_terms") if "trigger_terms" in raw else DEFAULT_TRIGGER_TERMS
    engines = raw.get("engine_start_commands", [])
    if not isinstance(engines, list):
        raise ConfigError("engine_start_commands must be a list of regular expressions")
    return Config(
        strategy_globs=_str_list(raw.get("strategy_globs", []), "strategy_globs"),
        detect_strategies=detect,
        trigger_terms=terms,
        shared_module_globs=_str_list(raw.get("shared_module_globs", []), "shared_module_globs"),
        backtest_commands=_parse_backtest(raw.get("backtest_commands", [])),
        engine_start_commands=tuple(_regex(e, f"engine_start_commands[{i}]") for i, e in enumerate(engines)),
        guard=_parse_guard(raw["guard"]) if raw.get("guard") is not None else None,
        receipt_required=_parse_receipt_required(raw.get("receipt_required", {})),
        claim_check=_parse_claim_check(raw.get("claim_check", {})),
        runaway=parse_runaway(raw["runaway"]) if "runaway" in raw else None,
        publish_guard=parse_publish_guard(raw["publish_guard"]) if "publish_guard" in raw else None,
    )


# --- logging --------------------------------------------------------------------------------------

def logs_dir(home: str | Path) -> Path:
    return Path(home) / "logs"


def _append_json(path: Path, record: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never break a session


def log_invocation(home: str | Path, record: Mapping[str, Any]) -> None:
    _append_json(logs_dir(home) / "hooks.jsonl", record)


def tag_session_trial(
    home: str | Path, session_id: str, trial_id: str | None, stage: str | None, kind: str, ts: datetime | None = None
) -> None:
    """Append a session<->trial link (feeds the AI usage metrics of a later phase)."""
    when = (ts or datetime.now(timezone.utc)).isoformat()
    _append_json(logs_dir(home) / "session_trials.jsonl",
                 {"session_id": session_id, "trial_id": trial_id, "stage": stage, "kind": kind, "ts": when})
