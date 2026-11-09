"""Thresholds and prices for usage metrics. There are no numeric fallbacks in code.

A missing or invalid thresholds file raises SettingsError naming the file: a threshold that cannot be
read is not a threshold that was not crossed. A missing prices file only means no estimate can be made.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TIERS = ("local", "small", "mid", "frontier")
PRICE_KEYS = ("input", "output", "cache_read", "cache_write_5m", "cache_write_1h")
THRESHOLD_KEYS = ("cache_cold_pct", "context_high_tokens", "mechanical_tool_share", "mechanical_min_tools")
WASTE_KEYS = ("waste_reread_k", "waste_large_output_tokens", "waste_repeat_fail_k")
EXAMPLE_NOTE = (
    "EXAMPLE values to review before use. Replace them with your own; do not treat them as facts."
)
PRICES_NOTE = (
    "EXAMPLE placeholders, not real prices. Rename the models to the ones you use and fill every price "
    "(USD per million tokens) from the provider's pricing page. Sessions with a harness cost never use these."
)


class SettingsError(ValueError):
    """A usage settings file is missing, unreadable or invalid."""


@dataclass(frozen=True)
class Thresholds:
    cache_cold_pct: float
    context_high_tokens: int
    mechanical_tool_share: float
    mechanical_min_tools: int


@dataclass(frozen=True)
class WasteThresholds:
    reread_k: int  # a path read more than this many times with no change between is flagged
    large_output_tokens: int  # a tool result above this many (estimated) tokens is flagged
    repeat_fail_k: int  # a command that failed at least this many times is flagged


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float
    tier: str


@dataclass(frozen=True)
class Prices:
    models: dict[str, ModelPrice]


def usage_dir(home: str | Path) -> Path:
    return Path(home) / "usage"


def thresholds_path(home: str | Path) -> Path:
    return usage_dir(home) / "thresholds.json"


def prices_path(home: str | Path) -> Path:
    return usage_dir(home) / "prices.json"


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _read(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SettingsError(f"cannot read {path}: {exc}") from exc
    except ValueError as exc:
        raise SettingsError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise SettingsError(f"{path} must contain a JSON object")
    return raw


def load_thresholds(home: str | Path) -> Thresholds:
    path = thresholds_path(home)
    raw = _read(path)
    extra = sorted(set(raw) - set(THRESHOLD_KEYS) - set(WASTE_KEYS) - {"_note"})
    if extra:
        raise SettingsError(f"{path}: unknown key(s) {extra}")
    missing = [k for k in THRESHOLD_KEYS if k not in raw]
    if missing:
        raise SettingsError(f"{path}: missing key(s) {missing}")
    for k in THRESHOLD_KEYS:
        if not _is_num(raw[k]):
            raise SettingsError(f"{path}: {k} must be a number")
    if not 0 <= raw["cache_cold_pct"] <= 100:
        raise SettingsError(f"{path}: cache_cold_pct must be between 0 and 100")
    if not 0 <= raw["mechanical_tool_share"] <= 1:
        raise SettingsError(f"{path}: mechanical_tool_share must be a fraction between 0 and 1")
    for k in ("context_high_tokens", "mechanical_min_tools"):
        if raw[k] < 1 or raw[k] != int(raw[k]):
            raise SettingsError(f"{path}: {k} must be a positive integer")
    return Thresholds(
        float(raw["cache_cold_pct"]), int(raw["context_high_tokens"]), float(raw["mechanical_tool_share"]),
        int(raw["mechanical_min_tools"]),
    )


def load_waste_thresholds(home: str | Path) -> WasteThresholds:
    """The waste-report thresholds in thresholds.json. Missing or invalid raises SettingsError: fail closed."""
    path = thresholds_path(home)
    raw = _read(path)
    for k in WASTE_KEYS:
        v = raw.get(k)
        if not _is_num(v) or v < 1 or v != int(v):
            raise SettingsError(f"{path}: {k} is missing or not a positive integer (needed by `qrh usage waste`)")
    return WasteThresholds(int(raw["waste_reread_k"]), int(raw["waste_large_output_tokens"]), int(raw["waste_repeat_fail_k"]))


def load_prices(home: str | Path) -> Prices | None:
    """None when prices.json does not exist (no estimates possible); SettingsError when it is invalid."""
    path = prices_path(home)
    if not path.exists():
        return None
    raw = _read(path)
    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise SettingsError(f"{path}: 'models' must be a non-empty mapping")
    out: dict[str, ModelPrice] = {}
    for name, entry in models.items():
        if not isinstance(entry, dict):
            raise SettingsError(f"{path}: models.{name} must be a mapping")
        for k in PRICE_KEYS:
            if not _is_num(entry.get(k)) or entry[k] < 0:
                raise SettingsError(f"{path}: models.{name}.{k} must be a non-negative number")
        if entry.get("tier") not in TIERS:
            raise SettingsError(f"{path}: models.{name}.tier must be one of {list(TIERS)} (tier)")
        out[name] = ModelPrice(**{k: float(entry[k]) for k in PRICE_KEYS}, tier=entry["tier"])
    return Prices(out)


def price_cost(price: ModelPrice, *, input: int, output: int, cache_read: int, cache_write_5m: int,
               cache_write_1h: int) -> float:
    """USD for the given token counts at ``price`` (prices are per million tokens)."""
    return (
        input * price.input + output * price.output + cache_read * price.cache_read
        + cache_write_5m * price.cache_write_5m + cache_write_1h * price.cache_write_1h
    ) / 1_000_000


def init_files(home: str | Path, force: bool = False) -> list[Path]:
    """Write example settings files; existing files are kept unless ``force``. Returns files written."""
    examples: dict[Path, dict[str, Any]] = {
        thresholds_path(home): {
            "_note": EXAMPLE_NOTE, "cache_cold_pct": 60, "context_high_tokens": 300000,
            "mechanical_tool_share": 0.8, "mechanical_min_tools": 5,
            "waste_reread_k": 3, "waste_large_output_tokens": 10000, "waste_repeat_fail_k": 3,
        },
        prices_path(home): {
            "_note": PRICES_NOTE,
            "models": {
                "example-frontier-model": {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write_5m": 1.25,
                                           "cache_write_1h": 2.0, "tier": "frontier"},
                "example-small-model": {"input": 0.1, "output": 0.5, "cache_read": 0.01, "cache_write_5m": 0.125,
                                        "cache_write_1h": 0.2, "tier": "small"},
            },
        },
    }
    written = []
    for path, body in examples.items():
        if path.exists() and not force:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written
