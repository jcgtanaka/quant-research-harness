"""Public-safety lint for packs marked ``visibility: public``.

Flags what must never ship in a public pack: ticker-like tokens, dollar and percent figures in lesson text, ISO
dates, account numbers, e-mail addresses and home paths; runs ``scripts/denylist_check.py`` when ``QRH_DENYLIST`` is
set. Matches are reported by kind and location only, never echoed (a finding must not leak what it found).
It is a safety net for accidents, not a proof of anonymity.

The ticker rule is configurable: ``QRH_SAFETY_CONFIG`` (or ``config=``) points to YAML with ``ticker_patterns``
(extra regexes) and ``allow`` (extra upper-case words that are not tickers).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from qrh.packs.model import Pack

DEFAULT_TICKER_PATTERNS = (r"\$[A-Z]{1,6}\b", r"\b[A-Z]{1,5}\.(?:US|L|TO|DE|HK)\b")
BARE_UPPER = re.compile(r"\b[A-Z]{2,5}\b")
ALLOW = frozenset(
    "PASS FAIL BLOCK BLOCKED WARN INFO NOTE TODO PBO DSR OOS CSCV SR PIT API CSV JSON YAML AST OHLC OHLCV EOD ETF "
    "USD EUR GBP FX NAV VWAP TWAP ATM OTM ITM IV HV DTE ML AI LLM DST UTC UI PDF CPU QRH SKILL MIT NONE TRUE FALSE "
    "YES NO OK PNL PL BPS CAGR IS KPI ID IDS URL ISO CFD DD MAE MFE ADV EMA SMA RSI VS ETL".split()
)
_FIGURES = (
    ("dollar figure", re.compile(r"\$\s?\d|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|PEN|dollars?)\b", re.IGNORECASE)),
    ("percent figure", re.compile(r"\d(?:\.\d+)?\s?%")),
)
_EVERYWHERE = (
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("home path", re.compile(r"(?:/home/|/Users/|[A-Za-z]:\\Users\\|~/)[\w.-]+")),
    ("account number", re.compile(r"\b(?:[A-Z]{1,3}\d{6,}|\d{9,})\b")),
)


@dataclass(frozen=True)
class SafetyFinding:
    file: str
    line: int
    kind: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.kind}"


def _config(config: str | Path | None) -> tuple[list[re.Pattern[str]], frozenset[str]]:
    path = config or os.environ.get("QRH_SAFETY_CONFIG")
    extra: list[str] = []
    allow = ALLOW
    if path:
        doc: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        extra = [str(x) for x in doc.get("ticker_patterns", [])]
        allow = ALLOW | {str(x).upper() for x in doc.get("allow", [])}
    return [re.compile(p) for p in (*DEFAULT_TICKER_PATTERNS, *extra)], allow


def _text_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".yaml", ".yml", ".md", ".txt"))


def scan_pack(pack: Pack, config: str | Path | None = None) -> list[SafetyFinding]:
    patterns, allow = _config(config)
    found: list[SafetyFinding] = []
    root = pack.path
    for p in _text_files(root):
        rel = p.relative_to(root).as_posix()
        is_lesson = rel.startswith("lessons/")
        for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            kinds: list[str] = []
            if any(rx.search(line) for rx in patterns[:]) or any(m.group(0) not in allow for m in BARE_UPPER.finditer(line)):
                kinds.append("ticker-like token")
            if is_lesson:
                kinds += [k for k, rx in _FIGURES if rx.search(line)]
            kinds += [k for k, rx in _EVERYWHERE if rx.search(line)]
            found += [SafetyFinding(rel, n, k) for k in kinds]
    return found


def denylist_findings(pack: Pack) -> list[SafetyFinding]:
    if not os.environ.get("QRH_DENYLIST"):
        return []
    script = Path(__file__).resolve().parents[3] / "scripts" / "denylist_check.py"
    if not script.is_file():
        return [SafetyFinding("(denylist)", 0, "QRH_DENYLIST is set but scripts/denylist_check.py was not found")]
    files = [str(p) for p in _text_files(pack.path)]
    if not files:
        return []
    res = subprocess.run([sys.executable, str(script), *files], capture_output=True, text=True, check=False)
    if res.returncode == 0:
        return []
    out = [SafetyFinding(Path(m.group(1)).relative_to(pack.path).as_posix(), int(m.group(2)), "denylist hit")
           for m in re.finditer(r"^(.+?):(\d+): denylist hit", res.stdout, re.MULTILINE)]
    return out or [SafetyFinding("(denylist)", 0, "denylist check failed")]


def public_safety_findings(pack: Pack, config: str | Path | None = None) -> list[SafetyFinding]:
    if pack.visibility != "public":
        return []
    return scan_pack(pack, config) + denylist_findings(pack)
