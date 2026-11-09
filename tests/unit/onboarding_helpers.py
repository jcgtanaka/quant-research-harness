"""Shared builders for the onboarding tests (synthetic data only)."""

from __future__ import annotations

import json
from pathlib import Path

from qrh.memory.catalog import Catalog
from qrh.synthetic import instruments as inst

MAKERS = {"equity": inst.equity_daily, "futures": inst.futures_daily, "fx": inst.fx_quotes,
          "option": inst.option_snapshots, "perp": inst.crypto_perp}


def make_catalog(tmp_path: Path) -> Catalog:
    (tmp_path / "home").mkdir(exist_ok=True)
    return Catalog(tmp_path / "home" / "registry.sqlite3")


def register(cat: Catalog, tmp_path: Path, kind: str = "equity", name: str = "ds", privacy: str | None = "public", **kw):
    """Write a synthetic dataset as CSV, register it, return (proposal dict, csv path)."""
    frame, proposal = MAKERS[kind](**kw)
    path = tmp_path / f"{name}.csv"
    inst.write_csv(frame, path)
    cat.register_dataset(path, name=name, privacy=privacy)
    return proposal, path


def proposal_file(tmp_path: Path, proposal: dict, name: str = "proposal.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(proposal))
    return p
