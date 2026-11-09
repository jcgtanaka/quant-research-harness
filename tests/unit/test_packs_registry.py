"""Imported attempts live in the registry: counted in family trials and searchable."""

import pytest

from qrh.memory.registry import HypothesisRecord, Registry
from qrh.memory.search import FtsSearcher
from qrh.packs.imports import imported_rows, remove_imported, sync_imported
from qrh.packs.model import Attempt


@pytest.fixture
def reg(tmp_path):
    r = Registry(tmp_path / "r.sqlite3")
    yield r
    r.close()


def attempt(**kw):
    d = dict(text="regime detector with momentum overlay", strategy_family="trend", asset_class="equity",
             frequency="1D", variants_tried=12, verdict="FAIL", date="2020-01-01", notes="")
    d.update(kw)
    return Attempt(**d)


def test_sync_is_idempotent_and_replaces_by_pack(reg):
    sync_imported(reg.conn, "demo", "h1", [attempt(), attempt(text="other idea")])
    sync_imported(reg.conn, "demo", "h1", [attempt(), attempt(text="other idea")])
    assert len(imported_rows(reg.conn)) == 2
    sync_imported(reg.conn, "demo", "h2", [attempt()])
    assert [r["pack_hash"] for r in imported_rows(reg.conn)] == ["h2"]
    remove_imported(reg.conn, "demo")
    assert imported_rows(reg.conn) == []


def test_imported_variants_raise_the_family_trial_count(reg):
    assert reg.family_trial_count("trend|equity|1D") == 0
    sync_imported(reg.conn, "demo", "h1", [attempt(variants_tried=12), attempt(text="b", variants_tried=3)])
    assert reg.family_trial_count("trend|equity|1D") == 15
    assert reg.family_trial_count("trend|fx|1D") == 0


def test_rows_are_marked_imported(reg):
    sync_imported(reg.conn, "demo", "h1", [attempt()])
    assert imported_rows(reg.conn)[0]["imported"] == 1


def test_prior_attempts_search_finds_imported(reg):
    sync_imported(reg.conn, "demo", "h1", [attempt()])
    hits = FtsSearcher(reg.conn).search("momentum regime detector", limit=3)
    assert hits and hits[0].trial_id.startswith("imported:demo:") and hits[0].verdict == "FAIL"
