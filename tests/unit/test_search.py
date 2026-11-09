import itertools

import pytest
from typer.testing import CliRunner

from qrh.cli import app
from qrh.memory.registry import HypothesisRecord, Registry
from qrh.memory.search import FtsSearcher, SearchHit, Searcher

runner = CliRunner()


def rec(text, family="trend", **kw):
    base = dict(
        text=text, strategy_family=family, asset_class="synthetic", frequency="1D",
        parameter_space_size=4, thresholds={"min_score": 0.5},
        difference_statement="Differs from the baseline by a longer lookback.",
    )
    base.update(kw)
    return HypothesisRecord(**base)


@pytest.fixture
def reg(tmp_path):
    counter = itertools.count(1)
    r = Registry(tmp_path / "r.sqlite3", id_factory=lambda: f"trial{next(counter):04d}")
    yield r
    r.close()


@pytest.fixture
def populated(reg):
    reg.register_hypothesis(rec("Momentum persists after volatility regime shifts", "momentum"))
    reg.register_hypothesis(rec("Mean reversion of spreads in quiet regime", "reversion"))
    reg.register_hypothesis(rec("Seasonality in overnight returns", "seasonal"))
    return reg


def test_fts_table_created_by_migration(reg):
    row = reg.conn.execute("SELECT name FROM sqlite_master WHERE name='hypotheses_fts'").fetchone()
    assert row is not None


def test_registration_keeps_index_in_sync(reg):
    reg.register_hypothesis(rec("alpha decay after publication"))
    hits = FtsSearcher(reg.conn).search("alpha")
    assert [h.trial_id for h in hits] == ["trial0001"]


def test_searcher_protocol(reg):
    s = FtsSearcher(reg.conn)
    assert isinstance(s, Searcher)


def test_ranking_prefers_more_matching_terms(populated):
    hits = FtsSearcher(populated.conn).search("momentum volatility regime")
    assert hits[0].trial_id == "trial0001"
    assert {h.trial_id for h in hits} == {"trial0001", "trial0002"}
    assert hits[0].score > hits[1].score  # higher is better


def test_family_field_is_searched(populated):
    hits = FtsSearcher(populated.conn).search("seasonal")
    assert [h.trial_id for h in hits] == ["trial0003"]


def test_stemming_matches_word_forms(populated):
    assert {h.trial_id for h in FtsSearcher(populated.conn).search("regimes")} == {"trial0001", "trial0002"}
    hits = FtsSearcher(populated.conn).search("return")
    assert [h.trial_id for h in hits] == ["trial0003"]


def test_limit_and_no_hits(populated):
    s = FtsSearcher(populated.conn)
    assert len(s.search("regime", limit=1)) == 1
    assert s.search("zzzznothing") == []
    assert s.search("") == []


def test_hostile_query_does_not_crash(populated):
    s = FtsSearcher(populated.conn)
    for q in ['"', "regime AND (", "NEAR(", "a*b'c", "regime OR", "col:regime", "-"]:
        assert isinstance(s.search(q), list)


def test_hit_carries_verdict_and_family(populated):
    populated.index_receipt("trial0001", "S0", "h", "PASS", "p")
    populated.index_receipt("trial0001", "S1", "h", "FAIL", "p")
    hit = FtsSearcher(populated.conn).search("momentum")[0]
    assert isinstance(hit, SearchHit)
    assert hit.verdict == "FAIL" and hit.family == "momentum" and "Momentum" in hit.text
    none = FtsSearcher(populated.conn).search("seasonality")[0]
    assert none.verdict is None


def test_reindex_rebuilds_from_hypotheses(populated):
    populated.conn.execute("DELETE FROM hypotheses_fts")
    s = FtsSearcher(populated.conn)
    assert s.search("momentum") == []
    assert s.reindex() == 3
    assert [h.trial_id for h in s.search("momentum")] == ["trial0001"]


def test_cli_search_and_reindex(tmp_path, monkeypatch):
    monkeypatch.setenv("QRH_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    r = Registry(tmp_path / "home" / "registry.sqlite3", id_factory=lambda: "abcdef0123456789")
    r.register_hypothesis(rec("Momentum persists after volatility regime shifts", "momentum"))
    r.close()
    res = runner.invoke(app, ["search", "momentum volatility"])
    assert res.exit_code == 0 and "abcdef01" in res.output and "momentum" in res.output
    res = runner.invoke(app, ["search", "--reindex"])
    assert res.exit_code == 0 and "1" in res.output
    assert runner.invoke(app, ["search", "nothing-matches-zzz"]).exit_code == 0
    assert runner.invoke(app, ["search"]).exit_code == 2  # no query and no --reindex
