"""Readiness warnings for strategies that use a language model."""

import pytest
from readiness_helpers import TID, clean_run, put_receipt

from qrh.llm import cache as lc
from qrh.memory.registry import HypothesisRecord, Registry
from qrh.readiness.collect import collect
from qrh.readiness.model import KNOWN_LIMITS

LLM = {
    "models": [{"id": "m1", "provider": "fake", "training_cutoff": "2019-01-01", "cutoff_source": "provider_documented"}],
    "live_retrieval": False,
}


def make(home, llm=None, margin=365):
    home.mkdir(parents=True, exist_ok=True)
    reg = Registry(home / "registry.sqlite3")
    reg.register_hypothesis(HypothesisRecord(
        text="A synthetic rule carries information.", strategy_family="trend", asset_class="synthetic",
        frequency="1D", parameter_space_size=9, thresholds={"max_pbo": 0.2, "llm_cutoff_margin_days": margin},
        difference_statement="Baseline.", datasets=[{"name": "prices"}], llm=llm,
    ), trial_id=TID)
    reg.conn.execute(
        "INSERT OR REPLACE INTO datasets(name, path, content_hash, schema, null_rates, status, reasons)"
        " VALUES (?,?,?,?,?,?,?)", ("prices", "/synthetic/prices.parquet", "hash-prices", "{}", "{}", "PASS", "[]"))
    return reg


def warn(home, reg, code):
    return [w for w in collect(home, TID, reg).warnings if w.code == code]


def test_a_strategy_without_llm_gets_no_llm_warnings_and_no_llm_limits(tmp_path):
    reg = make(tmp_path, None)
    clean_run(tmp_path, reg)
    codes = [w.code for w in collect(tmp_path, TID, reg).warnings]
    assert not [c for c in codes if c.startswith("llm")]


def test_the_two_llm_limits_are_in_the_table_and_listed_for_llm_strategies_only(tmp_path):
    by_code = {lim["code"]: lim for lim in KNOWN_LIMITS}
    for code in ("llm-cutoff-claim", "llm-post-cutoff-knowledge"):
        lim = by_code[code]
        assert lim["when"] == "llm" and "\n" not in lim["text"] and len(lim["text"]) <= 200
    assert "documented cutoff can be wrong" in by_code["llm-cutoff-claim"]["text"]
    assert "after their stated cutoff" in by_code["llm-post-cutoff-knowledge"]["text"]
    reg = make(tmp_path, LLM)
    clean_run(tmp_path, reg)
    listed = {w.code for w in collect(tmp_path, TID, reg).warnings if w.kind == "limit"}
    assert {"llm-cutoff-claim", "llm-post-cutoff-knowledge"} <= listed


def test_a_margin_below_365_days_is_a_medium_warning(tmp_path):
    reg = make(tmp_path, LLM, margin=180)
    clean_run(tmp_path, reg)
    [w] = warn(tmp_path, reg, "llm-short-margin")
    assert w.severity == "medium" and w.stage == "S2" and "180" in w.message and "365" in w.message


def test_a_margin_of_365_or_more_is_not_warned_about(tmp_path):
    reg = make(tmp_path, LLM, margin=365)
    clean_run(tmp_path, reg)
    assert warn(tmp_path, reg, "llm-short-margin") == []


def test_live_retrieval_evidence_is_a_high_warning_that_quotes_the_text(tmp_path):
    reg = make(tmp_path, {**LLM, "live_retrieval": {"point_in_time_evidence": "vendor snapshots keyed by publication"}})
    clean_run(tmp_path, reg)
    [w] = warn(tmp_path, reg, "llm-live-retrieval")
    assert w.severity == "high" and "vendor snapshots keyed by publication" in w.message and w.stage == "S2"


def test_a_recorded_cache_is_a_low_warning_with_its_size_and_hash(tmp_path):
    reg = make(tmp_path, LLM)
    clean_run(tmp_path, reg)
    assert warn(tmp_path, reg, "llm-recorded-cache") == []  # nothing recorded yet
    cache = lc.ReplayCache(lc.cache_dir(tmp_path, TID))
    for i in range(3):
        key = lc.cache_key("m1", {}, f"p{i}", dataset_hash="d" * 64, code_hash="c" * 64)
        cache.add({"key": key, "model_id": "m1", "params": {}, "prompt": f"p{i}", "response": "UP", "as_of": None,
                   "dataset_hash": "d" * 64, "code_hash": "c" * 64})
    cache.flush()
    [w] = warn(tmp_path, reg, "llm-recorded-cache")
    assert w.severity == "low" and "3 recorded" in w.message and cache.content_hash() in w.message
    assert "results depend on a recorded cache" in w.message


def test_a_corrupt_cache_is_a_high_warning_not_a_crash(tmp_path):
    reg = make(tmp_path, LLM)
    clean_run(tmp_path, reg)
    d = lc.cache_dir(tmp_path, TID)
    d.mkdir(parents=True)
    (d / "entries.jsonl").write_text("not json\n")
    [w] = warn(tmp_path, reg, "llm-cache-unreadable")
    assert w.severity == "high" and "qrh llm verify" in w.resolution


def test_an_allowed_llm001_is_a_medium_warning_and_replaces_the_generic_suppression_warning(tmp_path):
    reg = make(tmp_path, None)
    put_receipt(tmp_path, reg, "S2", params={"lint": {"verdict": "PASS", "warnings": [], "suppressed": [
        {"rule": "LLM001", "at": "adapter.py:3", "reason": "only the type hints are used"},
        {"rule": "LA005", "at": "adapter.py:9", "reason": "reviewed"}]}})
    ws = collect(tmp_path, TID, reg).warnings
    [w] = [x for x in ws if x.code == "llm-undeclared-allowed"]
    assert w.severity == "medium" and "only the type hints are used" in w.message and "adapter.py:3" in w.message
    suppressed = [x for x in ws if x.code == "lint-suppressed"]
    assert len(suppressed) == 1 and "LA005" in suppressed[0].message  # LLM001 is reported once, in its own warning


def test_direct_provider_clients_are_one_medium_warning_listing_the_files(tmp_path):
    reg = make(tmp_path, LLM)
    put_receipt(tmp_path, reg, "S2", params={"lint": {"verdict": "PASS", "suppressed": [], "warnings": [
        {"rule": "LLM002", "at": "helper.py:2", "snippet": "import openai"},
        {"rule": "LLM002", "at": "helper.py:9", "snippet": "openai.chat()"},
        {"rule": "LLM002", "at": "other.py:1", "snippet": "import ollama"},
        {"rule": "LA005", "at": "adapter.py:9", "snippet": "x / x.mean()"}]}})
    ws = collect(tmp_path, TID, reg).warnings
    [w] = [x for x in ws if x.code == "llm-direct-client"]
    assert w.severity == "medium" and w.stage == "S2" and w.kind == "llm"
    assert "helper.py" in w.message and "other.py" in w.message and "adapter.py" not in w.message
    assert "qrh.llm.complete" in w.message and "qrh.llm.complete" in w.resolution
    assert [x.code for x in ws if x.kind == "lint"] == ["lint-warn"]  # only LA005 stays a generic lint warning


def test_a_suppressed_llm002_still_appears_in_readiness_with_its_reason(tmp_path):
    reg = make(tmp_path, LLM)
    put_receipt(tmp_path, reg, "S2", params={"lint": {"verdict": "PASS", "warnings": [], "suppressed": [
        {"rule": "LLM002", "at": "helper.py:2", "reason": "type hints only"}]}})
    [w] = [x for x in collect(tmp_path, TID, reg).warnings if x.code == "lint-suppressed"]
    assert w.severity == "medium" and "LLM002" in w.message and "type hints only" in w.message
