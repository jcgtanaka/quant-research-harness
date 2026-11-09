"""The unbound-hash placeholder never reaches a pipeline or bench result, and a replay miss says how to fix it."""

import json
from types import SimpleNamespace

import pytest

from qrh import llm
from qrh.llm import cache as lc
from qrh.llm.runtime import UNBOUND_HASH, require_bound
from qrh.pipeline.stages import MISS_TEXT, Pipeline

REAL = "a" * 64


def test_require_bound_refuses_the_placeholder_in_either_hash_and_names_it():
    for ds, code in ((UNBOUND_HASH, REAL), (REAL, UNBOUND_HASH)):
        with pytest.raises(llm.LLMError, match="placeholder"):
            require_bound(ds, code)
    require_bound(REAL, REAL)  # real hashes pass


def _pipeline(tmp_path):
    trial = {"llm": {"models": [{"id": "m", "provider": "p"}]}, "datasets": []}
    reg = SimpleNamespace(get_trial=lambda tid: trial, path=tmp_path / "r.sqlite3")
    return Pipeline(reg, tmp_path)


def test_pipeline_session_refuses_a_placeholder_code_hash(tmp_path, monkeypatch):
    import qrh.pipeline.stages as stages
    import qrh.pipeline.stages_s1_s2 as s12

    monkeypatch.setattr(s12, "_dataset_hashes", lambda cat, ds: ({"d": REAL}, []))
    monkeypatch.setattr(stages, "package_code_hash", lambda: UNBOUND_HASH)
    with pytest.raises(llm.LLMError, match="placeholder"):
        _pipeline(tmp_path)._llm_session("t1")


def test_pipeline_session_with_real_hashes_is_not_refused(tmp_path, monkeypatch):
    import qrh.pipeline.stages as stages
    import qrh.pipeline.stages_s1_s2 as s12

    monkeypatch.setattr(s12, "_dataset_hashes", lambda cat, ds: ({"d": REAL}, []))
    monkeypatch.setattr(stages, "package_code_hash", lambda: REAL)
    assert _pipeline(tmp_path)._llm_session("t1").code_hash == REAL


def test_bench_detect_llm_refuses_a_placeholder_code_hash(tmp_path, monkeypatch):
    from qrh.bench import llm_corpus as llmc
    from qrh.bench import suites
    from qrh.methods.selection import MethodSelector

    case = next(c for c in llmc.llm_corpus() if c.name == "trend_prompt")  # a control: its gate passes
    m = llmc.materialize_llm(case, 1)
    monkeypatch.setattr(suites, "_package_code_hash", lambda: UNBOUND_HASH)
    sel = MethodSelector(tmp_path / "home")
    with pytest.raises(llm.LLMError, match="placeholder"):
        suites._detect_llm(m, sel, tmp_path, "case")


def _write_entry(tmp_path, dataset_hash, code_hash):
    entry = {"key": lc.cache_key("m", {}, "p", dataset_hash=dataset_hash, code_hash=code_hash), "model_id": "m",
             "params": {}, "prompt": "p", "response": "UP", "as_of": None,
             "dataset_hash": dataset_hash, "code_hash": code_hash}
    cache = lc.ReplayCache(tmp_path / "c")
    cache.add(entry)
    cache.flush()
    return cache


def test_verify_flags_an_entry_stored_under_the_placeholder(tmp_path):
    problems = _write_entry(tmp_path, UNBOUND_HASH, REAL).verify()
    assert any("unbound entry" in p for p in problems)
    problems = _write_entry(tmp_path / "x", REAL, UNBOUND_HASH).verify()
    assert any("unbound entry" in p for p in problems)


def test_verify_accepts_an_entry_with_real_hashes(tmp_path):
    assert _write_entry(tmp_path, REAL, REAL).verify() == []


def test_the_replay_miss_text_says_what_is_missing_how_to_record_and_the_cost():
    text = MISS_TEXT.format(trial="t1")
    assert "this trial has no recorded LLM answer for this data and code version" in text
    assert "re-run with --llm-mode record to record it" in text
    assert "this calls the model and may cost money" in text
