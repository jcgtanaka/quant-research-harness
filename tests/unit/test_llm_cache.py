"""The replay cache and the single entry point ``qrh.llm.complete`` (no network, fake backends only)."""

import json

import pytest

from qrh import llm
from qrh.llm import cache as lc
from qrh.llm.stub import StubBackend

PROVIDERS = {"m1": "fake"}


class Counting:
    def __init__(self, answer="UP"):
        self.calls = []
        self.answer = answer

    def __call__(self, model_id, prompt, params):
        self.calls.append((model_id, prompt, dict(params)))
        return self.answer


@pytest.fixture(autouse=True)
def clean_backends(monkeypatch):
    monkeypatch.setattr(llm.runtime, "_BACKENDS", {})


def session(tmp_path, mode="record", providers=None):
    return llm.llm_session(mode, tmp_path / "cache", providers if providers is not None else PROVIDERS)


def test_complete_outside_a_run_is_refused():
    with pytest.raises(llm.LLMError, match="outside"):
        llm.complete("m1", "hello")


DS, CODE = "d" * 64, "c" * 64


def key(model="m1", params=None, prompt="p", **hashes):
    return lc.cache_key(model, {} if params is None else params, prompt,
                        **{"dataset_hash": DS, "code_hash": CODE, **hashes})


def test_key_is_canonical_json_of_model_params_prompt_and_both_hashes():
    a = key(params={"t": 0, "a": 1})
    assert a == key(params={"a": 1, "t": 0})
    assert a != key("m2", {"a": 1, "t": 0})
    assert a != key(params={"a": 1, "t": 1})
    assert a != key(params={"a": 1, "t": 0}, prompt="p ")
    import hashlib

    expected = hashlib.sha256(json.dumps(
        {"code_hash": CODE, "dataset_hash": DS, "model_id": "m1", "params": {}, "prompt": "p"}, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    assert key() == expected


def test_a_different_dataset_hash_gives_a_different_key():
    assert key(dataset_hash="e" * 64) != key()


def test_a_different_code_hash_gives_a_different_key():
    assert key(code_hash="e" * 64) != key()


def test_identical_inputs_give_the_identical_key():
    assert key(params={"t": 0}) == key(params={"t": 0}) == lc.cache_key(
        "m1", {"t": 0}, "p", dataset_hash=DS, code_hash=CODE)


def test_as_of_is_not_part_of_the_key():
    import inspect

    assert "as_of" not in inspect.signature(lc.cache_key).parameters
    with pytest.raises(TypeError):
        lc.cache_key("m1", {}, "p", dataset_hash=DS, code_hash=CODE, as_of="2020-01-01")


@pytest.mark.parametrize("missing", ["dataset_hash", "code_hash"])
@pytest.mark.parametrize("bad", [None, "", "  ", 3])
def test_a_missing_hash_is_refused_never_replaced_by_an_empty_value(missing, bad):
    with pytest.raises(llm.LLMError, match=missing):
        key(**{missing: bad})


@pytest.mark.parametrize("missing", ["dataset_hash", "code_hash"])
def test_hashes_are_required_arguments(missing):
    args = {"dataset_hash": DS, "code_hash": CODE}
    del args[missing]
    with pytest.raises(TypeError):
        lc.cache_key("m1", {}, "p", **args)


def test_an_entry_without_its_hashes_fails_verification(tmp_path):
    root = tmp_path / "cache"
    cache = lc.ReplayCache(root)
    entry = {"key": key(), "model_id": "m1", "params": {}, "prompt": "p", "response": "UP", "as_of": None,
             "dataset_hash": DS, "code_hash": CODE}
    cache.add(entry)
    cache.flush()
    assert lc.ReplayCache(root).verify() == []
    stale = {k: v for k, v in entry.items() if k != "code_hash"}
    (root / "entries.jsonl").write_text(json.dumps({"entry_hash": lc.entry_hash(stale), "entry": stale}) + "\n")
    assert any("malformed" in p for p in lc.ReplayCache(root).verify())
    forged = {**entry, "dataset_hash": "e" * 64}
    (root / "entries.jsonl").write_text(json.dumps({"entry_hash": lc.entry_hash(forged), "entry": forged}) + "\n")
    assert any("key" in p for p in lc.ReplayCache(root).verify())


def test_record_calls_the_backend_once_and_replay_serves_it(tmp_path):
    backend = Counting("DOWN")
    llm.register_backend("fake", backend)
    with session(tmp_path) as s:
        assert llm.complete("m1", "q", params={"t": 0}, as_of="2020-01-02") == "DOWN"
        assert llm.complete("m1", "q", params={"t": 0}, as_of="2020-01-03") == "DOWN"  # same key: a hit
        assert (s.recorded, s.hits, s.misses) == (1, 1, 0)
    assert len(backend.calls) == 1
    llm.runtime._BACKENDS.clear()  # replay needs no backend at all
    with session(tmp_path, "replay") as s:
        assert llm.complete("m1", "q", params={"t": 0}) == "DOWN"
        assert (s.recorded, s.hits, s.misses) == (0, 1, 0)


def test_replay_miss_raises_with_the_decision_time(tmp_path):
    with session(tmp_path, "replay"):
        with pytest.raises(llm.LLMCacheMiss) as exc:
            llm.complete("m1", "never seen", as_of="2021-05-04")
    assert exc.value.as_of == "2021-05-04" and exc.value.model_id == "m1"
    assert "replay cache miss" in str(exc.value)


def test_replay_never_calls_a_backend_even_when_one_is_registered(tmp_path):
    backend = Counting()
    llm.register_backend("fake", backend)
    with session(tmp_path, "replay"):
        with pytest.raises(llm.LLMCacheMiss):
            llm.complete("m1", "q")
    assert backend.calls == []


def test_record_without_a_registered_backend_is_an_error(tmp_path):
    with session(tmp_path):
        with pytest.raises(llm.LLMError, match="no backend registered for provider 'fake'"):
            llm.complete("m1", "q")


def test_a_model_that_the_hypothesis_does_not_declare_is_refused(tmp_path):
    llm.register_backend("fake", Counting())
    with session(tmp_path):
        with pytest.raises(llm.LLMError, match="not declared"):
            llm.complete("other-model", "q")


def test_backend_must_return_text(tmp_path):
    llm.register_backend("fake", lambda m, p, params: 3)
    with session(tmp_path):
        with pytest.raises(llm.LLMError, match="must return a string"):
            llm.complete("m1", "q")


def test_entries_are_append_only_jsonl_with_hashes_and_an_index(tmp_path):
    llm.register_backend("fake", Counting("UP"))
    with session(tmp_path):
        llm.complete("m1", "a", as_of="2020-01-01")
        llm.complete("m1", "b", as_of="2020-01-02")
    root = tmp_path / "cache"
    lines = (root / "entries.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["entry_hash"] == lc.entry_hash(first["entry"])
    assert first["entry"]["as_of"] == "2020-01-01" and first["entry"]["response"] == "UP"
    index = json.loads((root / "index.json").read_text())
    assert set(index) == {json.loads(l)["entry"]["key"] for l in lines}
    # a second session appends, never rewrites
    with session(tmp_path):
        llm.complete("m1", "c")
    assert (root / "entries.jsonl").read_text().splitlines()[:2] == lines


def test_cache_hash_is_order_independent_and_changes_with_content(tmp_path):
    llm.register_backend("fake", Counting())
    a, b = tmp_path / "a", tmp_path / "b"
    for root, order in ((a, "xy"), (b, "yx")):
        with llm.llm_session("record", root, PROVIDERS):
            for ch in order:
                llm.complete("m1", ch)
    assert lc.ReplayCache(a).content_hash() == lc.ReplayCache(b).content_hash()
    with llm.llm_session("record", b, PROVIDERS):
        llm.complete("m1", "z")
    assert lc.ReplayCache(a).content_hash() != lc.ReplayCache(b).content_hash()
    assert lc.ReplayCache(tmp_path / "empty").content_hash() == lc.ReplayCache(tmp_path / "other").content_hash()


def test_status_counts_entries_and_lists_models_but_never_prompts(tmp_path):
    llm.register_backend("fake", Counting())
    with session(tmp_path):
        llm.complete("m1", "SECRET LICENSED PROMPT TEXT")
    st = lc.ReplayCache(tmp_path / "cache").status()
    assert st["entries"] == 1 and st["models"] == ["m1"] and len(st["cache_hash"]) == 64
    assert "SECRET" not in json.dumps(st)


def test_verify_detects_a_changed_entry_a_forged_key_and_a_bad_index(tmp_path):
    llm.register_backend("fake", Counting("UP"))
    # real hashes: an entry under the unbound placeholder is itself flagged by verify (test_llm_unbound_guard)
    with llm.llm_session("record", tmp_path / "cache", PROVIDERS, dataset_hash=DS, code_hash=CODE):
        llm.complete("m1", "a")
        llm.complete("m1", "b")
    root = tmp_path / "cache"
    assert lc.ReplayCache(root).verify() == []
    original = (root / "entries.jsonl").read_text()
    # 1: a response edited in place
    lines = original.splitlines()
    doc = json.loads(lines[0])
    doc["entry"]["response"] = "DOWN"
    (root / "entries.jsonl").write_text("\n".join([json.dumps(doc), *lines[1:]]) + "\n")
    assert any("hash" in p for p in lc.ReplayCache(root).verify())
    # 2: an entry whose hash was recomputed but whose key does not match its content
    doc["entry"]["response"] = "UP"
    doc["entry"]["prompt"] = "a different prompt"
    doc["entry_hash"] = lc.entry_hash(doc["entry"])
    (root / "entries.jsonl").write_text("\n".join([json.dumps(doc), *lines[1:]]) + "\n")
    assert any("key" in p for p in lc.ReplayCache(root).verify())
    # 3: an index that disagrees
    (root / "entries.jsonl").write_text(original)
    (root / "index.json").write_text(json.dumps({"deadbeef": "00"}))
    assert any("index" in p for p in lc.ReplayCache(root).verify())
    # 4: not JSON at all
    (root / "entries.jsonl").write_text("not json\n")
    assert any("line 1" in p for p in lc.ReplayCache(root).verify())


def test_a_tampered_cache_blocks_a_run_instead_of_serving_it(tmp_path):
    llm.register_backend("fake", Counting("UP"))
    with session(tmp_path):
        llm.complete("m1", "a")
    root = tmp_path / "cache"
    doc = json.loads((root / "entries.jsonl").read_text().splitlines()[0])
    doc["entry"]["response"] = "DOWN"
    (root / "entries.jsonl").write_text(json.dumps(doc) + "\n")
    with pytest.raises(llm.LLMCacheCorrupt, match="qrh llm verify"):
        with session(tmp_path, "replay"):
            llm.complete("m1", "a")


def test_used_hash_covers_only_the_entries_this_session_used(tmp_path):
    llm.register_backend("fake", Counting())
    with session(tmp_path) as s1:
        llm.complete("m1", "a")
        llm.complete("m1", "b")
        used_ab = s1.used_hash()
    with session(tmp_path, "replay") as s2:
        llm.complete("m1", "a")
        llm.complete("m1", "b")
        assert s2.used_hash() == used_ab  # the same calls give the same hash, in record or replay
    with session(tmp_path, "replay") as s3:
        llm.complete("m1", "a")
        assert s3.used_hash() != used_ab


def test_mode_can_be_forced_to_replay_for_one_block(tmp_path):
    backend = Counting()
    llm.register_backend("fake", backend)
    with session(tmp_path) as s:
        llm.complete("m1", "seen")
        with llm.replay_only():
            assert llm.complete("m1", "seen") == "UP"
            with pytest.raises(llm.LLMCacheMiss):
                llm.complete("m1", "new")
        assert s.mode == "record"
        llm.complete("m1", "new")  # record again after the block
    assert len(backend.calls) == 2


def test_replay_only_is_a_no_op_outside_a_session():
    with llm.replay_only():
        pass


def test_mode_is_a_context_variable_not_an_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("QRH_LLM_MODE", "record")
    llm.register_backend("fake", Counting())
    with session(tmp_path, "replay"):
        with pytest.raises(llm.LLMCacheMiss):
            llm.complete("m1", "q")


def test_bad_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="record or replay"):
        llm.llm_session("live", tmp_path, PROVIDERS)


def test_params_with_numpy_scalars_are_accepted_and_nan_is_refused(tmp_path):
    import numpy as np

    llm.register_backend("fake", Counting())
    with session(tmp_path):
        llm.complete("m1", "q", params={"temperature": np.float64(0.0), "n": np.int64(3)})
        with pytest.raises(ValueError):
            llm.complete("m1", "q", params={"temperature": float("nan")})


def test_stats_for_diagnostics(tmp_path):
    llm.register_backend("fake", Counting())
    with session(tmp_path) as s:
        llm.complete("m1", "a")
        llm.complete("m1", "a")
        assert s.stats() == {"calls": 2, "hits": 1, "recorded": 1, "misses": 0}


# --- the stub -------------------------------------------------------------------------------------------


PROMPT = "Decision date: {d}\nCloses, oldest first: {c}\nFlat threshold (fraction): {t}\nReply: UP, DOWN or FLAT.\n"


def test_stub_follows_a_trend_rule_from_the_prompt_only():
    stub = StubBackend()
    up = PROMPT.format(d="2021-01-05", c="100, 101, 103, 106", t=0.01)
    down = PROMPT.format(d="2021-01-05", c="106, 103, 101, 100", t=0.01)
    flat = PROMPT.format(d="2021-01-05", c="100, 100.2, 100.1, 100.3", t=0.01)
    assert [stub("m", p, {}) for p in (up, down, flat)] == ["UP", "DOWN", "FLAT"]
    assert stub("m", up, {}) == stub("other-model", up, {"x": 1})  # a pure function of the prompt text


def test_stub_without_a_recognizable_prompt_is_still_deterministic():
    stub = StubBackend()
    assert stub("m", "free text", {}) == stub("m", "free text", {}) in ("UP", "DOWN")


def test_stub_can_know_the_future_before_its_cutoff():
    oracle = {"2021-01-05": "DOWN", "2021-09-01": "DOWN"}
    stub = StubBackend(oracle=oracle, oracle_until="2021-06-30")
    rising = lambda d: PROMPT.format(d=d, c="100, 101, 103, 106", t=0.01)  # noqa: E731
    assert stub("m", rising("2021-01-05"), {}) == "DOWN"   # before the cutoff: the true next-bar sign
    assert stub("m", rising("2021-09-01"), {}) == "UP"     # after it: the trend rule
    assert stub("m", rising("2021-01-06"), {}) == "UP"     # not in the oracle: the trend rule
