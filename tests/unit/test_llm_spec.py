"""The ``llm`` field of the hypothesis: validation, storage and hashing."""

import copy
import datetime

import pytest

from qrh.memory.registry import HypothesisRecord, Registry
from qrh.pipeline.gates import check_hypothesis_spec

BASE = dict(
    text="t", strategy_family="trend", asset_class="synthetic", frequency="1D", parameter_space_size=1,
    thresholds={"a": 1}, difference_statement="d",
)
LLM = {
    "models": [{"id": "m1", "provider": "fake", "training_cutoff": "2020-01-31", "cutoff_source": "provider_documented"}],
    "prompt_files": ["prompts/a.txt"],
    "params": {"temperature": 0},
    "live_retrieval": False,
}


def spec(**llm_changes):
    llm = copy.deepcopy(LLM)
    llm.update(llm_changes)
    return {**BASE, "llm": llm}


def reasons(raw):
    rec, why = check_hypothesis_spec(raw)
    assert rec is None, "expected the spec to be refused"
    return " | ".join(why)


def test_a_valid_llm_section_is_accepted_and_normalized():
    rec, why = check_hypothesis_spec(spec())
    assert why == [] and rec is not None
    assert rec.llm.models[0].training_cutoff == "2020-01-31"
    assert rec.llm.live_retrieval is False and rec.llm.variants == []


def test_hypothesis_without_llm_is_unchanged_and_keeps_its_hash():
    plain = HypothesisRecord(**BASE)
    assert plain.llm is None
    # the hash of a hypothesis that never had the field must not move: this is the digest of exactly the payload
    # the pre-field algorithm hashed (BASE plus its resolved family id), computed independently of the model
    from qrh.hashing import hash_json

    assert plain.hypothesis_hash() == hash_json({**BASE, "family_id": "trend|synthetic|1D"})
    assert plain.hypothesis_hash() == "2c849f676f976bca0962398ba276e399068b32200944a87835e60d7639bb1325"
    assert "llm" not in plain.binding()


def test_llm_section_is_hashed_when_present():
    a = HypothesisRecord(**spec())
    b = HypothesisRecord(**spec(params={"temperature": 1}))
    assert a.hypothesis_hash() != HypothesisRecord(**BASE).hypothesis_hash()
    assert a.hypothesis_hash() != b.hypothesis_hash()
    assert a.hypothesis_hash() == HypothesisRecord(**spec()).hypothesis_hash()
    assert a.binding()["llm"]["models"][0]["id"] == "m1"


@pytest.mark.parametrize("missing", ["id", "provider", "training_cutoff", "cutoff_source"])
def test_each_model_key_is_required_with_no_default(missing):
    model = dict(LLM["models"][0])
    del model[missing]
    assert missing in reasons(spec(models=[model]))


def test_live_retrieval_is_required():
    raw = spec()
    del raw["llm"]["live_retrieval"]
    assert "live_retrieval" in reasons(raw)


def test_models_are_required_and_not_empty():
    raw = spec()
    del raw["llm"]["models"]
    assert "models" in reasons(raw)
    assert "models" in reasons(spec(models=[]))


@pytest.mark.parametrize("bad", ["2020-13-01", "2020/01/31", "yesterday", "", "2020-1-5", 20200131, None])
def test_bad_cutoff_dates_are_errors(bad):
    model = {**LLM["models"][0], "training_cutoff": bad}
    assert "training_cutoff" in reasons(spec(models=[model]))


def test_unknown_is_an_allowed_cutoff_and_source():
    model = {**LLM["models"][0], "training_cutoff": "unknown", "cutoff_source": "unknown"}
    rec, why = check_hypothesis_spec(spec(models=[model]))
    assert why == [] and rec.llm.models[0].training_cutoff == "unknown"


def test_an_unquoted_yaml_date_is_accepted():
    import yaml

    doc = yaml.safe_load("training_cutoff: 2020-01-31")
    assert isinstance(doc["training_cutoff"], datetime.date)
    model = {**LLM["models"][0], "training_cutoff": doc["training_cutoff"]}
    rec, why = check_hypothesis_spec(spec(models=[model]))
    assert why == [] and rec.llm.models[0].training_cutoff == "2020-01-31"


def test_a_datetime_is_not_a_date():
    model = {**LLM["models"][0], "training_cutoff": datetime.datetime(2020, 1, 31, 12)}
    assert "training_cutoff" in reasons(spec(models=[model]))


def test_unknown_source_value_is_an_error():
    model = {**LLM["models"][0], "cutoff_source": "rumour"}
    assert "cutoff_source" in reasons(spec(models=[model]))


def test_unknown_keys_are_errors_at_every_level():
    raw = spec()
    raw["llm"]["temperature"] = 0
    assert "temperature" in reasons(raw)
    model = {**LLM["models"][0], "version": "x"}
    assert "version" in reasons(spec(models=[model]))


def test_duplicate_model_ids_are_errors():
    assert "duplicate" in reasons(spec(models=[LLM["models"][0], LLM["models"][0]]))


@pytest.mark.parametrize("bad", ["/etc/passwd", "../up.txt", "a/../../b.txt", "", "C:\\x.txt"])
def test_prompt_files_must_be_relative_paths_inside_the_hypothesis_folder(bad):
    assert "prompt_files" in reasons(spec(prompt_files=[bad]))


def test_live_retrieval_forms():
    ok, why = check_hypothesis_spec(spec(live_retrieval={"point_in_time_evidence": "vendor snapshots by date"}))
    assert why == [] and ok.llm.live_retrieval == {"point_in_time_evidence": "vendor snapshots by date"}
    ok, why = check_hypothesis_spec(spec(live_retrieval=True))  # accepted here; S2 BLOCKS it (no evidence)
    assert why == [] and ok.llm.live_retrieval == {}
    assert "live_retrieval" in reasons(spec(live_retrieval="yes"))
    assert "live_retrieval" in reasons(spec(live_retrieval={"evidence": "x"}))
    assert "live_retrieval" in reasons(spec(live_retrieval=None))


def test_variants_must_reference_declared_models_and_prompt_files():
    v = {"name": "v1", "model": "m1", "prompt_file": "prompts/a.txt", "params": {"temperature": 0}}
    rec, why = check_hypothesis_spec(spec(variants=[v]))
    assert why == [] and rec.llm.variants[0].name == "v1"
    assert "model" in reasons(spec(variants=[{**v, "model": "other"}]))
    assert "prompt_file" in reasons(spec(variants=[{**v, "prompt_file": "prompts/zzz.txt"}]))
    assert "duplicate" in reasons(spec(variants=[v, v]))
    assert "name" in reasons(spec(variants=[{**v, "name": " "}]))


def test_params_must_be_json_numbers_and_strings():
    assert "params" in reasons(spec(params={"t": float("nan")}))
    assert "params" in reasons(spec(params={"f": object()}))


def test_llm_is_stored_in_the_registry_binding_and_survives_a_round_trip(tmp_path):
    reg = Registry(tmp_path / "r.sqlite3")
    rec, _ = check_hypothesis_spec(spec())
    tid = reg.register_hypothesis(rec)
    trial = reg.get_trial(tid)
    assert trial["llm"] == rec.binding()["llm"]
    again = HypothesisRecord.model_validate({k: trial[k] for k in HypothesisRecord.model_fields})
    assert again.hypothesis_hash() == trial["hypothesis_hash"]
    plain = reg.register_hypothesis(HypothesisRecord(**BASE))
    assert reg.get_trial(plain)["llm"] is None


def test_the_stored_llm_section_is_immutable(tmp_path):
    import sqlite3

    reg = Registry(tmp_path / "r.sqlite3")
    rec, _ = check_hypothesis_spec(spec())
    tid = reg.register_hypothesis(rec)
    with pytest.raises(sqlite3.DatabaseError):
        reg.conn.execute("UPDATE hypotheses SET binding='{}' WHERE trial_id=?", (tid,))
