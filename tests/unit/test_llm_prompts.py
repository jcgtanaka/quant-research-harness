"""Rule 2: prompt files are code (hashed with the adapter), and prompts, models and variants are trials."""

import textwrap

import pytest

from qrh.adapters.loader import AdapterLoadError, load_adapter
from qrh.hashing import code_hash, hash_json, sha256_bytes
from qrh.llm.variants import llm_fingerprint, resolve_variants

CODE = textwrap.dedent(
    """
    from qrh.adapters.base import DatasetRef
    class A:
        def parameter_space(self): return {"k": [1, 2]}
        def required_data(self): return [DatasetRef("px")]
        def run(self, params, data, start, end): raise NotImplementedError
    """
)


def setup(tmp_path, prompts=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "adapter.py").write_text(CODE)
    for name, text in (prompts or {}).items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path


# --- the loader tracks prompt files ----------------------------------------------------------------------


def test_without_dependencies_the_code_hash_is_unchanged(tmp_path):
    d = setup(tmp_path)
    assert load_adapter("adapter:A", d).code_hash == code_hash([d / "adapter.py"])
    assert load_adapter("adapter:A", d, dependencies=()).code_hash == code_hash([d / "adapter.py"])


def test_a_prompt_edit_changes_the_code_hash(tmp_path):
    d = setup(tmp_path, {"prompts/a.txt": "Answer UP or DOWN."})
    first = load_adapter("adapter:A", d, dependencies=["prompts/a.txt"])
    assert first.code_hash != load_adapter("adapter:A", d).code_hash
    (d / "prompts/a.txt").write_text("Answer UP or DOWN. Be brief.")
    second = load_adapter("adapter:A", d, dependencies=["prompts/a.txt"])
    assert second.code_hash != first.code_hash
    (d / "prompts/a.txt").write_text("Answer UP or DOWN.")
    assert load_adapter("adapter:A", d, dependencies=["prompts/a.txt"]).code_hash == first.code_hash


def test_dependency_files_are_listed_but_not_treated_as_source(tmp_path):
    d = setup(tmp_path, {"p.txt": "x"})
    loaded = load_adapter("adapter:A", d, dependencies=["p.txt"])
    assert [p.name for p in loaded.source_files] == ["adapter.py"]
    assert [p.name for p in loaded.dependency_files] == ["p.txt"]


def test_the_hash_does_not_depend_on_the_order_the_files_are_listed(tmp_path):
    d = setup(tmp_path, {"a.txt": "1", "b.txt": "2"})
    one = load_adapter("adapter:A", d, dependencies=["a.txt", "b.txt"]).code_hash
    assert one == load_adapter("adapter:A", d, dependencies=["b.txt", "a.txt"]).code_hash


@pytest.mark.parametrize("bad", ["missing.txt", "../outside.txt", "/etc/hostname"])
def test_a_missing_or_escaping_prompt_file_is_a_load_error(tmp_path, bad):
    d = setup(tmp_path / "inside")
    (tmp_path / "outside.txt").write_text("secret")
    with pytest.raises(AdapterLoadError, match="prompt"):
        load_adapter("adapter:A", d, dependencies=[bad])


def test_a_symlink_out_of_the_folder_is_refused(tmp_path):
    d = setup(tmp_path / "inside")
    (tmp_path / "outside.txt").write_text("secret")
    (d / "link.txt").symlink_to(tmp_path / "outside.txt")
    with pytest.raises(AdapterLoadError, match="prompt"):
        load_adapter("adapter:A", d, dependencies=["link.txt"])


# --- variants -------------------------------------------------------------------------------------------

MODELS = [
    {"id": "m1", "provider": "fake", "training_cutoff": "2020-01-31", "cutoff_source": "provider_documented"},
    {"id": "m2", "provider": "fake", "training_cutoff": "2020-02-29", "cutoff_source": "estimated"},
]


def llm(**kw):
    return {"models": MODELS, "prompt_files": [], "params": {"temperature": 0}, "live_retrieval": False,
            "variants": [], **kw}


def test_no_variants_declared_means_exactly_one_variant_formed_by_the_declaration(tmp_path):
    d = setup(tmp_path, {"a.txt": "AAA", "b.txt": "BBB"})
    [v] = resolve_variants(llm(prompt_files=["a.txt", "b.txt"]), d)
    assert v["name"] == "default" and v["models"] == ["m1", "m2"] and v["prompt_files"] == ["a.txt", "b.txt"]
    assert v["params"] == {"temperature": 0} and v["implicit"] is True
    assert v["prompt_hash"] == hash_json(sorted([sha256_bytes(b"AAA"), sha256_bytes(b"BBB")]))


def test_a_single_prompt_file_hash_is_the_file_hash(tmp_path):
    d = setup(tmp_path, {"a.txt": "AAA"})
    [v] = resolve_variants(llm(prompt_files=["a.txt"]), d)
    assert v["prompt_hash"] == sha256_bytes(b"AAA")


def test_declared_variants_each_carry_their_model_prompt_hash_and_merged_params(tmp_path):
    d = setup(tmp_path, {"a.txt": "AAA", "b.txt": "BBB"})
    spec = llm(prompt_files=["a.txt", "b.txt"], variants=[
        {"name": "terse", "model": "m1", "prompt_file": "a.txt", "params": {}},
        {"name": "verbose", "model": "m2", "prompt_file": "b.txt", "params": {"temperature": 0.7, "top_p": 1}},
    ])
    terse, verbose = resolve_variants(spec, d)
    assert (terse["name"], terse["models"], terse["prompt_files"]) == ("terse", ["m1"], ["a.txt"])
    assert terse["params"] == {"temperature": 0} and terse["prompt_hash"] == sha256_bytes(b"AAA")
    assert verbose["params"] == {"temperature": 0.7, "top_p": 1} and verbose["prompt_hash"] == sha256_bytes(b"BBB")


def test_a_prompt_edit_changes_the_variant_hash_and_the_fingerprint(tmp_path):
    d = setup(tmp_path, {"a.txt": "AAA"})
    spec = llm(prompt_files=["a.txt"])
    before_v, before_f = resolve_variants(spec, d)[0]["prompt_hash"], llm_fingerprint(spec, d)
    (d / "a.txt").write_text("AAB")
    assert resolve_variants(spec, d)[0]["prompt_hash"] != before_v
    assert llm_fingerprint(spec, d)["fingerprint"] != before_f["fingerprint"]
    assert llm_fingerprint(spec, d)["prompt_hash"] != before_f["prompt_hash"]


def test_the_fingerprint_also_moves_with_models_and_params(tmp_path):
    d = setup(tmp_path, {"a.txt": "AAA"})
    base = llm_fingerprint(llm(prompt_files=["a.txt"]), d)
    assert base["fingerprint"] == llm_fingerprint(llm(prompt_files=["a.txt"]), d)["fingerprint"]
    other_model = llm_fingerprint(llm(prompt_files=["a.txt"], models=MODELS[:1]), d)
    assert other_model["prompt_hash"] == base["prompt_hash"] and other_model["fingerprint"] != base["fingerprint"]
    other_params = llm_fingerprint(llm(prompt_files=["a.txt"], params={"temperature": 1}), d)
    assert other_params["fingerprint"] != base["fingerprint"]


def test_resolving_a_missing_prompt_file_fails_closed(tmp_path):
    with pytest.raises(AdapterLoadError, match="prompt"):
        resolve_variants(llm(prompt_files=["nope.txt"]), setup(tmp_path))


# --- the adapter reads what was hashed ---------------------------------------------------------------------


def test_read_prompt_returns_the_text_when_the_hash_matches(tmp_path):
    from qrh.llm import read_prompt

    d = setup(tmp_path, {"p/a.txt": "Hello {x}"})
    [v] = resolve_variants(llm(prompt_files=["p/a.txt"]), d)
    assert read_prompt(v, d / "adapter.py") == "Hello {x}"


def test_read_prompt_refuses_text_edited_after_it_was_hashed(tmp_path):
    from qrh.llm import LLMError, read_prompt

    d = setup(tmp_path, {"a.txt": "one"})
    [v] = resolve_variants(llm(prompt_files=["a.txt"]), d)
    (d / "a.txt").write_text("two")
    with pytest.raises(LLMError, match="changed after"):
        read_prompt(v, d / "adapter.py")


def test_read_prompt_joins_several_files_in_declared_order(tmp_path):
    from qrh.llm import read_prompt

    d = setup(tmp_path, {"sys.txt": "SYSTEM", "user.txt": "USER"})
    [v] = resolve_variants(llm(prompt_files=["sys.txt", "user.txt"]), d)
    assert read_prompt(v, d / "adapter.py") == "SYSTEM\n\nUSER"
