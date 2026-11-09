"""Pack folders: manifest, content files, hashing and layering of thresholds."""

import pytest

from packs_helpers import lesson, make_pack, question, write_yaml
from qrh.packs.layering import layer_thresholds, loosenings
from qrh.packs.model import PackError, load_pack
from qrh.validation.defaults import DEFAULT_THRESHOLDS


def test_loads_manifest_and_content(tmp_path):
    d = make_pack(tmp_path, lessons=[lesson()], questions=[[question()]], attempts=[{
        "text": "tried a thing", "strategy_family": "trend", "asset_class": "equity", "frequency": "1D",
        "variants_tried": 7, "verdict": "FAIL", "date": "2020-01-01", "notes": "n"}],
        account={"capital": 1000, "base_currency": "USD", "min_order_size": {"equity": 1}, "fee_schedule": {},
                 "margin_rules": {}, "caps": {"max_gross_leverage": 0.5}},
        thresholds={"all": {"max_pbo": 0.1}})
    pack = load_pack(d)
    assert pack.manifest.name == "demo" and pack.manifest.visibility == "public"
    assert [x.id for x in pack.lessons] == ["L1"] and len(pack.questions) == 1
    assert pack.attempts[0].variants_tried == 7 and pack.account.caps == {"max_gross_leverage": 0.5}
    assert len(pack.hash) == 64


def test_hash_changes_with_content(tmp_path):
    d = make_pack(tmp_path)
    h1 = load_pack(d).hash
    write_yaml(d / "lessons" / "x.yaml", lesson("L9"))
    assert load_pack(d).hash != h1


def test_invalid_manifest_lists_every_problem(tmp_path):
    d = make_pack(tmp_path, version="one")
    write_yaml(d / "qrh-pack.yaml", {"name": "x", "version": "one", "schema_version": 1, "visibility": "secret",
                                     "description": "", "authors": [], "license": ""})
    with pytest.raises(PackError) as e:
        load_pack(d)
    assert "version" in str(e.value) and "visibility" in str(e.value)


def test_require_lesson_needs_requirement(tmp_path):
    d = make_pack(tmp_path, lessons=[lesson(action="require")])
    with pytest.raises(PackError, match="requirement"):
        load_pack(d)


def test_draft_lessons_are_excluded_until_approved(tmp_path):
    d = make_pack(tmp_path, lessons=[lesson("A"), lesson("B", status="draft")])
    pack = load_pack(d)
    assert [x.id for x in pack.lessons] == ["A"] and pack.draft_count == 1


def test_missing_manifest_is_error(tmp_path):
    with pytest.raises(PackError, match="qrh-pack.yaml"):
        load_pack(tmp_path)


def test_lint_rule_needs_regex_or_ast(tmp_path):
    d = make_pack(tmp_path, lint=[{"id": "R1", "severity": "WARN", "message": "m"}])
    with pytest.raises(PackError, match="regex or ast"):
        load_pack(d)


def test_tightening_is_free_and_loosening_is_detected(tmp_path):
    tight = load_pack(make_pack(tmp_path, "t", thresholds={"all": {"max_pbo": 0.1, "min_dsr": 0.97}}))
    assert loosenings([tight], {}) == []
    loose = load_pack(make_pack(tmp_path, "l", thresholds={"all": {"max_pbo": 0.3, "min_dsr": 0.9}}))
    got = {(x.name, x.value) for x in loosenings([loose], {})}
    assert got == {("max_pbo", 0.3), ("min_dsr", 0.9)}


def test_layering_order_and_dimensions(tmp_path):
    pub = load_pack(make_pack(tmp_path, "pub", thresholds={
        "all": {"max_pbo": 0.15}, "asset_class": {"equity": {"min_dsr": 0.97}, "fx": {"min_dsr": 0.99}}}))
    priv = load_pack(make_pack(tmp_path, "priv", visibility="private", thresholds={"all": {"max_pbo": 0.1}}))
    attrs = {"asset_class": "equity", "strategy_family": "trend", "frequency": "1D"}
    eff, info = layer_thresholds(DEFAULT_THRESHOLDS, [pub, priv], attrs, {}, approved=set())
    assert eff["max_pbo"] == 0.1 and eff["min_dsr"] == 0.97
    assert info["tightened"]["max_pbo"] == {"from": 0.15, "to": 0.1, "pack": "priv"}


def test_a_later_pack_cannot_relax_an_earlier_tightening(tmp_path):
    a = load_pack(make_pack(tmp_path, "a", thresholds={"all": {"max_pbo": 0.1}}))
    b = load_pack(make_pack(tmp_path, "b", visibility="private", thresholds={"all": {"max_pbo": 0.15}}))
    eff, _ = layer_thresholds(DEFAULT_THRESHOLDS, [a, b], {}, {}, approved=set())
    assert eff["max_pbo"] == 0.1


def test_unapproved_loosening_is_rejected_and_approved_is_recorded(tmp_path):
    loose = load_pack(make_pack(tmp_path, "l", thresholds={"all": {"max_pbo": 0.3}}))
    with pytest.raises(PackError, match="loosen"):
        layer_thresholds(DEFAULT_THRESHOLDS, [loose], {}, {}, approved=set())
    eff, info = layer_thresholds(DEFAULT_THRESHOLDS, [loose], {}, {}, approved={"l:max_pbo=0.3"})
    assert eff["max_pbo"] == 0.3 and info["approved_loosening"] == ["l:max_pbo=0.3"]


def test_hypothesis_is_the_last_layer_and_overrides_are_recorded(tmp_path):
    p = load_pack(make_pack(tmp_path, "p", thresholds={"all": {"max_pbo": 0.1}}))
    eff, info = layer_thresholds(DEFAULT_THRESHOLDS, [p], {}, {"max_pbo": 0.18}, approved=set())
    assert eff["max_pbo"] == 0.18 and info["hypothesis_looser_than_pack"] == ["max_pbo"]
