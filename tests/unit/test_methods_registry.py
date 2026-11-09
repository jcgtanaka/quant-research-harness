import pytest

from qrh.methods.registry import (
    Method,
    MethodError,
    MethodRegistry,
    Reference,
    parse_ref,
)
from qrh.methods.builtin import BUILTIN, builtin_registry


def _m(id="s9.demo", version="alpha@1", status="default", **kw):
    base = dict(
        id=id, version=version, stage="S5", kind="statistic", entry=lambda x: x, parameters={},
        status=status, references=(), added_in="0.0.1",
    )
    base.update(kw)
    return Method(**base)


def test_parse_ref_accepts_equals_and_at_forms():
    assert parse_ref("s5.pbo=cscv@1") == ("s5.pbo", "cscv@1")
    assert parse_ref("s5.pbo@cscv@2") == ("s5.pbo", "cscv@2")
    for bad in ("", "s5.pbo", "s5.pbo=cscv", "s5.pbo=cscv@0", "=cscv@1", "S5.PBO=cscv@1", "s5.pbo=cscv@x"):
        with pytest.raises(MethodError):
            parse_ref(bad)


def test_method_validates_its_fields():
    with pytest.raises(MethodError):
        _m(status="bogus")
    with pytest.raises(MethodError):
        _m(version="nover")
    with pytest.raises(MethodError):
        _m(stage="S9")
    with pytest.raises(MethodError):
        _m(id="Has Space")


def test_reference_verified_requires_an_identifier():
    with pytest.raises(MethodError):
        Reference(citation="Someone (2020)", verified=True)
    ok = Reference(citation="Someone (2020)", doi="10.1/abc", verified=True)
    assert ok.verified and ok.as_dict()["doi"] == "10.1/abc"
    assert Reference(citation="x").verified is False


def test_registry_rejects_duplicates_and_looks_up():
    reg = MethodRegistry()
    reg.register(_m())
    with pytest.raises(MethodError):
        reg.register(_m())
    reg.register(_m(version="alpha@2", status="experimental"))
    assert reg.get("s9.demo", "alpha@2").status == "experimental"
    assert [m.version for m in reg.versions("s9.demo")] == ["alpha@1", "alpha@2"]
    with pytest.raises(MethodError):
        reg.get("s9.demo", "alpha@3")
    with pytest.raises(MethodError):
        reg.get("s9.none", "alpha@1")


def test_registry_default_is_the_single_default_version():
    reg = MethodRegistry()
    reg.register(_m())
    reg.register(_m(version="alpha@2", status="experimental"))
    assert reg.default("s9.demo").version == "alpha@1"


def test_registry_refuses_two_defaults_for_one_id():
    reg = MethodRegistry()
    reg.register(_m())
    with pytest.raises(MethodError):
        reg.register(_m(version="alpha@2", status="default"))


def test_builtin_registry_has_the_v1_implementations_as_defaults():
    expected = {
        "s2.lookahead_lint": "ast_rules@1",
        "s2.truncation": "prefix_truncation@1",
        "s2.sanity": "bounds@1",
        "s4.walkforward": "rolling_anchored@1",
        "s5.pbo": "cscv@1",
        "s5.dsr": "bailey_lopezdeprado2014@1",
        "s6.bootstrap": "stationary_politis_romano1994@1",
        "metrics.periods_per_year": "observed_density@1",
    }
    for mid, ver in expected.items():
        assert BUILTIN.default(mid).version == ver, mid
    rule_ids = {m.id for m in BUILTIN.all() if m.id.startswith("s1.quality.")}
    assert rule_ids == {f"s1.quality.{t}" for t in ("schema", "range", "relationship", "null_rate_drift", "gaps", "timezone")}
    for mid in rule_ids:
        assert BUILTIN.default(mid).version == f"{mid.rsplit('.', 1)[1]}@1"


def test_builtin_registry_shape():
    for m in BUILTIN.all():
        assert callable(m.entry) and m.added_in and m.stage in {"S1", "S2", "S3", "S4", "S5", "S6"}
        assert m.status in {"default", "experimental"}
        for ref in m.references:
            assert ref.citation.strip()
            assert ref.verified is False or (ref.doi or ref.arxiv or ref.url)
    assert builtin_registry() is not BUILTIN  # fresh copies for tests that register extras


def test_builtin_references_for_the_cited_methods():
    pbo = BUILTIN.get("s5.pbo", "cscv@1").references[0]
    assert "Bailey" in pbo.citation and "2017" in pbo.citation
    assert BUILTIN.get("s6.bootstrap", "stationary_politis_romano1994@1").references[0].citation.startswith("Politis")
    assert BUILTIN.get("s5.dsr", "bailey_lopezdeprado2014@1").references


def test_dois_are_never_unverified_guesses():
    # A DOI may be present only when it was checked against the publisher record (verified=True).
    for m in BUILTIN.all():
        for ref in m.references:
            if ref.doi:
                assert ref.verified, (m.id, ref.citation)


def test_experimental_alternative_is_registered_but_not_default():
    m = BUILTIN.get("s4.walkforward", "purged_embargo@1")
    assert m.status == "experimental"
    assert any("Advances in Financial Machine Learning" in r.citation for r in m.references)
    assert BUILTIN.default("s4.walkforward").version == "rolling_anchored@1"
