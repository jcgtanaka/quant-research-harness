"""Rule 1: the training-cutoff gate (``s2.llm_gate@1``), a pure function of the declaration and the windows."""

import copy

import pytest

from qrh.llm.gate import evaluate_gate
from qrh.llm.spec import LLMSpec
from qrh.packs.layering import direction, is_looser
from qrh.validation.defaults import DEFAULT_THRESHOLDS, fill_thresholds

MODEL = {"id": "m1", "provider": "fake", "training_cutoff": "2020-01-31", "cutoff_source": "provider_documented"}
SPEC = {"models": [MODEL], "live_retrieval": False}
WINDOWS = {"development": ("2021-01-01", "2023-12-31")}


def gate(spec=None, windows=None, margin=180, **changes):
    raw = copy.deepcopy(SPEC if spec is None else spec)
    raw.update(changes)
    return evaluate_gate(LLMSpec.model_validate(raw).model_dump(mode="json"), windows or WINDOWS, margin)


def test_default_margin_is_a_conservative_documented_threshold():
    assert DEFAULT_THRESHOLDS["llm_cutoff_margin_days"] == 180
    assert fill_thresholds({})["llm_cutoff_margin_days"] == 180


def test_margin_is_tightened_by_raising_it_and_loosening_needs_the_approval_flow():
    assert direction("llm_cutoff_margin_days") == "min"  # higher is stricter, like min_* floors
    assert is_looser("llm_cutoff_margin_days", 180, 90) and not is_looser("llm_cutoff_margin_days", 180, 365)


def test_window_after_cutoff_plus_margin_passes_and_records_the_evidence():
    g = gate()
    assert g.verdict == "PASS" and g.reasons == []
    ev = g.evidence
    assert ev["margin_days"] == 180 and ev["binding_cutoff"] == "2020-01-31" and ev["binding_date"] == "2020-07-29"
    assert ev["binding_model"] == "m1"
    assert ev["models"] == [{"id": "m1", "provider": "fake", "training_cutoff": "2020-01-31",
                             "cutoff_source": "provider_documented"}]
    assert ev["windows"][0]["name"] == "development" and ev["windows"][0]["status"] == "PASS"
    assert ev["windows"][0]["earliest_contaminated_date"] is None


def test_window_starting_inside_the_margin_fails_and_names_everything():
    g = gate(windows={"development": ("2020-05-01", "2023-12-31")})
    assert g.verdict == "FAIL"
    text = " ".join(g.reasons)
    for needle in ("window 'development'", "2020-05-01", "model 'm1'", "2020-01-31", "180 days", "2020-07-29",
                   "earliest contaminated date 2020-05-01"):
        assert needle in text, needle


def test_the_start_must_be_strictly_after_the_binding_date():
    assert gate(windows={"development": ("2020-07-29", "2023-12-31")}).verdict == "FAIL"   # equal: still contaminated
    assert gate(windows={"development": ("2020-07-30", "2023-12-31")}).verdict == "PASS"


def test_the_margin_comes_from_the_threshold():
    w = {"development": ("2020-09-01", "2023-12-31")}
    assert gate(windows=w, margin=180).verdict == "PASS"
    assert gate(windows=w, margin=365).verdict == "FAIL"
    assert gate(windows=w, margin=0).verdict == "PASS"


def test_every_window_in_the_spec_is_checked_not_only_the_development_window():
    w = {"development": ("2022-01-01", "2022-12-31"), "holdout": ("2023-01-01", "2023-12-31"),
         "stress": ("2020-02-01", "2020-06-30")}
    g = gate(windows=w)
    assert g.verdict == "FAIL"
    by_name = {x["name"]: x for x in g.evidence["windows"]}
    assert by_name["development"]["status"] == "PASS" and by_name["holdout"]["status"] == "PASS"
    assert by_name["stress"]["status"] == "FAIL" and "window 'stress'" in " ".join(g.reasons)


def test_the_binding_cutoff_is_the_latest_declared_cutoff_and_variants_models_count():
    late = {**MODEL, "id": "m2", "training_cutoff": "2021-03-31"}
    g = gate(models=[MODEL, late])
    assert g.evidence["binding_model"] == "m2" and g.evidence["binding_date"] == "2021-09-27"
    assert g.verdict == "FAIL" and "model 'm2'" in " ".join(g.reasons)
    both = gate(models=[MODEL, late], prompt_files=["p.txt"],
                variants=[{"name": "only-old", "model": "m1", "prompt_file": "p.txt"},
                          {"name": "newer", "model": "m2", "prompt_file": "p.txt"}])
    assert both.evidence["binding_model"] == "m2"  # a model used by a variant is part of the gate


@pytest.mark.parametrize("model", [
    {**MODEL, "training_cutoff": "unknown"},
    {**MODEL, "cutoff_source": "unknown"},
    {**MODEL, "training_cutoff": "unknown", "cutoff_source": "unknown"},
])
def test_an_unknown_cutoff_or_source_blocks_and_names_the_model(model):
    g = gate(models=[model])
    assert g.verdict == "BLOCKED"
    assert "model 'm1'" in " ".join(g.reasons) and "unknown" in " ".join(g.reasons)


def test_one_unknown_model_blocks_even_when_the_others_are_fine():
    g = gate(models=[MODEL, {**MODEL, "id": "m2", "training_cutoff": "unknown"}])
    assert g.verdict == "BLOCKED" and "model 'm2'" in " ".join(g.reasons) and "model 'm1'" not in " ".join(g.reasons)


def test_blocked_wins_over_fail():
    g = gate(models=[MODEL, {**MODEL, "id": "m2", "cutoff_source": "unknown"}],
             windows={"development": ("2020-02-01", "2023-12-31")})
    assert g.verdict == "BLOCKED"
    assert any("starts 2020-02-01" in r for r in g.reasons)  # the contamination is still reported


def test_an_estimated_cutoff_is_usable():
    assert gate(models=[{**MODEL, "cutoff_source": "estimated"}]).verdict == "PASS"


def test_live_retrieval_without_evidence_blocks():
    g = gate(live_retrieval={})
    assert g.verdict == "BLOCKED" and "point_in_time_evidence" in " ".join(g.reasons)
    assert gate(live_retrieval={"point_in_time_evidence": "   "}).verdict == "BLOCKED"


def test_live_retrieval_with_evidence_passes_the_gate_but_leaves_a_warning():
    g = gate(live_retrieval={"point_in_time_evidence": "vendor snapshots keyed by publication time"})
    assert g.verdict == "PASS"
    assert g.evidence["live_retrieval"] == {"used": True, "evidence": "vendor snapshots keyed by publication time"}
    assert gate().evidence["live_retrieval"] == {"used": False, "evidence": None}


@pytest.mark.parametrize("margin", [None, "180", float("nan"), -1, True])
def test_a_bad_margin_threshold_blocks(margin):
    g = gate(margin=margin)
    assert g.verdict == "BLOCKED" and "llm_cutoff_margin_days" in " ".join(g.reasons)


def test_timezone_aware_window_starts_are_compared_by_their_utc_date():
    g = gate(windows={"development": ("2020-07-30T00:30:00+02:00", "2023-12-31T00:00:00+00:00")})  # 2020-07-29 UTC
    assert g.verdict == "FAIL"


def test_the_evidence_is_json_safe_and_deterministic():
    import json

    a, b = gate(), gate()
    assert json.dumps(a.evidence, sort_keys=True) == json.dumps(b.evidence, sort_keys=True)


def test_a_pack_may_tighten_the_margin_but_loosening_it_needs_a_human_approval(tmp_path):
    from packs_helpers import make_pack

    from qrh.packs.layering import layer_thresholds
    from qrh.packs.model import PackError, load_pack

    tight = load_pack(make_pack(tmp_path, "tight", thresholds={"all": {"llm_cutoff_margin_days": 365}}))
    eff, info = layer_thresholds(DEFAULT_THRESHOLDS, [tight], {}, {}, set())
    assert eff["llm_cutoff_margin_days"] == 365 and info["tightened"]["llm_cutoff_margin_days"]["to"] == 365
    loose = load_pack(make_pack(tmp_path, "loose", thresholds={"all": {"llm_cutoff_margin_days": 90}}))
    with pytest.raises(PackError, match="loosen core threshold llm_cutoff_margin_days"):
        layer_thresholds(DEFAULT_THRESHOLDS, [loose], {}, {}, set())
    eff, info = layer_thresholds(DEFAULT_THRESHOLDS, [loose], {}, {}, {"loose:llm_cutoff_margin_days=90"})
    assert eff["llm_cutoff_margin_days"] == 90 and info["approved_loosening"] == ["loose:llm_cutoff_margin_days=90"]
