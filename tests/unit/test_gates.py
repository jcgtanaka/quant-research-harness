import pytest

from qrh.pipeline.gates import check_hypothesis_spec, overall_verdict
from qrh.pipeline.receipt import Stage, Verdict

VALID = {
    "text": "h",
    "strategy_family": "s",
    "asset_class": "a",
    "frequency": "f",
    "parameter_space_size": 4,
    "thresholds": {"min": 1},
    "difference_statement": "differs by x",
}


def test_valid_spec():
    rec, reasons = check_hypothesis_spec(VALID)
    assert reasons == [] and rec.strategy_family == "s"


@pytest.mark.parametrize(
    "patch,needle",
    [
        ({"difference_statement": ""}, "difference_statement"),
        ({"difference_statement": "   "}, "difference_statement"),
        ({"thresholds": {}}, "thresholds"),
        ({"thresholds": None}, "thresholds"),
        ({"parameter_space_size": 0}, "parameter_space_size"),
        ({"surprise": 1}, "surprise"),
    ],
)
def test_invalid_specs(patch, needle):
    rec, reasons = check_hypothesis_spec({**VALID, **patch})
    assert rec is None and any(needle in r for r in reasons)


def test_missing_field_and_non_mapping():
    spec = dict(VALID)
    del spec["frequency"]
    assert any("frequency" in r for r in check_hypothesis_spec(spec)[1])
    rec, reasons = check_hypothesis_spec(["not", "a", "dict"])
    assert rec is None and reasons


S = Stage
P, F, B, I = Verdict.PASS, Verdict.FAIL, Verdict.BLOCKED, Verdict.INCONCLUSIVE
ALL = [Stage(f"S{i}") for i in range(7)]


def test_overall_pass_only_if_all_pass():
    assert overall_verdict({s: P for s in ALL}, ALL)[0] == P


def test_overall_blocked_beats_fail():
    v = {s: P for s in ALL}
    v[S.S2], v[S.S3] = F, B
    assert overall_verdict(v, ALL)[0] == B


def test_overall_fail_and_missing():
    assert overall_verdict({S.S0: P, S.S1: F}, ALL)[0] == F
    verdict, reasons = overall_verdict({S.S0: P}, ALL)
    assert verdict == B and any("S1" in r for r in reasons)


def test_overall_inconclusive_is_not_pass():
    v = {s: P for s in ALL}
    v[S.S4] = I
    assert overall_verdict(v, ALL)[0] == I
