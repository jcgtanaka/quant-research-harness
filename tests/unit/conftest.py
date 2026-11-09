import pytest

from qrh.pipeline.receipt import Stage, Verdict
from qrh.reports.inputs import StageData
from report_data import TRIAL, _receipt, build


@pytest.fixture
def make_data():
    def factory(stage, verdict=Verdict.PASS, variant=None) -> StageData:
        receipt, art = build(stage, verdict, variant)
        data = StageData(TRIAL, Stage(stage), receipt, art)
        if Stage(stage) is Stage.S7:
            raise ValueError("use make_summary")
        return data

    return factory


@pytest.fixture
def make_summary():
    def factory(final=Verdict.FAIL, failing="S5") -> StageData:
        receipts = {}
        for s in ("S0", "S1", "S2", "S3", "S4", "S5", "S6"):
            if failing and s > failing:
                continue
            v = final if s == failing else Verdict.PASS
            receipts[Stage(s)] = build(s, v)[0]
        s7 = _receipt("S7", final, ["S5: FAIL"] if failing else [], params={
            "stage_verdicts": {s.value: r.verdict.value for s, r in receipts.items()}, "failing_stage": failing})
        receipts[Stage.S7] = s7
        return StageData(TRIAL, Stage.S7, s7, {}, receipts)

    return factory


@pytest.fixture(autouse=True)
def no_code_exposure_noise(monkeypatch):
    """Tests run from an editable checkout, which session-context would warn about on every run; only the tests of
    that warning turn it back on."""
    from qrh.hooks import session_context

    monkeypatch.setattr(session_context, "_exposure_line", lambda: None)
