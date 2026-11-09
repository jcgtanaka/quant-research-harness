"""Readiness warnings about the parameter stability check: unconfirmed thresholds and checks that did not conclude."""

import pytest

from readiness_helpers import TID, clean_run, confirm_stability, make_registry, put_receipt
from qrh.interview import engine
from qrh.interview.stability import stability_questions
from qrh.packs.runtime import PackView
from qrh.readiness import collect as rc
from qrh.validation.defaults import DEFAULT_THRESHOLDS

NAMES = ("stability_closeness", "stability_plateau_share", "stability_neighbour_ratio", "stability_steps")
FROZEN = {"max_pbo": 0.2, **{n: DEFAULT_THRESHOLDS[n] for n in NAMES}}


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    return h


def registry(home, **kw):
    return make_registry(home, thresholds=FROZEN, **kw)


def warning(home, reg, code):
    return next((w for w in rc.collect(home, TID, reg).warnings if w.code == code), None)


def test_unanswered_stability_questions_raise_a_medium_warning_even_without_packs(home):
    reg = registry(home, stability=False)
    clean_run(home, reg)
    w = warning(home, reg, "stability-unconfirmed")
    assert w is not None and w.kind == "interview" and w.stage == "S5" and w.severity == "medium"
    assert "stability thresholds not confirmed by the trader" in w.message
    assert "default" in w.message and "unanswered" in w.message
    for qid in ("stability-closeness", "stability-plateau-share", "stability-neighbour-ratio", "stability-steps"):
        assert qid in w.message
    assert "qrh interview next --stage S0 --scope strategy:trend|synthetic|1D" in w.resolution
    assert "qrh new" in w.resolution  # the thresholds are frozen, so a new trial is needed
    reg.close()


def test_the_dont_know_answer_raises_the_same_warning_and_says_so(home):
    reg = registry(home, stability=False)
    clean_run(home, reg)
    confirm_stability(reg, answer="keep_default")
    q = stability_questions()[0]
    scope = "strategy:trend|synthetic|1D"
    engine.record_answer(reg.conn, q=q, scope_key=scope, identity=engine.identity_for(reg.conn, PackView.empty(), scope),
                         value="unknown", answered_by="human", provenance="human-cli", session_id=None)
    w = warning(home, reg, "stability-unconfirmed")
    assert w is not None and "stability-closeness" in w.message and "don't know" in w.message
    assert "stability-steps" not in w.message
    reg.close()


def test_confirmed_thresholds_raise_no_warning(home):
    reg = registry(home)
    clean_run(home, reg)
    assert warning(home, reg, "stability-unconfirmed") is None
    reg.close()


def test_an_answer_that_differs_from_the_frozen_threshold_is_flagged(home):
    reg = registry(home, stability=False)
    clean_run(home, reg)
    confirm_stability(reg)
    scope = "strategy:trend|synthetic|1D"  # then the trader asks for a stricter closeness than the one frozen
    engine.record_answer(reg.conn, q=stability_questions()[0], scope_key=scope,
                         identity=engine.identity_for(reg.conn, PackView.empty(), scope), value="stricter",
                         answered_by="human", provenance="human-cli", session_id=None)
    w = warning(home, reg, "stability-unconfirmed")
    assert w is not None and "differs from the frozen" in w.message
    reg.close()


def test_a_check_that_was_not_applicable_is_a_medium_warning(home):
    reg = registry(home)
    clean_run(home, reg)
    put_receipt(home, reg, "S5", params={"param_stability": {
        "status": "NOT_APPLICABLE", "reasons": ["no numeric parameter to vary"], "skipped": ["mode"], "unchecked": []}})
    w = warning(home, reg, "stability-not-applicable")
    assert w is not None and (w.stage, w.severity) == ("S5", "medium") and "no numeric parameter" in w.message
    reg.close()


def test_unchecked_and_skipped_parameters_are_listed(home):
    reg = registry(home)
    clean_run(home, reg)
    put_receipt(home, reg, "S5", params={"param_stability": {
        "status": "PASS", "reasons": [], "skipped": ["mode"],
        "unchecked": [{"dimension": "lag", "reason": "a neighbour value cannot be produced from 0"}]}})
    unchecked = warning(home, reg, "stability-unchecked")
    skipped = warning(home, reg, "stability-skipped")
    assert unchecked.severity == "medium" and "lag" in unchecked.message
    assert skipped.severity == "low" and "mode" in skipped.message
    reg.close()


def test_a_clean_check_adds_nothing(home):
    reg = registry(home)
    clean_run(home, reg)
    put_receipt(home, reg, "S5", params={"param_stability": {"status": "PASS", "reasons": [], "skipped": [], "unchecked": []}})
    codes = {w.code for w in rc.collect(home, TID, reg).warnings}
    assert not codes & {"stability-not-applicable", "stability-unchecked", "stability-skipped", "stability-unconfirmed"}
    reg.close()


def test_keep_the_default_against_a_pack_default_in_force_is_confirmed_only_when_frozen_equals_it(home):
    reg = make_registry(home, thresholds={**FROZEN, "stability_closeness": 0.8})  # confirms keep_default for all four
    clean_run(home, reg)
    put_receipt(home, reg, "S0", params={"stability_thresholds": {
        n: {"value": FROZEN[n], "source": "default", "default": FROZEN[n]} for n in NAMES}
        | {"stability_closeness": {"value": 0.8, "source": "pack strict", "default": 0.8}}})
    assert warning(home, reg, "stability-unconfirmed") is None
    put_receipt(home, reg, "S0", params={"stability_thresholds": {
        n: {"value": FROZEN[n], "source": "default", "default": FROZEN[n]} for n in NAMES}
        | {"stability_closeness": {"value": 0.8, "source": "hypothesis", "default": 0.7}}})
    w = warning(home, reg, "stability-unconfirmed")
    assert w is not None and "stability-closeness" in w.message
    reg.close()


def test_declared_neighbours_and_bounds_are_listed_for_the_trader(home):
    reg = make_registry(home, thresholds=FROZEN)
    reg.conn.execute("DELETE FROM hypotheses WHERE trial_id=?", (TID,))
    from qrh.memory.registry import HypothesisRecord

    reg.register_hypothesis(HypothesisRecord(
        text="A synthetic rule carries information.", strategy_family="trend", asset_class="synthetic", frequency="1D",
        parameter_space_size=1, thresholds=FROZEN, difference_statement="Baseline.", datasets=[{"name": "prices"}],
        stability={"neighbours": {"lookback": [15, 25]}, "bounds": {"lookback": [5, 60]}}), trial_id=TID)
    clean_run(home, reg)
    w = warning(home, reg, "stability-declared")
    assert w is not None and (w.stage, w.severity) == ("S5", "medium")
    assert "lookback" in w.message and "15" in w.message and "[5, 60]" in w.message
    reg.close()
