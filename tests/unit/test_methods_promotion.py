import copy
from datetime import datetime, timedelta, timezone

import pytest

from hook_helpers import human_approve_id
from qrh import approvals  # noqa: F401
from qrh.bench.corpus import BUG_CLASSES
from qrh.bench.profiles import PROFILES
from qrh.bench.receipt import BenchReceipt, write_bench
from qrh.bench.stats import metric
from qrh.methods.promotion import (
    ACTIONS,
    MIN_TIE_REASON,
    check_promotion,
    meets_full_profile,
    run_action,
)
from qrh.methods.selection import MethodSelector
from qrh.methods.state import load_state

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
CAND = ("s4.walkforward", "purged_embargo@1")
FULL = PROFILES["full"]


def suite_results(detect=60, controls=2, fpr=1, power=(10, 30, 60, 80, 90), n=100, exceeds=False):
    planted = {f"detect/{c}": metric(detect, n, direction="higher", gate=True) for c in BUG_CLASSES}
    planted["false_alarm/controls"] = metric(controls, n, direction="lower", gate=True)
    noise, cells = {}, {}
    for g in FULL["noise-fpr"]["grids"]:
        for t in FULL["noise-fpr"]["periods"]:
            key = f"g{g}_t{t}"
            noise[f"fpr/{key}"] = metric(fpr, n, direction="lower", gate=True)
            cells[key] = {"grid": g, "periods": t, "reps": n, "errors": 0, "exceeds_nominal": exceeds}
    noise["fpr/pooled"] = metric(fpr, n, direction="lower")
    pw, pcells = {}, {}
    for s, k in zip(FULL["known-edge-power"]["strengths"], power):
        pw[f"power/s{s:g}"] = metric(k, n, direction="higher", gate=True)
        pcells[f"s{s:g}"] = {"edge": s}
    return {
        "planted-bugs": {"metrics": planted, "reps": n, "variants": {}, "detectors": {}},
        "noise-fpr": {"metrics": noise, "cells": cells, "nominal": 0.05},
        "known-edge-power": {"metrics": pw, "cells": pcells, "strengths": list(FULL["known-edge-power"]["strengths"])},
    }


def bench(cand=None, base=None, overrides=None, config=None, baseline=True, methods=None, base_methods=None, profile="full"):
    cand = cand or suite_results(fpr=0)
    base = base or suite_results(fpr=1)
    return BenchReceipt(
        profile=profile, suites=["planted-bugs", "noise-fpr", "known-edge-power"], seed=1,
        config=config or copy.deepcopy(FULL),
        methods=methods or {CAND[0]: CAND[1], "s5.pbo": "cscv@1"},
        overrides=overrides if overrides is not None else {CAND[0]: CAND[1]},
        non_default_methods={CAND[0]: CAND[1]},
        baseline_methods=base_methods or {CAND[0]: "rolling_anchored@1", "s5.pbo": "cscv@1"},
        results=cand, baseline_results=base if baseline else None, code_hash="c",
    )


@pytest.fixture
def sel(tmp_path):
    return MethodSelector(tmp_path)


def test_a_clean_win_is_eligible(sel):
    d = check_promotion(sel, bench(), *CAND, code_hash="c")
    assert d.ok, d.reasons
    assert d.better and not d.worse and not d.tie_required


def test_any_worse_gated_metric_refuses_and_names_it(sel):
    cand = suite_results(fpr=0)
    cand["planted-bugs"]["metrics"]["detect/centered_rolling"] = metric(59, 100, direction="higher", gate=True)
    d = check_promotion(sel, bench(cand=cand), *CAND, code_hash="c")
    assert not d.ok and d.worse == ["planted-bugs: detect/centered_rolling"]
    assert any("worse" in r for r in d.reasons)


def test_worse_false_positive_rate_and_worse_power_both_count(sel):
    d = check_promotion(sel, bench(cand=suite_results(fpr=2), base=suite_results(fpr=1)), *CAND, code_hash="c")
    assert not d.ok and any("fpr/" in w for w in d.worse)
    d = check_promotion(sel, bench(cand=suite_results(fpr=1, power=(10, 30, 60, 80, 89)), base=suite_results(fpr=1)), *CAND, code_hash="c")
    assert not d.ok and d.worse == ["known-edge-power: power/s0.008"]


def test_all_ties_need_a_documented_reason(sel):
    same = bench(cand=suite_results(), base=suite_results())
    d = check_promotion(sel, same, *CAND, code_hash="c")
    assert not d.ok and d.tie_required and not d.better
    assert not check_promotion(sel, same, *CAND, tie_reason="short", code_hash="c").ok
    ok = check_promotion(sel, same, *CAND, tie_reason="x" * MIN_TIE_REASON, code_hash="c")
    assert ok.ok and ok.tie_required


def test_receipt_must_have_a_baseline_and_exactly_this_override(sel):
    assert not check_promotion(sel, bench(baseline=False), *CAND, code_hash="c").ok
    assert not check_promotion(sel, bench(overrides={}), *CAND, code_hash="c").ok
    assert not check_promotion(sel, bench(overrides={CAND[0]: CAND[1], "s5.pbo": "cscv@1"}), *CAND, code_hash="c").ok
    assert not check_promotion(sel, bench(), "s4.walkforward", "rolling_anchored@1", code_hash="c").ok


def test_other_methods_must_be_identical_between_the_arms(sel):
    r = bench(base_methods={CAND[0]: "rolling_anchored@1", "s5.pbo": "cscv@2"})
    d = check_promotion(sel, r, *CAND, code_hash="c")
    assert not d.ok and any("s5.pbo" in x for x in d.reasons)


def test_stale_baseline_is_refused(tmp_path):
    from qrh.methods.state import save_state

    save_state(tmp_path, {"schema_version": 1, "history": [], "statuses": {
        "s4.walkforward": {"purged_embargo@1": "deprecated", "rolling_anchored@1": "experimental"}}})
    assert any("default" in r for r in check_promotion(MethodSelector(tmp_path), bench(), *CAND, code_hash="c").reasons)


def test_profile_must_meet_the_full_profile(sel):
    quick = copy.deepcopy(FULL)
    quick["noise-fpr"]["reps"] = 10
    assert not meets_full_profile(quick) and meets_full_profile(copy.deepcopy(FULL))
    d = check_promotion(sel, bench(config=quick), *CAND, code_hash="c")
    assert not d.ok and any("profile" in r for r in d.reasons)
    fewer_cells = copy.deepcopy(FULL)
    fewer_cells["known-edge-power"]["strengths"] = [0.004]
    assert not meets_full_profile(fewer_cells)


def test_missing_suite_and_mismatched_cells_are_refused(sel):
    cand = suite_results(fpr=0)
    del cand["noise-fpr"]
    r = bench(cand=cand)
    r = r.model_copy(update={"suites": ["planted-bugs", "known-edge-power"]})
    assert any("noise-fpr" in x for x in check_promotion(sel, r, *CAND, code_hash="c").reasons)
    cand = suite_results(fpr=0)
    del cand["noise-fpr"]["metrics"]["fpr/g12_t500"]
    assert any("cells" in x or "metrics" in x for x in check_promotion(sel, bench(cand=cand), *CAND, code_hash="c").reasons)


def test_a_candidate_that_newly_exceeds_nominal_is_refused(sel):
    d = check_promotion(sel, bench(cand=suite_results(fpr=0, exceeds=True), base=suite_results(fpr=1, exceeds=False)), *CAND, code_hash="c")
    assert not d.ok and any("nominal" in r for r in d.reasons)
    ok = check_promotion(sel, bench(cand=suite_results(fpr=0, exceeds=True), base=suite_results(fpr=1, exceeds=True)), *CAND, code_hash="c")
    assert ok.ok


def test_a_non_experimental_candidate_cannot_be_promoted(sel):
    r = bench(overrides={"s5.pbo": "cscv@1"}, methods={"s5.pbo": "cscv@1"}, base_methods={"s5.pbo": "cscv@1"})
    assert not check_promotion(sel, r, "s5.pbo", "cscv@1", code_hash="c").ok


# --- the approval flow -----------------------------------------------------------------------


def promote(home, receipt, now=NOW, reason="measured win on every benchmark suite", **kw):
    folder = write_bench(receipt, home)
    return run_action(home, "promote", f"{CAND[0]}={CAND[1]}", reason=reason, bench=str(folder / "receipt.json"), now=now, code_hash="c", **kw)


def test_promotion_waits_for_a_human_then_applies(tmp_path):
    r = bench()
    first = promote(tmp_path, r)
    assert first.status == "pending" and first.approval_id
    assert load_state(tmp_path)["statuses"] == {}  # nothing changed yet
    again = promote(tmp_path, r)
    assert again.status == "pending" and again.approval_id == first.approval_id  # same open request
    human_approve_id(tmp_path, first.approval_id, "reviewed the bench report myself", now=NOW)
    done = promote(tmp_path, r, now=NOW + timedelta(minutes=1))
    assert done.status == "applied"
    st = load_state(tmp_path)
    assert st["statuses"]["s4.walkforward"] == {"purged_embargo@1": "default", "rolling_anchored@1": "deprecated"}
    h = st["history"][-1]
    assert h["action"] == "promote" and h["method"] == f"{CAND[0]}={CAND[1]}" and h["approval_id"] == first.approval_id
    assert h["bench_id"] == r.bench_id() and h["bench_receipt_hash"] == r.receipt_hash()
    assert h["reason"] and h["approval_reason"] == "reviewed the bench report myself" and h["previous_default"] == "rolling_anchored@1"
    assert MethodSelector(tmp_path).default("s4.walkforward").version == "purged_embargo@1"


def test_an_approval_is_single_use_and_bound_to_the_exact_action(tmp_path):
    r = bench()
    first = promote(tmp_path, r)
    human_approve_id(tmp_path, first.approval_id, "reviewed the bench report myself", now=NOW)
    other = promote(tmp_path, r, reason="a different reason than the one approved", now=NOW)
    assert other.status == "pending" and other.approval_id != first.approval_id  # different action, new request
    assert promote(tmp_path, r, now=NOW).status == "applied"
    assert promote(tmp_path, r, now=NOW).status == "refused"  # already the default: the approval is spent


def test_an_ineligible_promotion_creates_no_approval_request(tmp_path):
    out = promote(tmp_path, bench(cand=suite_results(fpr=5), base=suite_results(fpr=1)))
    assert out.status == "refused" and out.reasons
    assert not (tmp_path / "approvals").exists()


def test_reason_is_required(tmp_path):
    out = promote(tmp_path, bench(), reason="short")
    assert out.status == "refused" and any("reason" in r for r in out.reasons)


def test_deprecate_and_demote_use_the_same_flow_without_a_bench(tmp_path):
    out = run_action(tmp_path, "deprecate", "s4.walkforward=purged_embargo@1", reason="superseded by a better design", now=NOW)
    assert out.status == "pending"
    human_approve_id(tmp_path, out.approval_id, "agree it is superseded", now=NOW)
    assert run_action(tmp_path, "deprecate", "s4.walkforward=purged_embargo@1", reason="superseded by a better design", now=NOW).status == "applied"
    assert load_state(tmp_path)["statuses"]["s4.walkforward"]["purged_embargo@1"] == "deprecated"
    # the default cannot be deprecated or demoted without naming its replacement
    assert run_action(tmp_path, "deprecate", "s4.walkforward=rolling_anchored@1", reason="no longer wanted at all", now=NOW).status == "refused"
    assert run_action(tmp_path, "demote", "s4.walkforward=rolling_anchored@1", reason="rolling back for a reason", now=NOW).status == "refused"


def test_demote_restores_a_named_version_as_default(tmp_path):
    r = bench()
    first = promote(tmp_path, r)
    human_approve_id(tmp_path, first.approval_id, "reviewed the bench report myself", now=NOW)
    assert promote(tmp_path, r, now=NOW).status == "applied"
    args = dict(reason="regression found after promotion", restore="s4.walkforward=rolling_anchored@1", now=NOW)
    pending = run_action(tmp_path, "demote", "s4.walkforward=purged_embargo@1", **args)
    assert pending.status == "pending"
    human_approve_id(tmp_path, pending.approval_id, "rolling back on purpose", now=NOW)
    assert run_action(tmp_path, "demote", "s4.walkforward=purged_embargo@1", **args).status == "applied"
    assert load_state(tmp_path)["statuses"]["s4.walkforward"] == {"purged_embargo@1": "experimental", "rolling_anchored@1": "default"}
    assert load_state(tmp_path)["history"][-1]["action"] == "demote"


def test_actions_are_the_documented_set():
    assert set(ACTIONS) == {"promote", "demote", "deprecate"}


def test_unknown_action_or_method_is_refused(tmp_path):
    assert run_action(tmp_path, "explode", "s5.pbo=cscv@1", reason="a sufficiently long reason", now=NOW).status == "refused"
    assert run_action(tmp_path, "deprecate", "s5.pbo=cscv@9", reason="a sufficiently long reason", now=NOW).status == "refused"


def test_a_receipt_measured_on_other_code_is_refused(sel):
    d = check_promotion(sel, bench(), *CAND, code_hash="a-different-code-hash")
    assert not d.ok and any("code" in r for r in d.reasons)
    # by default the current harness code is the reference, and the synthetic receipt's hash is not it
    assert not check_promotion(sel, bench(), *CAND).ok


# --- an opt-in method (no shipped default): the baseline is "not selected" ----------------------

PNL = ("s2.pnl_reconstruction", "pnl_reconstruction@1")


def pnl_bench(**kw):
    kw.setdefault("overrides", {PNL[0]: PNL[1]})
    kw.setdefault("methods", {PNL[0]: PNL[1], "s5.pbo": "cscv@1"})
    kw.setdefault("base_methods", {"s5.pbo": "cscv@1"})
    cand = suite_results(detect=60, fpr=1)
    cand["planted-bugs"]["metrics"]["detect/same_bar_execution"] = metric(100, 100, direction="higher", gate=True)
    return bench(cand=cand, base=suite_results(detect=60, fpr=1), **kw)


def test_opt_in_candidate_with_a_clean_win_is_eligible(sel):
    d = check_promotion(sel, pnl_bench(), *PNL, code_hash="c")
    assert d.ok, d.reasons
    assert d.better == ["planted-bugs: detect/same_bar_execution"] and not d.worse


def test_opt_in_baseline_must_not_already_include_it(sel):
    r = pnl_bench(base_methods={PNL[0]: "pnl_reconstruction@1", "s5.pbo": "cscv@1"})
    assert not check_promotion(sel, r, *PNL, code_hash="c").ok


def test_opt_in_promotion_waits_for_approval_then_becomes_the_default_with_nothing_deprecated(tmp_path):
    r = pnl_bench()
    folder = write_bench(r, tmp_path)
    kw = dict(reason="closes the same-bar and units gaps", bench=str(folder / "receipt.json"), now=NOW, code_hash="c")
    first = run_action(tmp_path, "promote", f"{PNL[0]}={PNL[1]}", **kw)
    assert first.status == "pending" and first.approval_id
    assert load_state(tmp_path)["statuses"] == {}
    human_approve_id(tmp_path, first.approval_id, "reviewed the bench report myself", now=NOW)
    assert run_action(tmp_path, "promote", f"{PNL[0]}={PNL[1]}", **kw).status == "applied"
    st = load_state(tmp_path)
    assert st["statuses"][PNL[0]] == {PNL[1]: "default"} and st["history"][-1]["previous_default"] is None
    assert MethodSelector(tmp_path).default(PNL[0]).version == PNL[1]
