import pytest

from qrh.bench.corpus import BUG_CLASSES, corpus
from qrh.bench.profiles import PROFILES, resolve_profile
from qrh.bench.suites import run_known_edge_power, run_noise_fpr, run_planted_bugs
from qrh.methods.selection import MethodSelector

R = 1
TINY = {
    "planted-bugs": {"reps": R},
    "noise-fpr": {"grids": [12], "periods": [400], "reps": 12},
    "known-edge-power": {"grid": 12, "periods": 600, "strengths": [0.002, 0.008], "reps": 12},
}


@pytest.fixture
def arms(tmp_path):
    return {"candidate": MethodSelector(tmp_path)}


def test_profiles_define_every_suite_and_full_is_larger():
    for name in ("quick", "full"):
        p = PROFILES[name]
        assert set(p) >= {"planted-bugs", "noise-fpr", "known-edge-power"}
    q, f = PROFILES["quick"], PROFILES["full"]
    assert f["noise-fpr"]["reps"] > q["noise-fpr"]["reps"] and f["planted-bugs"]["reps"] > q["planted-bugs"]["reps"]
    assert set(q["noise-fpr"]["grids"]) <= set(f["noise-fpr"]["grids"])
    assert len(f["known-edge-power"]["strengths"]) >= 4


def test_resolve_profile_applies_reps_override_to_every_suite():
    cfg = resolve_profile("quick", reps=7)
    assert cfg["planted-bugs"]["reps"] == cfg["noise-fpr"]["reps"] == cfg["known-edge-power"]["reps"] == 7
    with pytest.raises(ValueError):
        resolve_profile("nope")
    with pytest.raises(ValueError):
        resolve_profile("quick", reps=0)


def test_planted_bugs_counts_are_consistent(arms):
    res = run_planted_bugs(arms, TINY["planted-bugs"], seed=1)["candidate"]
    metrics = res["metrics"]
    cases = corpus()
    for bug in BUG_CLASSES:
        variants = [c for c in cases if c.bug_class == bug and not c.expected_undetectable]
        m = metrics[f"detect/{bug}"]
        assert m["n"] == len(variants) * R and 0 <= m["k"] <= m["n"] and m["gate"] and m["direction"] == "higher"
        assert sum(metrics[f"detect/{bug}/{v.name}"]["k"] for v in variants) == m["k"]
    fa = metrics["false_alarm/controls"]
    assert fa["direction"] == "lower" and fa["gate"] and fa["n"] > 0
    assert res["detectors"]["lookahead_shift"]["truncation"] > 0


def test_planted_bugs_obvious_defects_are_found_and_controls_are_clean(arms):
    res = run_planted_bugs(arms, TINY["planted-bugs"], seed=2)["candidate"]
    m = res["metrics"]
    assert m["detect/lookahead_shift/shift_minus_1"]["rate"] == 1.0
    assert m["detect/centered_rolling/center_window_21"]["rate"] == 1.0
    assert m["detect/backward_fill/bfill"]["rate"] == 1.0
    assert m["detect/gaps_duplicates/gap_10"]["rate"] == 1.0
    assert m["detect/relationship_corruption/mag_0.1_len_5"]["rate"] == 1.0
    assert m["false_alarm/controls"]["k"] == 0, {k: v for k, v in m.items() if k.startswith("false_alarm") and v["k"]}


def test_planted_bugs_is_deterministic_and_seeded(arms, tmp_path):
    a = run_planted_bugs(arms, TINY["planted-bugs"], seed=3)
    b = run_planted_bugs({"candidate": MethodSelector(tmp_path)}, TINY["planted-bugs"], seed=3)
    assert a == b


def test_noise_fpr_reports_cells_nominal_and_gate_breakdown(arms):
    res = run_noise_fpr(arms, TINY["noise-fpr"], seed=4)["candidate"]
    assert res["nominal"] == pytest.approx(0.05)
    cell = res["metrics"]["fpr/g12_t400"]
    assert cell["n"] == 12 and cell["direction"] == "lower" and cell["gate"]
    pooled = res["metrics"]["fpr/pooled"]
    assert pooled["k"] == cell["k"] and pooled["gate"] is False
    s4, pbo, dsr, s5 = (res["metrics"][f"{k}/pooled"] for k in ("pass_s4", "pass_pbo", "pass_dsr", "pass_s5"))
    assert pooled["k"] <= min(s4["k"], s5["k"]) and s5["k"] <= min(pbo["k"], dsr["k"])
    assert set(res["cells"]["g12_t400"]) >= {"grid", "periods", "reps", "errors"}
    assert res["cells"]["g12_t400"]["exceeds_nominal"] in (True, False)


def test_noise_fpr_shares_data_between_arms_and_methods_differ_only_where_selected(tmp_path):
    arms = {
        "candidate": MethodSelector(tmp_path, overrides={"s4.walkforward": "purged_embargo@1"}),
        "baseline": MethodSelector(tmp_path),
    }
    out = run_noise_fpr(arms, TINY["noise-fpr"], seed=5)
    assert out["candidate"]["methods"]["s4.walkforward"] == "purged_embargo@1"
    assert out["baseline"]["methods"]["s4.walkforward"] == "rolling_anchored@1"
    same = run_noise_fpr({"candidate": MethodSelector(tmp_path), "baseline": MethodSelector(tmp_path)}, TINY["noise-fpr"], seed=5)
    assert same["candidate"]["metrics"] == same["baseline"]["metrics"]


def test_known_edge_power_rises_with_strength(arms):
    res = run_known_edge_power(arms, TINY["known-edge-power"], seed=6)["candidate"]
    weak, strong = res["metrics"]["power/s0.002"], res["metrics"]["power/s0.008"]
    assert weak["direction"] == "higher" and weak["gate"] and weak["n"] == 12
    assert strong["rate"] > weak["rate"]
    assert res["strengths"] == [0.002, 0.008]


def test_expected_undetectable_variants_are_reported_separately_and_never_as_misses(arms):
    res = run_planted_bugs(arms, TINY["planted-bugs"], seed=7)["candidate"]
    m = res["metrics"]
    hidden = [c for c in corpus() if c.expected_undetectable]
    assert hidden and not any(f"detect/relationship_corruption/{c.name}" in m for c in hidden)
    agg = m["expected_undetectable/relationship_corruption"]
    assert agg["n"] == len(hidden) * R and agg["gate"] is False and agg["direction"] == "higher"
    for c in hidden:
        assert m[f"expected_undetectable/relationship_corruption/{c.name}"]["n"] == R
    detectable = [c for c in corpus() if c.bug_class == "relationship_corruption" and not c.expected_undetectable]
    assert m["detect/relationship_corruption"]["n"] == len(detectable) * R
    assert m["detect/relationship_corruption"]["rate"] == 1.0  # every variant above the tolerance is found
    assert res["expected_undetectable"] == {"relationship_corruption": [c.name for c in hidden]}


def test_pnl_reconstruction_closes_the_same_bar_and_units_gaps_without_false_alarms(tmp_path):
    arms = {
        "candidate": MethodSelector(tmp_path, overrides={"s2.pnl_reconstruction": "pnl_reconstruction@1"}),
        "baseline": MethodSelector(tmp_path),
    }
    out = run_planted_bugs(arms, TINY["planted-bugs"], seed=8)
    base, cand = out["baseline"]["metrics"], out["candidate"]["metrics"]
    # the default arm used to detect no same-bar execution at all (0/27); the default same-bar check now catches it
    assert base["detect/same_bar_execution"]["rate"] == 1.0 and cand["detect/same_bar_execution"]["rate"] == 1.0
    assert cand["detect/percent_units"]["rate"] == 1.0 and base["detect/percent_units"]["rate"] >= 0.6
    assert cand["false_alarm/controls"]["k"] == 0, {k: v for k, v in cand.items() if k.startswith("false_alarm") and v["k"]}
    for cls in BUG_CLASSES:  # never a regression elsewhere
        assert cand[f"detect/{cls}"]["k"] >= base[f"detect/{cls}"]["k"]
    assert out["candidate"]["detectors"]["same_bar_execution"]["pnl"] > 0
    assert out["baseline"]["detectors"]["same_bar_execution"]["pnl"] == 0  # the opt-in method is not in the default arm
    assert out["baseline"]["detectors"]["same_bar_execution"]["same_bar"] > 0  # the default check is
    assert out["candidate"]["methods"]["s2.pnl_reconstruction"] == "pnl_reconstruction@1"
    assert "s2.pnl_reconstruction" not in out["baseline"]["methods"]


def test_the_default_arm_detects_same_bar_execution_and_sparse_lookahead_with_no_false_alarm(arms):
    res = run_planted_bugs(arms, TINY["planted-bugs"], seed=3)["candidate"]
    m, det = res["metrics"], res["detectors"]
    assert m["detect/same_bar_execution"]["k"] == m["detect/same_bar_execution"]["n"] > 0
    assert det["same_bar_execution"]["same_bar"] > 0 and det["same_bar_execution"]["pnl"] == 0
    for name in ("np_roll_sparse", "named_negative_shift_sparse"):
        assert m[f"detect/lookahead_shift/{name}"]["rate"] == 1.0
    assert det["lookahead_shift"]["perturbation"] >= det["lookahead_shift"]["truncation"]
    assert m["false_alarm/controls"]["k"] == 0, {k: v for k, v in m.items() if k.startswith("false_alarm") and v["k"]}
