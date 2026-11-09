"""Review round 2: per-parameter neighbour ratio, local plateau, step-distance neighbours, numeric strings."""

import pytest

from qrh.pipeline.stages_s3_s7 import grid_combos
from qrh.validation.param_stability import (
    StabilityBlocked, StabilityThresholds, generate_neighbours, numeric_dimensions,
)
from test_param_stability import FakeAdapter, TH, grid_matrix, plateau, single, spike, stability

WIDE = {"fast": [3, 10, 30], "slow": [60, 120, 240]}


def centre(grid, **want):
    return next(label for label, p in grid if all(float(p[k]) == float(v) for k, v in want.items()))


# --- 1: the neighbour ratio is decided per parameter -----------------------------------------------------------------


def test_an_inert_parameter_cannot_hide_a_spike_in_another():
    grid, matrix = single({"w": 20, "k": 1.0}, edge=spike({"w": 20}))  # k changes nothing: its neighbours look perfect
    res = stability(grid, matrix, [grid[0][0]], adapter=FakeAdapter(spike({"w": 20})))
    nb = res.neighbour
    by = {p["dimension"]: p for p in nb["per_parameter"]}
    assert by["w"]["ratio"] < 0.2 and by["k"]["ratio"] > 0.5
    assert nb["worst_parameter"] == "w" and nb["ratio"] == pytest.approx(by["w"]["ratio"])
    assert nb["pooled_mean_ratio"] > nb["ratio"]  # the pooled mean is information only
    assert nb["passed"] is False and res.status == "FAIL"


def test_the_decision_uses_the_worst_parameter_even_when_the_pooled_mean_would_pass():
    grid, matrix = single({"w": 20, "k": 1.0, "v": 3.0}, edge=spike({"w": 20}))
    res = stability(grid, matrix, [grid[0][0]], adapter=FakeAdapter(spike({"w": 20})))
    assert res.neighbour["pooled_mean_ratio"] >= 0.5 > res.neighbour["ratio"] and res.status == "FAIL"


# --- 2: the plateau is local ----------------------------------------------------------------------------------------


def test_a_wide_log_spaced_grid_of_a_robust_strategy_passes():
    grid, matrix = grid_matrix(WIDE, plateau)
    res = stability(grid, matrix, [centre(grid, fast=10, slow=120)] * 3, adapter=FakeAdapter(plateau))
    assert res.status == "PASS" and res.plateau["share"] == 1.0 and res.plateau["passed"] is True


def test_the_global_share_is_information_only_and_never_decides():
    def edge(p):
        return 0.002 if 8 <= p["fast"] <= 12 else 0.0  # good near fast=10 only: 3 of 9 grid points are good

    grid, matrix = grid_matrix(WIDE, edge)
    res = stability(grid, matrix, [centre(grid, fast=10, slow=120)] * 3, adapter=FakeAdapter(edge))
    g = res.plateau["global_info"]
    assert g["share"] < 0.5 and "information" in g["note"]
    assert res.status == "PASS" and res.plateau["share"] >= 0.5


def test_the_local_share_counts_nearby_settings_against_the_selected_sharpe_not_the_best():
    def edge(p):  # the selected setting is mediocre, a far grid point is great, its neighbours match it
        return 0.004 if p["fast"] == 30 else 0.001

    grid, matrix = grid_matrix(WIDE, edge)
    res = stability(grid, matrix, [centre(grid, fast=10, slow=120)] * 3, adapter=FakeAdapter(edge))
    pl = res.plateau
    assert pl["reference_sharpe"] == pytest.approx(res.neighbour["selected_sharpe"]) and pl["share"] == 1.0
    assert res.status == "PASS"


def test_a_local_spike_fails_the_local_plateau_too():
    grid, matrix = single({"w": 20}, edge=spike({"w": 20}))
    res = stability(grid, matrix, [grid[0][0]], adapter=FakeAdapter(spike({"w": 20})))
    assert res.plateau["applicable"] and res.plateau["share"] == 0.0 and res.plateau["passed"] is False
    assert res.plateau["n_neighbours"] == 4
    assert any("plateau" in r for r in res.reasons)


def test_grid_points_at_about_the_step_distance_are_used_without_running_the_adapter():
    space = {"fast": [9, 10, 11]}
    grid, matrix = grid_matrix(space, plateau)
    adapter = FakeAdapter(plateau)
    th = StabilityThresholds(steps=(0.10,))
    res = stability(grid, matrix, [centre(grid, fast=10)], adapter=adapter, thresholds=th)
    assert adapter.calls == [] and {n["source"] for n in res.neighbour["neighbours"]} == {"grid"}


def test_missing_step_points_are_generated_as_non_trial_runs_for_a_grid_too():
    grid, matrix = grid_matrix(WIDE, plateau)
    adapter = FakeAdapter(plateau)
    res = stability(grid, matrix, [centre(grid, fast=10, slow=120)], adapter=adapter)
    assert {n["source"] for n in res.neighbour["neighbours"]} == {"generated"}
    assert sorted({c["fast"] for c in adapter.calls}) == [8, 9, 10, 11, 12]  # slow nudges keep fast=10
    assert "not counted" in res.to_dict()["trial_count_note"]


def test_a_grid_needing_generated_neighbours_without_an_adapter_is_blocked():
    grid, matrix = grid_matrix(WIDE, plateau)
    with pytest.raises(StabilityBlocked, match="adapter"):
        stability(grid, matrix, [centre(grid, fast=10, slow=120)])


# --- 3: no hairline neighbours ---------------------------------------------------------------------------------------


def test_grid_points_closer_than_the_smallest_step_are_ignored_and_replaced_by_step_distance_ones():
    space = {"w": [9.9, 10.0, 10.1]}
    grid, matrix = grid_matrix(space, spike({"w": 10.0}))
    adapter = FakeAdapter(spike({"w": 10.0}))
    res = stability(grid, matrix, [centre(grid, w=10.0)], adapter=adapter)
    assert sorted(c["w"] for c in adapter.calls) == pytest.approx([8.0, 9.0, 11.0, 12.0])
    assert res.status == "FAIL" and all(n["source"] == "generated" for n in res.neighbour["neighbours"])


def test_declared_neighbours_closer_than_the_smallest_step_are_refused():
    grid, matrix = single({"w": 10.0}, edge=spike({"w": 10.0}))
    with pytest.raises(StabilityBlocked, match="too close"):
        stability(grid, matrix, [grid[0][0]], adapter=FakeAdapter(plateau),
                  declared={"neighbours": {"w": [9.999, 10.001]}})


def test_generated_integer_neighbours_are_never_closer_than_the_step():
    nb, _ = generate_neighbours({"w": 15}, {"w": [15]}, (0.10,))
    assert sorted(n.value for n in nb) == [13, 17]  # 13.5 and 16.5 round away from 15, never toward it


def test_bounds_that_squeeze_a_neighbour_below_the_step_leave_it_unchecked():
    nb, unchecked = generate_neighbours({"w": 20}, {"w": [20]}, (0.10, 0.20), bounds={"w": [19, 21]})
    assert nb == [] and unchecked and "closer than" in unchecked[0]["reason"]


# --- 4: numbers written as text, categorical parameters, nothing to vary -----------------------------------------------


def test_numeric_strings_are_numeric_dimensions():
    dims, skipped = numeric_dimensions([{"w": "10", "m": "a"}, {"w": "20.5", "m": "b"}])
    assert dims == {"w": ["10", "20.5"]} and skipped == ["m"]


def test_a_spike_hidden_behind_a_string_typed_parameter_still_fails():
    grid, matrix = single({"w": "20"}, edge=spike({"w": 20}))

    def edge(p):
        return 0.002 if float(p["w"]) == 20 else 0.0

    adapter = FakeAdapter(edge)
    res = stability(grid, matrix, [grid[0][0]], adapter=adapter)
    assert res.status == "FAIL" and all(isinstance(c["w"], str) for c in adapter.calls)
    assert sorted(float(c["w"]) for c in adapter.calls) == [16.0, 18.0, 22.0, 24.0]


def test_a_non_numeric_parameter_with_several_grid_values_is_varied_one_at_a_time():
    grid, matrix = grid_matrix({"mode": ["a", "b", "c"]}, lambda p: 0.002 if p["mode"] == "a" else 0.0)
    res = stability(grid, matrix, [centre_mode(grid, "a")], adapter=FakeAdapter(plateau))
    assert res.status == "FAIL" and res.neighbour["n_neighbours"] == 2
    assert res.neighbour["per_parameter"][0]["dimension"] == "mode"
    assert res.plateau["applicable"] is False


def centre_mode(grid, v):
    return next(label for label, p in grid if p["mode"] == v)


def test_parameters_with_nothing_numeric_and_nothing_variable_are_blocked_not_undecided():
    grid, matrix = single({"mode": "a"})
    with pytest.raises(StabilityBlocked, match="none is numeric"):
        stability(grid, matrix, [grid[0][0]], adapter=FakeAdapter(plateau))


def test_an_adapter_with_no_parameters_is_not_applicable():
    grid = grid_combos({})
    import pandas as pd
    from test_param_stability import INDEX, series

    matrix = pd.DataFrame({grid[0][0]: series({}, plateau).to_numpy()}, index=INDEX)
    res = stability(grid, matrix, [grid[0][0]], adapter=FakeAdapter(plateau))
    assert res.status == "NOT_APPLICABLE" and "no parameters" in res.reasons[0]


def test_a_non_positive_selected_sharpe_is_not_applicable_even_without_numeric_parameters():
    grid, matrix = single({"mode": "a"}, edge=lambda p: -0.002)
    res = stability(grid, matrix, [grid[0][0]], adapter=FakeAdapter(plateau))
    assert res.status == "NOT_APPLICABLE"
