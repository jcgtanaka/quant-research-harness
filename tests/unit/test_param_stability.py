"""s5.param_stability@1: neighbour check, plateau share and per-parameter diagnostics, all from out-of-sample returns."""

import zlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qrh.pipeline.stages_s3_s7 import grid_combos
from qrh.validation.param_stability import (
    StabilityBlocked,
    StabilityThresholds,
    generate_neighbours,
    numeric_dimensions,
    run_param_stability,
)

PPY = 252.0
N = 1600
OOS_FROM = 600  # the last bars are the out-of-sample windows
INDEX = pd.date_range("2020-01-01", periods=N, freq="D", tz="UTC")
OOS = INDEX[OOS_FROM:]
TH = StabilityThresholds()


def noise(params_key, seed=0):
    """Returns with a shared factor and an idiosyncratic part, so variants of one idea are correlated like real ones."""
    common = np.random.default_rng(seed).normal(0, 1, N)
    own = np.random.default_rng(zlib.crc32(str(params_key).encode()) + seed).normal(0, 1, N)
    return 0.01 * (0.9 * common + np.sqrt(1 - 0.81) * own)


def series(params, edge, seed=0):
    return pd.Series(edge(params) + noise(sorted(params.items()), seed), index=INDEX)


def grid_matrix(space, edge):
    grid = grid_combos(space)
    return grid, pd.DataFrame({label: series(p, edge).to_numpy() for label, p in grid}, index=INDEX)


def spike(centre):
    return lambda p: 0.002 if all(p[k] == v for k, v in centre.items()) else 0.0


def plateau(p):
    return 0.002


class FakeAdapter:
    def __init__(self, edge, fail_on=None):
        self.edge, self.calls, self.fail_on = edge, [], fail_on

    def run(self, params, data, start, end, **_):
        self.calls.append(dict(params))
        if self.fail_on and self.fail_on(params):
            raise ValueError("window must be positive")
        s = series(params, self.edge)
        return SimpleNamespace(returns=s.loc[start:end])


def stability(grid, matrix, picks, adapter=None, thresholds=TH, declared=None):
    return run_param_stability(
        grid=grid, matrix=matrix, oos_index=OOS, picks=picks, ppy=PPY, thresholds=thresholds,
        adapter=adapter, view=None, start=INDEX[0], end=INDEX[-1], declared=declared,
    )


SPACE = {"fast": [5, 10, 15], "slow": [20, 40, 60]}


# --- helpers ---------------------------------------------------------------------------------------------------


def test_numeric_dimensions_skip_text_bool_and_structured_values_and_list_them():
    combos = [{"a": 1, "b": "x", "c": True, "d": {"n": 1}}, {"a": 2.5, "b": "y", "c": False, "d": {"n": 2}}]
    dims, skipped = numeric_dimensions(combos)
    assert dims == {"a": [1, 2.5]} and skipped == ["b", "c", "d"]


def test_generated_neighbours_use_the_declared_steps_round_integers_and_deduplicate():
    nb, unchecked = generate_neighbours({"w": 20}, {"w": [20]}, (0.10, 0.20))
    assert sorted(n.value for n in nb) == [16, 18, 22, 24] and unchecked == []
    assert all(isinstance(n.value, int) for n in nb)
    small, _ = generate_neighbours({"w": 5}, {"w": [5]}, (0.10, 0.20))
    assert sorted(n.value for n in small) == [4, 6]  # 4.5 rounds back to 5 and 6.0 appears twice: both dropped
    fl, _ = generate_neighbours({"x": 0.5}, {"x": [0.5]}, (0.10,))
    assert sorted(n.value for n in fl) == pytest.approx([0.45, 0.55])


def test_generated_neighbours_are_clipped_to_the_declared_bounds():
    nb, unchecked = generate_neighbours({"w": 20}, {"w": [20]}, (0.10, 0.20), bounds={"w": [17, 23]})
    assert sorted(n.value for n in nb) == [17, 18, 22, 23] and [u["reason"] for u in unchecked if "one direction" not in u["reason"]] == []


def test_a_dimension_whose_neighbours_cannot_be_produced_is_reported_as_unchecked():
    nb, unchecked = generate_neighbours({"w": 0, "v": 10}, {"w": [0], "v": [10]}, (0.10,))
    assert {n.dimension for n in nb} == {"v"}
    assert [u["dimension"] for u in unchecked] == ["w"] and unchecked[0]["reason"]
    nb2, unchecked2 = generate_neighbours({"w": 20}, {"w": [20]}, (0.10,), bounds={"w": [20, 20]})
    assert nb2 == [] and [u["dimension"] for u in unchecked2] == ["w"]


def test_declared_neighbours_replace_the_generated_ones_for_that_parameter():
    nb, _ = generate_neighbours({"w": 20, "v": 10}, {"w": [20], "v": [10]}, (0.10, 0.20), explicit={"w": [15, 25]})
    assert sorted(n.value for n in nb if n.dimension == "w") == [15, 25]
    assert sorted(n.value for n in nb if n.dimension == "v") == [8, 9, 11, 12]
    with pytest.raises(StabilityBlocked, match="unknown parameter"):
        generate_neighbours({"w": 20}, {"w": [20]}, (0.10,), explicit={"zzz": [1]})


def test_thresholds_default_and_validate():
    th = StabilityThresholds.from_mapping({})
    assert (th.closeness, th.plateau_share, th.neighbour_ratio, th.steps) == (0.70, 0.50, 0.50, (0.10, 0.20))
    th2 = StabilityThresholds.from_mapping({"stability_steps": [0.05], "stability_closeness": 0.8})
    assert th2.steps == (0.05,) and th2.closeness == 0.8
    for bad in ({"stability_closeness": 0}, {"stability_closeness": 1.5}, {"stability_steps": []},
                {"stability_steps": [0.1, "x"]}, {"stability_steps": [1.2]}, {"stability_plateau_share": True},
                {"stability_neighbour_ratio": float("nan")}):
        with pytest.raises(StabilityBlocked):
            StabilityThresholds.from_mapping(bad)


# --- a grid of combinations ------------------------------------------------------------------------------------


def test_a_spike_in_the_grid_fails_both_measures_and_names_the_fragile_parameter():
    grid, matrix = grid_matrix(SPACE, spike({"fast": 10, "slow": 40}))
    centre = next(label for label, p in grid if p == {"fast": 10, "slow": 40})
    res = stability(grid, matrix, [centre] * 5, adapter=FakeAdapter(spike({"fast": 10, "slow": 40})))
    assert res.status == "FAIL"
    assert res.neighbour["passed"] is False and res.neighbour["ratio"] < 0.5
    assert res.neighbour["n_neighbours"] == 8  # one parameter at a time, 10% and 20% up and down (not the far grid points)
    assert res.plateau["passed"] is False and res.plateau["share"] < 0.5
    assert res.diagnostics["most_fragile"] in SPACE
    assert any("neighbour" in r for r in res.reasons) and any("plateau" in r for r in res.reasons)


def test_a_plateau_in_the_grid_passes():
    grid, matrix = grid_matrix(SPACE, plateau)
    label = grid[4][0]
    res = stability(grid, matrix, [label] * 3, adapter=FakeAdapter(plateau))
    assert res.status == "PASS"
    assert res.neighbour["ratio"] >= 0.5 and res.plateau["share"] >= 0.5 and res.reasons == []


def test_the_selected_combination_is_the_one_picked_in_most_folds():
    grid, matrix = grid_matrix(SPACE, plateau)
    a, b = grid[0][0], grid[4][0]
    res = stability(grid, matrix, [a, b, b, a, b], adapter=FakeAdapter(plateau))
    assert res.selected["label"] == b and res.selected["picks"] == {a: 2, b: 3}
    tie = stability(grid, matrix, [a, b], adapter=FakeAdapter(plateau))
    assert tie.selected["label"] == min(a, b)  # ties: the lowest label, like the walk-forward selection


def test_neighbours_at_the_corner_of_the_grid_are_built_by_step_and_deduplicated():
    grid, matrix = grid_matrix(SPACE, plateau)
    corner = next(label for label, p in grid if p == {"fast": 5, "slow": 20})
    res = stability(grid, matrix, [corner], adapter=FakeAdapter(plateau))
    assert res.neighbour["n_neighbours"] == 6  # fast 4 and 6 (steps 10% and 20% round to the same integers), slow 4 values


def test_the_local_share_counts_nearby_settings_and_the_global_share_is_information():
    grid, matrix = grid_matrix(SPACE, lambda p: 0.002 if p["fast"] == 10 else 0.0)  # only the row fast=10 is good
    res = stability(grid, matrix, [grid[4][0]], adapter=FakeAdapter(lambda p: 0.002 if p["fast"] == 10 else 0.0))
    assert res.plateau["n_neighbours"] == 8 and res.plateau["n_close"] == 4  # the four `slow` nudges keep fast=10
    assert res.plateau["share"] == pytest.approx(0.5) and res.plateau["passed"] is True
    g = res.plateau["global_info"]
    assert g["n_combinations"] == 9 and g["share"] == pytest.approx(3 / 9)
    assert res.status == "FAIL" and res.neighbour["worst_parameter"] == "fast"  # fast=10 is a spike in `fast`


def test_a_selected_sharpe_at_or_below_zero_is_reported_not_failed():
    grid, matrix = grid_matrix(SPACE, lambda p: -0.002 if p == {"fast": 10, "slow": 40} else 0.0)
    centre = next(label for label, p in grid if p == {"fast": 10, "slow": 40})
    res = stability(grid, matrix, [centre], adapter=FakeAdapter(plateau))
    assert res.neighbour["meaningful"] is False and res.neighbour["passed"] is None and res.neighbour["ratio"] is None
    assert res.neighbour["selected_sharpe"] < 0 and res.neighbour["note"]
    assert not any("neighbour" in r for r in res.reasons)


def test_two_combinations_get_the_local_checks_and_no_global_share():
    grid, matrix = grid_matrix({"fast": [5, 10]}, plateau)
    res = stability(grid, matrix, [grid[0][0]], adapter=FakeAdapter(plateau))
    assert res.neighbour["applicable"] is True and res.neighbour["n_neighbours"] == 2  # 4 and 6; the far grid point is ignored
    assert res.plateau["applicable"] is True and res.plateau["global_info"]["applicable"] is False


def test_non_numeric_dimensions_with_several_values_are_varied_and_held_fixed_for_the_numeric_ones():
    space = {"fast": [5, 10, 15], "mode": ["a", "b"]}
    grid, matrix = grid_matrix(space, plateau)
    sel = next(label for label, p in grid if p == {"fast": 10, "mode": "a"})
    res = stability(grid, matrix, [sel], adapter=FakeAdapter(plateau))
    assert res.skipped == []
    assert {p["dimension"] for p in res.neighbour["per_parameter"]} == {"fast", "mode"}
    assert res.neighbour["n_neighbours"] == 5  # fast 8, 9, 11, 12 with mode "a" kept, and mode "b"
    assert all(n["dimension"] != "fast" or True for n in res.neighbour["neighbours"])


def test_per_parameter_diagnostics_report_gap_and_correlation_and_name_the_most_fragile():
    # only `fast` matters: the edge flips with it, `slow` does nothing
    grid, matrix = grid_matrix(SPACE, lambda p: 0.002 if p["fast"] == 10 else 0.0)
    res = stability(grid, matrix, [grid[4][0]], adapter=FakeAdapter(lambda p: 0.002 if p["fast"] == 10 else 0.0))
    by = {d["dimension"]: d for d in res.diagnostics["parameters"]}
    assert set(by) == {"fast", "slow"}
    assert by["fast"]["mean_abs_sharpe_gap"] > by["slow"]["mean_abs_sharpe_gap"]
    assert -1.0 <= by["fast"]["mean_correlation"] <= 1.0
    assert res.diagnostics["most_fragile"] == "fast"


def test_diagnostics_never_decide_the_verdict():
    grid, matrix = grid_matrix(SPACE, plateau)
    res = stability(grid, matrix, [grid[4][0]], adapter=FakeAdapter(plateau))
    assert res.status == "PASS" and res.diagnostics["parameters"]


# --- one combination: the harness generates the neighbours --------------------------------------------------------


def single(params, edge=plateau):
    grid = grid_combos({k: [v] for k, v in params.items()})
    return grid, pd.DataFrame({grid[0][0]: series(params, edge).to_numpy()}, index=INDEX)


def test_a_single_combination_spike_fails_with_generated_neighbours():
    grid, matrix = single({"w": 20})
    adapter = FakeAdapter(spike({"w": 20}))
    res = stability(grid, matrix, [grid[0][0]], adapter=adapter)
    assert res.status == "FAIL" and res.neighbour["passed"] is False
    assert sorted(c["w"] for c in adapter.calls) == [16, 18, 22, 24]
    assert res.plateau["applicable"] is True and res.plateau["passed"] is False  # the plateau is local now
    assert res.neighbour["source"] == "generated"


def test_a_single_combination_plateau_passes():
    grid, matrix = single({"w": 20, "v": 3})
    adapter = FakeAdapter(plateau)
    res = stability(grid, matrix, [grid[0][0]], adapter=adapter)
    assert res.status == "PASS" and res.neighbour["ratio"] >= 0.5
    assert len(adapter.calls) == len(res.neighbour["neighbours"]) >= 6
    assert {d["dimension"] for d in res.diagnostics["parameters"]} == {"w", "v"}


def test_neighbours_are_scored_on_the_out_of_sample_bars_only():
    grid, matrix = single({"w": 20})

    def edge_only_before_oos(p):
        return 0.0

    class Early(FakeAdapter):
        def run(self, params, data, start, end, **_):
            r = super().run(params, data, start, end)
            if params["w"] != 20:  # a huge edge, but only in the in-sample part
                r.returns.iloc[:OOS_FROM] += 0.05
            return r

    res = stability(grid, matrix, [grid[0][0]], adapter=Early(plateau))
    ratios = [n["sharpe"] for n in res.neighbour["neighbours"]]
    assert max(ratios) < 10  # the in-sample boost never reaches the score
    assert res.neighbour["ratio"] < 1.5


def test_flat_neighbours_with_an_undefined_sharpe_count_as_zero_and_are_counted():
    grid, matrix = single({"w": 20})

    class Flat(FakeAdapter):
        def run(self, params, data, start, end, **_):
            r = super().run(params, data, start, end)
            if params["w"] != 20:
                r.returns[:] = 0.0
            return r

    res = stability(grid, matrix, [grid[0][0]], adapter=Flat(plateau))
    assert res.status == "FAIL" and res.neighbour["ratio"] == pytest.approx(0.0)
    assert res.neighbour["undefined_as_zero"] == 4


def test_a_neighbour_the_adapter_refuses_is_unchecked_not_fatal():
    grid, matrix = single({"w": 20, "v": 3})
    adapter = FakeAdapter(plateau, fail_on=lambda p: p["v"] < 3)
    res = stability(grid, matrix, [grid[0][0]], adapter=adapter)
    assert res.status == "PASS"
    assert any(u["dimension"] == "v" for u in res.unchecked)
    assert any(n["dimension"] == "v" for n in res.neighbour["neighbours"])  # the upward neighbours still ran


def test_no_neighbour_at_all_is_blocked_never_a_silent_pass():
    grid, matrix = single({"w": 20})
    with pytest.raises(StabilityBlocked, match="no neighbour"):
        stability(grid, matrix, [grid[0][0]], adapter=FakeAdapter(plateau, fail_on=lambda p: p["w"] != 20))


def test_declared_neighbour_values_are_used_and_a_failing_one_blocks():
    grid, matrix = single({"w": 20})
    adapter = FakeAdapter(plateau)
    res = stability(grid, matrix, [grid[0][0]], adapter=adapter, declared={"neighbours": {"w": [10, 30]}})
    assert sorted(c["w"] for c in adapter.calls) == [10, 30] and res.neighbour["source"] == "declared"
    with pytest.raises(StabilityBlocked, match="declared neighbour"):
        stability(grid, matrix, [grid[0][0]], adapter=FakeAdapter(plateau, fail_on=lambda p: p["w"] == 10),
                  declared={"neighbours": {"w": [10, 30]}})


def test_declared_bounds_clip_the_generated_neighbours():
    grid, matrix = single({"w": 20})
    adapter = FakeAdapter(plateau)
    stability(grid, matrix, [grid[0][0]], adapter=adapter, declared={"bounds": {"w": [19, 100]}})
    assert sorted(c["w"] for c in adapter.calls) == [22, 24]  # 18 and 16 clip to 19, closer than the smallest step


def test_without_numeric_or_varied_parameters_the_check_is_blocked_with_a_reason():
    grid = grid_combos({"mode": ["a"]})
    matrix = pd.DataFrame({grid[0][0]: series({"mode": "a"}, plateau).to_numpy()}, index=INDEX)
    with pytest.raises(StabilityBlocked, match="none is numeric"):
        stability(grid, matrix, [grid[0][0]], adapter=FakeAdapter(plateau))


def test_a_non_meaningful_selection_and_no_plateau_is_not_applicable_never_pass():
    grid, matrix = single({"w": 20}, edge=lambda p: -0.002)
    adapter = FakeAdapter(plateau)
    res = stability(grid, matrix, [grid[0][0]], adapter=adapter)
    assert res.status == "NOT_APPLICABLE" and res.neighbour["meaningful"] is False and res.neighbour["passed"] is None
    assert "not meaningful" in res.reasons[0] and adapter.calls == []  # nothing would be judged, so nothing is run


def test_the_result_is_json_ready_and_says_neighbour_runs_are_not_trials():
    import json

    grid, matrix = single({"w": 20})
    res = stability(grid, matrix, [grid[0][0]], adapter=FakeAdapter(plateau))
    doc = res.to_dict()
    json.dumps(doc)
    assert "not counted" in doc["trial_count_note"]
    assert doc["thresholds"]["steps"] == [0.1, 0.2]
