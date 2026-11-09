import json

import pytest

from qrh.pipeline.receipt import Verdict
from qrh.reports import chartdata as cd

BASE_KEYS = {"schema_version", "key", "title", "stage", "trial_id", "receipt_hash", "chart_type", "disclaimer"}


def check(d, chart_type, data):
    assert BASE_KEYS <= set(d), BASE_KEYS - set(d)
    assert d["chart_type"] == chart_type and d["trial_id"] == data.trial_id and d["stage"] == data.stage.value
    assert d["receipt_hash"] == data.receipt_hash and len(d["receipt_hash"]) == 64
    json.dumps(d, allow_nan=False)  # JSON-serializable, no NaN/inf
    return d


def axis(d, name):
    assert {"label", "unit", "type"} <= set(d[name]), d[name]


def test_s0_card_lists_hypothesis_fields_and_every_threshold(make_data):
    data = make_data("S0")
    d = check(cd.s0_card(data), "text", data)
    fields = {f["label"]: f["value"] for f in d["fields"]}
    assert fields["Family"] == "trend|synthetic|1D" and fields["Asset class"] == "synthetic"
    assert fields["Frequency"] == "1D" and "synthetic rule" in fields["Hypothesis"]
    assert {t["name"] for t in d["thresholds"]} == set(data.artifacts["hypothesis"]["thresholds"])


def test_s1_coverage_has_a_row_per_dataset_with_gaps(make_data):
    data = make_data("S1")
    d = check(cd.s1_coverage(data), "timeline", data)
    axis(d, "x_axis")
    assert [r["label"] for r in d["rows"]] == ["prices", "macro"]
    assert d["rows"][0]["gaps"][0]["missing"] == 2 and d["rows"][0]["rows"] == 118


def test_s1_rules_table_has_text_status_and_offenders(make_data):
    data = make_data("S1", variant="fail")
    d = check(cd.s1_rules(data), "table", data)
    assert [c["key"] for c in d["columns"]][:4] == ["dataset", "rule", "type", "status"]
    statuses = {(r["dataset"], r["rule"]): r["status"] for r in d["rows"]}
    assert statuses[("prices", "gaps#1")] == "FAIL" and statuses[("prices", "range#2")] == "PASS"
    failing = next(r for r in d["rows"] if r["status"] == "FAIL")
    assert failing["offenders"] == 1 and "2021-02-03" in failing["first_offenders"]
    assert d["status_column"] == "status"


def test_s2_lint_table_and_empty_state(make_data):
    d = check(cd.s2_lint(make_data("S2")), "table", make_data("S2"))
    assert [r["status"] for r in d["rows"]] == ["WARN", "ALLOWED"]
    empty = cd.s2_lint(make_data("S2", variant="empty_lint"))
    assert empty["rows"] == [] and "no findings" in empty["empty_message"].lower()


def test_s2_truncation_line_chart_marks_first_divergence(make_data):
    data = make_data("S2", variant="diverge")
    assert cd.truncation_count(data) == 3
    d = check(cd.s2_truncation(data, 1), "line", data)
    axis(d, "x_axis"), axis(d, "y_axis")
    assert [s["name"] for s in d["series"]] == ["Full-data run", "Truncated-data run"]
    assert all(len(s["x"]) == len(s["y"]) == 30 for s in d["series"])
    (m,) = d["markers"]
    assert m["type"] == "vline" and m["x"].startswith("2021-01-21") and "first divergence" in m["label"].lower()
    clean = cd.s2_truncation(data, 0)
    assert clean["markers"] == [] and clean["cutoff_status"] == "PASS"


def test_s2_sanity_table(make_data):
    data = make_data("S2")
    d = check(cd.s2_sanity(data), "table", data)
    assert [r["status"] for r in d["rows"]] == ["PASS"] * 3 and d["rows"][0]["bound"] == 0.5


def test_s3_heatmap_for_two_parameters(make_data):
    data = make_data("S3")
    d = check(cd.s3_heatmap(data), "heatmap", data)
    assert d["x_axis"]["label"] == "slow" and d["y_axis"]["label"] == "fast"
    assert d["x_labels"] == ["40", "80", "160"] and d["y_labels"] == ["5", "10", "20"]
    assert len(d["z"]) == 3 and all(len(r) == 3 for r in d["z"])
    assert d["z_min"] == min(v for r in d["z"] for v in r) and d["colorscale"] == list(__import__("qrh.reports.theme", fromlist=["x"]).SEQUENTIAL)


def test_s3_chart_choice_follows_parameter_count(make_data):
    assert cd.s3_primary_key(make_data("S3")) == "s3_heatmap"
    assert cd.s3_primary_key(make_data("S3", variant="wide")) == "s3_ranked"


def test_s3_ranked_shows_top_30_and_counts_the_rest(make_data):
    data = make_data("S3", variant="wide")  # 80 combinations
    d = check(cd.s3_ranked(data), "dot", data)
    assert len(d["rows"]) == 30 and d["rest_count"] == 50
    sharpes = [r["value"] for r in d["rows"]]
    assert sharpes == sorted(sharpes, reverse=True)


def test_s3_histogram_counts_all_combinations(make_data):
    data = make_data("S3")
    d = check(cd.s3_hist(data), "histogram", data)
    assert sum(d["counts"]) == 9 and len(d["bins"]) == len(d["counts"]) + 1
    axis(d, "x_axis"), axis(d, "y_axis")


def test_s4_charts(make_data):
    data = make_data("S4")
    f = check(cd.s4_folds(data), "timeline", data)
    assert [r["label"] for r in f["rows"]] == ["Fold 0", "Fold 1"]
    assert {"train", "test"} <= set(f["rows"][0]) and f["series_names"] == ["Train", "Test"]
    e = check(cd.s4_equity(data), "line", data)
    assert len(e["series"]) == 1 and len(e["series"][0]["x"]) == 80 and e["y_axis"]["label"].startswith("Equity")
    t = check(cd.s4_params(data), "table", data)
    assert t["rows"][0]["params"] == "fast=5, slow=40" and "train_sharpe" in t["rows"][0]


def test_s5_lambda_histogram_labels_pbo_at_zero(make_data):
    data = make_data("S5")
    d = check(cd.s5_lambda(data), "histogram", data)
    zero = next(m for m in d["markers"] if m["x"] == 0)
    pbo = data.artifacts["dsr_inputs"]["pbo"]
    assert zero["type"] == "vline" and f"PBO = {pbo:.3f}" in zero["label"]
    assert sum(d["counts"]) == 500


def test_s5_dsr_bars_compare_observed_sr_with_sr0(make_data):
    data = make_data("S5")
    d = check(cd.s5_dsr(data), "bar", data)
    assert [b["name"] for b in d["bars"]] == ["Observed per-period SR", "SR0 (expected best of N null trials)"]
    assert d["bars"][0]["value"] == 0.08 and d["bars"][1]["value"] == 0.03
    assert d["annotations"]["dsr"] == 0.97 and "DSR = 0.970" in d["annotations"]["label"]
    assert [i["name"] for i in d["inputs"]] == ["Observed SR", "SR0", "N (trials)", "V (Sharpe variance)", "Skewness", "Kurtosis", "Observations"]


def test_s6_bootstrap_panels_have_ci_markers(make_data):
    data = make_data("S6")
    d = check(cd.s6_bootstrap(data), "histogram_panels", data)
    assert [p["metric"] for p in d["panels"]] == ["sharpe", "cagr", "max_drawdown"]
    for p in d["panels"]:
        assert sum(p["counts"]) == 300 and len(p["bins"]) == len(p["counts"]) + 1
        assert [m["label"].split()[0] for m in p["markers"] if m["role"] == "ci"] == ["CI", "CI"]
        axis(p, "x_axis"), axis(p, "y_axis")


def test_s6_histogram_ignores_undefined_samples(make_data):
    data = make_data("S6", variant="nan")
    d = cd.s6_bootstrap(data)
    assert sum(d["panels"][0]["counts"]) == 300 - 6 and d["panels"][0]["undefined_share"] == pytest.approx(6 / 300)


def test_s6_equity_two_series_on_one_axis(make_data):
    data = make_data("S6")
    d = check(cd.s6_equity(data), "line", data)
    assert [s["name"] for s in d["series"]] == ["Base OOS equity", "Cost-stressed (x2) OOS equity"]
    assert "y2_axis" not in d


def test_s7_summary(make_summary):
    data = make_summary()
    d = check(cd.s7_summary(data), "summary", data)
    assert d["final"] == {"verdict": "FAIL", "failing_stage": "S5"}
    rows = {r["stage"]: r for r in d["stages"]}
    assert rows["S3"]["verdict"] == "PASS" and rows["S5"]["verdict"] == "FAIL" and rows["S6"]["verdict"] == "NOT RUN"
    assert rows["S5"]["key_metrics"]["pbo"] > 0 and rows["S4"]["key_metrics"]["oos_sharpe"] == 1.2


def test_status_card_for_a_stage_without_artifacts(make_data):
    data = make_data("S3")
    data.artifacts = {}
    data.receipt = data.receipt.model_copy(update={"verdict": Verdict.BLOCKED, "reasons": ["adapter failed"]})
    d = check(cd.status_card(data), "text", data)
    assert d["verdict"] == "BLOCKED" and d["reasons"] == ["adapter failed"]


def test_all_numbers_are_finite_in_every_chartdata(make_data, make_summary):
    from qrh.reports.catalog import figure_specs

    for stage in ("S0", "S1", "S2", "S3", "S4", "S5", "S6"):
        for spec in figure_specs(make_data(stage)):
            json.dumps(spec.data(), allow_nan=False)
    for spec in figure_specs(make_summary()):
        json.dumps(spec.data(), allow_nan=False)


def test_s5_dsr_inputs_show_both_variances_and_the_one_used_when_the_receipt_has_them(make_data):
    data = make_data("S5", variant="stability")
    d = check(cd.s5_dsr(data), "bar", data)
    by = {i["name"]: i["value"] for i in d["inputs"]}
    assert by["V used (the larger: estimator)"] == 0.001 and by["V, grid (cross-variant)"] == 0.0004
    assert by["V, sampling estimate"] == 0.001


def test_s5_stability_table_lists_the_selection_the_neighbours_and_the_measures(make_data):
    data = make_data("S5", variant="stability")
    d = check(cd.s5_stability(data), "table", data)
    items = [r["item"] for r in d["rows"]]
    assert items[0] == "Selected combination" and items.count("Neighbour (generated)") == 2
    assert {"Neighbour ratio (worst parameter decides)", "Local plateau share (decides)",
            "Global plateau share (information only)", "Neighbour ratio, per parameter", "Most fragile parameter",
            "Unchecked", "Skipped (not numeric)"} <= set(items)
    assert d["title"].endswith("FAIL") and "not counted" in d["annotations"]["note"]


def test_a_receipt_without_the_stability_artifact_gets_no_stability_page(make_data):
    from qrh.reports.catalog import figure_specs

    assert [s.key for s in figure_specs(make_data("S5"))] == ["s5_lambda", "s5_dsr"]
    assert [s.key for s in figure_specs(make_data("S5", variant="stability"))] == ["s5_lambda", "s5_dsr", "s5_stability"]
