import io
import warnings

import pytest
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

from qrh.pipeline.receipt import Verdict
from qrh.reports import chartdata, charts, theme
from qrh.reports.catalog import figure_specs

STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")


def texts(fig):
    return [t for t in fig.findobj(Text) if t.get_text()]


def strings(fig):
    return [t.get_text() for t in texts(fig)]


def all_specs(make_data, make_summary):
    out = []
    for s in STAGES:
        out += [(f"{s}:{spec.key}", spec) for spec in figure_specs(make_data(s))]
    out += [(f"S7:{spec.key}", spec) for spec in figure_specs(make_summary())]
    return out


def test_every_figure_builds_and_renders_to_pdf_without_warnings(make_data, make_summary):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # missing glyphs, layout and deprecation problems become failures
        for name, spec in all_specs(make_data, make_summary):
            fig = spec.chart()
            assert isinstance(fig, Figure), name
            fig.savefig(io.BytesIO(), format="pdf")


def test_every_page_has_title_and_the_standard_footer(make_data, make_summary):
    for name, spec in all_specs(make_data, make_summary):
        fig, d = spec.chart(), spec.data()
        s = strings(fig)
        assert d["title"] in s, name
        footer = f"Trial {d['trial_id']}  |  Stage {d['stage']}  |  Receipt {d['receipt_hash'][:12]}"
        assert footer in s and theme.DISCLAIMER in s, name


def test_pages_use_the_surface_color(make_data):
    fig = figure_specs(make_data("S4"))[1].chart()
    assert fig.get_facecolor()[:3] == pytest.approx(tuple(int(theme.SURFACE[i : i + 2], 16) / 255 for i in (1, 3, 5)))
    assert all(ax.get_facecolor()[:3] == fig.get_facecolor()[:3] for ax in fig.axes)


def test_truncation_chart_uses_slots_one_and_two_a_legend_and_marks_divergence(make_data):
    data = make_data("S2", variant="diverge")
    fig = charts.s2_truncation(data, 1)
    (ax,) = fig.axes
    colors = {l.get_color() for l in ax.get_lines() if len(l.get_xdata()) > 2}
    assert colors == {theme.CATEGORICAL[0], theme.CATEGORICAL[1]}
    assert ax.get_legend() is not None
    assert any("First divergence" in s for s in strings(fig))
    assert any(len(l.get_xdata()) == 2 and l.get_xdata()[0] == l.get_xdata()[1] for l in ax.get_lines())  # vertical rule
    assert all(l.get_linewidth() == pytest.approx(1.5) for l in ax.get_lines() if len(l.get_xdata()) > 2)


def test_clean_truncation_chart_has_no_divergence_rule(make_data):
    fig = charts.s2_truncation(make_data("S2", variant="diverge"), 0)
    assert not any("First divergence" in s for s in strings(fig))


def test_no_chart_uses_a_second_y_axis(make_data, make_summary):
    for name, spec in all_specs(make_data, make_summary):
        fig = spec.chart()
        for ax in fig.axes:
            siblings = [a for a in fig.axes if a is not ax and a.bbox.bounds == ax.bbox.bounds and a.get_visible()]
            assert not [a for a in siblings if a.yaxis.get_ticks_position() == "right"], name


def test_heatmap_shows_a_value_label_in_every_cell_with_the_blue_ramp(make_data):
    data = make_data("S3")
    fig = charts.s3_heatmap(data)
    d = chartdata.s3_heatmap(data)
    labels = {f"{v:.2f}" for row in d["z"] for v in row if v is not None}
    assert labels <= set(strings(fig))
    ax = fig.axes[0]
    (image,) = ax.get_images()
    assert image.get_cmap().name == "qrh_blue"


def test_ranked_chart_shows_thirty_rows_and_the_count_of_the_rest(make_data):
    fig = charts.s3_ranked(make_data("S3", variant="wide"))
    assert any("50 more combinations" in s for s in strings(fig))


def test_histogram_with_one_series_has_no_legend_but_rules_are_labeled(make_data):
    fig = charts.s5_lambda(make_data("S5"))
    (ax,) = fig.axes
    assert any(len(l.get_xdata()) == 2 and l.get_xdata()[0] == 0 == l.get_xdata()[1] for l in ax.get_lines())
    assert any(s.startswith("PBO = ") for s in strings(fig))


def test_dsr_bar_chart_labels_dsr_and_compares_two_bars(make_data):
    fig = charts.s5_dsr(make_data("S5"))
    s = strings(fig)
    assert any("DSR = 0.970" in x for x in s)
    (ax,) = fig.axes
    assert len(ax.patches) == 2 and ax.get_legend() is not None


def test_bootstrap_has_three_panels_with_ci_labels(make_data):
    fig = charts.s6_bootstrap(make_data("S6"))
    assert len(fig.axes) == 3
    assert sum("CI low" in s for s in strings(fig)) == 3 and sum("CI high" in s for s in strings(fig)) == 3


def test_stressed_equity_has_two_series_one_axis_and_end_labels(make_data):
    fig = charts.s6_equity(make_data("S6"))
    (ax,) = fig.axes
    assert ax.get_legend() is not None
    assert {l.get_color() for l in ax.get_lines()} == {theme.CATEGORICAL[0], theme.CATEGORICAL[1]}
    assert any("Cost-stressed (x2)" in s for s in strings(fig))


def test_statuses_carry_icon_text_and_status_color(make_summary):
    fig = charts.s7_summary(make_summary())
    by_text = {t.get_text(): t for t in texts(fig)}
    for verdict in ("PASS", "FAIL", "NOT RUN"):
        label = theme.verdict_label(verdict)
        assert label in by_text, label
    assert by_text[theme.verdict_label("PASS")].get_color() == theme.STATUS["good"]
    assert by_text[theme.verdict_label("FAIL")].get_color() == theme.STATUS["critical"]
    assert any(s.startswith("Final verdict") and "FAIL" in s for s in strings(fig))
    assert any("S5" in s and "first non-PASS" in s for s in strings(fig))


def test_table_status_cells_use_icon_text_and_color(make_data):
    fig = charts.s1_rules(make_data("S1", variant="fail"))
    by_text = {t.get_text(): t for t in texts(fig)}
    assert by_text[theme.verdict_label("FAIL")].get_color() == theme.STATUS["critical"]
    assert by_text[theme.verdict_label("PASS")].get_color() == theme.STATUS["good"]


def test_lint_table_maps_block_and_warn_to_status_words(make_data):
    fig = charts.s2_lint(make_data("S2"))
    s = strings(fig)
    assert theme.verdict_label("WARN") in s and "• ALLOWED" in s


def test_empty_lint_table_says_so(make_data):
    fig = charts.s2_lint(make_data("S2", variant="empty_lint"))
    assert any("No findings" in s for s in strings(fig))


def test_text_card_lists_hypothesis_and_every_threshold(make_data):
    data = make_data("S0")
    s = " ".join(strings(charts.s0_card(data)))
    assert "synthetic rule carries information" in s and "synthetic" in s and "1D" in s
    for name in data.artifacts["hypothesis"]["thresholds"]:
        assert name in s


def test_status_card_shows_verdict_and_reasons(make_data):
    data = make_data("S3")
    data.artifacts = {}
    data.receipt = data.receipt.model_copy(update={"verdict": Verdict.BLOCKED, "reasons": ["adapter failed to run"]})
    s = " ".join(strings(charts.status_card(data)))
    assert theme.verdict_label("BLOCKED") in s and "adapter failed to run" in s


def test_folds_timeline_has_train_and_test_in_the_legend(make_data):
    fig = charts.s4_folds(make_data("S4"))
    (ax,) = fig.axes
    legend = {t.get_text() for t in ax.get_legend().get_texts()}
    assert legend == {"Train", "Test"}


def test_line_end_label_for_the_single_equity_series(make_data):
    fig = charts.s4_equity(make_data("S4"))
    assert any(s.startswith("Out-of-sample equity") for s in strings(fig))
    (ax,) = fig.axes
    assert ax.get_legend() is None


def test_stability_page_draws_a_table(make_data):
    fig = charts.s5_stability(make_data("S5", variant="stability"))
    s = strings(fig)
    assert any("Neighbour ratio" in x for x in s) and any("Parameter stability" in x for x in s)
