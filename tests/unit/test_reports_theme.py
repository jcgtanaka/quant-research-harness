import pytest

from qrh.reports import theme


def test_palettes_are_exactly_the_specified_ramps():
    assert theme.SURFACE == "#fcfcfb"
    assert theme.CATEGORICAL == ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
    assert theme.SEQUENTIAL == (
        "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
    )
    assert theme.STATUS == {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


def test_series_slots_are_fixed_and_never_cycle():
    assert [theme.series_color(i) for i in range(3)] == list(theme.CATEGORICAL[:3])
    with pytest.raises(ValueError):
        theme.series_color(len(theme.CATEGORICAL))
    with pytest.raises(ValueError):
        theme.series_color(-1)


def test_at_most_three_series_on_one_chart():
    theme.check_series_count(3)
    with pytest.raises(ValueError, match="at most 3"):
        theme.check_series_count(4)


def test_verdicts_map_to_status_colors_and_always_carry_text():
    assert theme.verdict_color("PASS") == theme.STATUS["good"]
    assert theme.verdict_color("INCONCLUSIVE") == theme.STATUS["warning"]
    assert theme.verdict_color("FAIL") == theme.STATUS["critical"]
    assert theme.verdict_color("BLOCKED") == theme.STATUS["serious"]
    labels = {v: theme.verdict_label(v) for v in ("PASS", "INCONCLUSIVE", "FAIL", "BLOCKED")}
    assert all(v in lab for v, lab in labels.items())
    assert len({lab.split()[0] for lab in labels.values()}) == 4  # four distinct icon glyphs
    assert theme.verdict_label("NOT RUN").endswith("NOT RUN")


def test_status_colors_are_never_series_colors():
    assert not set(theme.STATUS.values()) & set(theme.CATEGORICAL)


def test_rc_pins_surface_font_and_deterministic_output():
    rc = theme.RC
    assert rc["figure.facecolor"] == theme.SURFACE and rc["axes.facecolor"] == theme.SURFACE
    assert rc["font.family"] == ["DejaVu Sans"] and rc["svg.hashsalt"]
    assert rc["lines.linewidth"] == 1.5 and rc["lines.markersize"] >= 8


def test_sequential_colormap_runs_light_to_dark():
    cmap = theme.sequential_cmap()
    light, dark = cmap(0.0), cmap(1.0)
    assert sum(light[:3]) > sum(dark[:3])


def test_text_color_contrasts_with_the_cell_background():
    assert theme.text_on(theme.SEQUENTIAL[0]) != theme.text_on(theme.SEQUENTIAL[-1])


def test_agg_backend_and_no_dual_axes_anywhere():
    import matplotlib
    from pathlib import Path

    assert matplotlib.get_backend().lower() == "agg"
    root = Path(theme.__file__).parent
    assert not any("twinx" in p.read_text() or "twiny" in p.read_text() for p in root.glob("*.py"))
