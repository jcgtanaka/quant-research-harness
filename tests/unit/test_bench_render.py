import json

import pytest

from qrh.bench.charts import draw_figure, render_pdf
from qrh.bench.chartdata import build_figures
from qrh.bench.receipt import write_bench
from qrh.bench.render import render_bench
from qrh.bench.run import run_bench
from qrh.bench.textreport import format_comparison, format_receipt
from qrh.bench.compare import within_receipt

CFG = {
    "planted-bugs": {"reps": 1},
    "noise-fpr": {"grids": [12], "periods": [400], "reps": 8},
    "known-edge-power": {"grid": 12, "periods": 600, "strengths": [0.002, 0.008], "reps": 8},
}


@pytest.fixture(scope="module")
def receipt(tmp_path_factory):
    home = tmp_path_factory.mktemp("bench_home")
    return run_bench(home, config=CFG, overrides={"s4.walkforward": "purged_embargo@1"}, compare_to_default=True, seed=9)


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    return run_bench(tmp_path_factory.mktemp("bench_home2"), config=CFG, seed=9)


def test_text_report_shows_methods_flags_and_every_suite(receipt, plain):
    text = format_receipt(receipt)
    assert receipt.bench_id() in text and receipt.receipt_hash() in text
    assert "s4.walkforward=purged_embargo@1" in text and "EXPERIMENTAL" in text
    for heading in ("Planted bugs", "Noise false positives", "Known-edge power"):
        assert heading in text
    assert "lookahead_shift" in text and "controls" in text and "nominal" in text and "g12_t400" in text
    assert "s0.002" in text and "Default vs candidate" in text
    assert "EXPERIMENTAL" not in format_receipt(plain)


def test_comparison_text_has_deltas_p_values_and_distinguishability(receipt):
    text = format_comparison(within_receipt(receipt))
    assert "p=" in text and "delta" in text.lower() and "distinguishable" in text.lower()
    assert "s4.walkforward" in text


def test_chart_data_is_json_deterministic_and_covers_every_figure(receipt, plain):
    figs = build_figures(receipt)
    keys = [f["key"] for f in figs]
    assert keys[0] == "bench_card"
    for k in ("planted_detection", "noise_fpr", "power_curve", "comparison"):
        assert k in keys
    assert "comparison" not in [f["key"] for f in build_figures(plain)]
    text = json.dumps(figs, sort_keys=True, allow_nan=False)
    assert json.dumps(build_figures(receipt), sort_keys=True, allow_nan=False) == text
    for f in figs:
        assert f["receipt_hash"] == receipt.receipt_hash() and f["schema_version"] == 1 and "disclaimer" in f
    card = figs[0]
    assert {m["id"] for m in card["methods"] if m["experimental"]} == {"s4.walkforward"}


def test_noise_figure_carries_the_nominal_level_and_intervals(receipt):
    fig = next(f for f in build_figures(receipt) if f["key"] == "noise_fpr")
    assert fig["nominal"] == pytest.approx(0.05) and fig["categories"][0] == "g12_t400"
    s = fig["series"][0]
    assert len(s["values"]) == len(s["ci_low"]) == len(s["ci_high"]) == len(fig["categories"])
    assert len(fig["series"]) == 2  # default and candidate


def test_pdf_is_drawn_from_chart_data_only_and_is_deterministic(receipt, tmp_path):
    figs = build_figures(receipt)
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    render_pdf(a, figs, receipt.bench_id(), receipt.receipt_hash())
    # round-trip through JSON: the PDF cannot depend on anything but the chart data
    render_pdf(b, json.loads(json.dumps(figs)), receipt.bench_id(), receipt.receipt_hash())
    assert a.read_bytes() == b.read_bytes() and a.stat().st_size > 5000
    assert receipt.receipt_hash().encode() in a.read_bytes()  # full hash in the PDF metadata
    for f in figs:
        assert draw_figure(f, receipt.bench_id(), receipt.receipt_hash()) is not None


def test_render_bench_writes_text_json_pdf_and_chart_json(receipt, tmp_path):
    folder = write_bench(receipt, tmp_path)
    files = render_bench(receipt, folder)
    names = {p.name for p in files}
    assert names == {"report.txt", "report.pdf", "chart.json"}
    assert (folder / "receipt.json").is_file()
    doc = json.loads((folder / "chart.json").read_text())
    assert doc["receipt_hash"] == receipt.receipt_hash() and doc["figures"]
    before = {p.name: p.read_bytes() for p in files}
    render_bench(receipt, folder)
    assert {p.name: p.read_bytes() for p in files} == before  # byte-identical on re-render


def test_history_figure_appears_when_the_suite_ran(receipt):
    fake = receipt.model_copy(update={"suites": [*receipt.suites, "history"], "results": {**receipt.results, "history": {
        "source": "home", "input_hash": "h", "methods": {}, "trials": [
            {"trial_id": "t1", "status": "evaluated", "changed": True, "direction": "stricter", "reproduced": True,
             "stored": {"pass_all": True}, "default": {"pass_all": True}, "candidate": {"pass_all": False}},
            {"trial_id": "t2", "status": "skipped", "reason": "hash mismatch"}],
        "summary": {"evaluated": 1, "skipped": 1, "reproduced": 1, "changed": 1, "stricter": 1, "looser": 0}}}})
    figs = build_figures(fake)
    assert "history" in [f["key"] for f in figs]
    assert "stricter" in format_receipt(fake)


def test_expected_undetectable_variants_are_listed_apart_from_the_misses(receipt):
    text = format_receipt(receipt)
    assert "expected undetectable" in text and "pnl=" in text
    pages = [f for f in build_figures(receipt) if f["key"].startswith("planted_variants")]
    rows = [r for p in pages for r in p["rows"]]
    hidden = [r for r in rows if r["bug_class"] == "relationship_corruption" and r["variant"].startswith("mag_0.005")]
    assert hidden and all(r["kind"] == "expected undetectable" for r in hidden)
    assert all(r["kind"] == "detect" for r in rows if r["bug_class"] == "lookahead_shift")
