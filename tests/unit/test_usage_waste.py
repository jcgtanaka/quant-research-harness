import json

import pytest
from typer.testing import CliRunner

from qrh.cli import app
from qrh.usage.harvest import harvest
from qrh.usage.report import write_chart_files
from qrh.usage.settings import SettingsError, load_waste_thresholds
from qrh.usage.waste import build_waste, render_waste_text, waste_charts
from usage_helpers import (SECRET, SECRET_PATH, assistant, compact, cost_state, default_thresholds, flat, prompt, tool_call,
                           write_jsonl, write_settings)

M = "model-big"
T = "2026-03-01T10:{:02d}:00.000Z"
WASTE = {"waste_reread_k": 2, "waste_large_output_tokens": 1000, "waste_repeat_fail_k": 2}
runner = CliRunner()


def th(**extra):
    return {**default_thresholds(), **WASTE, **extra}


def read(i, path=SECRET_PATH):
    return tool_call(f"r{i}", M, T.format(i % 60), "Read", f"tr{i}", {"file_path": path}, inp=1, out=1, cr=100)


@pytest.fixture
def home(tmp_path):
    claude = tmp_path / "claude"
    recs = flat(
        prompt(T.format(0)),
        *[read(i) for i in range(1, 5)],                                              # 4 reads of one path: run 4 > 2
        read(5, "/b.py"),
        tool_call("ed", M, T.format(5), "Edit", "ted", {"file_path": "/c.py", "old_string": "a", "new_string": "b"},
                  inp=1, out=1, cr=100),
        tool_call("big", M, T.format(6), "Bash", "tb", {"command": "cat big"}, result="x" * 8000, inp=1, out=1, cr=100),
        *[tool_call(f"f{i}", M, T.format(10 + i), "Bash", f"tf{i}", {"command": "make test"}, result="Exit code 2", error=True,
                    inp=1, out=1, cr=100) for i in range(3)],
        assistant("cold", M, T.format(20), inp=10, out=2, cr=0, cw5=900),
        assistant("cold2", M, T.format(21), inp=10, out=2, cr=0, cw5=900),
        compact(T.format(22)),
        cost_state(10.0, {M: {"inp": 100, "out": 20, "cr": 10, "cw": 5000, "cost": 10.0}}),
    )
    write_jsonl(claude / "projects" / "-work-projA" / "s1.jsonl", recs)
    # a session whose cost is unknown and which is a mechanical read-only run on the frontier model
    write_jsonl(claude / "projects" / "-work-projB" / "s2.jsonl", flat(
        prompt(T.format(0)), *[tool_call(f"g{i}", M, T.format(i), "Grep", f"tg{i}", {"pattern": f"p{i}"}, inp=1, out=100, cr=100)
                               for i in range(1, 8)]))
    h = tmp_path / "home"
    write_settings(h, thresholds=th())
    harvest(claude, h)
    return h


def test_thresholds_for_waste_are_required_and_fail_closed(tmp_path):
    write_settings(tmp_path)
    with pytest.raises(SettingsError, match="waste_reread_k"):
        load_waste_thresholds(tmp_path)
    write_settings(tmp_path, thresholds=th(waste_reread_k=0))
    with pytest.raises(SettingsError, match="waste_reread_k"):
        load_waste_thresholds(tmp_path)
    write_settings(tmp_path, thresholds=th())
    w = load_waste_thresholds(tmp_path)
    assert (w.reread_k, w.large_output_tokens, w.repeat_fail_k) == (2, 1000, 2)


def test_regular_thresholds_still_load_with_or_without_the_waste_keys(tmp_path):
    from qrh.usage.settings import load_thresholds

    write_settings(tmp_path, thresholds=th())
    assert load_thresholds(tmp_path).cache_cold_pct == 60


def test_waste_types_per_session_and_totals(home):
    rep = build_waste(home, env={})
    s1, s2 = rep["sessions"]
    assert (s1["session"], s2["session"]) == ("S001", "S002")
    types = {w["type"]: w for w in s1["waste"]}
    assert types["re-reads"]["paths"] == 1 and types["re-reads"]["worst_run"] == 4
    assert types["large-tool-outputs"]["count"] == 1 and types["large-tool-outputs"]["tokens"] == 2000
    assert types["repeated-failing-commands"]["commands"] == 1 and types["repeated-failing-commands"]["worst"] == 3
    assert types["cache-cold-events"]["count"] == 2 and types["cache-cold-events"]["estimate"] is True
    assert 0 < types["cache-cold-events"]["cost_usd"] < 10.0
    assert types["compactions"]["count"] == 1
    assert "frontier-on-mechanical" not in types
    assert {w["type"] for w in s2["waste"]} == {"frontier-on-mechanical"}
    tot = rep["totals"]
    assert tot["re-reads"]["count"] == 1 and tot["compactions"]["count"] == 1 and rep["totals_sessions"] == 2


def test_every_waste_type_carries_a_one_line_fix(home):
    rep = build_waste(home, env={})
    for s in rep["sessions"]:
        for w in s["waste"]:
            assert w["fix"] and "\n" not in w["fix"] and len(w["fix"]) < 200


def test_cold_cache_excess_cost_is_hand_computed_per_model(home):
    # two cold requests rewrote 900 tokens each at the 5m write price; the excess over reading them from cache is
    # 1800 x (12.5 - 1.0) / 1e6 = 0.0207 USD. The old formula (session cost x token share) gave about 9.9 USD.
    rep = build_waste(home, env={})
    cold = next(w for w in rep["sessions"][0]["waste"] if w["type"] == "cache-cold-events")
    assert cold["cold_write_tokens"] == 1800 and cold["estimate"] is True
    assert cold["cost_usd"] == pytest.approx(0.0207) and cold["cost_label"] == "estimated excess cost"
    assert "cache_write_price - cache_read_price" in cold["basis"] and "cost-state" not in cold["basis"]
    assert rep["totals"]["cache-cold-events"]["cost_usd"] == pytest.approx(0.0207)
    assert rep["totals"]["cache-cold-events"]["cold_write_tokens"] == 1800
    assert "cache-cold-events" not in [w["type"] for w in rep["sessions"][1]["waste"]]


def _cold_home(tmp_path, model, prices, cw5=0, cw1=0):
    claude = tmp_path / "c2"
    write_jsonl(claude / "projects" / "-w-p" / "x.jsonl", flat(
        prompt(T.format(0)), assistant("a", model, T.format(1), inp=1, out=1, cr=100),
        assistant("b", model, T.format(2), inp=1, out=1, cr=0, cw5=cw5, cw1=cw1),
        cost_state(50.0, {model: {"inp": 2, "out": 2, "cr": 100, "cw": cw5 + cw1, "cost": 50.0}})))
    h = tmp_path / "h2"
    write_settings(h, thresholds=th(), prices=prices)
    harvest(claude, h)
    return h


def test_one_hour_and_five_minute_writes_use_their_own_prices(tmp_path):
    h = _cold_home(tmp_path, M, "default", cw5=1000, cw1=500)
    cold = next(w for w in build_waste(h, env={})["sessions"][0]["waste"] if w["type"] == "cache-cold-events")
    assert cold["cold_write_tokens"] == 1500
    assert cold["cost_usd"] == pytest.approx((1000 * (12.5 - 1.0) + 500 * (20.0 - 1.0)) / 1e6)


def test_example_or_missing_prices_give_tokens_but_no_dollars(tmp_path):
    ex = {"models": {"example-frontier-model": {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write_5m": 1.25,
                                                "cache_write_1h": 2.0, "tier": "frontier"}}}
    for model, prices in (("example-frontier-model", ex), ("unpriced-model", "default"), ("model-big", None)):
        h = _cold_home(tmp_path / model, model, prices, cw5=1000)
        rep = build_waste(h, env={})
        cold = next(w for w in rep["sessions"][0]["waste"] if w["type"] == "cache-cold-events")
        assert cold["cold_write_tokens"] == 1000 and cold["cost_usd"] is None and "n/a (no price)" in cold["basis"], model
        assert "n/a (no price)" in render_waste_text(rep)
        assert rep["totals"]["cache-cold-events"]["cost_usd"] == 0.0 and rep["totals"]["cache-cold-events"]["unpriced_sessions"] == 1


def test_cost_state_never_feeds_the_estimate(tmp_path):
    cheap = _cold_home(tmp_path / "a", M, "default", cw5=1000)
    cold = next(w for w in build_waste(cheap, env={})["sessions"][0]["waste"] if w["type"] == "cache-cold-events")
    assert cold["cost_usd"] == pytest.approx(1000 * 11.5 / 1e6)  # the 50 USD cost-state is irrelevant


def test_paths_and_commands_are_aliases_only(home):
    rep = build_waste(home, env={})
    blob = json.dumps(rep) + render_waste_text(rep)
    assert SECRET_PATH not in blob and "make test" not in blob and SECRET not in blob and "/work" not in blob
    rr = next(w for w in rep["sessions"][0]["waste"] if w["type"] == "re-reads")
    assert all(a.startswith("f-") and len(a) == 6 for a in rr["aliases"])


def test_sessions_are_not_listed_by_id(home):
    blob = json.dumps(build_waste(home, env={}))
    assert "s1" not in json.dumps([s["session"] for s in build_waste(home, env={})["sessions"]]) and '"session_id"' not in blob


def test_protected_labels_apply(home):
    rep = build_waste(home, env={"QRH_PROTECTED_LABELS": "projA"})
    assert rep["sessions"][0]["project"] == "private-1" and "projA" not in json.dumps(rep)


def test_missing_waste_thresholds_stop_the_report(tmp_path):
    write_settings(tmp_path)
    with pytest.raises(SettingsError, match="waste_"):
        build_waste(tmp_path, env={})


def test_charts_and_files(home, tmp_path):
    rep = build_waste(home, env={})
    charts = waste_charts(rep)
    assert [c["key"] for c in charts] == ["waste_summary", "waste_by_type", "waste_sessions"]
    paths = write_chart_files(home, charts)
    assert all(p.name.endswith(".chart.json") for p in paths)
    assert json.loads(paths[1].read_text())["chart_type"] in ("bar", "bar_ranked")


def test_cli_text_json_pdf_and_determinism(home, tmp_path, monkeypatch):
    monkeypatch.setenv("QRH_HOME", str(home))
    r = runner.invoke(app, ["usage", "waste"])
    assert r.exit_code == 0 and "re-reads" in r.output and "fix:" in r.output
    a = runner.invoke(app, ["usage", "waste", "--json"])
    assert json.loads(a.stdout)["totals_sessions"] == 2 and runner.invoke(app, ["usage", "waste", "--json"]).stdout == a.stdout
    pdf1, pdf2 = tmp_path / "a.pdf", tmp_path / "b.pdf"
    assert runner.invoke(app, ["usage", "waste", "--pdf", str(pdf1)]).exit_code == 0
    runner.invoke(app, ["usage", "waste", "--pdf", str(pdf2)])
    assert pdf1.read_bytes().startswith(b"%PDF") and pdf1.read_bytes() == pdf2.read_bytes()
    assert (home / "usage" / "reports" / "waste_by_type.chart.json").is_file()


def test_cli_without_thresholds_is_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("QRH_HOME", str(tmp_path))
    write_settings(tmp_path)
    r = runner.invoke(app, ["usage", "waste"])
    assert r.exit_code == 2 and "waste_reread_k" in r.output


def test_summary_chart_carries_cold_tokens_and_the_excess_label(home):
    summary = waste_charts(build_waste(home, env={}))[0]
    vals = {f["label"]: f["value"] for f in summary["fields"]}
    assert vals["Cold-cache tokens re-written"] == 1800 and vals["Estimated excess cost (priced sessions)"] == "USD 0.02"
