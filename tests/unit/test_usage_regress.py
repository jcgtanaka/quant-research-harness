import json
from datetime import datetime, timedelta, timezone

import pytest
from scipy import stats
from typer.testing import CliRunner

from qrh.cli import app
from qrh.usage.harvest import _write_archive
from qrh.usage.regress import MIN_TURNS, build_regress, regress_charts, render_regress_text
from qrh.usage.settings import SettingsError
from usage_helpers import SECRET, write_settings

runner = CliRunner()
M1, M2 = "model-big", "model-small"
BASE = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)  # a Monday: ISO week 10


def turn(i, ts, model=M1, out=100, inp=10, cr=800, cw=100, tools=3, failures=0):
    return {"ts": ts.isoformat(), "tools": tools, "failures": failures,
            "models": {model: {"input": inp, "output": out, "cache_read": cr, "cache_write_5m": cw, "cache_write_1h": 0,
                               "requests": 1}}}


def session(sid, day, n=25, model=M1, cfg="cfg-a", cost=10.0, **kw):
    start = BASE + timedelta(days=day)
    turns = [turn(i, start + timedelta(minutes=i), model=model, **{k: (v(i) if callable(v) else v) for k, v in kw.items()})
             for i in range(n)]
    return {"session_id": sid, "project_label": "projA", "start": start.isoformat(), "end": start.isoformat(),
            "models": {model: turns[0]["models"][model] | {"output": 100 * n}}, "turns": turns, "cost_usd": cost,
            "config_hash": cfg, "cost_source": "harness"}


def put(home, *sessions, thresholds=True):
    if thresholds:
        write_settings(home)
    _write_archive(home, {s["session_id"]: s for s in sessions})


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


def test_week_split_orders_groups_and_compares_consecutive_pairs(home):
    put(home, session("a", 0), session("b", 1), session("c", 8), session("d", 15))
    rep = build_regress(home, "week", env={})
    assert [g["key"] for g in rep["groups"]] == ["2026-W10", "2026-W11", "2026-W12"]
    assert [(g["sessions"], g["turns"]) for g in rep["groups"]] == [(2, 50), (1, 25), (1, 25)]
    assert [(c["before"], c["after"]) for c in rep["comparisons"]] == [("2026-W10", "2026-W11"), ("2026-W11", "2026-W12")]


def test_mann_whitney_matches_scipy_and_reports_effect_size(home):
    # failures-free; the "after" side writes more output tokens per turn
    put(home, session("a", 0, out=lambda i: 100 + i), session("b", 8, out=lambda i: 140 + i))
    c = build_regress(home, "week", env={})["comparisons"][0]
    m = next(x for x in c["metrics"] if x["metric"] == "output_tokens_per_turn" or x["metric"] == "new_tokens_per_turn")
    before = [10 + 100 + i + 100 for i in range(25)]
    after = [10 + 140 + i + 100 for i in range(25)]
    u, p = stats.mannwhitneyu(after, before, alternative="two-sided")
    assert m["test"] == "mann-whitney-u" and m["p"] == pytest.approx(p)
    assert m["effect_size"] == pytest.approx(2 * u / (25 * 25) - 1)       # rank-biserial, after vs before
    assert m["effect_size"] > 0.9 and m["before"]["median"] < m["after"]["median"]


def test_fisher_exact_for_rates(home):
    put(home, session("a", 0, failures=lambda i: 1 if i < 5 else 0), session("b", 8, failures=lambda i: 1 if i < 15 else 0))
    c = build_regress(home, "week", env={})["comparisons"][0]
    m = next(x for x in c["metrics"] if x["metric"] == "turns_with_a_tool_failure")
    odds, p = stats.fisher_exact([[15, 10], [5, 20]])
    assert m["test"] == "fisher-exact" and m["p"] == pytest.approx(p) and m["effect_size"] == pytest.approx(odds)
    assert m["before"]["rate"] == pytest.approx(5 / 25) and m["after"]["rate"] == pytest.approx(15 / 25)


def test_hook_blocks_per_turn_come_from_the_hook_log_by_timestamp(home):
    put(home, session("a", 0), session("b", 8))
    (home / "logs").mkdir()
    start_b = BASE + timedelta(days=8)
    rows = [{"session_id": "b", "decision": "block", "hook": "data-gate", "ts": (start_b + timedelta(minutes=i, seconds=30)).isoformat()}
            for i in range(10)]
    rows += [{"session_id": "b", "decision": "allow", "hook": "data-gate", "ts": start_b.isoformat()},
             {"session_id": "other", "decision": "block", "hook": "data-gate", "ts": start_b.isoformat()}]
    (home / "logs" / "hooks.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    c = build_regress(home, "week", env={})["comparisons"][0]
    m = next(x for x in c["metrics"] if x["metric"] == "turns_with_a_hook_block")
    assert m["before"]["rate"] == 0 and m["after"]["rate"] == pytest.approx(10 / 25)
    per = next(x for x in c["metrics"] if x["metric"] == "hook_blocks_per_turn")
    assert per["after"]["mean"] == pytest.approx(10 / 25)


def test_cost_per_turn_splits_the_session_cost_by_weight_and_skips_unknown_cost(home):
    a, b = session("a", 0, cost=25.0), session("b", 8, cost=None)
    put(home, a, b)
    c = build_regress(home, "week", env={})["comparisons"][0]
    m = next(x for x in c["metrics"] if x["metric"] == "cost_per_turn_usd")
    assert m["before"]["mean"] == pytest.approx(1.0) and m["after"]["n"] == 0 and m["test"] == "not-run"
    assert "fewer than" in m["note"] or "no data" in m["note"]


def test_cache_hit_per_turn(home):
    put(home, session("a", 0, cr=900, cw=0, inp=100), session("b", 8, cr=0, cw=900, inp=100))
    m = next(x for x in build_regress(home, "week", env={})["comparisons"][0]["metrics"] if x["metric"] == "cache_hit_pct")
    assert m["before"]["median"] == pytest.approx(90.0) and m["after"]["median"] == 0.0 and m["effect_size"] == pytest.approx(-1.0)


def test_small_samples_warn_and_never_claim_cause(home):
    put(home, session("a", 0, n=MIN_TURNS - 1), session("b", 8, n=30))
    rep = build_regress(home, "week", env={})
    c = rep["comparisons"][0]
    assert c["small_sample"] is True and f"fewer than {MIN_TURNS} turns" in c["warning"]
    text = render_regress_text(rep)
    assert "small sample" in text.lower() and "does not show cause" in text


def test_no_small_sample_warning_when_both_sides_are_large(home):
    put(home, session("a", 0, n=MIN_TURNS), session("b", 8, n=MIN_TURNS))
    assert build_regress(home, "week", env={})["comparisons"][0]["small_sample"] is False


def test_model_split_uses_the_leading_model_of_each_session(home):
    put(home, session("a", 0, model=M1), session("b", 1, model=M2), session("c", 2, model=M2))
    rep = build_regress(home, "model", env={})
    assert [g["key"] for g in rep["groups"]] == [M1, M2] and rep["comparisons"][0]["after"] == M2


def test_config_split_follows_the_hash_in_time_order_and_shows_a_short_hash(home):
    put(home, session("a", 0, cfg="aaaaaaaaaaaaaaaa"), session("b", 1, cfg="bbbbbbbbbbbbbbbb"), session("c", 2, cfg="aaaaaaaaaaaaaaaa"))
    rep = build_regress(home, "config", env={})
    assert [g["key"] for g in rep["groups"]] == ["cfg-aaaaaaaa", "cfg-bbbbbbbb"]      # a recurs: still one group
    assert rep["groups"][0]["sessions"] == 2


def test_sessions_without_a_config_hash_group_as_unknown(home):
    s = session("a", 0)
    del s["config_hash"]
    put(home, s, session("b", 1, cfg="bbbbbbbbbbbbbbbb"))
    assert [g["key"] for g in build_regress(home, "config", env={})["groups"]] == ["cfg-bbbbbbbb", "unknown"]  # unknown last


def test_one_group_means_no_comparison_and_says_so(home):
    put(home, session("a", 0), session("b", 1))
    rep = build_regress(home, "week", env={})
    assert rep["comparisons"] == [] and "only one group" in render_regress_text(rep)


def test_bad_grouping_and_missing_thresholds(home):
    put(home, session("a", 0))
    with pytest.raises(ValueError):
        build_regress(home, "nope", env={})
    (home / "usage" / "thresholds.json").unlink()
    with pytest.raises(SettingsError):
        build_regress(home, "week", env={})


def test_old_sessions_without_turns_are_counted_and_skipped(home):
    old = session("old", 0)
    del old["turns"]
    put(home, old, session("b", 8), session("c", 15))
    rep = build_regress(home, "week", env={})
    assert any("predate" in n for n in rep["notes"]) and [g["key"] for g in rep["groups"]] == ["2026-W11", "2026-W12"]


def test_report_has_no_ids_text_or_paths(home):
    s = session("session-secret-id", 0)
    s["project_label"] = "projA"
    put(home, s, session("b", 8))
    blob = json.dumps(build_regress(home, "week", env={})) + render_regress_text(build_regress(home, "week", env={}))
    assert "session-secret-id" not in blob and SECRET not in blob


def test_charts_json_and_cli(home, tmp_path, monkeypatch):
    put(home, session("a", 0), session("b", 8))
    charts = regress_charts(build_regress(home, "week", env={}))
    assert [c["key"] for c in charts] == ["regress_summary", "regress_groups", "regress_comparisons"]
    monkeypatch.setenv("QRH_HOME", str(home))
    r = runner.invoke(app, ["usage", "regress", "--by", "week"])
    assert r.exit_code == 0 and "2026-W10" in r.output and "mann-whitney" in r.output
    j = runner.invoke(app, ["usage", "regress", "--by", "week", "--json"])
    assert json.loads(j.stdout)["by"] == "week" and runner.invoke(app, ["usage", "regress", "--json"]).stdout == j.stdout
    pdf = tmp_path / "r.pdf"
    p1 = runner.invoke(app, ["usage", "regress", "--by", "week", "--pdf", str(pdf)])
    first = pdf.read_bytes()
    runner.invoke(app, ["usage", "regress", "--by", "week", "--pdf", str(pdf)])
    assert p1.exit_code == 0 and first.startswith(b"%PDF") and pdf.read_bytes() == first
    assert (home / "usage" / "reports" / "regress_comparisons.chart.json").is_file()
    assert runner.invoke(app, ["usage", "regress", "--by", "nope"]).exit_code == 2


def test_constant_metrics_say_there_is_nothing_to_test(home):
    put(home, session("a", 0), session("b", 8))
    m = next(x for x in build_regress(home, "week", env={})["comparisons"][0]["metrics"] if x["metric"] == "hook_blocks_per_turn")
    assert m["p"] is None and "identical" in m["note"]
