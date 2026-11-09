import pytest

from qrh.usage.aggregate import group, totals
from qrh.usage.privacy import PrivacyError, label_map

C = lambda i=0, o=0, cr=0, cw=0, n=1: {"input": i, "output": o, "cache_read": cr, "cache_write_5m": cw,  # noqa: E731
                                       "cache_write_1h": 0, "requests": n}


def s(sid, start, label="p", cost=1.0, source="harness", models=None, subs=None, trials=None, by_model=None):
    models = models or {"m1": C(10, 5, 90, 0)}
    return {"session_id": sid, "project_label": label, "start": start, "cost_usd": cost, "cost_source": source,
            "models": models, "subagents": subs or {}, "cost_by_model": by_model if by_model is not None else
            ({"m1": cost} if cost is not None else {}), "trials": trials or []}


SESSIONS = {
    "a": s("a", "2026-03-02T10:00:00Z", "alpha", 2.0),   # Monday, ISO week 10
    "b": s("b", "2026-03-02T23:00:00Z", "beta", 1.0, subs={"ag": {"models": {"m2": C(100, 50, 0, 0)}}},
           by_model={"m1": 0.5, "m2": 0.5}),
    "c": s("c", "2026-03-09T01:00:00Z", "alpha", None, "none"),
}


def test_totals_count_cost_and_sessions_without_cost():
    t = totals(SESSIONS.values(), [])
    assert t["sessions"] == 3 and t["cost_usd"] == pytest.approx(3.0) and t["sessions_without_cost"] == 1
    assert t["cost_sources"] == {"harness": 2, "none": 1}


def test_group_by_day_week_project_with_tokens_from_all_chains():
    day = {r["key"]: r for r in group(SESSIONS.values(), [], "day")}
    assert day["2026-03-02"]["sessions"] == 2 and day["2026-03-02"]["cost_usd"] == pytest.approx(3.0)
    assert day["2026-03-02"]["input"] == 120  # main plus subagent
    week = {r["key"]: r for r in group(SESSIONS.values(), [], "week", plays={"a": 2, "c": 1})}
    assert set(week) == {"2026-W10", "2026-W11"} and week["2026-W10"]["plays"] == 2
    proj = {r["key"]: r for r in group(SESSIONS.values(), [], "project")}
    assert proj["alpha"]["sessions"] == 2 and proj["alpha"]["sessions_without_cost"] == 1


def test_group_by_model_uses_cost_by_model_and_local_rows():
    local = [{"ts": "2026-03-02T09:00:00+00:00", "model": "ollama-x", "prompt_tokens": 7, "completion_tokens": 3}]
    rows = {r["key"]: r for r in group(SESSIONS.values(), local, "model")}
    assert rows["m1"]["cost_usd"] == pytest.approx(2.5) and rows["m2"]["cost_usd"] == pytest.approx(0.5)
    assert rows["ollama-x"]["cost_usd"] == 0 and rows["ollama-x"]["input"] == 7 and rows["ollama-x"]["output"] == 3
    days = {r["key"]: r for r in group(SESSIONS.values(), local, "day")}
    assert days["2026-03-02"]["input"] == 127


def test_group_by_trial_and_session():
    tr = lambda tid, stage, cost: {"trial_id": tid, "stage": stage, "models": {"m1": C(1, 1)}, "cost_usd": cost}  # noqa: E731
    sess = {"a": s("a", "2026-03-02T10:00:00Z", trials=[tr("t1", "S3", 1.5), tr("unattributed", None, 0.5)])}
    rows = {r["key"]: r for r in group(sess.values(), [], "trial")}
    assert rows["t1"]["cost_usd"] == pytest.approx(1.5) and rows["unattributed"]["cost_usd"] == pytest.approx(0.5)
    by_stage = {r["key"]: r for r in group(sess.values(), [], "trial-stage")}
    assert "t1/S3" in by_stage
    srows = group(SESSIONS.values(), [], "session")
    assert [r["key"] for r in srows] == ["S001", "S002", "S003"]


def test_unknown_grouping_is_rejected():
    with pytest.raises(ValueError):
        group([], [], "nope")


def test_protected_labels_are_numbered_by_first_start(monkeypatch):
    env = {"QRH_PROTECTED_LABELS": "^secret"}
    m = label_map([
        s("x", "2026-03-05T00:00:00Z", "secret-b"), s("y", "2026-03-01T00:00:00Z", "secret-a"),
        s("z", "2026-03-02T00:00:00Z", "open")], env)
    assert m == {"secret-a": "private-1", "secret-b": "private-2"}
    rows = group([s("y", "2026-03-01T00:00:00Z", "secret-a")], [], "project", labels=m)
    assert rows[0]["key"] == "private-1"
    assert label_map([s("z", "2026-03-02T00:00:00Z", "open")], {}) == {}


def test_invalid_protected_regex_is_an_error():
    with pytest.raises(PrivacyError, match="QRH_PROTECTED_LABELS"):
        label_map([], {"QRH_PROTECTED_LABELS": "("})


def test_model_rows_include_cost_only_models_so_they_add_up_to_the_total():
    sess = {"a": s("a", "2026-03-02T10:00:00Z", cost=3.0, by_model={"m1": 2.0, "background": 1.0})}
    rows = {r["key"]: r for r in group(sess.values(), [], "model")}
    assert rows["background"]["cost_usd"] == pytest.approx(1.0) and rows["background"]["input"] == 0
    assert sum(r["cost_usd"] for r in rows.values()) == pytest.approx(totals(sess.values(), [])["cost_usd"])
