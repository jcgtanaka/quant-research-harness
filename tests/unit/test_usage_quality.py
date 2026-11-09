from qrh.usage.quality import evaluate, evaluate_all
from qrh.usage.settings import Prices, ModelPrice, Thresholds

TH = Thresholds(60.0, 300000, 0.8, 5)
PRICES = Prices({
    "model-big": ModelPrice(1, 1, 1, 1, 1, "frontier"),
    "model-small": ModelPrice(1, 1, 1, 1, 1, "small"),
})


def session(**over):
    base = {
        "session_id": "s", "project_label": "p", "start": "2026-03-02T10:00:00Z",
        "models": {"model-big": {"output": 100, "requests": 3}},
        "tools": {"Read": 4, "Grep": 3, "Bash": 3}, "cold_cache_events": 0, "peak_context": 1000,
        "hooks": {"block_total": 0, "lint_blocks": 0, "gate_blocks": 0}, "compactions": 0,
    }
    base.update(over)
    return base


def names(s, prices=PRICES):
    plays, _notes = evaluate(s, TH, prices)
    return {p.play for p in plays}


def test_strong_on_mechanical_fires_with_numbers_and_heuristic_label():
    plays, _ = evaluate(session(), TH, PRICES)
    p = next(p for p in plays if p.play == "strong-on-mechanical")
    assert p.numbers["tool_calls"] == 10 and p.numbers["read_only_share"] == 1.0
    assert "heuristic" in p.text and "not a verdict" in p.text and "frontier" in p.text


def test_strong_on_mechanical_boundaries():
    assert "strong-on-mechanical" not in names(session(tools={"Read": 4}))  # below the minimum tool count
    assert "strong-on-mechanical" not in names(session(tools={"Read": 7, "Edit": 1, "Grep": 2}))  # mutating tool
    assert "strong-on-mechanical" not in names(session(tools={"Read": 7, "Agent": 1, "Grep": 2}))
    assert "strong-on-mechanical" not in names(session(tools={"Read": 3, "Other": 2}))  # share 0.6 < 0.8
    assert "strong-on-mechanical" not in names(session(models={"model-small": {"output": 100}}))
    assert "strong-on-mechanical" in names(session(tools={"Read": 4, "Other": 1, "Grep": 1}))  # 5/6 >= 0.8


def test_mechanical_not_evaluated_without_tier_information():
    plays, notes = evaluate(session(), TH, None)
    assert "strong-on-mechanical" not in {p.play for p in plays} and any("tier" in n for n in notes)
    _, notes = evaluate(session(models={"unknown": {"output": 5}}), TH, PRICES)
    assert any("unknown" in n for n in notes)


def test_threshold_plays():
    assert "cold-cache" not in names(session(cold_cache_events=3))
    assert "cold-cache" in names(session(cold_cache_events=4))
    assert "high-context" not in names(session(peak_context=299999))
    assert "high-context" in names(session(peak_context=300000))
    assert "many-hook-blocks" not in names(session(hooks={"block_total": 4, "lint_blocks": 0, "gate_blocks": 0}))
    assert "many-hook-blocks" in names(session(hooks={"block_total": 5, "lint_blocks": 0, "gate_blocks": 0}))
    assert "gate-fail-after-edit" in names(session(hooks={"block_total": 1, "lint_blocks": 1, "gate_blocks": 0}))
    assert "gate-fail-after-edit" in names(session(hooks={"block_total": 1, "lint_blocks": 0, "gate_blocks": 1}))
    assert "gate-fail-after-edit" not in names(session())


def test_every_play_names_its_rule_and_numbers():
    s = session(cold_cache_events=9, peak_context=400000)
    for p in evaluate(s, TH, PRICES)[0]:
        assert p.rule and p.numbers and p.session_id == "s"
    cold = next(p for p in evaluate(s, TH, PRICES)[0] if p.play == "cold-cache")
    assert "9" in cold.text and "3" in cold.text


def test_evaluate_all_collects_plays_and_notes():
    plays, notes = evaluate_all([session(session_id="a", cold_cache_events=5), session(session_id="b")], TH, None)
    assert [p.session_id for p in plays] == ["a"] and len(notes) == 2
