import json

import pytest

from qrh.usage.attribution import (
    UNATTRIBUTED, Tag, allocate_cost, attribute, hook_stats, load_hook_stats, load_session_tags,
)
from qrh.usage.parse import Request, TokenCounts
from qrh.usage.settings import load_prices
from usage_helpers import write_settings


def req(ts, model="model-big", inp=1, out=0):
    return Request("id", ts, model, TokenCounts(input=inp, output=out, requests=1), False)


def tag(ts, trial, stage=None):
    from qrh.usage.parse import parse_ts
    return Tag(parse_ts(ts), trial, stage)


def test_message_goes_to_most_recent_earlier_tag_and_untagged_is_unattributed():
    tags = [tag("2026-03-01T10:00:10Z", "t1", "S3"), tag("2026-03-01T10:00:30Z", "t2", None)]
    reqs = [req("2026-03-01T10:00:05Z"), req("2026-03-01T10:00:10Z"), req("2026-03-01T10:00:20Z", inp=5),
            req("2026-03-01T10:00:40Z", inp=7), req(None)]
    got = attribute(reqs, tags)
    assert got[(UNATTRIBUTED, None)]["model-big"].input == 2
    assert got[("t1", "S3")]["model-big"].input == 6
    assert got[("t2", None)]["model-big"].input == 7


def test_no_tags_means_everything_unattributed():
    got = attribute([req("2026-03-01T10:00:05Z")], [])
    assert list(got) == [(UNATTRIBUTED, None)]


def test_tag_without_trial_id_does_not_attribute(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    rows = [{"session_id": "s", "trial_id": None, "stage": None, "kind": "new", "ts": "2026-03-01T10:00:00+00:00"},
            {"session_id": "s", "trial_id": "t9", "stage": "S2", "kind": "run", "ts": "2026-03-01T10:01:00+00:00"},
            {"session_id": "other", "trial_id": "t8", "stage": None, "kind": "run", "ts": "2026-03-01T10:00:00+00:00"}]
    (logs / "session_trials.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    tags = load_session_tags(tmp_path)
    assert [(t.trial_id, t.stage) for t in tags["s"]] == [("t9", "S2")] and "other" in tags


def test_hook_stats_count_blocks_overrides_and_gate_failures():
    rows = [
        {"hook": "lookahead-lint", "decision": "block"}, {"hook": "lookahead-lint", "decision": "block"},
        {"hook": "data-gate", "decision": "block"}, {"hook": "data-gate", "decision": "allow", "override": True},
        {"hook": "receipt-required", "decision": "warn"},
    ]
    s = hook_stats(rows)
    assert s == {"blocks": {"data-gate": 1, "lookahead-lint": 2}, "block_total": 3, "overrides": 1,
                 "lint_blocks": 2, "gate_blocks": 1}


def test_load_hook_stats_groups_by_session_and_tolerates_bad_lines(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "hooks.jsonl").write_text(
        json.dumps({"session_id": "a", "hook": "data-gate", "decision": "block"}) + "\n{bad\n"
        + json.dumps({"session_id": "b", "hook": "x", "decision": "allow"}) + "\n")
    stats = load_hook_stats(tmp_path)
    assert stats["a"]["block_total"] == 1 and stats["b"]["block_total"] == 0


def test_allocate_cost_price_weighted_when_all_models_priced(tmp_path):
    write_settings(tmp_path)
    prices = load_prices(tmp_path)
    attrib = {("t1", None): {"model-small": TokenCounts(input=1_000_000)},
              (UNATTRIBUTED, None): {"model-big": TokenCounts(input=1_000_000)}}
    cost, how = allocate_cost(attrib, 11.0, prices)
    assert how == "price-weighted"
    assert cost[("t1", None)] == pytest.approx(1.0) and cost[(UNATTRIBUTED, None)] == pytest.approx(10.0)


def test_allocate_cost_falls_back_to_tokens_without_prices_and_none_without_cost():
    attrib = {("t1", None): {"m": TokenCounts(input=1, output=2)}, ("t2", None): {"m": TokenCounts(output=9)}}
    cost, how = allocate_cost(attrib, 12.0, None)
    assert how == "token-weighted" and cost[("t1", None)] == pytest.approx(3.0) and cost[("t2", None)] == pytest.approx(9.0)
    assert allocate_cost(attrib, None, None) == ({k: None for k in attrib}, "none")
