import io
import json
from datetime import datetime, timezone

import pytest

from qrh.usage.local import (
    LocalLogError, append_local, load_local, record_from_ollama,
)
from qrh.usage.verify import verify_sessions

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


def test_record_from_chat_response():
    resp = {"model": "qwen-x", "prompt_eval_count": 120, "eval_count": 30, "total_duration": 2_500_000_000,
            "message": {"role": "assistant", "content": "SECRET-PROMPT-TEXT"}, "done": True}
    rec = record_from_ollama(json.dumps(resp), task="summarize log", session_id="s1", now=NOW)
    assert rec == {"ts": "2026-03-01T12:00:00+00:00", "session_id": "s1", "model": "qwen-x", "prompt_tokens": 120,
                   "completion_tokens": 30, "duration_ms": 2500.0, "task": "summarize log"}


def test_record_from_streamed_ndjson_uses_final_object_and_optional_fields_are_omitted():
    lines = json.dumps({"model": "m", "response": "partial"}) + "\n" + json.dumps(
        {"model": "m", "done": True, "prompt_eval_count": 5, "eval_count": 6, "total_duration": 1_000_000})
    rec = record_from_ollama(lines, now=NOW)
    assert "session_id" not in rec and rec["task"] == "" and rec["prompt_tokens"] == 5


@pytest.mark.parametrize("text", ["", "not json", "[1]", json.dumps({"model": "m"}),
                                  json.dumps({"prompt_eval_count": 1, "eval_count": 1})])
def test_bad_ollama_input_is_an_error(text):
    with pytest.raises(LocalLogError):
        record_from_ollama(text, now=NOW)


def test_model_override_and_roundtrip(tmp_path):
    rec = record_from_ollama(json.dumps({"prompt_eval_count": 1, "eval_count": 2, "total_duration": 5}),
                             model="override", now=NOW)
    append_local(tmp_path, rec)
    append_local(tmp_path, rec)
    (tmp_path / "logs" / "local_llm.jsonl").open("a").write("{bad\n")
    assert len(load_local(tmp_path)) == 2 and load_local(tmp_path)[0]["model"] == "override"


def test_load_local_missing_file_is_empty(tmp_path):
    assert load_local(tmp_path) == []


def sess(models, harness, subs=None, raw=None, label="p"):
    return {"session_id": "x", "project_label": label, "start": "2026-03-01T00:00:00Z", "models": models,
            "models_raw": raw or models, "subagents": subs or {}, "harness_usage": harness}


def counts(inp=0, out=0, cr=0, cw=0):
    return {"input": inp, "output": out, "cache_read": cr, "cache_write_5m": cw, "cache_write_1h": 0, "requests": 1}


def hu(inp=0, out=0, cr=0, cw=0):
    return {"input": inp, "output": out, "cache_read": cr, "cache_creation": cw, "cost_usd": 0.0}


def test_exact_match_and_inflation():
    s = sess({"m": counts(10, 20, 30, 40)}, {"m": hu(10, 20, 30, 40)}, raw={"m": counts(20, 40, 60, 80)})
    r = verify_sessions({"a": s}, scope="strict")
    assert r.verified == 1 and r.matched == 1 and r.mismatched == 0 and r.ok
    assert r.inflation_factor == pytest.approx(2.0)
    assert r.rows[0].best_diff_pct == 0


def test_subagent_inclusion_is_detected_per_session_and_overall():
    s = sess({"m": counts(10, 10, 10, 10)}, {"m": hu(15, 15, 15, 15)},
             subs={"agent-1": {"models": {"m": counts(5, 5, 5, 5)}, "cost_state": False}})
    r = verify_sessions({"a": s}, scope="strict")
    assert r.rows[0].better == "main+subagents" and r.ok
    assert r.subagent_finding.startswith("cost-state includes subagents")
    s2 = sess({"m": counts(10, 10, 10, 10)}, {"m": hu(10, 10, 10, 10)},
              subs={"agent-1": {"models": {"m": counts(5, 5, 5, 5)}, "cost_state": True}})
    r2 = verify_sessions({"b": s2})
    assert r2.rows[0].better == "main" and r2.subagent_finding.startswith("cost-state excludes subagents")
    assert r2.subagent_files == 1 and r2.subagent_files_with_cost_state == 1


def test_mismatch_beyond_one_percent_fails_and_tolerance_edge():
    ok = verify_sessions({"a": sess({"m": counts(100, 0, 0, 0)}, {"m": hu(101, 0, 0, 0)})})
    assert ok.ok  # 1/101 is under 1% (diff is measured against the harness figure)
    bad = verify_sessions({"a": sess({"m": counts(100, 0, 0, 0)}, {"m": hu(102, 0, 0, 0)})})
    assert not bad.ok and bad.mismatched == 1 and bad.rows[0].best_diff_pct == pytest.approx(100 * 2 / 102)


def test_missing_model_counts_as_full_mismatch_and_sessions_without_cost_state_are_skipped():
    r = verify_sessions({"a": sess({"m": counts(10)}, {"other": hu(10)}),
                         "b": sess({"m": counts(10)}, None)}, scope="strict")
    assert r.verified == 1 and r.mismatched == 1


def test_transcript_scope_ignores_cost_state_only_models_and_output_tokens():
    s = sess({"m": counts(10, 5, 30, 40)}, {"m": hu(10, 50, 30, 40), "background": hu(99, 9, 0, 0)})
    assert not verify_sessions({"a": s}, scope="strict").ok  # strict: output differs and a model is missing from the transcripts
    r = verify_sessions({"a": s}, scope="transcript")
    assert r.ok and r.rows[0].best_diff_pct == 0


def test_causes_of_mismatch_are_counted_per_session():
    s1 = sess({"m": counts(10, 5, 30, 40)}, {"m": hu(10, 50, 30, 40), "background": hu(99, 9, 0, 0)})
    s2 = sess({"m": counts(20, 5, 30, 40)}, {"m": hu(10, 5, 30, 40)})
    s3 = sess({"m": counts(5, 5, 30, 40)}, {"m": hu(10, 5, 30, 40)})
    r = verify_sessions({"a": s1, "b": s2, "c": s3}, scope="strict")
    assert r.causes == {"cost-state has a model the transcripts lack": 1, "output tokens below cost-state": 1,
                        "transcript above cost-state": 1, "input side below cost-state": 1}


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError):
        verify_sessions({}, scope="nope")


def test_difference_below_tolerance_between_hypotheses_is_not_a_preference():
    s = sess({"m": counts(1000, 0, 0, 0)}, {"m": hu(1000, 0, 0, 0)},
             subs={"agent-1": {"models": {"m": counts(5, 0, 0, 0)}, "cost_state": False}})
    r = verify_sessions({"a": s})
    assert r.rows[0].better == "indistinguishable" and r.subagent_finding.startswith("not determined")


def test_finding_is_mostly_when_decisive_sessions_agree_at_least_four_to_one():
    from qrh.usage.verify import VerifyResult
    assert VerifyResult(better_counts={"main+subagents": 8, "main": 2}).subagent_finding.startswith(
        "cost-state mostly includes subagents")
    assert VerifyResult(better_counts={"main+subagents": 5, "main": 5}).subagent_finding.startswith("mixed")
    assert VerifyResult(better_counts={"main+subagents": 1, "main": 9}).subagent_finding.startswith(
        "cost-state mostly excludes subagents")


# --- default scope and the gate ---------------------------------------------------------------------

SUB = {"agent-1": {"models": {"m": counts(5, 5, 5, 5)}, "cost_state": False}}


def test_default_scope_is_transcript():
    s = sess({"m": counts(10, 5, 30, 40)}, {"m": hu(10, 50, 30, 40), "background": hu(99, 9, 0, 0)})
    r = verify_sessions({"a": s})
    assert r.scope == "transcript" and r.ok and r.rows[0].best_diff_pct == 0


def test_gate_fails_on_a_session_without_subagents_that_mismatches_in_transcript_scope():
    r = verify_sessions({"a": sess({"m": counts(100)}, {"m": hu(120)})})
    assert not r.ok and r.gating_mismatched == 1 and r.gating_verified == 1


def test_sessions_with_subagents_are_reported_but_do_not_gate():
    good = sess({"m": counts(100)}, {"m": hu(100)})
    sub_bad = sess({"m": counts(100)}, {"m": hu(500)}, subs=SUB)
    r = verify_sessions({"a": good, "b": sub_bad})
    assert r.ok and r.mismatched == 1 and r.gating_mismatched == 0
    assert r.subagent_sessions == 1 and r.subagent_matched == 0
    assert [row.gating for row in r.rows] == [True, False]


def test_subagent_match_rate_counts_matches():
    one = {"agent-1": {"models": {"m": counts(5)}, "cost_state": False}}
    a = sess({"m": counts(10)}, {"m": hu(15)}, subs=one)           # main+subagents matches
    b = sess({"m": counts(10)}, {"m": hu(900)}, subs=one)          # does not
    r = verify_sessions({"a": a, "b": b})
    assert r.subagent_sessions == 2 and r.subagent_matched == 1 and r.ok


def test_strict_scope_still_gates_every_session():
    s = sess({"m": counts(100)}, {"m": hu(500)}, subs=SUB)
    assert not verify_sessions({"a": s}, scope="strict").ok
    assert verify_sessions({"a": s}, scope="transcript").ok


def test_alignment_explains_a_mismatch_only_in_transcript_scope_and_only_when_it_matches():
    base = sess({"m": counts(100, 0, 100, 10)}, {"m": hu(50, 0, 50, 5)})
    base["alignment"] = {"kind": "counter-restart", "models": {"m": counts(50, 0, 50, 5)}}
    r = verify_sessions({"a": base})
    assert r.ok and r.rows[0].explained == "counter-restart" and r.explained == {"counter-restart": 1}
    assert not verify_sessions({"a": base}, scope="strict").ok
    wrong = sess({"m": counts(100, 0, 100, 10)}, {"m": hu(50, 0, 50, 5)})
    wrong["alignment"] = {"kind": "stale-cost-state", "models": {"m": counts(70, 0, 70, 7)}}
    r2 = verify_sessions({"a": wrong})
    assert not r2.ok and r2.rows[0].explained is None


def test_render_labels_lower_bound_and_unexplained(tmp_path):
    from qrh.usage.verify import render_verify

    a = sess({"m": counts(100)}, {"m": hu(120)})
    b = sess({"m": counts(100)}, {"m": hu(500)}, subs=SUB)
    a["session_id"], b["session_id"] = "a", "b"
    a["start"], b["start"] = "2026-03-01T00:00:00Z", "2026-03-02T00:00:00Z"
    text = render_verify(verify_sessions({"a": a, "b": b}), {"a": 1, "b": 2}, {})
    assert "lower bound" in text and "not gating" in text and "unexplained" in text
    assert "S001" in text and "S002" in text and "scope               transcript" in text
