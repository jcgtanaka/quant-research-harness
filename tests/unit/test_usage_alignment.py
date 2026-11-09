"""Deterministic causes of a no-subagent mismatch between the transcript and the harness cost-state.

Each cause was confirmed on real sessions (numbers only): the cost-state is stale (the transcript went on after
its last write), the harness cost counter restarted at a resume (it covers only the later part of the file),
or a message never completed (the harness does not bill an interrupted stream, the transcript still records it).
"""

from qrh.usage.harvest import harvest, load_archive
from qrh.usage.parse import parse_transcript
from qrh.usage.verify import verify_sessions
from usage_helpers import assistant, cost_state, flat, write_jsonl, write_settings

M = "model-big"


def ts(day, minute):
    return f"2026-03-{day:02d}T10:{minute:02d}:00.000Z"


def req(mid, day, minute, cr, cw=10, inp=5, **kw):
    return assistant(mid, M, ts(day, minute), inp=inp, out=3, cr=cr, cw5=cw, **kw)


def cs(*reqs):
    """A cost-state whose per-model totals are the sum of the given (inp, cr, cw) triples."""
    inp = sum(r[0] for r in reqs)
    cr = sum(r[1] for r in reqs)
    cw = sum(r[2] for r in reqs)
    return cost_state(1.0, {M: {"inp": inp, "out": 9, "cr": cr, "cw": cw, "cost": 1.0}})


def archive_of(tmp_path, records):
    claude = tmp_path / "claude"
    write_jsonl(claude / "projects" / "-work-p" / "sx.jsonl", records)
    home = tmp_path / "home"
    write_settings(home)
    harvest(claude, home)
    return load_archive(home)


def verified(archive):
    return verify_sessions(archive, scope="transcript")


def test_parse_marks_incomplete_messages_and_cost_state_position(tmp_path):
    recs = flat(req("a", 1, 0, 100), req("b", 1, 1, 200, stop=None), req("c", 1, 2, 300, stop="end_turn"),
                cs((5, 100, 10)), req("d", 1, 3, 400))
    t = parse_transcript(write_jsonl(tmp_path / "t.jsonl", recs))
    assert [r.complete for r in t.requests] == [True, False, True, True]  # no stop_reason key at all counts as complete
    assert t.cost_state_main_requests == 3


def test_a_copy_with_a_stop_reason_completes_the_message(tmp_path):
    first = assistant("a", M, ts(1, 0), inp=1, cr=1, cw5=1, stop=None)
    last = assistant("a", M, ts(1, 0), inp=1, cr=1, cw5=1, stop="end_turn")
    t = parse_transcript(write_jsonl(tmp_path / "t.jsonl", flat(first, last)))
    assert len(t.requests) == 1 and t.requests[0].complete


def test_stale_cost_state_matches_the_prefix_up_to_its_last_write(tmp_path):
    a, b, c = (5, 100, 10), (5, 200, 10), (5, 300, 10)
    recs = flat(req("a", 1, 0, 100), req("b", 1, 1, 200), cs(a, b), req("c", 1, 2, 300))
    arc = archive_of(tmp_path, recs)
    al = arc["sx"]["alignment"]
    assert al["kind"] == "stale-cost-state" and al["requests_after"] == 1 and al["requests_total"] == 3
    res = verified(arc)
    row = res.rows[0]
    assert row.ok and row.explained == "stale-cost-state" and res.ok
    assert res.explained == {"stale-cost-state": 1}
    del c


def test_counter_restart_matches_the_suffix_exactly(tmp_path):
    b, c = (5, 200, 10), (5, 300, 10)
    recs = flat(req("a", 1, 0, 100), req("b", 3, 0, 200), req("c", 3, 1, 300), cs(b, c))
    arc = archive_of(tmp_path, recs)
    al = arc["sx"]["alignment"]
    assert al["kind"] == "counter-restart" and al["requests_before"] == 1 and al["requests_total"] == 3
    assert al["gap_before_s"] == 2 * 24 * 3600
    row = verified(arc).rows[0]
    assert row.ok and row.explained == "counter-restart"


def test_incomplete_message_is_not_billed(tmp_path):
    a, c = (5, 100, 10), (5, 300, 10)
    recs = flat(req("a", 1, 0, 100, stop="end_turn"), req("b", 1, 1, 200, stop=None), req("c", 1, 2, 300, stop="end_turn"),
                cs(a, c))
    arc = archive_of(tmp_path, recs)
    al = arc["sx"]["alignment"]
    assert al["kind"] == "incomplete-message" and al["incomplete_requests"] == 1
    row = verified(arc).rows[0]
    assert row.ok and row.explained == "incomplete-message"


def test_a_session_that_matches_plainly_needs_no_alignment(tmp_path):
    recs = flat(req("a", 1, 0, 100), req("b", 1, 1, 200), cs((5, 100, 10), (5, 200, 10)))
    arc = archive_of(tmp_path, recs)
    assert arc["sx"]["alignment"] is None
    row = verified(arc).rows[0]
    assert row.ok and row.explained is None


def test_an_unexplained_gap_stays_a_mismatch_and_fails_the_gate(tmp_path):
    # the cost-state is 300 cache-read tokens above the transcript: no stale tail, no restart, no incomplete message
    recs = flat(req("a", 1, 0, 100), req("b", 1, 1, 200), cs((5, 400, 10), (5, 200, 10)))
    arc = archive_of(tmp_path, recs)
    assert arc["sx"]["alignment"] is None
    res = verified(arc)
    assert not res.rows[0].ok and res.rows[0].explained is None and not res.ok


def test_search_for_a_restart_demands_an_exact_match(tmp_path):
    # theirs is 3% below the full total: no suffix equals it exactly, so it is not "explained" by chance
    recs = flat(*[req(f"m{i}", 1, i, 1000) for i in range(30)], cs(*[(5, 1000, 10)] * 29, (5, 0, 10)))
    arc = archive_of(tmp_path, recs)
    assert arc["sx"]["alignment"] is None
    assert not verified(arc).rows[0].ok


def test_sessions_with_subagents_get_no_alignment(tmp_path):
    claude = tmp_path / "claude"
    write_jsonl(claude / "projects" / "-work-p" / "sx.jsonl", flat(
        req("a", 1, 0, 100), req("b", 1, 1, 200), cs((5, 100, 10), (5, 200, 10)), req("c", 1, 2, 300)))
    write_jsonl(claude / "projects" / "-work-p" / "sx" / "subagents" / "agent-1.jsonl",
                flat(assistant("z", M, ts(1, 1), inp=1, sidechain=True)))
    home = tmp_path / "home"
    write_settings(home)
    harvest(claude, home)
    assert load_archive(home)["sx"]["alignment"] is None
