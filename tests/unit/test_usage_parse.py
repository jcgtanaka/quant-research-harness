from qrh.usage.parse import parse_transcript
from usage_helpers import SECRET, SECRET_PATH, assistant, compact, cost_state, flat, user, write_jsonl

T = "2026-03-01T10:00:0{}.000Z"


def parse(tmp_path, records, raw=()):
    return parse_transcript(write_jsonl(tmp_path / "s.jsonl", records, raw))


def test_repeated_records_with_same_id_count_once_and_raw_shows_inflation(tmp_path):
    recs = flat(assistant("m1", "model-big", T.format(1), inp=10, out=5, cr=100, cw5=7, repeat=3),
                assistant("m2", "model-big", T.format(2), inp=1, out=1, repeat=1))
    t = parse(tmp_path, recs)
    assert [r.msg_id for r in t.requests] == ["m1", "m2"]
    assert t.requests[0].tokens.input == 10 and t.requests[0].tokens.cache_read == 100
    assert t.raw_main["model-big"].input == 31 and t.raw_main["model-big"].requests == 4


def test_records_without_id_each_count_once(tmp_path):
    recs = flat(assistant(None, "model-big", T.format(1), inp=3, repeat=2))
    assert len(parse(tmp_path, recs).requests) == 2


def test_tool_names_are_counted_across_repeated_records_once_per_tool_use_id(tmp_path):
    recs = flat(assistant("m1", "model-big", T.format(1), tools=("Read", "Grep"), repeat=3),
                assistant("m2", "model-big", T.format(2), tools=("Read",), tool_ids=("tu_m1_0",), repeat=2))
    t = parse(tmp_path, recs)
    assert t.tools == {"Read": 1, "Grep": 1}


def test_sidechain_requests_are_flagged_and_excluded_from_main_tools(tmp_path):
    recs = flat(assistant("m1", "model-big", T.format(1), inp=1),
                assistant("m2", "model-small", T.format(2), inp=2, sidechain=True, tools=("Edit",)))
    t = parse(tmp_path, recs)
    assert [r.sidechain for r in t.requests] == [False, True]
    assert t.tools == {} and "model-small" not in t.raw_main


def test_cache_write_split_and_fallback_to_five_minute(tmp_path):
    recs = flat(assistant("a", "model-big", T.format(1), cw5=4, cw1=6),
                assistant("b", "model-big", T.format(2), cw5=9, split=False))
    a, b = parse(tmp_path, recs).requests
    assert (a.tokens.cache_write_5m, a.tokens.cache_write_1h) == (4, 6)
    assert (b.tokens.cache_write_5m, b.tokens.cache_write_1h) == (9, 0)


def test_compaction_boundary_marks_next_main_request(tmp_path):
    recs = flat(assistant("a", "model-big", T.format(1)), compact(T.format(2)),
                assistant("b", "model-big", T.format(3), sidechain=True),
                assistant("c", "model-big", T.format(4)), assistant("d", "model-big", T.format(5)))
    t = parse(tmp_path, recs)
    assert t.compactions == 1
    assert [r.after_compaction for r in t.requests] == [False, False, True, False]


def test_last_cost_state_wins(tmp_path):
    recs = flat(cost_state(1.0, {"model-big": {"inp": 1, "cost": 1.0}}),
                cost_state(2.5, {"model-big": {"inp": 9, "out": 8, "cr": 7, "cw": 6, "cost": 2.0},
                                 "model-small": {"cost": 0.5}}))
    cs = parse(tmp_path, recs).cost_state
    assert cs["total_cost_usd"] == 2.5
    assert cs["models"]["model-big"] == {"input": 9, "output": 8, "cache_read": 7, "cache_creation": 6,
                                         "cost_usd": 2.0}


def test_malformed_lines_counted_unknown_types_ignored_blank_lines_skipped(tmp_path):
    recs = flat(assistant("a", "model-big", T.format(1), inp=1), {"type": "mystery", "x": 1})
    t = parse(tmp_path, recs, raw=("{not json", "", "[1, 2]", '"str"'))
    assert t.malformed == 3 and len(t.requests) == 1


def test_synthetic_model_and_missing_usage_are_skipped(tmp_path):
    recs = flat(assistant("a", "<synthetic>", T.format(1), inp=1), assistant("b", "model-big", T.format(2),
                with_usage=False), assistant("c", "model-big", T.format(3), inp=2))
    assert [r.msg_id for r in parse(tmp_path, recs).requests] == ["c"]


def test_cwd_and_time_bounds(tmp_path):
    recs = flat(user(T.format(1), cwd="/a/b"), assistant("a", "model-big", T.format(5), cwd="/a/b"),
                assistant("b", "model-big", T.format(3), cwd="/c"))
    t = parse(tmp_path, recs)
    assert t.cwds["/a/b"] == 2 and t.first_ts == T.format(1) and t.last_ts == T.format(5)


def test_no_message_text_or_paths_are_retained(tmp_path):
    recs = flat(user(T.format(1)), assistant("a", "model-big", T.format(2), tools=("Read",), repeat=2))
    text = repr(parse(tmp_path, recs).__dict__)
    assert SECRET not in text and SECRET_PATH not in text


def test_repeated_records_whose_output_grows_keep_the_final_value(tmp_path):
    first = assistant("m1", "model-big", T.format(1), inp=10, out=5, cr=100, sidechain=True)
    later = assistant("m1", "model-big", T.format(2), inp=10, out=40, cr=100, sidechain=True)
    t = parse(tmp_path, flat(first, later, assistant("m2", "model-big", T.format(3), inp=1, out=2)))
    assert [r.msg_id for r in t.requests] == ["m1", "m2"]
    assert t.requests[0].tokens.output == 40 and t.requests[0].tokens.input == 10
    assert t.requests[0].ts == T.format(1)
