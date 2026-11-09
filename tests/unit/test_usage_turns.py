"""Per-turn and waste facts captured at harvest: counts and hashes only, never text or paths."""

import json

import pytest

from qrh.usage.harvest import harvest, load_archive
from qrh.usage.parse import parse_transcript
from usage_helpers import (SECRET, SECRET_PATH, assistant, compact, cost_state, flat, prompt, tool_call, write_jsonl,
                           write_settings)

M = "model-big"
T = "2026-03-01T10:{:02d}:00.000Z"


def read(mid, minute, path=SECRET_PATH, tid=None, **kw):
    return tool_call(mid, M, T.format(minute), "Read", tid or f"t{mid}", {"file_path": path}, **kw)


def test_turns_split_at_typed_prompts_only(tmp_path):
    recs = flat(
        prompt(T.format(0)), assistant("a", M, T.format(1), inp=10, out=2, cr=100, cw5=5),
        tool_call("b", M, T.format(2), "Bash", "tb", {"command": "ls"}, inp=1, out=1),   # its tool_result is a user record
        {**prompt(T.format(3), "reminder"), "isMeta": True},
        prompt(T.format(4)), assistant("c", M, T.format(5), inp=20, out=3, cr=200),
    )
    t = parse_transcript(write_jsonl(tmp_path / "t.jsonl", recs))
    assert [tr.requests for tr in t.turns] == [2, 1]
    assert [r.turn for r in t.requests] == [0, 0, 1]
    assert t.turns[0].ts == T.format(0)


def test_requests_before_any_prompt_get_turn_zero_without_crashing(tmp_path):
    t = parse_transcript(write_jsonl(tmp_path / "t.jsonl", flat(assistant("a", M, T.format(1), inp=1))))
    assert len(t.turns) == 1 and t.requests[0].turn == 0


@pytest.mark.parametrize("text", ["<system-reminder>x</system-reminder>", "<command-name>/clear</command-name>",
                                  "Caveat: The messages below were generated", "<local-command-stdout>x"])
def test_harness_text_does_not_start_a_turn(tmp_path, text):
    t = parse_transcript(write_jsonl(tmp_path / "t.jsonl", flat(prompt(T.format(0)), prompt(T.format(1), text))))
    assert len(t.turns) == 1


def test_failures_count_per_turn_but_denials_interrupts_and_hook_blocks_do_not(tmp_path):
    recs = flat(
        prompt(T.format(0)),
        tool_call("a", M, T.format(1), "Bash", "t1", {"command": "false"}, result="Exit code 1", error=True),
        tool_call("b", M, T.format(2), "Bash", "t2", {"command": "x"}, result="The user doesn't want to proceed", error=True),
        tool_call("c", M, T.format(3), "Bash", "t3", {"command": "y"}, result="[Request interrupted by user]", error=True),
        tool_call("d", M, T.format(4), "Bash", "t4", {"command": "z"}, result="BLOCKED by qrh hook data-gate: x", error=True),
        tool_call("e", M, T.format(5), "Bash", "t5", {"command": "ok"}, result="fine"),
        prompt(T.format(6)),
        tool_call("f", M, T.format(7), "Bash", "t6", {"command": "false"}, result="Exit code 2", error=True),
    )
    t = parse_transcript(write_jsonl(tmp_path / "t.jsonl", recs))
    assert [(tr.tools, tr.failures) for tr in t.turns] == [(5, 1), (1, 1)]


def test_rereads_count_runs_per_hashed_path_and_reset_on_edit_prompt_and_compaction(tmp_path):
    recs = flat(
        prompt(T.format(0)),
        read("a", 1), read("b", 2), read("c", 3),
        tool_call("e", M, T.format(4), "Edit", "te", {"file_path": SECRET_PATH, "old_string": "x", "new_string": "y"}),
        read("d", 5), read("f", 6),
        prompt(T.format(7)), read("g", 8),
        read("h", 9, path="/other/file.py"),
    )
    t = parse_transcript(write_jsonl(tmp_path / "t.jsonl", recs))
    runs = sorted(t.reread_runs.values())
    assert runs == [1, 3] or runs == [1, 1, 3] or sorted(runs) == [1, 1, 3]  # 3 reads before the edit is the longest run
    assert max(t.reread_runs.values()) == 3 and len(t.reread_runs) == 2
    assert all(len(k) == 8 and SECRET_PATH not in k for k in t.reread_runs)


def test_failed_reads_are_not_rereads(tmp_path):
    recs = flat(prompt(T.format(0)), read("a", 1, error=True, result="File not found"), read("b", 2, error=True, result="x"))
    assert parse_transcript(write_jsonl(tmp_path / "t.jsonl", recs)).reread_runs == {}


def test_failing_commands_are_counted_by_hash_across_turns(tmp_path):
    recs = flat(prompt(T.format(0)),
                *[tool_call(f"m{i}", M, T.format(i + 1), "Bash", f"x{i}", {"command": "make   build\n"}, result="Exit code 1", error=True)
                  for i in range(3)],
                tool_call("ok", M, T.format(9), "Bash", "xo", {"command": "make build"}, result="fine"))
    t = parse_transcript(write_jsonl(tmp_path / "t.jsonl", recs))
    assert list(t.failing_commands.values()) == [3] and all(len(k) == 8 for k in t.failing_commands)


def test_output_sizes_are_estimated_tokens_of_tool_results(tmp_path):
    recs = flat(prompt(T.format(0)), tool_call("a", M, T.format(1), "Bash", "t1", {"command": "ls"}, result="x" * 400),
                tool_call("b", M, T.format(2), "Bash", "t2", {"command": "ls"}, result="y" * 40_000))
    assert parse_transcript(write_jsonl(tmp_path / "t.jsonl", recs)).output_tokens == [100, 10_000]


# --- harvest ------------------------------------------------------------------------------------

@pytest.fixture
def env(tmp_path):
    claude = tmp_path / "claude"
    recs = flat(
        prompt(T.format(0)), assistant("a", M, T.format(1), inp=10, out=2, cr=0, cw5=1000),
        assistant("b", M, T.format(2), inp=10, out=2, cr=10, cw5=900),       # cold: 10 of 920 read from cache
        tool_call("c", M, T.format(3), "Bash", "t1", {"command": "boom"}, result="Exit code 1", error=True, inp=5, out=1, cr=900),
        prompt(T.format(4)), assistant("d", M, T.format(5), inp=5, out=4, cr=900),
        compact(T.format(6)),
        cost_state(2.0, {M: {"inp": 30, "out": 9, "cr": 1810, "cw": 1900, "cost": 2.0}}),
    )
    write_jsonl(claude / "projects" / "-work-projA" / "s1.jsonl", recs)
    (claude / "settings.json").write_text(json.dumps({"hooks": {"Stop": [{"hooks": [{"command": "x"}]}]}, "theme": "dark"}))
    (claude / "CLAUDE.md").write_text("user rules v1")
    home = tmp_path / "home"
    write_settings(home)
    return claude, home


def test_harvest_stores_turns_and_waste_facts(env):
    claude, home = env
    harvest(claude, home)
    s = load_archive(home)["s1"]
    assert len(s["turns"]) == 2 and s["turns"][0]["tools"] == 1 and s["turns"][0]["failures"] == 1
    assert s["turns"][0]["models"][M]["cache_write_5m"] == 1900 and s["turns"][1]["models"][M]["cache_read"] == 900
    w = s["waste"]
    assert w["cold_writes"] == {M: {"cache_write_5m": 900, "cache_write_1h": 0}}
    assert w["output_tokens"] == [2] and list(w["failing_commands"].values()) == [1] and w["reread_runs"] == {}


def test_harvest_blob_has_no_text_or_paths(env):
    claude, home = env
    harvest(claude, home)
    raw = (home / "usage" / "sessions.json").read_text()
    assert SECRET not in raw and SECRET_PATH not in raw and "boom" not in raw


def test_config_hash_covers_hooks_section_and_claude_md_and_is_only_a_hash(env):
    claude, home = env
    harvest(claude, home)
    h1 = load_archive(home)["s1"]["config_hash"]
    assert len(h1) == 16 and "dark" not in h1
    harvest(claude, home)
    assert load_archive(home)["s1"]["config_hash"] == h1                       # unchanged config, same hash
    (claude / "settings.json").write_text(json.dumps({"hooks": {"Stop": []}, "theme": "light"}))
    harvest(claude, home)
    assert load_archive(home)["s1"]["config_hash"] == h1                       # captured once: a later harvest keeps it


def test_config_hash_differs_for_a_new_session_after_the_config_changed(env, tmp_path):
    claude, home = env
    harvest(claude, home)
    h1 = load_archive(home)["s1"]["config_hash"]
    (claude / "CLAUDE.md").write_text("user rules v2")
    write_jsonl(claude / "projects" / "-work-projA" / "s2.jsonl", flat(prompt(T.format(0)), assistant("z", M, T.format(1), inp=1)))
    harvest(claude, home)
    a = load_archive(home)
    assert a["s2"]["config_hash"] != h1 and a["s1"]["config_hash"] == h1


def test_non_hook_settings_do_not_change_the_hash(env):
    claude, home = env
    harvest(claude, home)
    h1 = load_archive(home)["s1"]["config_hash"]
    (claude / "settings.json").write_text(json.dumps({"hooks": {"Stop": [{"hooks": [{"command": "x"}]}]}, "theme": "light"}))
    write_jsonl(claude / "projects" / "-work-projA" / "s2.jsonl", flat(prompt(T.format(0)), assistant("z", M, T.format(1), inp=1)))
    harvest(claude, home)
    assert load_archive(home)["s2"]["config_hash"] == h1


def test_missing_config_files_hash_as_absent_not_as_an_error(tmp_path):
    claude = tmp_path / "c"
    write_jsonl(claude / "projects" / "-w-p" / "a.jsonl", flat(prompt(T.format(0)), assistant("a", M, T.format(1), inp=1)))
    home = tmp_path / "h"
    write_settings(home)
    harvest(claude, home)
    assert len(load_archive(home)["a"]["config_hash"]) == 16
