"""Runaway guard: loop, failure-streak and spend trip wires, config rules, approvals, fail modes."""

import json
import os

import pytest

from hook_helpers import approval_id, bash_payload, hook_log, human_approve, payload, run, write_config
from qrh.hooks.common import ConfigError, load_config

CFG = "runaway:\n  repeat_limit: 3\n  failure_streak: 3\n  spend_cap_usd: 10\n"


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    write_config(h, CFG)
    return h


def pre(home, tool="Bash", n=1, sid="sess-1", transcript="", **tool_input):
    tool_input = tool_input or {"command": "pytest -q"}
    return run("runaway-guard", payload("PreToolUse", session_id=sid, tool_name=tool, tool_input=tool_input,
                                        tool_use_id=f"tu-{n}", transcript_path=transcript), home)


def fail(home, n, sid="sess-1", tool="Bash", interrupt=False, **tool_input):
    data = payload("PostToolUseFailure", session_id=sid, tool_name=tool, tool_input=tool_input or {"command": "x"},
                   tool_use_id=f"tu-{n}", error="boom", is_interrupt=interrupt)
    return run("runaway-failure", data, home)


def transcript(tmp_path, *records, pad=0):
    p = tmp_path / "t.jsonl"
    lines = [json.dumps({"type": "user", "message": {"content": "x" * pad}})] if pad else []
    lines += [json.dumps(r) for r in records]
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def cost_rec(total):
    return {"type": "cost-state", "totalCostUSD": total, "modelUsage": {}}


# --- configuration ------------------------------------------------------------------------------

def test_section_absent_means_disabled_and_nothing_is_tracked(tmp_path):
    h = tmp_path / "h"
    write_config(h, "trigger_terms: [alpha]\n")
    for i in range(10):
        assert pre(h, n=i).exit_code == 0
    assert "disabled" in hook_log(h)[-1]["reason"]
    assert not (h / "runaway").exists()


def test_valid_section_parses(home):
    rc = load_config(home).runaway
    assert (rc.repeat_limit, rc.failure_streak, rc.spend_cap_usd) == (3, 3, 10.0)


def test_each_limit_is_optional_but_there_are_no_defaults(tmp_path):
    h = tmp_path / "h"
    write_config(h, "runaway:\n  spend_cap_usd: 5.5\n")
    rc = load_config(h).runaway
    assert rc.spend_cap_usd == 5.5 and rc.repeat_limit is None and rc.failure_streak is None
    for i in range(10):  # no loop limit configured, so no loop wire
        assert pre(h, n=i).exit_code == 0


@pytest.mark.parametrize("text", [
    "runaway: {}\n", "runaway: 3\n", "runaway:\n  repeat_limit: 1\n", "runaway:\n  repeat_limit: 2.5\n",
    "runaway:\n  failure_streak: 0\n", "runaway:\n  spend_cap_usd: 0\n", "runaway:\n  spend_cap_usd: -1\n",
    "runaway:\n  spend_cap_usd: true\n", "runaway:\n  spend_cap_usd: .nan\n", "runaway:\n  spend_cap_usd: .inf\n",
    "runaway:\n  repeat_limit: '3'\n", "runaway:\n  repeat_limit: 3\n  surprise: 1\n",
])
def test_invalid_section_is_a_config_error(tmp_path, text):
    h = tmp_path / "h"
    write_config(h, text)
    with pytest.raises(ConfigError):
        load_config(h)


def test_invalid_config_blocks_the_guard(tmp_path):
    h = tmp_path / "h"
    write_config(h, "runaway: {}\n")
    assert pre(h).exit_code == 2


def test_project_config_may_lower_but_never_raise(home, tmp_path):
    proj = tmp_path / "proj"
    (proj / ".qrh").mkdir(parents=True)
    (proj / ".qrh" / "config.yaml").write_text("runaway:\n  repeat_limit: 2\n  failure_streak: 99\n  spend_cap_usd: 1000\n")

    def call(n, **kw):
        return run("runaway-guard", payload("PreToolUse", cwd=str(proj), tool_name="Bash", tool_use_id=f"t{n}",
                                            tool_input={"command": "ls"}, **kw), home)

    assert call(1).exit_code == 0
    assert call(2).exit_code == 2  # the project's lower repeat_limit (2) applies
    t = transcript(proj, cost_rec(11.0))
    out = call(3, transcript_path=t)
    assert out.exit_code == 2 and "10" in out.stderr  # the user's cap of 10 stands; the project's 1000 is ignored


def test_project_config_cannot_enable_a_wire_the_user_left_off(tmp_path):
    h = tmp_path / "h"
    write_config(h, "runaway:\n  spend_cap_usd: 10\n")
    proj = tmp_path / "proj"
    (proj / ".qrh").mkdir(parents=True)
    (proj / ".qrh" / "config.yaml").write_text("runaway:\n  repeat_limit: 2\n")
    for n in range(4):
        out = run("runaway-guard", payload("PreToolUse", cwd=str(proj), tool_name="Bash", tool_use_id=f"t{n}",
                                           tool_input={"command": "ls"}), h)
        assert out.exit_code == 0


def test_project_config_cannot_enable_the_guard_and_bad_files_are_ignored_safely(tmp_path):
    h = tmp_path / "h"
    write_config(h, "trigger_terms: [alpha]\n")
    proj = tmp_path / "proj"
    (proj / ".qrh").mkdir(parents=True)
    (proj / ".qrh" / "config.yaml").write_text("runaway:\n  repeat_limit: 2\n")
    assert run("runaway-guard", payload("PreToolUse", cwd=str(proj), tool_name="Bash", tool_input={"command": "a"}), h).exit_code == 0
    write_config(h, CFG)
    (proj / ".qrh" / "config.yaml").write_text("runaway: [not a mapping\n")
    out = run("runaway-guard", payload("PreToolUse", cwd=str(proj), tool_name="Bash", tool_input={"command": "a"}), h)
    assert out.exit_code == 0 and "project_config_error" in hook_log(h)[-1]


# --- loop ---------------------------------------------------------------------------------------

def test_third_identical_call_in_a_row_is_blocked_with_the_approval_flow(home):
    assert pre(home, n=1).exit_code == 0
    assert pre(home, n=2).exit_code == 0
    out = pre(home, n=3)
    assert out.exit_code == 2 and "3 times in a row" in out.stderr and "limit 3" in out.stderr
    assert "qrh approve" in out.stderr and "OWN terminal" in out.stderr
    from qrh.banner import BANNER, RELAY
    assert out.stderr.startswith(BANNER) and "Hook: runaway-guard" in out.stderr and RELAY.strip() in out.stderr


def test_a_different_call_in_between_resets_the_count(home):
    assert pre(home, n=1).exit_code == 0 and pre(home, n=2).exit_code == 0
    assert pre(home, command="ls", n=3).exit_code == 0
    assert pre(home, n=4).exit_code == 0 and pre(home, n=5).exit_code == 0
    assert pre(home, n=6).exit_code == 2


def test_same_tool_with_other_input_is_not_a_repeat(home):
    for i in range(6):
        assert pre(home, command=f"pytest tests/test_{i}.py", n=i).exit_code == 0


def test_whitespace_differences_are_the_same_call(home):
    pre(home, command="pytest   -q", n=1)
    pre(home, command="pytest -q\n", n=2)
    assert pre(home, command="pytest -q", n=3).exit_code == 2


def test_deliberate_waits_and_agents_never_count(home):
    for i in range(6):
        assert pre(home, tool="Task", n=i, description="d", prompt="p").exit_code == 0
        assert pre(home, tool="TaskOutput", n=100 + i, task_id="1").exit_code == 0


def test_sessions_are_counted_separately(home):
    pre(home, n=1, sid="a"), pre(home, n=2, sid="a")
    assert pre(home, n=3, sid="b").exit_code == 0
    assert pre(home, n=4, sid="a").exit_code == 2


def test_human_approval_lets_one_call_through_and_restarts_the_count(home):
    pre(home, n=1), pre(home, n=2)
    blocked = pre(home, n=3)
    assert blocked.exit_code == 2
    human_approve(home, blocked.stderr)
    assert pre(home, n=4).exit_code == 0           # the approved call
    assert pre(home, n=5).exit_code == 0           # count restarted: this is the 1st again
    assert pre(home, n=6).exit_code == 0
    again = pre(home, n=7)
    assert again.exit_code == 2 and approval_id(again.stderr) != approval_id(blocked.stderr)


# --- failures -----------------------------------------------------------------------------------

def test_consecutive_failures_trip_the_next_call(home):
    for n in (1, 2, 3):
        assert pre(home, command=f"make step{n}", n=n).exit_code == 0
        assert fail(home, n).exit_code == 0
    out = pre(home, command="make other", n=4)
    assert out.exit_code == 2 and "3 tool calls in a row failed" in out.stderr


def test_a_call_that_did_not_fail_ends_the_streak(home):
    for n in (1, 2):
        pre(home, command=f"make step{n}", n=n)
        fail(home, n)
    pre(home, command="make ok", n=3)          # no failure event for tu-3: it succeeded
    pre(home, command="make b", n=4)
    fail(home, 4)
    assert pre(home, command="make c", n=5).exit_code == 0


def test_interrupts_are_not_failures(home):
    for n in (1, 2, 3):
        pre(home, command=f"make s{n}", n=n)
        fail(home, n, interrupt=True)
    assert pre(home, command="make z", n=4).exit_code == 0


def test_failure_hook_never_blocks_and_logs(home):
    out = fail(home, 1)
    assert out.exit_code == 0
    assert hook_log(home)[-1]["hook"] == "runaway-failure" and hook_log(home)[-1]["decision"] == "info"


def test_failure_without_a_tool_use_id_marks_the_latest_call(home):
    for n in (1, 2, 3):
        pre(home, command=f"make s{n}", n=n)
        run("runaway-failure", payload("PostToolUseFailure", tool_name="Bash", tool_input={"command": "x"}, error="e"),
            home)
    assert pre(home, command="make z", n=4).exit_code == 2


def test_failure_hook_with_the_guard_disabled_records_nothing(tmp_path):
    h = tmp_path / "h"
    write_config(h, "trigger_terms: [alpha]\n")
    assert fail(h, 1).exit_code == 0 and not (h / "runaway").exists()


def test_approval_clears_the_failure_streak(home):
    for n in (1, 2, 3):
        pre(home, command=f"make s{n}", n=n)
        fail(home, n)
    blocked = pre(home, command="make z", n=4)
    human_approve(home, blocked.stderr)
    assert pre(home, command="make z", n=5).exit_code == 0
    assert pre(home, command="make y", n=6).exit_code == 0


# --- spend --------------------------------------------------------------------------------------

def test_spend_at_or_over_the_cap_blocks(home, tmp_path):
    t = transcript(tmp_path, cost_rec(10.0))
    out = pre(home, transcript=t)
    assert out.exit_code == 2 and "10.00" in out.stderr and "cost-state" in out.stderr


def test_spend_under_the_cap_passes_and_uses_the_latest_cost_state(home, tmp_path):
    t = transcript(tmp_path, cost_rec(50.0), {"type": "assistant"}, cost_rec(3.25))
    assert pre(home, transcript=t).exit_code == 0
    t2 = transcript(tmp_path, cost_rec(3.25), cost_rec(12.0))
    assert pre(home, transcript=t2, command="other").exit_code == 2


def test_latest_cost_state_is_found_in_a_large_transcript(home, tmp_path):
    t = transcript(tmp_path, cost_rec(10.5), *[{"type": "assistant", "n": i} for i in range(3000)], pad=600_000)
    assert pre(home, transcript=t).exit_code == 2


def test_no_cost_state_yet_never_trips_on_spend(home, tmp_path):
    t = transcript(tmp_path, {"type": "user"}, {"type": "assistant"})
    assert pre(home, transcript=t).exit_code == 0


def test_unreadable_transcript_allows_and_logs(home, tmp_path):
    out = pre(home, transcript=str(tmp_path / "missing.jsonl"))
    assert out.exit_code == 0
    rec = hook_log(home)[-1]
    assert rec["decision"] == "error" and "transcript unreadable" in rec["reason"]


def test_a_missing_transcript_path_is_also_unreadable_but_loops_still_trip(home):
    pre(home, n=1), pre(home, n=2)
    assert pre(home, n=3, transcript="").exit_code == 2  # the loop wire does not need the transcript


def test_spend_wire_off_when_no_cap_configured(tmp_path):
    h = tmp_path / "h"
    write_config(h, "runaway:\n  repeat_limit: 3\n")
    t = transcript(tmp_path, cost_rec(999.0))
    assert pre(h, transcript=t).exit_code == 0


def test_several_wires_are_all_listed(home, tmp_path):
    t = transcript(tmp_path, cost_rec(11.0))
    for n in (1, 2, 3):
        pre(home, command="a", n=n, transcript=t)
    out = pre(home, command="a", n=4, transcript=t)
    assert "in a row" in out.stderr and "spend" in out.stderr


# --- registration and state ---------------------------------------------------------------------

def test_registered_for_every_tool_and_for_failures():
    from qrh.hooks.specs import HOOKS

    g, f = HOOKS["runaway-guard"], HOOKS["runaway-failure"]
    assert (g.event, g.matcher, g.enforcement, g.approvable) == ("PreToolUse", None, True, True)
    assert (f.event, f.matcher, f.enforcement) == ("PostToolUseFailure", None, False)


def test_a_corrupt_state_file_is_reset_not_fatal(home):
    pre(home, n=1)
    state_file = next((home / "runaway").iterdir())
    state_file.write_text("{not json")
    assert pre(home, n=2).exit_code == 0
    assert hook_log(home)[-1].get("state_reset") is True


def test_state_dir_is_not_world_readable(home):
    pre(home, n=1)
    if os.name == "posix":
        assert oct((home / "runaway").stat().st_mode & 0o777) == "0o700"


def test_hook_log_for_blocks_does_not_leak_commands_beyond_the_redacted_form(home):
    pre(home, command="curl -H 'Authorization: Bearer abcdef123456' x", n=1)  # gitleaks:allow (fake token)
    pre(home, command="curl -H 'Authorization: Bearer abcdef123456' x", n=2)  # gitleaks:allow (fake token)
    pre(home, command="curl -H 'Authorization: Bearer abcdef123456' x", n=3)  # gitleaks:allow (fake token)
    assert "abcdef123456" not in (home / "logs" / "hooks.jsonl").read_text()
    assert "abcdef123456" not in json.dumps([p.read_text() for p in (home / "runaway").iterdir()])
    # the pending request keeps the raw action (so the human signs exactly what was blocked): private files only
    import stat
    assert stat.S_IMODE((home / "approvals").stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in (home / "approvals").iterdir())
