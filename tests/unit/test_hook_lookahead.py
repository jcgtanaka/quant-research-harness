import pytest

from hook_helpers import edit_payload, hook_log, payload, run, write_config

STRATEGY_HEAD = """\
import pandas as pd
from qrh.adapters.base import DataView, RunResult


class Adapter:
    def run(self, params, data, start, end):
        px = data.as_of(end)["close"]
{body}
"""


def strategy(tmp_path, body, name="strat.py"):
    p = tmp_path / name
    p.write_text(STRATEGY_HEAD.format(body=body), encoding="utf-8")
    return p


def lint(tmp_path, home, path, tool="Edit", env=None):
    return run("lookahead-lint", edit_payload("PostToolUse", str(path), tool=tool), home, env=env)


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


@pytest.mark.parametrize("tool", ["Edit", "Write", "MultiEdit"])
def test_block_finding_exits_2_with_file_line_rule(tmp_path, home, tool):
    p = strategy(tmp_path, "        sig = px.shift(-1)")
    out = lint(tmp_path, home, p, tool=tool)
    assert out.exit_code == 2
    assert f"{p}:8" in out.stderr and "LA001" in out.stderr and "negative shift" in out.stderr
    assert hook_log(home)[-1]["decision"] == "block"


def test_clean_strategy_allowed(tmp_path, home):
    out = lint(tmp_path, home, strategy(tmp_path, "        sig = px.shift(1)"))
    assert out.exit_code == 0 and out.stderr == ""
    assert hook_log(home)[-1]["decision"] == "allow"


def test_warn_findings_only_in_the_log(tmp_path, home):
    p = strategy(tmp_path, "        m = px.mean()\n        z = px - m")
    out = lint(tmp_path, home, p)
    assert out.exit_code == 0 and out.stderr == "" and out.stdout == ""
    rec = hook_log(home)[-1]
    assert rec["decision"] == "warn" and "LA005" in rec["reason"]
    assert rec["warnings"] and rec["warnings"][0].startswith("LA005:")


def test_suppressed_block_finding_does_not_block(tmp_path, home):
    p = strategy(tmp_path, "        y = px.shift(-1)  # qrh: allow LA001 evaluation label only")
    assert lint(tmp_path, home, p).exit_code == 0


def test_non_python_file_ignored_and_not_read(tmp_path, home):
    p = tmp_path / "notes.md"
    p.write_text("x.shift(-1)")
    out = lint(tmp_path, home, p)
    assert out.exit_code == 0 and "not a python file" in hook_log(home)[-1]["reason"]
    assert lint(tmp_path, home, tmp_path / "does_not_exist.txt").exit_code == 0


def test_python_file_that_is_not_a_strategy_is_ignored(tmp_path, home):
    p = tmp_path / "util.py"
    p.write_text("def f(x):\n    return x.shift(-1)\n")
    out = lint(tmp_path, home, p)
    assert out.exit_code == 0 and "not a strategy" in hook_log(home)[-1]["reason"]


def test_adapter_import_without_run_method_is_not_a_strategy(tmp_path, home):
    p = tmp_path / "helpers.py"
    p.write_text("from qrh.adapters.base import DataView\n\nclass H:\n    def go(self, x):\n        return x.shift(-1)\n")
    assert lint(tmp_path, home, p).exit_code == 0


def test_explicit_glob_lints_any_python_file(tmp_path, home):
    write_config(home, 'strategy_globs: ["signals/*.py"]\n')
    d = tmp_path / "signals"
    d.mkdir()
    p = d / "sig.py"
    p.write_text("def f(x):\n    return x.shift(-1)\n")
    assert lint(tmp_path, home, p).exit_code == 2
    other = tmp_path / "other.py"
    other.write_text("def f(x):\n    return x.shift(-1)\n")
    assert lint(tmp_path, home, other).exit_code == 0


def test_detection_can_be_disabled(tmp_path, home):
    write_config(home, "detect_strategies: false\n")
    assert lint(tmp_path, home, strategy(tmp_path, "        sig = px.shift(-1)")).exit_code == 0


def test_syntax_error_in_strategy_blocks(tmp_path, home):
    p = strategy(tmp_path, "        sig = (")
    out = lint(tmp_path, home, p)
    assert out.exit_code == 2 and "LA000" in out.stderr


def test_unreadable_python_file_fails_closed(tmp_path, home):
    out = lint(tmp_path, home, tmp_path / "gone.py")
    assert out.exit_code == 2 and "cannot read" in out.stderr


def test_relative_path_resolves_against_payload_cwd(tmp_path, home):
    strategy(tmp_path, "        sig = px.shift(-1)", name="rel.py")
    data = edit_payload("PostToolUse", "rel.py")
    data["cwd"] = str(tmp_path)
    assert run("lookahead-lint", data, home).exit_code == 2


def test_wrong_event_or_tool_is_ignored(tmp_path, home):
    p = strategy(tmp_path, "        sig = px.shift(-1)")
    assert run("lookahead-lint", edit_payload("PreToolUse", str(p)), home).exit_code == 0
    assert run("lookahead-lint", payload("PostToolUse", tool_name="Bash", tool_input={"command": "x"}), home).exit_code == 0


def test_malformed_config_blocks(tmp_path, home):
    write_config(home, "strategy_globs: [unclosed")
    p = strategy(tmp_path, "        sig = px.shift(1)")
    out = lint(tmp_path, home, p)
    assert out.exit_code == 2 and "config.yaml" in out.stderr


def test_human_approval_allows_one_blocked_edit(tmp_path, home):
    p = strategy(tmp_path, "        sig = px.shift(-1)")
    from hook_helpers import human_approve

    first = lint(tmp_path, home, p)
    assert first.exit_code == 2
    human_approve(home, first.stderr, "label for evaluation, reviewed")
    assert lint(tmp_path, home, p).exit_code == 0
    assert lint(tmp_path, home, p).exit_code == 2
    assert hook_log(home)[1]["override"] is True
