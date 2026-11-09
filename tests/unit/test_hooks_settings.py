import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qrh.cli import app
from qrh.hooks.settings import SettingsError, desired_settings, merge_settings
from qrh.hooks.specs import HOOKS

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[2]


def commands(settings, event, matcher=None):
    out = []
    for group in settings["hooks"].get(event, []):
        if group.get("matcher") == matcher:
            out += [h["command"] for h in group["hooks"]]
    return out


def test_every_hook_registered_with_event_matcher_timeout():
    s = desired_settings()
    for spec in HOOKS.values():
        groups = [g for g in s["hooks"][spec.event] if g.get("matcher") == spec.matcher]
        assert len(groups) == 1, spec.name
        entry = [h for h in groups[0]["hooks"] if h["command"] == f"qrh hook {spec.name}"]
        assert len(entry) == 1 and entry[0]["type"] == "command" and entry[0]["timeout"] == spec.timeout
    assert commands(s, "PreToolUse", "Bash") == ["qrh hook data-gate", "qrh hook pre-restart-guard"]
    assert commands(s, "PostToolUse", "Edit|Write|MultiEdit") == ["qrh hook lookahead-lint"]
    assert commands(s, "Stop") == ["qrh hook claim-check", "qrh hook receipt-required"]
    assert commands(s, "PreToolUse", "Bash|Read|Edit|Write|MultiEdit|NotebookEdit|Grep|Glob|WebFetch|mcp__.*") == ["qrh hook approval-guard"]
    assert commands(s, "PreToolUse", "Bash|Write|Edit|MultiEdit") == ["qrh hook publish-guard"]
    assert commands(s, "PreToolUse") == ["qrh hook runaway-guard"]
    assert commands(s, "PostToolUseFailure") == ["qrh hook runaway-failure"]
    assert commands(s, "UserPromptSubmit") == ["qrh hook prior-attempts"]
    assert commands(s, "SessionStart", "startup|resume|clear|compact") == ["qrh hook session-context"]
    assert "matcher" not in s["hooks"]["Stop"][0]


def test_example_file_matches_generated_settings():
    example = json.loads((ROOT / "hooks" / "claude_code" / "settings.example.json").read_text())
    assert example == desired_settings()


EXISTING = {
    "model": "some-model",
    "permissions": {"allow": ["Bash(ls:*)"]},
    "hooks": {
        "PreToolUse": [
            {"matcher": "Bash", "hooks": [{"type": "command", "command": "other-tool check", "timeout": 3}]},
            {"matcher": "Write", "hooks": [{"type": "command", "command": "writer-guard"}]},
        ],
        "Notification": [{"hooks": [{"type": "command", "command": "notify-me"}]}],
    },
}


def test_merge_keeps_everything_else_and_adds_ours():
    before = copy.deepcopy(EXISTING)
    merged = merge_settings(EXISTING)
    assert EXISTING == before  # input not mutated
    assert merged["model"] == "some-model" and merged["permissions"] == EXISTING["permissions"]
    assert merged["hooks"]["Notification"] == EXISTING["hooks"]["Notification"]
    bash = commands(merged, "PreToolUse", "Bash")
    assert bash[0] == "other-tool check" and "qrh hook data-gate" in bash and "qrh hook pre-restart-guard" in bash
    assert commands(merged, "PreToolUse", "Write") == ["writer-guard"]
    assert commands(merged, "PreToolUse", "Edit|Write|MultiEdit") == ["qrh hook impact-check"]


def test_merge_is_idempotent_and_never_duplicates():
    once = merge_settings(EXISTING)
    assert merge_settings(once) == once
    assert merge_settings(merge_settings(once)) == once
    assert merge_settings({}) == desired_settings()
    all_cmds = [h["command"] for ev in once["hooks"].values() for g in ev for h in g["hooks"]]
    assert len(all_cmds) == len(set(all_cmds))


def test_merge_refreshes_our_stale_entry_in_place():
    stale = {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "qrh hook receipt-required", "timeout": 1}]}]}}
    merged = merge_settings(stale)
    entry = next(h for h in merged["hooks"]["Stop"][0]["hooks"] if h["command"] == "qrh hook receipt-required")
    assert len(merged["hooks"]["Stop"][0]["hooks"]) == 2
    assert entry["timeout"] == HOOKS["receipt-required"].timeout and len(merged["hooks"]["Stop"]) == 1


@pytest.mark.parametrize("bad", [
    {"hooks": []}, {"hooks": {"Stop": {}}}, {"hooks": {"Stop": ["x"]}}, {"hooks": {"Stop": [{"hooks": "x"}]}}, [],
])
def test_merge_rejects_unexpected_shapes(bad):
    with pytest.raises(SettingsError):
        merge_settings(bad)


def test_cli_print_without_settings_reads_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    res = runner.invoke(app, ["hooks", "print-settings"])
    assert res.exit_code == 0
    assert json.loads(res.stdout) == desired_settings()
    assert not (tmp_path / ".claude").exists()


@pytest.mark.parametrize("scope", ["user", "project"])
def test_cli_scopes(scope):
    res = runner.invoke(app, ["hooks", "print-settings", "--scope", scope])
    assert res.exit_code == 0 and json.loads(res.stdout) == desired_settings()
    assert scope in res.stderr


def test_cli_bad_scope():
    assert runner.invoke(app, ["hooks", "print-settings", "--scope", "galaxy"]).exit_code == 2


def test_cli_print_with_settings_does_not_write(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps(EXISTING))
    res = runner.invoke(app, ["hooks", "print-settings", "--settings", str(f)])
    assert res.exit_code == 0 and json.loads(res.stdout) == merge_settings(EXISTING)
    assert json.loads(f.read_text()) == EXISTING
    assert list(tmp_path.iterdir()) == [f]


def test_cli_write_makes_timestamped_backup_then_is_idempotent(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps(EXISTING))
    res = runner.invoke(app, ["hooks", "print-settings", "--settings", str(f), "--write"])
    assert res.exit_code == 0, res.output
    backups = [p for p in tmp_path.iterdir() if p.name.startswith("settings.json.bak-")]
    assert len(backups) == 1 and json.loads(backups[0].read_text()) == EXISTING
    assert json.loads(f.read_text()) == merge_settings(EXISTING)
    res = runner.invoke(app, ["hooks", "print-settings", "--settings", str(f), "--write"])
    assert res.exit_code == 0 and "up to date" in res.output
    assert len([p for p in tmp_path.iterdir() if p.name.startswith("settings.json.bak-")]) == 1


def test_cli_write_creates_missing_file_without_backup(tmp_path):
    f = tmp_path / "new" / "settings.json"
    res = runner.invoke(app, ["hooks", "print-settings", "--settings", str(f), "--write"])
    assert res.exit_code == 0 and json.loads(f.read_text()) == desired_settings()
    assert [p.name for p in f.parent.iterdir()] == ["settings.json"]


def test_cli_write_requires_explicit_settings_path():
    res = runner.invoke(app, ["hooks", "print-settings", "--write"])
    assert res.exit_code == 2 and "--settings" in res.output


def test_cli_invalid_existing_json_is_refused_and_untouched(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text("{ not json")
    res = runner.invoke(app, ["hooks", "print-settings", "--settings", str(f), "--write"])
    assert res.exit_code == 2 and f.read_text() == "{ not json"
    assert list(tmp_path.iterdir()) == [f]


def test_merge_moves_our_entry_from_an_outdated_matcher_and_stays_idempotent():
    old = "Bash|Read|Edit|Write|MultiEdit|NotebookEdit|Grep|Glob|mcp__.*"  # approval-guard before WebFetch was added
    stale = {"hooks": {"PreToolUse": [
        {"matcher": old, "hooks": [{"type": "command", "command": "qrh hook approval-guard", "timeout": 10},
                                   {"type": "command", "command": "other-tool check"}]},
        {"matcher": "Glob", "hooks": [{"type": "command", "command": "qrh hook approval-guard", "timeout": 10}]},
    ]}}
    merged = merge_settings(stale)
    assert commands(merged, "PreToolUse", old) == ["other-tool check"]  # foreign entries stay, ours leaves
    assert "Glob" not in [g.get("matcher") for g in merged["hooks"]["PreToolUse"]]  # emptied group is dropped
    assert commands(merged, "PreToolUse", HOOKS["approval-guard"].matcher) == ["qrh hook approval-guard"]
    assert merge_settings(merged) == merged
    count = sum(h["command"] == "qrh hook approval-guard" for g in merged["hooks"]["PreToolUse"] for h in g["hooks"])
    assert count == 1
