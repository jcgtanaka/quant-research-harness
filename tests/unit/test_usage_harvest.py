import json
from pathlib import Path

import pytest

from qrh.usage.harvest import harvest, load_archive
from qrh.usage.settings import SettingsError
from usage_helpers import SECRET, SECRET_PATH, T, assistant, build_claude, flat, write_jsonl, write_settings


@pytest.fixture
def env(tmp_path):
    claude = build_claude(tmp_path / "claude")
    home = tmp_path / "home"
    write_settings(home)
    return claude, home


def test_session_metrics(env):
    claude, home = env
    harvest(claude, home)
    s = load_archive(home)["s1"]
    assert s["project_label"] == "projA"
    assert s["models"]["model-big"]["input"] == 2310 and s["models"]["model-big"]["requests"] == 5
    assert s["models_raw"]["model-big"]["input"] == 3010  # r1 and r3 repeat once per content block
    assert s["models"]["model-big"]["cache_write_1h"] == 400 and s["models"]["model-big"]["cache_write_5m"] == 1150
    assert s["peak_context"] == 1010
    assert s["hit_pct"] == pytest.approx(100 * 1150 / 5010)
    assert s["cold_cache_events"] == 2 and s["compactions"] == 1 and s["malformed_lines"] == 1
    assert s["tools"] == {"Edit": 1, "Grep": 1, "Read": 1}
    assert s["start"].startswith("2026-03-01T10:00") and s["end"].startswith("2026-03-01T10:07")
    assert s["duration_s"] == 7 * 60
    assert s["subagents"]["agent-a1"]["models"]["model-small"]["input"] == 20
    assert s["subagents"]["_inline"]["models"]["model-small"]["input"] == 7
    assert s["subagents"]["agent-a1"]["cost_state"] is False


def test_cost_sources(env):
    claude, home = env
    harvest(claude, home)
    a = load_archive(home)
    assert (a["s1"]["cost_source"], a["s1"]["cost_usd"]) == ("harness", 3.5)
    assert a["s1"]["cost_by_model"] == {"model-big": 3.0, "model-small": 0.5}
    assert a["s1"]["harness_usage"]["model-big"]["cache_creation"] == 1550
    assert a["s2"]["cost_source"] == "estimate" and a["s2"]["cost_usd"] == pytest.approx(10 + 5 + 1)
    assert a["s3"]["cost_source"] == "none" and a["s3"]["cost_usd"] is None and "mystery" in a["s3"]["cost_reason"]


def test_no_prices_file_means_no_estimate_with_reason(env):
    claude, home = env
    (home / "usage" / "prices.json").unlink()
    harvest(claude, home)
    s2 = load_archive(home)["s2"]
    assert s2["cost_source"] == "none" and "prices.json" in s2["cost_reason"]


def test_missing_thresholds_fail_closed_before_writing(env):
    claude, home = env
    (home / "usage" / "thresholds.json").unlink()
    with pytest.raises(SettingsError, match="thresholds.json"):
        harvest(claude, home)
    assert not (home / "usage" / "sessions.json").exists()


def test_archive_survives_transcript_sweep_and_newer_harvest_replaces(env):
    claude, home = env
    harvest(claude, home)
    (claude / "projects" / "-work-projA" / "s3.jsonl").unlink()
    p = claude / "projects" / "-work-projA" / "s2.jsonl"
    write_jsonl(p, flat(assistant("e1", "model-big", T.format(10), inp=2_000_000)))
    summary = harvest(claude, home)
    a = load_archive(home)
    assert "s3" in a and summary.archived_only == 1
    assert (summary.new, summary.updated) == (0, 2)
    assert a["s2"]["models"]["model-big"]["input"] == 2_000_000
    assert a["s2"]["cost_usd"] == pytest.approx(20 + 1)


def test_harvest_is_idempotent_and_deterministic(env):
    claude, home = env
    harvest(claude, home)
    first = (home / "usage" / "sessions.json").read_bytes()
    harvest(claude, home)
    assert (home / "usage" / "sessions.json").read_bytes() == first


def test_attribution_and_hook_stats_are_stored(env):
    claude, home = env
    logs = home / "logs"
    logs.mkdir()
    (logs / "session_trials.jsonl").write_text(json.dumps(
        {"session_id": "s1", "trial_id": "t1", "stage": "S3", "kind": "run", "ts": "2026-03-01T10:03:00+00:00"}) + "\n")
    (logs / "hooks.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"session_id": "s1", "hook": "lookahead-lint", "decision": "block"},
        {"session_id": "s1", "hook": "data-gate", "decision": "block"}]) + "\n")
    harvest(claude, home)
    s = load_archive(home)["s1"]
    by = {(t["trial_id"], t["stage"]): t for t in s["trials"]}
    assert by[("unattributed", None)]["models"]["model-big"]["requests"] == 2
    assert by[("t1", "S3")]["models"]["model-big"]["requests"] == 3
    assert sum(t["cost_usd"] for t in s["trials"]) == pytest.approx(3.5)
    assert s["hooks"]["lint_blocks"] == 1 and s["hooks"]["gate_blocks"] == 1


def test_label_for_home_directory_and_most_frequent_cwd(tmp_path):
    claude = tmp_path / "c"
    write_jsonl(claude / "projects" / "p" / "h.jsonl", flat(
        assistant("a", "model-big", T.format(1), inp=1, cwd=str(Path.home())),
        assistant("b", "model-big", T.format(2), inp=1, cwd=str(Path.home()))))
    write_jsonl(claude / "projects" / "p" / "m.jsonl", flat(
        assistant("a", "model-big", T.format(1), inp=1, cwd="/x/rare"),
        assistant("b", "model-big", T.format(2), inp=1, cwd="/y/common"),
        assistant("c", "model-big", T.format(3), inp=1, cwd="/y/common")))
    home = tmp_path / "home"
    write_settings(home)
    harvest(claude, home)
    a = load_archive(home)
    assert a["h"]["project_label"] == "home" and a["m"]["project_label"] == "common"


def test_archive_holds_no_text_or_paths(env):
    claude, home = env
    harvest(claude, home)
    blob = (home / "usage" / "sessions.json").read_text()
    for forbidden in (SECRET, SECRET_PATH, "/work/", "-work-projA", str(claude)):
        assert forbidden not in blob


def test_empty_claude_dir_is_fine(tmp_path):
    write_settings(tmp_path / "home")
    summary = harvest(tmp_path / "nothing", tmp_path / "home")
    assert summary.scanned == 0
