import json

import pytest

from hook_helpers import approval_id, bash_payload, hook_log, human_approve, payload, run, write_config
from qrh.hooks.common import (
    ConfigError,
    Payload,
    PayloadError,
    glob_match,
    load_config,
    parse_payload,
    redact,
    redact_command,
)
from qrh.hooks.runner import HookResult, HookSpec

# Fake secrets are assembled at runtime so no literal looks like a real credential in source.
SK = "sk-" + "a1B2c3D4" * 4
GHP = "ghp_" + "x9Y8z7W6" * 4
GHO = "gho_" + "q1w2e3r4" * 4
GHS = "ghs_" + "t5y6u7i8" * 4
PAT = "github_pat_" + "A1b2C3d4" * 5
SLACK = "xo" + "xb-" + "1234567890-abcdefghij"
AKIA = "AK" + "IA" + "ABCDEFGHIJKLMNOP"
ASIA = "AS" + "IA" + "ABCDEFGHIJKLMNOP"


def test_parse_payload_fields():
    p = parse_payload(json.dumps(payload("UserPromptSubmit", prompt="hello", stop_hook_active=True)))
    assert isinstance(p, Payload)
    assert p.session_id == "sess-1" and p.event == "UserPromptSubmit" and p.prompt == "hello"
    assert p.stop_hook_active is True and p.tool_input == {}


@pytest.mark.parametrize("bad", ["", "not json", "[1, 2]", "null", '"str"'])
def test_parse_payload_rejects_non_objects(bad):
    with pytest.raises(PayloadError):
        parse_payload(bad)


def test_payload_tolerates_wrong_types():
    p = parse_payload(json.dumps({"tool_input": "oops", "session_id": 5, "stop_hook_active": "yes"}))
    assert p.tool_input == {} and p.session_id == "" and p.stop_hook_active is False


@pytest.mark.parametrize("secret", [SK, GHP, GHO, GHS, PAT, SLACK, AKIA, ASIA])
def test_redacts_provider_keys(secret):
    out = redact(f"curl -H x https://h/{secret}/y --flag {secret}")  # gitleaks:allow (fake keys)
    assert secret not in out and "***" in out


def test_redacts_bearer_basic_and_key_values():
    assert "abc123tok" not in redact("curl -H 'Authorization: Bearer abc123tok'")
    assert "dXNlcjpwYXNz" not in redact("Authorization: Basic dXNlcjpwYXNz")
    for text in ("token=hunter2xx", "PASSWORD=hunter2xx", "api_key: hunter2xx", "my_secret='hunter2xx'",
                 "--password hunter2xx", "--token=hunter2xx", "--secret hunter2xx"):
        assert "hunter2xx" not in redact(text), text


def test_redaction_keeps_harmless_text():
    assert redact("uv run qrh run abc12345 --stage S4") == "uv run qrh run abc12345 --stage S4"


def test_command_redacted_then_truncated_to_160():
    out = redact_command("echo " + "x" * 300)
    assert len(out) <= 160 and out.startswith("echo x")
    # a secret that would straddle the cut must not leak its head
    cmd = "a" * 150 + " token=" + "S3CRETVALUE" * 3
    assert "S3CRET" not in redact_command(cmd)
    assert redact_command(f"run {SK}") == "run sk-***"


def test_glob_match_semantics():
    assert glob_match("/w/strategies/a.py", "strategies/*.py", cwd="/w")
    assert glob_match("/w/x/strategies/a.py", "strategies/*.py", cwd="/w")  # relative globs match any depth
    assert not glob_match("/w/strategies/sub/a.py", "strategies/*.py", cwd="/w")
    assert glob_match("/w/strategies/sub/deep/a.py", "strategies/**/*.py", cwd="/w")
    assert glob_match("/abs/x/a.py", "/abs/x/*.py", cwd="/w")
    assert not glob_match("/abs/y/a.py", "/abs/x/*.py", cwd="/w")
    assert glob_match("/w/core/util.py", "core/util.p?", cwd="/w")


def test_config_defaults_when_missing(tmp_path):
    cfg = load_config(tmp_path)
    assert cfg.detect_strategies is True and cfg.strategy_globs == ()
    assert "walk-forward" in cfg.trigger_terms and "alpha" in cfg.trigger_terms
    assert cfg.guard is None and cfg.receipt_required.enabled is True
    assert cfg.receipt_required.abs_tol == 0.01 and cfg.receipt_required.rel_tol == 0.005
    assert cfg.receipt_required.recent_days == 30


FULL = """
strategy_globs: ["strategies/*.py"]
detect_strategies: false
trigger_terms: [alpha, "walk-forward"]
shared_module_globs: ["core/**/*.py"]
backtest_commands:
  - pattern: 'python\\s+bt\\.py'
    datasets: [prices, fx]
  - pattern: 'make bt (?P<trial>\\w+)'
    trial_id: from_args
engine_start_commands:
  - 'engine\\s+start'
guard:
  source_globs: ["src/**/*.py"]
  build_hash_file: build/HASH
  required_datasets: [prices]
  max_dataset_age: {prices: 3D}
  test_receipt_max_age: 1D
  parity: {disabled: "parity tooling not built yet"}
receipt_required:
  enabled: false
  metric_patterns: ["calmar"]
  tolerance: {abs: 0.02, rel: 0.01}
  recent_days: 7
"""


def test_config_full(tmp_path):
    write_config(tmp_path, FULL)
    cfg = load_config(tmp_path)
    assert cfg.strategy_globs == ("strategies/*.py",) and cfg.detect_strategies is False
    assert cfg.trigger_terms == ("alpha", "walk-forward")
    assert cfg.backtest_commands[0].datasets == ("prices", "fx") and not cfg.backtest_commands[0].trial_from_args
    assert cfg.backtest_commands[1].trial_from_args
    assert cfg.engine_start_commands[0].search("engine start --paper")
    g = cfg.guard
    assert g and g.build_hash_file == "build/HASH" and g.max_dataset_age == {"prices": "3D"}
    assert g.parity_disabled_reason == "parity tooling not built yet" and not g.parity_required
    rr = cfg.receipt_required
    assert rr.enabled is False and rr.metric_patterns == ("calmar",) and rr.abs_tol == 0.02 and rr.recent_days == 7


@pytest.mark.parametrize("text", [
    "strategy_globs: [unclosed",                         # yaml error
    "- just\n- a list\n",                                # not a mapping
    "strategy_globs: notalist\n",
    "backtest_commands:\n  - pattern: '(unclosed'\n    datasets: [a]\n",
    "backtest_commands:\n  - pattern: 'x'\n",            # neither datasets nor trial_id
    "backtest_commands:\n  - pattern: 'x'\n    datasets: [a]\n    trial_id: from_args\n",
    "guard:\n  parity: maybe\n",
    "guard:\n  parity: {disabled: short}\n",              # reason must be given (>= 10 chars)
    "guard:\n  max_dataset_age: [1D]\n",
    "receipt_required:\n  recent_days: -1\n",
    "receipt_required:\n  tolerance: {abs: -1}\n",
    "unknown_key: 1\n",
])
def test_config_malformed_raises(tmp_path, text):
    write_config(tmp_path, text)
    with pytest.raises(ConfigError):
        load_config(tmp_path)


# --- dispatcher ---------------------------------------------------------------------------------

def _stub(decision="allow", enforcement=True, exc=None, reason="because"):
    def fn(ctx):
        if exc:
            raise exc
        return HookResult(decision, reason, stdout="ctx text" if decision == "info" else "")

    return {"stub": HookSpec("stub", "PreToolUse", "Bash", 5, enforcement, "unused", run=fn)}


def test_unknown_hook_is_blocked(tmp_path):
    out = run("nope", payload("PreToolUse"), tmp_path)
    assert out.exit_code == 2 and "unknown hook" in out.stderr


def test_allow_logs_one_json_line(tmp_path):
    out = run("stub", bash_payload("ls"), tmp_path, hooks=_stub())
    assert out.exit_code == 0 and out.stderr == ""
    (rec,) = hook_log(tmp_path)
    assert {"ts", "session_id", "hook", "event", "decision", "reason", "trial_id", "duration_ms"} <= set(rec)
    assert rec["hook"] == "stub" and rec["decision"] == "allow" and rec["session_id"] == "sess-1"


def test_block_exit_2_with_stderr_and_approval_instructions(tmp_path):
    out = run("stub", bash_payload("ls"), tmp_path, hooks=_stub("block", reason="nope nope"))
    assert out.exit_code == 2 and "nope nope" in out.stderr and "QRH_OVERRIDE" not in out.stderr
    rid = approval_id(out.stderr)
    assert f'qrh approve {rid} --reason "' in out.stderr and "OWN terminal" in out.stderr
    assert (tmp_path / "approvals" / f"{rid}.json").is_file()
    assert hook_log(tmp_path)[0]["approval_pending"] == rid
    assert hook_log(tmp_path)[0]["decision"] == "block"


def test_info_prints_stdout_exit_0(tmp_path):
    out = run("stub", payload("SessionStart"), tmp_path, hooks=_stub("info", enforcement=False))
    assert out.exit_code == 0 and out.stdout.strip() == "ctx text"


def test_enforcement_exception_blocks_informational_does_not(tmp_path):
    out = run("stub", bash_payload("ls"), tmp_path, hooks=_stub(exc=RuntimeError("boom")))
    assert out.exit_code == 2 and "boom" in out.stderr
    out = run("stub", bash_payload("ls"), tmp_path, hooks=_stub(exc=RuntimeError("boom"), enforcement=False))
    assert out.exit_code == 0 and out.stderr == ""
    assert hook_log(tmp_path)[-1]["decision"] == "error"


def test_bad_payload_blocks_enforcement_only(tmp_path):
    assert run("stub", "{not json", tmp_path, hooks=_stub()).exit_code == 2
    assert run("stub", "{not json", tmp_path, hooks=_stub(enforcement=False)).exit_code == 0


def test_malformed_config_blocks_enforcement_skips_informational(tmp_path):
    write_config(tmp_path, "strategy_globs: [unclosed")
    out = run("stub", bash_payload("ls"), tmp_path, hooks=_stub())
    assert out.exit_code == 2 and "config.yaml" in out.stderr
    out = run("stub", bash_payload("ls"), tmp_path, hooks=_stub(enforcement=False))
    assert out.exit_code == 0 and out.stderr == ""
    rec = hook_log(tmp_path)[-1]
    assert rec["decision"] == "info" and "config" in rec["reason"]


def test_human_approval_allows_once_and_is_recorded(tmp_path):
    blocked = _stub("block")
    first = run("stub", bash_payload("ls"), tmp_path, hooks=blocked)
    rid = human_approve(tmp_path, first.stderr, "vendor outage, approved by owner")
    out = run("stub", bash_payload("ls"), tmp_path, hooks=blocked)
    assert out.exit_code == 0
    rec = hook_log(tmp_path)[-1]
    assert rec["decision"] == "allow" and rec["override"] is True and rec["approval_id"] == rid
    assert "vendor outage" in rec["reason"]
    from qrh.memory.registry import Registry

    reg = Registry(tmp_path / "registry.sqlite3")
    ov = reg.list_hook_overrides(since="1970-01-01T00:00:00+00:00")
    reg.close()
    assert ov[0]["hook"] == "stub" and ov[0]["session_id"] == "sess-1" and "vendor outage" in ov[0]["reason"]
    assert ov[0]["approval_id"] == rid
    # one blocked action only: the same retry is blocked again, with a fresh approval id
    again = run("stub", bash_payload("ls"), tmp_path, hooks=blocked)
    assert again.exit_code == 2 and approval_id(again.stderr) != rid


def test_approval_covers_only_the_identical_action(tmp_path):
    blocked = _stub("block")
    first = run("stub", bash_payload("ls"), tmp_path, hooks=blocked)
    human_approve(tmp_path, first.stderr)
    assert run("stub", bash_payload("ls -l"), tmp_path, hooks=blocked).exit_code == 2
    assert run("stub", bash_payload("ls"), tmp_path, hooks=blocked).exit_code == 0


def test_environment_variable_no_longer_overrides(tmp_path):
    env = {"QRH_OVERRIDE": "stub:vendor outage, approved by owner"}
    out = run("stub", bash_payload("ls"), tmp_path, env=env, hooks=_stub("block"))
    assert out.exit_code == 2 and "QRH_OVERRIDE" not in out.stderr


def test_unapproved_retry_stays_blocked_and_reuses_the_pending_id(tmp_path):
    a = run("stub", bash_payload("ls"), tmp_path, hooks=_stub("block"))
    b = run("stub", bash_payload("ls"), tmp_path, hooks=_stub("block"))
    assert a.exit_code == b.exit_code == 2 and approval_id(a.stderr) == approval_id(b.stderr)


def test_non_approvable_hook_offers_no_approval(tmp_path):
    from dataclasses import replace

    hooks = {"stub": replace(_stub("block")["stub"], approvable=False)}
    out = run("stub", bash_payload("ls"), tmp_path, hooks=hooks)
    assert out.exit_code == 2 and "qrh approve" not in out.stderr and "cannot be overridden" in out.stderr
    assert not (tmp_path / "approvals").exists()


def test_stop_style_hooks_supply_their_own_fingerprint(tmp_path):
    def fn(ctx):
        return HookResult("block", "claims unsupported", action_parts=("stub", "2 claims"))

    hooks = {"stub": HookSpec("stub", "Stop", None, 5, True, "unused", run=fn)}
    stop = payload("Stop", stop_hook_active=False)
    first = run("stub", stop, tmp_path, hooks=hooks)
    assert first.exit_code == 2
    human_approve(tmp_path, first.stderr)
    assert run("stub", stop, tmp_path, hooks=hooks).exit_code == 0


def test_blocks_without_a_fingerprint_offer_no_approval(tmp_path):
    out = run("stub", payload("Stop"), tmp_path, hooks={"stub": HookSpec(
        "stub", "Stop", None, 5, True, "unused", run=lambda ctx: HookResult("block", "x"))})
    assert out.exit_code == 2 and "qrh approve" not in out.stderr


def test_on_override_callback_runs_when_a_token_is_consumed(tmp_path):
    seen = []

    def fn(ctx):
        return HookResult("block", "x", on_override=lambda: seen.append(1))

    hooks = {"stub": HookSpec("stub", "PreToolUse", "Bash", 5, True, "unused", run=fn)}
    first = run("stub", bash_payload("ls"), tmp_path, hooks=hooks)
    assert seen == []
    human_approve(tmp_path, first.stderr)
    assert run("stub", bash_payload("ls"), tmp_path, hooks=hooks).exit_code == 0 and seen == [1]


def test_approval_with_unreadable_registry_still_logged(tmp_path):
    first = run("stub", bash_payload("ls"), tmp_path, hooks=_stub("block"))
    human_approve(tmp_path, first.stderr)
    (tmp_path / "registry.sqlite3").write_bytes(b"this is not a sqlite database at all" * 20)
    out = run("stub", bash_payload("ls"), tmp_path, hooks=_stub("block"))
    assert out.exit_code == 0
    rec = hook_log(tmp_path)[-1]
    assert rec["override"] is True and rec["override_recorded_in_registry"] is False


def test_malformed_public_key_keeps_the_block_and_says_why(tmp_path):
    first = run("stub", bash_payload("ls"), tmp_path, hooks=_stub("block"))
    human_approve(tmp_path, first.stderr)
    (tmp_path / "keys" / "approval.pub").write_text("{not json")
    out = run("stub", bash_payload("ls"), tmp_path, hooks=_stub("block"))
    assert out.exit_code == 2 and "public key" in out.stderr


def test_tampered_token_keeps_the_block_and_says_why(tmp_path):
    import json as _json

    first = run("stub", bash_payload("ls"), tmp_path, hooks=_stub("block"))
    rid = human_approve(tmp_path, first.stderr)
    path = tmp_path / "approvals" / f"{rid}.json"
    rec = _json.loads(path.read_text())
    rec["approval_reason"] = "forged reason that was not typed by a human"
    path.write_text(_json.dumps(rec))
    out = run("stub", bash_payload("ls"), tmp_path, hooks=_stub("block"))
    assert out.exit_code == 2 and "signature" in out.stderr


def test_a_config_error_block_can_also_be_approved(tmp_path):
    write_config(tmp_path, "strategy_globs: [unclosed")
    first = run("stub", bash_payload("ls"), tmp_path, hooks=_stub())
    assert first.exit_code == 2
    human_approve(tmp_path, first.stderr)
    assert run("stub", bash_payload("ls"), tmp_path, hooks=_stub()).exit_code == 0


def test_needs_config_false_ignores_a_broken_config(tmp_path):
    from dataclasses import replace

    write_config(tmp_path, "strategy_globs: [unclosed")
    hooks = {"stub": replace(_stub()["stub"], needs_config=False)}
    assert run("stub", bash_payload("ls"), tmp_path, hooks=hooks).exit_code == 0


def test_log_never_contains_prompt_or_full_command(tmp_path):
    secret_cmd = "echo " + "y" * 400 + " token=hunter2xx"
    run("stub", bash_payload(secret_cmd, ), tmp_path, hooks=_stub())
    run("stub", payload("UserPromptSubmit", prompt="my private prompt text"), tmp_path, hooks=_stub())
    raw = (tmp_path / "logs" / "hooks.jsonl").read_text()
    assert "hunter2xx" not in raw and "my private prompt text" not in raw
    assert all(len(json.loads(line).get("command", "")) <= 160 for line in raw.splitlines())


def _stop_stub(decision):
    return {"stub": HookSpec("stub", "Stop", None, 5, True, "unused",
                             run=lambda ctx: HookResult(decision, "unsupported claim", action_parts=("stub", "c")))}


def test_a_second_stop_is_still_checked_and_blocked(tmp_path):
    hooks = _stop_stub("block")
    assert run("stub", payload("Stop"), tmp_path, hooks=hooks).exit_code == 2
    out = run("stub", payload("Stop", stop_hook_active=True), tmp_path, hooks=hooks)
    assert out.exit_code == 2 and "unsupported claim" in out.stderr


def test_after_the_cap_the_block_says_the_user_must_decide(tmp_path):
    from qrh.hooks.runner import STOP_BLOCK_CAP

    assert STOP_BLOCK_CAP == 3
    hooks = _stop_stub("block")
    outs = [run("stub", payload("Stop", stop_hook_active=i > 0), tmp_path, hooks=hooks) for i in range(STOP_BLOCK_CAP + 1)]
    assert all(o.exit_code == 2 for o in outs)
    assert "cannot be verified" not in outs[0].stderr
    last = outs[-1].stderr
    assert "cannot be verified" in last and "user must decide" in last and str(STOP_BLOCK_CAP) in last
    assert "unsupported claim" in last


def test_a_clean_second_stop_passes_and_resets_the_count(tmp_path):
    from qrh.hooks.runner import STOP_BLOCK_CAP

    run("stub", payload("Stop"), tmp_path, hooks=_stop_stub("block"))
    assert run("stub", payload("Stop", stop_hook_active=True), tmp_path, hooks=_stop_stub("allow")).exit_code == 0
    for i in range(STOP_BLOCK_CAP - 1):  # the count restarted: still below the cap
        out = run("stub", payload("Stop", stop_hook_active=i > 0), tmp_path, hooks=_stop_stub("block"))
    assert "cannot be verified" not in out.stderr


def test_the_stop_block_count_is_per_session(tmp_path):
    from qrh.hooks.runner import STOP_BLOCK_CAP

    hooks = _stop_stub("block")
    for _ in range(STOP_BLOCK_CAP):
        run("stub", payload("Stop", session_id="a"), tmp_path, hooks=hooks)
    out = run("stub", payload("Stop", session_id="b"), tmp_path, hooks=hooks)
    assert "cannot be verified" not in out.stderr
