import json
import os
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from hook_helpers import DB, edit_payload, hook_log, payload, register_dataset, run, seed_trial, write_config
from qrh.hooks.impact_check import run_codegraph
from qrh.memory.registry import Registry
from qrh.pipeline.receipt import Verdict

NOW = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


def broken_registry(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / DB).write_bytes(b"garbage, not sqlite " * 50)


# --- session-context ----------------------------------------------------------------------------

def ctx_run(home, source="startup", env=None):
    return run("session-context", payload("SessionStart", source=source), home, env=env, now=lambda: NOW)


def test_session_context_empty_registry_only_says_the_guards_are_off(home):
    out = ctx_run(home)
    assert out.exit_code == 0 and out.stderr == ""
    lines = out.stdout.splitlines()
    assert lines[0] == "qrh context:" and lines[-4:] == [
        "Publish guard:", "- disabled: no `publish_guard` section in config.yaml (no protected repos: pushes, hook "
        "changes and publication commands are not gated)",
        "Runaway guard:", "- disabled: no `runaway` section in config.yaml (no loop, failure or spend limits)"]
    assert all("qrh approve --init" in x or "qrh config sign" in x or x == "Security setup:" for x in lines[1:-4])


PUBLISH_CFG = "publish_guard:\n  protected_repos: [/tmp/some/repo]\n"


def test_session_context_says_nothing_when_both_guards_are_on_and_the_only_news_is_the_key(home, tmp_path):
    write_config(home, "runaway:\n  repeat_limit: 3\n" + PUBLISH_CFG)
    _secure(home, tmp_path)  # key and signed baseline in place, nothing to warn about
    out = ctx_run(home)
    lines = out.stdout.splitlines()
    assert out.exit_code == 0 and lines[0] == "qrh context:" and lines[1] == "Security setup:"
    assert len(lines) == 3 and "fingerprint" in lines[2]


def test_session_context_prints_the_key_fingerprint_so_a_replaced_key_is_noticeable(home, tmp_path):
    from qrh import signing

    write_config(home, "runaway:\n  repeat_limit: 3\n" + PUBLISH_CFG)
    _secure(home, tmp_path)
    kid = signing.load_public(home).key_id
    first = ctx_run(home).stdout
    assert f"approval key fingerprint {kid}" in first
    assert "note" in first.lower() and "agent cannot write" in first
    for f in (home / "keys").iterdir():
        f.unlink()
    from hook_helpers import ensure_test_keys

    ensure_test_keys(home)  # a different key now
    second = ctx_run(home).stdout
    assert f"fingerprint {signing.load_public(home).key_id}" in second and kid not in second.replace("changed", "")


def test_session_context_has_no_fingerprint_line_without_a_key(home):
    assert "fingerprint" not in ctx_run(home).stdout


def test_publish_guard_line_disappears_once_the_section_exists(home, tmp_path):
    write_config(home, "runaway:\n  repeat_limit: 3\n" + PUBLISH_CFG)
    assert "Publish guard" not in ctx_run(home).stdout


# --- editable install / writable code -----------------------------------------------------------------

def _pkg(root, parts):
    d = root.joinpath(*parts)
    d.mkdir(parents=True)
    (d / "signing.py").write_text("x = 1\n")
    return d


def test_code_exposure_names_a_writable_source_checkout(tmp_path):
    from qrh.hooks.session_context import code_exposure

    text = code_exposure(_pkg(tmp_path, ["checkout", "src", "qrh"]))
    assert "source checkout" in text and "editable" in text and "could edit" in text
    assert "Deployment hardening" in text


def test_code_exposure_names_a_user_writable_install_too(tmp_path):
    from qrh.hooks.session_context import code_exposure

    text = code_exposure(_pkg(tmp_path, ["venv", "lib", "python3.12", "site-packages", "qrh"]))
    assert "install directory" in text and "could edit" in text


def test_code_exposure_is_silent_when_the_user_cannot_write_the_code(tmp_path):
    import os
    import stat

    from qrh.hooks.session_context import code_exposure

    if os.geteuid() == 0:
        pytest.skip("root can write everything")
    pkg = _pkg(tmp_path, ["ro", "site-packages", "qrh"])
    (pkg / "signing.py").chmod(0o444)
    for d in (pkg, pkg.parent):
        d.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        assert code_exposure(pkg) is None
    finally:
        for d in (pkg.parent, pkg):
            d.chmod(0o755)


def test_session_context_prints_the_code_exposure_line(home, monkeypatch):
    from qrh.hooks import session_context

    monkeypatch.setattr(session_context, "_exposure_line", lambda: "- qrh runs from a source checkout that is writable")
    assert "- qrh runs from a source checkout that is writable" in ctx_run(home).stdout


def test_session_context_lists_open_trials_with_latest_verdict(home):
    seed_trial(home, "open00000001", verdicts={"S0": Verdict.PASS, "S1": Verdict.PASS, "S2": Verdict.FAIL})
    seed_trial(home, "done00000002", verdicts={"S0": Verdict.PASS, "S7": Verdict.PASS})
    seed_trial(home, "fresh0000003")
    text = ctx_run(home).stdout
    assert "open000" in text and "S2 FAIL" in text and "fresh000" in text
    assert "done0000" not in text  # a trial whose final verdict is PASS is not open


def test_session_context_caps_trials_at_ten_and_total_lines_at_25(home, tmp_path):
    for i in range(40):
        seed_trial(home, f"trial{i:07d}", verdicts={"S0": Verdict.PASS})
    for i in range(12):
        register_dataset(home, tmp_path, f"ds{i}", status="BLOCKED")
    reg = Registry(home / DB)
    for i in range(12):
        reg.log_hook_override("data-gate", f"reason number {i} is long enough", "s1")
    reg.close()
    text = ctx_run(home).stdout
    assert len(text.strip().splitlines()) <= 25
    assert sum(1 for line in text.splitlines() if line.strip().startswith("- trial")) <= 10


def test_session_context_blocked_datasets(home, tmp_path):
    register_dataset(home, tmp_path, "prices", status="BLOCKED")
    register_dataset(home, tmp_path, "fx", status="PASS")
    text = ctx_run(home).stdout
    assert "prices" in text and "BLOCKED" in text and "fx" not in text


def test_session_context_overrides_last_seven_days_only(home):
    seed_trial(home, "t0000000001")
    reg = Registry(home / DB)
    reg.log_hook_override("data-gate", "recent hook override reason", "sess-1")
    reg.log_override("t0000000001", "S2", "recent trial override reason", "me")
    reg.log_hook_override("pre-restart-guard", "old hook override reason", "sess-0")
    old = (NOW - timedelta(days=9)).isoformat()
    recent = (NOW - timedelta(days=2)).isoformat()
    reg.conn.execute("UPDATE overrides SET created_at=?", (recent,))
    reg.conn.execute("UPDATE overrides SET created_at=? WHERE reason LIKE 'old%'", (old,))
    reg.close()
    text = ctx_run(home).stdout
    assert "recent hook override reason" in text and "recent trial override reason" in text
    assert "old hook override reason" not in text and "data-gate" in text


def test_session_context_report_errors(home):
    seed_trial(home, "t0000000001", verdicts={"S0": Verdict.PASS, "S7": Verdict.PASS})
    d = home / "reports" / "t0000000001"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(
        {"files": [], "errors": [{"stage": "S4", "error": "RuntimeError: matplotlib missing"}]}))
    text = ctx_run(home).stdout
    assert "S4" in text and "matplotlib missing" in text


@pytest.mark.parametrize("source", ["startup", "resume", "clear", "compact"])
def test_session_context_for_every_source(home, source):
    seed_trial(home, "t0000000001")
    assert "t0000000" in ctx_run(home, source).stdout


def test_session_context_never_breaks_the_session(home):
    broken_registry(home)
    out = ctx_run(home)
    assert out.exit_code == 0 and out.stderr == "" and "Datasets" not in out.stdout
    assert any(r["decision"] == "error" for r in hook_log(home))


def test_session_context_malformed_config_skips_with_a_note(home):
    write_config(home, "trigger_terms: [unclosed")
    out = ctx_run(home)
    assert out.exit_code == 0 and "config.yaml" in out.stdout


# --- prior-attempts -----------------------------------------------------------------------------

def prompt_run(home, prompt, env=None):
    return run("prior-attempts", payload("UserPromptSubmit", prompt=prompt), home, env=env)


@pytest.fixture
def seeded(home):
    seed_trial(home, "aaaa1111bbbb", text="Momentum persists after volatility regime shifts", family="momentum",
               verdicts={"S0": Verdict.PASS, "S5": Verdict.FAIL})
    seed_trial(home, "cccc2222dddd", text="Mean reversion of spreads in a quiet regime", family="reversion")
    for i in range(4):
        seed_trial(home, f"eeee{i}333ffff", text=f"Calendar effect number {i} around holidays", family="seasonal")
    return home


def test_prior_attempts_lists_matches_with_verdict_family_text(seeded):
    out = prompt_run(seeded, "Let me test a new momentum strategy for volatility regimes")
    assert out.exit_code == 0
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("- ")]
    assert lines[0].startswith("- aaaa1111") and "FAIL" in lines[0] and "momentum" in lines[0]
    assert "Momentum persists" in lines[0]
    assert len(lines) <= 3


def test_prior_attempts_top_three_only(seeded):
    out = prompt_run(seeded, "backtest a calendar effect holidays strategy")
    assert len([ln for ln in out.stdout.splitlines() if ln.startswith("- ")]) == 3


def test_prior_attempts_requires_a_trigger_term(seeded):
    out = prompt_run(seeded, "please rename momentum.py to mom.py")
    assert out.stdout == ""


def test_prior_attempts_trigger_is_case_insensitive_and_does_not_touch_registry_otherwise(home):
    broken_registry(home)
    assert prompt_run(home, "fix the typo in README").exit_code == 0       # no trigger: registry never opened
    assert hook_log(home)[-1]["decision"] != "error"
    out = prompt_run(home, "WALK-FORWARD results?")
    assert out.exit_code == 0 and out.stdout == "" and hook_log(home)[-1]["decision"] == "error"


def test_prior_attempts_custom_trigger_terms(seeded):
    write_config(seeded, "trigger_terms: [holiday]\n")
    assert prompt_run(seeded, "momentum strategy").stdout == ""
    assert "Calendar effect" in prompt_run(seeded, "holiday calendar").stdout


def test_prior_attempts_no_hits_prints_nothing(seeded):
    assert prompt_run(seeded, "backtest zzzqqq").stdout == ""


def test_prior_attempts_hostile_prompt(seeded):
    out = prompt_run(seeded, 'strategy "; DROP TABLE hypotheses; -- NEAR( AND OR momentum')
    assert out.exit_code == 0


def test_prior_attempts_never_logs_the_prompt(seeded):
    prompt_run(seeded, "my confidential momentum strategy idea")
    assert "confidential" not in (seeded / "logs" / "hooks.jsonl").read_text()


# --- impact-check -------------------------------------------------------------------------------

SHARED = 'shared_module_globs: ["core/**/*.py"]\n'


def impact(home, path, runner, event="PreToolUse", tool="Edit"):
    return run("impact-check", edit_payload(event, path, tool=tool), home, codegraph=runner)


def test_impact_outside_shared_globs_does_not_call_codegraph(home):
    write_config(home, SHARED)
    calls = []
    out = impact(home, "/w/other/x.py", lambda f, cwd: calls.append(f))
    assert out.exit_code == 0 and calls == []


def test_impact_warns_with_counts_and_tests_in_log_only(home):
    write_config(home, SHARED)
    out = impact(home, "/w/core/util.py", lambda f, cwd: (0, "tests/test_a.py\ntests/test_b.py\nsrc/pipeline.py\n"))
    assert out.exit_code == 0 and out.stdout == "" and out.stderr == ""
    rec = hook_log(home)[-1]
    assert rec["decision"] == "warn" and "3 affected" in rec["reason"]
    assert rec["affected_tests"] == ["tests/test_a.py", "tests/test_b.py"]


def test_impact_unavailable_is_logged_and_exit_0(home):
    write_config(home, SHARED)
    for result in (None, (1, "✗ CodeGraph not initialized in /w"), (2, "")):
        out = impact(home, "/w/core/util.py", lambda f, cwd, r=result: r)
        assert out.exit_code == 0
        assert "impact check unavailable" in hook_log(home)[-1]["reason"]


def test_impact_runner_exception_never_blocks(home):
    write_config(home, SHARED)

    def boom(f, cwd):
        raise RuntimeError("boom")

    assert impact(home, "/w/core/util.py", boom).exit_code == 0


def test_impact_ignores_other_events(home):
    write_config(home, SHARED)
    calls = []
    impact(home, "/w/core/util.py", lambda f, cwd: calls.append(f), event="PostToolUse")
    assert calls == []


def test_impact_malformed_config_is_skipped_not_blocked(home):
    write_config(home, "shared_module_globs: [oops")
    assert impact(home, "/w/core/util.py", lambda f, cwd: (0, "")).exit_code == 0


def test_run_codegraph_default_handles_absence_failure_and_timeout(tmp_path):
    assert run_codegraph("f.py", tmp_path, which=lambda name: None) is None

    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="codegraph", timeout=5)

    assert run_codegraph("f.py", tmp_path, which=lambda n: "/bin/codegraph", runner=timeout) is None

    class Done:
        returncode, stdout = 0, "tests/test_a.py\n"

    seen = {}

    def ok(cmd, **kw):
        seen["cmd"], seen["timeout"] = cmd, kw.get("timeout")
        return Done()

    assert run_codegraph("f.py", tmp_path, which=lambda n: "/bin/codegraph", runner=ok) == (0, "tests/test_a.py\n")
    assert seen["cmd"][:2] == ["/bin/codegraph", "affected"] and seen["timeout"] == 5


def test_run_codegraph_disables_codegraph_telemetry_in_its_subprocess(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEGRAPH_TELEMETRY", "1")  # even if the user's environment enables it
    monkeypatch.setenv("QRH_TEST_MARKER", "kept")
    seen = {}

    class Done:
        returncode, stdout = 0, ""

    def ok(cmd, **kw):
        seen["env"] = kw.get("env")
        return Done()

    run_codegraph("f.py", tmp_path, which=lambda n: "/bin/codegraph", runner=ok)
    assert seen["env"] is not None and seen["env"]["CODEGRAPH_TELEMETRY"] == "0"
    assert seen["env"]["QRH_TEST_MARKER"] == "kept"  # the rest of the environment is inherited
    assert os.environ["CODEGRAPH_TELEMETRY"] == "1"  # and the parent process is untouched


# --- security status in session-context -------------------------------------------------------------

def _secure(home, tmp_path):
    from hook_helpers import TEST_PASSPHRASE, ensure_test_keys
    from qrh import integrity, signing

    ensure_test_keys(home)
    integrity.sign_baseline(home, signing.unlock(home, TEST_PASSPHRASE, NOW), [], NOW)


def test_session_context_warns_without_an_approval_key_or_a_signed_baseline(home):
    write_config(home, "runaway:\n  repeat_limit: 3\n")
    out = ctx_run(home).stdout
    assert "qrh approve --init" in out and "qrh config sign" in out


def test_session_context_reports_a_changed_baseline(home, tmp_path):
    write_config(home, "runaway:\n  repeat_limit: 3\n")
    _secure(home, tmp_path)
    write_config(home, "runaway:\n  repeat_limit: 9\n")
    assert "config or hook registration changed since it was signed; run qrh config sign" in ctx_run(home).stdout


def test_session_context_reports_stop_answers_the_loop_guard_let_through(home, tmp_path):
    write_config(home, "runaway:\n  repeat_limit: 3\n")
    _secure(home, tmp_path)
    # records written by an older qrh that let a second Stop through; the current runner never writes one
    from qrh.hooks.common import log_invocation

    log_invocation(home, {"ts": (NOW - timedelta(days=1)).isoformat(), "session_id": "sess-1", "hook": "claim-check",
                          "event": "Stop", "decision": "allow", "reason": "LOOP-GUARD: legacy", "loop_guard": True})
    out = ctx_run(home).stdout
    assert "loop guard" in out.lower() and "unchecked" in out.lower()
