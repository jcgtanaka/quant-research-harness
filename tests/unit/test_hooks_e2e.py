"""`qrh hook <name>` end to end: a real subprocess, JSON on stdin, exit code and streams as Claude Code sees them."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hook_helpers import approval_id, bash_payload, human_approve_id, edit_payload, payload, register_dataset, seed_trial
from qrh.pipeline.receipt import Verdict

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
TID = "a1b2c3d4e5f60718293a4b5c6d7e8f90"


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


def hook(name, data, home, extra_env=None):
    env = {k: v for k, v in os.environ.items() if k not in ("QRH_OVERRIDE", "QRH_HOME")}
    env.update({"QRH_HOME": str(home), "QRH_REPORTS": "0", **(extra_env or {})})
    text = data if isinstance(data, str) else json.dumps(data)
    t0 = time.perf_counter()
    done = subprocess.run([sys.executable, "-c", "from qrh.cli import app; app()", "hook", name],
                          input=text, capture_output=True, text=True, env=env, timeout=60)
    done.wall = time.perf_counter() - t0  # type: ignore[attr-defined]
    return done


def test_lookahead_lint_blocks_the_bad_example_and_passes_the_clean_one(home):
    bad = hook("lookahead-lint", edit_payload("PostToolUse", str(EXAMPLES / "02_lookahead" / "adapter.py")), home)
    assert bad.returncode == 2
    assert "adapter.py:" in bad.stderr and "LA001" in bad.stderr
    good = hook("lookahead-lint", edit_payload("PostToolUse", str(EXAMPLES / "01_clean_strategy" / "adapter.py")), home)
    assert good.returncode == 0 and good.stderr == ""
    lines = [json.loads(x) for x in (home / "logs" / "hooks.jsonl").read_text().splitlines()]
    assert [r["decision"] for r in lines] == ["block", "allow"] and all(r["duration_ms"] < 1000 for r in lines)


def test_data_gate_blocks_on_a_blocked_dataset(home, tmp_path):
    register_dataset(home, tmp_path, "prices", status="BLOCKED")
    seed_trial(home, TID, datasets=["prices"])
    out = hook("data-gate", bash_payload(f"qrh run {TID}"), home)
    assert out.returncode == 2 and "prices" in out.stderr


def test_session_context_and_prior_attempts_print_to_stdout(home):
    seed_trial(home, TID, text="Momentum persists after volatility regime shifts", family="momentum")
    ctx = hook("session-context", payload("SessionStart", source="startup"), home)
    assert ctx.returncode == 0 and TID[:8] in ctx.stdout
    prior = hook("prior-attempts", payload("UserPromptSubmit", prompt="new momentum strategy idea"), home)
    assert prior.returncode == 0 and TID[:8] in prior.stdout and "momentum" in prior.stdout


def test_receipt_required_stop_flow(home, tmp_path):
    seed_trial(home, TID, verdicts={"S4": Verdict.PASS}, metrics={"S4": {"oos_sharpe": 1.42}})
    t = tmp_path / "t.jsonl"

    def stop(text, active=False):
        t.write_text(json.dumps({"type": "assistant", "message": {"role": "assistant",
                                 "content": [{"type": "text", "text": text}]}}) + "\n")
        return hook("receipt-required", payload("Stop", transcript_path=str(t), stop_hook_active=active), home)

    assert stop("OOS Sharpe is 1.42.").returncode == 0
    bad = stop("OOS Sharpe is 2.75.")
    assert bad.returncode == 2 and "2.75" in bad.stderr
    assert stop("OOS Sharpe is 2.75.", active=True).returncode == 2  # a repeated Stop is still checked


def test_human_approval_through_the_cli(home, tmp_path):
    register_dataset(home, tmp_path, "prices", status="BLOCKED")
    seed_trial(home, TID, datasets=["prices"])
    blocked = hook("data-gate", bash_payload(f"qrh run {TID}"), home)
    assert blocked.returncode == 2 and "qrh approve" in blocked.stderr
    rid = approval_id(blocked.stderr)
    env = {**os.environ, "QRH_HOME": str(home)}
    # `qrh approve` from a process without a terminal (like an agent's shell) is refused ...
    agent = subprocess.run([sys.executable, "-c", "from qrh.cli import app; app()", "approve", rid, "--reason",
                            "vendor outage, owner approved"], capture_output=True, text=True, env=env, timeout=60,
                           start_new_session=True)
    assert agent.returncode == 2 and "terminal" in agent.stderr
    # ... the human signs in their own terminal (here through the code seam)
    human_approve_id(home, rid, "vendor outage, owner approved")
    assert hook("data-gate", bash_payload(f"qrh run {TID}"), home).returncode == 0
    assert hook("data-gate", bash_payload(f"qrh run {TID}"), home).returncode == 2


def test_the_override_environment_variable_does_nothing(home, tmp_path):
    register_dataset(home, tmp_path, "prices", status="BLOCKED")
    seed_trial(home, TID, datasets=["prices"])
    env = {"QRH_OVERRIDE": "data-gate:vendor outage, owner approved"}
    assert hook("data-gate", bash_payload(f"qrh run {TID}"), home, env).returncode == 2


def test_bad_input_and_unknown_hook(home):
    assert hook("data-gate", "{not json", home).returncode == 2           # enforcement: fail closed
    assert hook("session-context", "{not json", home).returncode == 0     # informational: never breaks
    out = hook("no-such-hook", "{}", home)
    assert out.returncode == 2 and "unknown hook" in out.stderr


def test_hook_does_not_import_pandas():
    code = ("import sys; from qrh.hooks.runner import run_hook; import json;"
            "run_hook('prior-attempts', json.dumps({'prompt':'strategy','hook_event_name':'UserPromptSubmit'}),"
            " home=__import__('pathlib').Path(sys.argv[1]));"
            "print('pandas' in sys.modules)")
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        done = subprocess.run([sys.executable, "-c", code, d], capture_output=True, text=True)
    assert done.stdout.strip() == "False", done.stderr
