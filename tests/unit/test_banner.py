"""The standing banner rule: a block that needs a HUMAN decision starts with this exact banner."""

from hook_helpers import approval_id, bash_payload, human_approve, payload, run
from qrh.banner import BANNER, RELAY, decision_banner
from qrh.hooks.runner import HookResult, HookSpec

EXACT = (
    "##################\n"
    "WARNING: DECISION TO BE MADE BY YOU!\n"
    "##################\n"
    "Content of the warning:\n"
)


def test_banner_text_is_exact():
    assert BANNER == EXACT
    text = decision_banner("details here")
    assert text.startswith(EXACT + "details here\n")
    assert RELAY in text and text.rstrip().endswith(RELAY.rstrip())
    assert "verbatim" in RELAY and "STOP" in RELAY


def _stub(name="stub", event="PreToolUse", **kw):
    return {name: HookSpec(name, event, "Bash", 5, True, "unused",
                           run=lambda ctx: HookResult("block", "dataset prices is BLOCKED", **kw))}


def test_an_approvable_block_starts_with_the_banner_and_names_everything(tmp_path):
    out = run("stub", bash_payload("qrh run abc12345 --token=sk-abcdefghijklmnop"), tmp_path, hooks=_stub())
    assert out.exit_code == 2 and out.stderr.startswith(EXACT)
    rid = approval_id(out.stderr)
    body = out.stderr[len(EXACT):]
    assert "Hook: stub" in body and "qrh run abc12345" in body and "sk-abcdefghijklmnop" not in body
    assert "dataset prices is BLOCKED" in body and "Risk:" in body
    assert f'qrh approve {rid} --reason "' in body and "OWN terminal" in body
    assert body.rstrip().endswith(RELAY.rstrip())


def test_a_final_block_needs_no_human_decision_and_has_no_banner(tmp_path):
    from dataclasses import replace

    hooks = {"stub": replace(_stub()["stub"], approvable=False)}
    out = run("stub", bash_payload("ls"), tmp_path, hooks=hooks)
    assert out.exit_code == 2 and not out.stderr.startswith(EXACT) and "cannot be overridden" in out.stderr


def test_approval_is_bound_to_the_session_that_was_blocked(tmp_path):
    first = run("stub", bash_payload("ls"), tmp_path, hooks=_stub())
    human_approve(tmp_path, first.stderr)
    other = run("stub", bash_payload("ls", session_id="sess-OTHER"), tmp_path, hooks=_stub())
    assert other.exit_code == 2
    assert run("stub", bash_payload("ls"), tmp_path, hooks=_stub()).exit_code == 0


def test_stop_hooks_supply_action_parts_that_the_human_sees(tmp_path):
    hooks = _stub(event="Stop", action_parts=("claim-check", "tests_pass"))
    hooks["stub"] = HookSpec("stub", "Stop", None, 5, True, "unused", run=hooks["stub"].run)
    stop = payload("Stop", stop_hook_active=False)
    first = run("stub", stop, tmp_path, hooks=hooks)
    assert first.exit_code == 2 and first.stderr.startswith(EXACT) and "tests_pass" in first.stderr
    human_approve(tmp_path, first.stderr)
    assert run("stub", stop, tmp_path, hooks=hooks).exit_code == 0
