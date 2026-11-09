"""Conditional receipts in hooks: receipt-required treats them as valid sources; session context lists conditions."""

import json
from datetime import datetime, timezone

import pytest

from hook_helpers import DB, payload, run, seed_trial
from qrh.memory.registry import Registry
from qrh.pipeline.receipt import Receipt, Stage, Verdict, write_receipt

NOW = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)
TID = "0123abcd4567ef890123abcd4567ef89"
COND = [{"question_id": "rates-source", "scope": "dataset:prices", "stage": "S1", "text": "How were rates produced?",
         "status": "unknown", "risk": "Possible look-ahead."},
        {"question_id": "fill-timing", "scope": "strategy:x", "stage": "S2", "text": "When is the order filled?",
         "status": "pending", "risk": ""}]


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    seed_trial(h, TID, verdicts={"S0": Verdict.PASS, "S4": Verdict.PASS, "S5": Verdict.PASS},
               metrics={"S4": {"oos_sharpe": 1.42}, "S5": {"pbo": 0.18, "dsr": 0.97}})
    reg = Registry(h / DB)
    try:
        h0 = reg.get_trial(TID)["hypothesis_hash"]
        write_receipt(Receipt(trial_id=TID, stage=Stage.S7, hypothesis_hash=h0, code_hash="c" * 64,
                              verdict=Verdict.PASS_CONDITIONAL, params={"conditions": COND}), h, reg)
    finally:
        reg.close()
    return h


def test_receipt_required_accepts_numbers_from_a_conditional_trial(home, tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "text", "text": f"Trial {TID[:8]}: out-of-sample Sharpe of 1.42, PASS · CONDITIONAL."}]}}) + "\n")
    out = run("receipt-required", payload("Stop", transcript_path=str(t), stop_hook_active=False), home)
    assert out.exit_code == 0 and out.stderr == ""


def test_session_context_mentions_the_conditions_of_a_conditional_pass(home):
    out = run("session-context", payload("SessionStart", source="startup"), home, now=lambda: NOW)
    text = out.stdout
    assert "PASS · CONDITIONAL" in text and TID[:8] in text
    assert "rates-source (unknown)" in text and "fill-timing (pending)" in text
    assert "quote" in text.lower()
    assert len(text.splitlines()) <= 25


def test_a_conditional_pass_is_not_listed_as_an_open_trial(home):
    text = run("session-context", payload("SessionStart", source="startup"), home, now=lambda: NOW).stdout
    assert "Open trials" not in text
