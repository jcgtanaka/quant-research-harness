import json

import pytest

from hook_helpers import DB, hook_log, payload, run, seed_trial, write_config
from qrh.hooks.receipt_required import (
    Claim,
    claim_matches,
    extract_claims,
    last_assistant_text,
)
from qrh.pipeline.receipt import Verdict

TID = "0123abcd4567ef890123abcd4567ef89"
OTHER = "fedc9876ba5432100fedc9876ba54321"


def claims(text, extra=()):
    return [(c.name, c.value, c.percent) for c in extract_claims(text, extra)]


# --- claim extraction -----------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("The Sharpe ratio of 1.42 holds.", [("sharpe", 1.42, False)]),
    ("Sharpe: 1.4", [("sharpe", 1.4, False)]),
    ("Sharpe = 0.8 out-of-sample", [("sharpe", 0.8, False)]),
    ("we got a 1.4 Sharpe", [("sharpe", 1.4, False)]),
    ("Sortino 2.1 and Sharpe 1.1", [("sortino", 2.1, False), ("sharpe", 1.1, False)]),
    ("CAGR of 12.5%", [("cagr", 12.5, True)]),
    ("max drawdown of -12.3%", [("drawdown", -12.3, True)]),
    ("a 12% max drawdown", [("drawdown", 12.0, True)]),
    ("win rate 55%", [("win rate", 55.0, True)]),
    ("55% win rate", [("win rate", 55.0, True)]),
    ("profit factor 1.8", [("profit factor", 1.8, False)]),
    ("PBO 0.18, DSR = 0.97", [("pbo", 0.18, False), ("dsr", 0.97, False)]),
    ("annualized volatility of 14%", [("volatility", 14.0, True)]),
    ("returns of 8.2%", [("return", 8.2, True)]),
    ("total return: 8.2 %", [("return", 8.2, True)]),
    ("Sharpe rose to 1.4", [("sharpe", 1.4, False)]),
])
def test_extracts_claims(text, expected):
    assert claims(text) == expected


@pytest.mark.parametrize("text", [
    "The Sharpe ratio measures risk-adjusted return.",
    "Sharpe improved in 3 files.",
    "This function returns 3 rows.",
    "returns 3 values",          # returns without a percent sign is not a performance claim
    "Added 12 tests for drawdown handling.",
    "Edited 2 files; the drawdown module was refactored in 4 places.",
    "",
])
def test_ignores_non_claims(text):
    assert claims(text) == []


def test_extra_metric_names_from_config():
    assert claims("Calmar 2.5", ("calmar",)) == [("calmar", 2.5, False)]
    assert claims("Calmar 2.5") == []


# --- matching -------------------------------------------------------------------------------------

def C(name, value, percent=False):
    return Claim(name, value, percent, f"{name} {value}")


@pytest.mark.parametrize("claim,metrics,ok", [
    (C("sharpe", 1.42), {"oos_sharpe": 1.42}, True),
    (C("sharpe", 1.43), {"oos_sharpe": 1.42}, True),            # within abs 0.01
    (C("sharpe", 1.50), {"oos_sharpe": 1.42}, False),
    (C("sharpe", 100.4), {"oos_sharpe": 100.0}, True),          # within relative 0.5%
    (C("sharpe", 101.0), {"oos_sharpe": 100.0}, False),
    (C("cagr", 12.3, True), {"oos_cagr": 0.123}, True),         # percent -> fraction
    (C("cagr", 12.3, True), {"oos_cagr": 12.3}, True),          # stored as percent
    (C("cagr", 12.3, False), {"oos_cagr": 0.123}, True),        # percent-type metric quoted without a sign
    (C("sharpe", 1.4, False), {"oos_sharpe": 0.014}, False),    # ratio metrics are never rescaled
    (C("drawdown", -12.3, True), {"oos_max_drawdown": 0.123}, True),   # drawdown sign is ignored
    (C("drawdown", 20.0, True), {"oos_max_drawdown": -0.20}, True),
    (C("sharpe", -1.4), {"oos_sharpe": 1.4}, False),
    (C("pbo", 18.0, True), {"pbo": 0.18}, True),
    (C("sharpe", 1.4), {}, False),
    # same family only
    (C("sharpe", 0.18), {"pbo": 0.18}, False),
    (C("pbo", 1.42), {"oos_sharpe": 1.42}, False),
    (C("sortino", 1.42), {"oos_sharpe": 1.42}, False),
    (C("dsr", 0.99), {"oos_sharpe": 0.99}, False),
    (C("dsr", 0.99), {"dsr": 0.99}, True),
    (C("return", 8.0, True), {"oos_cagr": 0.08}, True),
    (C("return", 8.0, True), {"total_return": 0.08}, True),
    (C("volatility", 14.0, True), {"ann_vol": 0.14}, True),
    (C("win rate", 55.0, True), {"win_rate": 0.55}, True),
    (C("profit factor", 1.8), {"profit_factor": 1.8}, True),
    # thresholds, limits, counts and null expectations are not observed metrics
    (C("dsr", 0.95), {"min_dsr": 0.95}, False),
    (C("pbo", 0.2), {"max_pbo": 0.2}, False),
    (C("sharpe", 0.0), {"min_oos_sharpe": 0.0}, False),
    (C("sharpe", 10.0), {"n_folds": 10.0, "oos_sharpe": 3.17}, False),
    (C("sharpe", 9.0), {"sharpe_count": 9.0, "grid_size": 9.0}, False),
    (C("sharpe", 0.0894), {"sr0": 0.0894}, False),
    (C("drawdown", 9.0, True), {"sr0": 0.0894, "oos_max_drawdown": -0.20}, False),
    (C("sharpe", 0.5), {"var_sharpe": 0.5}, False),
    (C("volatility", 14.0, True), {"volume": 0.14}, False),     # "vol" is a word, not a substring
])
def test_claim_matches(claim, metrics, ok):
    assert claim_matches(claim, metrics, abs_tol=0.01, rel_tol=0.005) is ok


def test_metric_map_overrides_and_extends_the_family_table():
    assert claim_matches(C("calmar", 2.5), {"calmar_ratio": 2.5}, abs_tol=0.01, rel_tol=0.005)  # default: the word itself
    assert not claim_matches(C("calmar", 2.5), {"calmar_ratio": 2.5}, abs_tol=0.01, rel_tol=0.005,
                             metric_map={"calmar": ("calmar_x",)})
    assert claim_matches(C("calmar", 2.5), {"calmar_ratio": 2.5}, abs_tol=0.01, rel_tol=0.005,
                         metric_map={"calmar": ("calmar_ratio",)})
    assert claim_matches(C("sharpe", 1.4), {"my_risk_adj": 1.4}, abs_tol=0.01, rel_tol=0.005,
                         metric_map={"sharpe": ("risk_adj",)})
    assert not claim_matches(C("sharpe", 1.4), {"oos_sharpe": 1.4}, abs_tol=0.01, rel_tol=0.005,
                             metric_map={"sharpe": ("risk_adj",)})


@pytest.mark.parametrize("text,expected", [
    ("Deflated Sharpe 0.99", [("dsr", 0.99, False)]),
    ("annualized return of 88%", [("cagr", 88.0, True)]),
    ("an 88% annualized return", [("return", 88.0, True)]),
    ("probability of backtest overfitting is 0.02", [("pbo", 0.02, False)]),
    ("max DD of 20%", [("drawdown", 20.0, True)]),
    ("vol of 14%", [("volatility", 14.0, True)]),
])
def test_more_claim_words(text, expected):
    assert claims(text) == expected


def test_word_without_number_and_number_without_word_are_ignored():
    assert claims("Sharpe is great, see 2.9 files") == []
    assert claims("There were 12 trials and 9 folds") == []


def test_tolerance_is_configurable():
    assert not claim_matches(C("sharpe", 1.45), {"oos_sharpe": 1.42}, abs_tol=0.01, rel_tol=0.005)
    assert claim_matches(C("sharpe", 1.45), {"oos_sharpe": 1.42}, abs_tol=0.05, rel_tol=0.005)


# --- transcripts ----------------------------------------------------------------------------------

def write_transcript(path, entries):
    path.write_text("\n".join(e if isinstance(e, str) else json.dumps(e) for e in entries) + "\n")
    return path


def asst(content, nested=True):
    return {"type": "assistant", "message": {"role": "assistant", "content": content}} if nested \
        else {"role": "assistant", "content": content}


def test_last_assistant_text_shapes(tmp_path):
    t = tmp_path / "t.jsonl"
    write_transcript(t, [{"type": "user", "message": {"role": "user", "content": "hi"}}, asst("plain string")])
    assert last_assistant_text(t) == "plain string"
    write_transcript(t, [asst([{"type": "text", "text": "block one"}, {"type": "text", "text": "block two"}])])
    assert last_assistant_text(t) == "block one\nblock two"
    write_transcript(t, [asst("final words"), asst([{"type": "tool_use", "id": "x", "name": "Bash", "input": {"command": "Sharpe 9.9"}}])])
    assert last_assistant_text(t) == "final words"  # tool_use skipped; last entry with text wins
    write_transcript(t, [asst([{"type": "text", "text": "A"}, {"type": "tool_use", "name": "x", "input": {}},
                               {"type": "text", "text": "B"}])])
    assert last_assistant_text(t) == "A\nB"
    write_transcript(t, [asst("flat shape", nested=False)])
    assert last_assistant_text(t) == "flat shape"
    write_transcript(t, ["not json {{", asst("survives bad lines"), "", "[1,2]"])
    assert last_assistant_text(t) == "survives bad lines"
    write_transcript(t, [{"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "Sharpe 5.5"}]}}])
    assert last_assistant_text(t) is None


# --- the hook -------------------------------------------------------------------------------------

@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    seed_trial(h, TID, verdicts={"S0": Verdict.PASS, "S4": Verdict.PASS, "S5": Verdict.PASS},
               metrics={"S4": {"oos_sharpe": 1.42}, "S5": {"pbo": 0.18, "dsr": 0.97}})
    return h


def stop(home, tmp_path, text, env=None, active=False, entries=None, **extra):
    t = write_transcript(tmp_path / "t.jsonl", entries or [asst([{"type": "text", "text": text}])])
    data = payload("Stop", transcript_path=str(t), stop_hook_active=active)
    return run("receipt-required", data, home, env=env, **extra)


def test_matching_claim_is_allowed(home, tmp_path):
    out = stop(home, tmp_path, "Out-of-sample Sharpe of 1.42 with PBO 0.18.")
    assert out.exit_code == 0 and out.stderr == ""


def test_fabricated_claim_blocks_with_instruction(home, tmp_path):
    out = stop(home, tmp_path, "The strategy has a Sharpe of 2.9.")
    assert out.exit_code == 2
    assert "2.9" in out.stderr and "run the harness or remove the claim" in out.stderr
    assert hook_log(home)[-1]["decision"] == "block"


def test_every_unmatched_claim_listed_and_matched_ones_not(home, tmp_path):
    out = stop(home, tmp_path, "Sharpe 1.42, CAGR 31%, win rate 71%.")
    assert out.exit_code == 2 and "31" in out.stderr and "71" in out.stderr and "1.42" not in out.stderr


def test_percent_conversion(home, tmp_path):
    seed_trial(home, OTHER, verdicts={"S6": Verdict.PASS}, metrics={"S6": {"cagr": 0.081}})
    assert stop(home, tmp_path, "CAGR of 8.1%").exit_code == 0


def test_mentioned_trial_scopes_the_lookup(home, tmp_path):
    seed_trial(home, OTHER, verdicts={"S6": Verdict.PASS}, metrics={"S6": {"oos_sharpe": 3.3}})
    assert stop(home, tmp_path, "Sharpe 3.3").exit_code == 0                      # recent receipts: found
    out = stop(home, tmp_path, f"Trial {TID[:8]} shows Sharpe 3.3")               # scoped to TID: not there
    assert out.exit_code == 2
    assert stop(home, tmp_path, f"Trial {OTHER[:8]} shows Sharpe 3.3").exit_code == 0
    assert stop(home, tmp_path, f"Trial {TID[:7]} Sharpe 3.3").exit_code == 0     # 7 chars is not an id prefix


def test_old_receipts_do_not_count(tmp_path):
    h = tmp_path / "home2"
    seed_trial(h, TID, verdicts={"S4": Verdict.PASS}, metrics={"S4": {"oos_sharpe": 1.42}},
               created_at="2020-01-01T00:00:00+00:00")
    assert stop(h, tmp_path, "Sharpe 1.42").exit_code == 2
    write_config(h, "receipt_required:\n  recent_days: 100000\n")
    assert stop(h, tmp_path, "Sharpe 1.42").exit_code == 0
    assert stop(h, tmp_path, f"Sharpe 1.42 for {TID[:8]}", ).exit_code == 0  # a mentioned trial ignores age


def test_tampered_receipt_cannot_back_a_claim(home, tmp_path):
    path = home / "receipts" / TID / "S4.json"
    doc = json.loads(path.read_text())
    doc["receipt"]["metrics"]["oos_sharpe"] = 9.9
    path.write_text(json.dumps(doc))
    assert stop(home, tmp_path, "Sharpe 9.9").exit_code == 2


def test_no_claims_does_not_touch_the_registry(tmp_path):
    h = tmp_path / "broken"
    h.mkdir()
    (h / DB).write_bytes(b"garbage, not sqlite " * 50)
    assert stop(h, tmp_path, "Refactored the adapter; 12 tests pass.").exit_code == 0


def test_claims_with_unreadable_registry_fail_closed(tmp_path):
    h = tmp_path / "broken"
    h.mkdir()
    (h / DB).write_bytes(b"garbage, not sqlite " * 50)
    out = stop(h, tmp_path, "Sharpe 1.4")
    assert out.exit_code == 2 and "registry" in out.stderr


def test_a_repeated_stop_is_still_checked(home, tmp_path):
    assert stop(home, tmp_path, "Sharpe 9.9").exit_code == 2
    assert stop(home, tmp_path, "Sharpe 9.9", active=True).exit_code == 2
    assert stop(home, tmp_path, "No numbers here.", active=True).exit_code == 0


def test_disabled_in_config(home, tmp_path):
    write_config(home, "receipt_required:\n  enabled: false\n")
    assert stop(home, tmp_path, "Sharpe 9.9").exit_code == 0
    assert "disabled" in hook_log(home)[-1]["reason"]


def test_malformed_config_blocks(home, tmp_path):
    write_config(home, "receipt_required: [oops")
    assert stop(home, tmp_path, "Sharpe 1.42").exit_code == 2


@pytest.mark.parametrize("path", ["", "/nonexistent/transcript.jsonl"])
def test_unreadable_transcript_exits_0_and_logs_error(home, path):
    out = run("receipt-required", payload("Stop", transcript_path=path), home)
    assert out.exit_code == 0 and out.stderr == ""
    rec = hook_log(home)[-1]
    assert rec["decision"] == "error" and "transcript" in rec["reason"]


def test_transcript_that_is_a_directory(home, tmp_path):
    assert run("receipt-required", payload("Stop", transcript_path=str(tmp_path)), home).exit_code == 0


def test_no_assistant_message_allows(home, tmp_path):
    out = stop(home, tmp_path, "", entries=[{"type": "user", "message": {"role": "user", "content": "Sharpe 9.9?"}}])
    assert out.exit_code == 0


def test_only_the_last_assistant_message_is_checked(home, tmp_path):
    entries = [asst("Early guess: Sharpe 9.9"), {"type": "user", "message": {"role": "user", "content": "ok"}},
               asst("Corrected: Sharpe 1.42")]
    assert stop(home, tmp_path, "", entries=entries).exit_code == 0


def test_extra_metric_patterns_from_config(home, tmp_path):
    write_config(home, 'receipt_required:\n  metric_patterns: ["calmar"]\n')
    assert stop(home, tmp_path, "Calmar 4.4").exit_code == 2


def test_human_approval_allows_one_stop(home, tmp_path):
    from hook_helpers import human_approve

    first = stop(home, tmp_path, "Sharpe 9.9")
    assert first.exit_code == 2
    human_approve(home, first.stderr, "figure quoted from vendor docs, checked")
    assert stop(home, tmp_path, "Sharpe 9.9").exit_code == 0
    assert stop(home, tmp_path, "Sharpe 9.9").exit_code == 2


def test_approval_is_tied_to_the_unsupported_claims(home, tmp_path):
    from hook_helpers import human_approve

    first = stop(home, tmp_path, "Sharpe 9.9")
    human_approve(home, first.stderr, "figure quoted from vendor docs, checked")
    assert stop(home, tmp_path, "Sharpe 8.8").exit_code == 2
    assert stop(home, tmp_path, "Sharpe 9.9").exit_code == 0


# --- name-aware matching through the hook -------------------------------------------------------

@pytest.fixture
def home_ex01(tmp_path):
    """Metrics shaped like example 01's receipts: observed values next to thresholds, counts and SR0."""
    h = tmp_path / "ex01"
    seed_trial(h, TID, verdicts={"S4": Verdict.PASS, "S5": Verdict.PASS}, metrics={
        "S4": {"oos_sharpe": 3.17, "min_oos_sharpe": 0.0, "n_folds": 10.0, "oos_max_drawdown": -0.20,
               "oos_cagr": 0.88},
        "S5": {"dsr": 0.992, "min_dsr": 0.95, "pbo": 0.02, "max_pbo": 0.2, "sr0": 0.0894},
    })
    return h


def test_example_01_shaped_claims_accepted(home_ex01, tmp_path):
    out = stop(home_ex01, tmp_path, "Sharpe of 3.17, PBO 0.02, a -20.0% max drawdown, CAGR 88%, DSR 0.99.")
    assert out.exit_code == 0, out.stderr


@pytest.mark.parametrize("text,needle", [
    ("A 9% max drawdown", "drawdown"),            # sr0 ~ 0.089 exists but is not a drawdown
    ("DSR 0.95", "dsr"),                          # only the threshold min_dsr equals it
    ("Sharpe 10", "sharpe"),                      # n_folds is a count
    ("Sharpe 0", "sharpe"),                       # min_oos_sharpe is a limit
    ("PBO 0.2", "pbo"),                           # max_pbo is a limit
    ("Sharpe 0.02", "sharpe"),                    # the pbo value, wrong family
])
def test_coincidental_values_do_not_accept_claims(home_ex01, tmp_path, text, needle):
    out = stop(home_ex01, tmp_path, text)
    assert out.exit_code == 2 and needle in out.stderr


def test_metric_map_in_config(tmp_path):
    h = tmp_path / "mm"
    seed_trial(h, TID, verdicts={"S4": Verdict.PASS}, metrics={"S4": {"calmar_ratio": 2.5}})
    write_config(h, 'receipt_required:\n  metric_patterns: ["calmar"]\n  metric_map: {calmar: [calmar_ratio]}\n')
    assert stop(h, tmp_path, "Calmar 2.5").exit_code == 0
    assert stop(h, tmp_path, "Calmar 2.9").exit_code == 2


def test_malformed_metric_map_blocks(home, tmp_path):
    write_config(home, "receipt_required:\n  metric_map: [oops]\n")
    assert stop(home, tmp_path, "Sharpe 1.42").exit_code == 2
