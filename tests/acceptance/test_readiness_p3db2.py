"""P3d-b2 part A end to end on example 01: the readiness report, the human acknowledgement and its invalidation."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qrh import signing, tty
from qrh.cli import app
from qrh.pipeline.receipt import Verdict
from qrh.readiness import collect as rc
from qrh.readiness import data as rd
from qrh.readiness import is_acknowledged

runner = CliRunner()
PACKS = Path(__file__).resolve().parents[2] / "packs"
SESSION = "sess-readiness"
PASSPHRASE = "a test passphrase, runtime only"  # gitleaks:allow


def cli(*args):
    return runner.invoke(app, list(args))


@pytest.fixture(autouse=True)
def fake_user_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.delenv("QRH_REPORTS", raising=False)  # the readiness report is generated automatically at S7


@pytest.fixture
def terminal(monkeypatch):
    """A mocked controlling terminal (the existing test seam): what the human sees and types."""
    state = {"shown": [], "typed": []}
    monkeypatch.setattr(tty, "has_terminal", lambda: True)
    monkeypatch.setattr(tty, "write", lambda text: state["shown"].append(text))

    def read_secret(prompt):
        state["shown"].append(f"<{prompt}>")
        return state["typed"].pop(0)

    monkeypatch.setattr(tty, "read_secret", read_secret)
    return state


@pytest.fixture
def key(home, monkeypatch):
    monkeypatch.setattr(signing, "KDF_N", 2 ** 4)
    signing.init_keys(home, PASSPHRASE, datetime.now(timezone.utc))


def human_acks(terminal, trial_id):
    terminal["typed"] = [PASSPHRASE]
    terminal["shown"].clear()
    res = cli("readiness", "ack", trial_id)
    assert res.exit_code == 0, res.output
    return "".join(terminal["shown"])


def say(tmp_path, text):
    from qrh.hooks.runner import run_hook

    ts = (datetime.now(timezone.utc) + timedelta(seconds=3)).isoformat()
    p = tmp_path / "user" / ".claude" / "projects" / "proj" / f"{SESSION}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"type": "user", "timestamp": ts, "sessionId": SESSION,
                             "message": {"role": "user", "content": text}}) + "\n")
    run_hook("session-context", json.dumps({"hook_event_name": "SessionStart", "session_id": SESSION,
                                            "transcript_path": str(p), "source": "startup"}),
             home=tmp_path / "qrh_home", env={})
    return p


def onboard_prices(tmp_path, name="prices"):
    """Onboard the example's synthetic dataset (a clean run has no unsettled dataset): propose, verify, confirm."""
    proposal = {"source": "llm", "extras": {}, "fields": {k: {"value": v} for k, v in {
        "asset_class": "index", "price_columns": ["close"], "price_units": "currency", "timestamp_semantics": "bar_close",
        "timezone": "UTC", "frequency": "1D", "adjustments": "none"}.items()}}
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(proposal))
    assert cli("data", "propose", name, "--file", str(path)).exit_code == 0
    assert cli("data", "verify", name).exit_code == 0
    for field in ("adjustments", "timestamp_semantics"):
        assert cli("data", "confirm", name, field).exit_code == 0


def ask_and_answer(tmp_path, tid, stage, scope, qid, value, said=None):
    res = cli("interview", "next", "--stage", stage, "--scope", scope, "--trial", tid, "--include-unknown", "--json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["question"]["id"] == qid
    t = say(tmp_path, said or value)
    res = cli("interview", "answer", qid, "--scope", scope, "--value", value, "--transcript", str(t), "--session", SESSION)
    assert res.exit_code == 0, res.output


def confirm_stability_thresholds(tmp_path, scope="strategy:trend|synthetic|1D"):
    """The trader's own answers (a transcript the harness can check), keeping the default for each of the four."""
    for qid in ("stability-closeness", "stability-plateau-share", "stability-neighbour-ratio", "stability-steps"):
        res = cli("interview", "next", "--stage", "S0", "--scope", scope, "--json")
        assert res.exit_code == 0 and json.loads(res.stdout)["question"]["id"] == qid, res.output
        t = say(tmp_path, "keep_default")
        res = cli("interview", "answer", qid, "--scope", scope, "--value", "keep_default", "--transcript", str(t),
                  "--session", SESSION)
        assert res.exit_code == 0, res.output


def collected(home, tid):
    return rd.build_data(rc.collect(home, tid))


def test_clean_strategy_has_few_low_warnings_and_the_acknowledgement_is_invalidated_by_a_new_answer(
        fresh_example, tmp_path, home, key, terminal):
    ex = fresh_example("01_clean_strategy")
    ex.register(ex.build_data(), "prices")
    onboard_prices(tmp_path)
    confirm_stability_thresholds(tmp_path)  # the trader answers the four stability questions before the hypothesis is registered
    tid = ex.new_trial()
    assert ex.run().exit_code == 0 and ex.receipt("S7").verdict is Verdict.PASS

    # generated automatically at S7, into the trial's report manifest
    base = home / "reports" / tid
    assert (base / "readiness.pdf").read_bytes().startswith(b"%PDF")
    manifest = json.loads((base / "manifest.json").read_text())
    assert {f["path"] for f in manifest["files"] if f["stage"] == "readiness"} == {
        "readiness.pdf", "readiness.chart.json", "readiness.json"}
    c = rc.collect(home, tid)
    assert c.verdict == "PASS" and not c.high()
    assert {w.severity for w in c.warnings} <= {"low", "medium"} and len(c.warnings) <= 4
    # The index claim on this dataset is unverifiable (no class-specific evidence), so it is an honest medium warning.
    assert [w.code for w in c.warnings if w.severity == "medium"] == ["non-default-method", "unverified-recommended"]
    stored = json.loads((base / "readiness.json").read_text())
    assert stored["report_hash"] == collected(home, tid).hash

    # pending until the human acknowledges, with the passphrase on the (mocked) terminal
    assert is_acknowledged(tid)[0] is False
    status = cli("readiness", "status", tid)
    assert status.exit_code == 1 and "PENDING" in status.output
    shown = human_acks(terminal, tid)
    assert shown.startswith("##################\nWARNING: DECISION TO BE MADE BY YOU!\n##################\nContent of the warning:")
    assert "No HIGH warnings." in shown.split("<")[0]
    ok, why = is_acknowledged(tid)
    assert ok, why
    assert cli("readiness", "status", tid).exit_code == 0

    # an identical re-run writes identical receipts: the acknowledgement survives it
    assert ex.run().exit_code == 0
    assert is_acknowledged(tid)[0]

    # a new question bank appears: unanswered critical questions are HIGH warnings and the acknowledgement is stale
    assert cli("pack", "add", str(PACKS / "retail-core")).exit_code == 0
    ok, why = is_acknowledged(tid)
    assert not ok and "warnings changed" in why
    assert any(w.code == "pending-critical" for w in rc.collect(home, tid).warnings)
    shown = human_acks(terminal, tid)
    assert "pending-critical" in shown.split("<")[0] and "HIGH warnings (" in shown
    assert is_acknowledged(tid)[0]

    # the user answers one question: the warning set changes again, the acknowledgement is invalid again
    before = collected(home, tid)
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "bar-type-declared", "time", "Time bars, one day each")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "data-history-revised", "never_revised", "No, it is as first published")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "data-timezone", "utc", "UTC")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "timestamp-semantics", "bar_close", "the bar close")
    after = collected(home, tid)
    assert after.hash != before.hash and len(after.doc["warnings"]) < len(before.doc["warnings"])
    ok, why = is_acknowledged(tid)
    assert not ok and "warnings changed" in why and "resolved" in why
    refreshed = cli("readiness", "report", tid)
    assert refreshed.exit_code == 0 and after.hash in refreshed.output
    assert "PENDING" in cli("readiness", "status", tid).output


def test_the_conditional_run_lists_its_high_warnings(fresh_example, tmp_path, home, key, terminal):
    assert cli("pack", "add", str(PACKS / "retail-core")).exit_code == 0
    assert cli("pack", "add", str(PACKS / "example-trader")).exit_code == 0
    ex = fresh_example("01_clean_strategy")
    ex.register(ex.build_data(), "prices")
    tid = ex.new_trial()
    fam = "trend|synthetic|1D"
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "bar-type-declared", "time", "Time bars, one day each")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "data-history-revised", "never_revised", "No, it is as first published")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "data-timezone", "utc", "UTC")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "timestamp-semantics", "bar_close", "the bar close")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "ex-rate-source", "unknown", "honestly I don't know")
    ask_and_answer(tmp_path, tid, "S2", f"strategy:{fam}", "signal-vs-fill", "next_bar", "Signal at one bar, fill at a later bar")
    ask_and_answer(tmp_path, tid, "S3", f"strategy:{fam}", "variants-outside-harness", "0", "I tried 0 variants outside")
    ask_and_answer(tmp_path, tid, "S4", f"strategy:{fam}", "reopt-cadence-live", "fixed", "never, parameters stay fixed")
    assert ex.run().exit_code == 0
    assert ex.receipt("S7").verdict is Verdict.PASS_CONDITIONAL

    c = rc.collect(home, tid)
    assert c.verdict == "PASS_CONDITIONAL" and c.verdict_display == "PASS · CONDITIONAL"
    high = {(w.kind, w.code, w.scope) for w in c.high()}
    assert ("interview", "unknown-critical", "ex-rate-source@dataset:prices") in high
    assert ("condition", "open-condition", "ex-rate-source@dataset:prices") in high
    medium = {(w.code, w.scope.split("@")[0]) for w in c.warnings if w.severity == "medium"}
    assert ("pending-recommended", "falsification-criterion") in medium and ("non-default-method", "s2.pnl_reconstruction=pnl_reconstruction") in medium
    assert any(w.kind == "limit" for w in c.warnings)
    assert all(w.resolution for w in c.warnings)

    # the generated report page 1 says PASS · CONDITIONAL, and the human is shown every HIGH warning first
    chart = json.loads((home / "reports" / tid / "readiness.chart.json").read_text())
    assert chart["figures"][0]["verdict_display"] == "PASS · CONDITIONAL" and chart["figures"][0]["ack"]["acknowledged"] is False
    shown = human_acks(terminal, tid).split("<")[0]
    assert "PASS · CONDITIONAL" in shown and "ex-rate-source" in shown
    assert shown.count("▲ [") == len(c.high())
    assert is_acknowledged(tid)[0]
    # the acknowledgement made no verdict change
    assert ex.receipt("S7").verdict is Verdict.PASS_CONDITIONAL


def receipt_hashes(ex):
    return {s: ex.receipt(s).receipt_hash() for s in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")}


def test_identical_reruns_keep_every_hash_and_the_acknowledgement(fresh_example, tmp_path, home, key, terminal):
    ex = fresh_example("01_clean_strategy")
    ex.register(ex.build_data(), "prices")
    tid = ex.new_trial()
    assert ex.run().exit_code == 0
    assert ex.receipt("S2").diagnostics["truncation_cached"] is False
    first, report1 = receipt_hashes(ex), collected(home, tid).hash
    human_acks(terminal, tid)
    assert ex.run().exit_code == 0
    assert ex.receipt("S2").diagnostics["truncation_cached"] is True      # served from the truncation cache
    assert receipt_hashes(ex) == first                                     # identical inputs, identical receipts
    assert collected(home, tid).hash == report1
    ok, why = is_acknowledged(tid)
    assert ok, why

    # changed evidence kills it: the adapter source
    adapter = ex.dir / "adapter.py"
    adapter.write_text(adapter.read_text() + "\n# a changed adapter source\n")
    assert ex.run().exit_code == 0
    assert not is_acknowledged(tid)[0]


def test_a_changed_dataset_or_answer_invalidates_the_acknowledgement(fresh_example, tmp_path, home, key, terminal):
    assert cli("pack", "add", str(PACKS / "retail-core")).exit_code == 0
    ex = fresh_example("01_clean_strategy")
    ex.register(ex.build_data(), "prices")
    tid = ex.new_trial()
    assert ex.run().exit_code == 0
    human_acks(terminal, tid)
    assert ex.run().exit_code == 0 and is_acknowledged(tid)[0]
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "bar-type-declared", "time", "Time bars, one day each")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "data-history-revised", "never_revised", "No, it is as first published")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "data-timezone", "utc", "UTC")
    ask_and_answer(tmp_path, tid, "S1", "dataset:prices", "timestamp-semantics", "bar_close", "the bar close")
    assert not is_acknowledged(tid)[0]
    assert ex.run().exit_code == 0
    human_acks(terminal, tid)
    assert is_acknowledged(tid)[0]
    # a changed dataset: the same file name with one value edited, registered again
    csv = ex.dir / "data" / "prices.csv"
    lines = csv.read_text().splitlines()
    head, last = lines[-1].rsplit(",", 1)
    lines[-1] = f"{head},{float(last) * 1.01}"
    csv.write_text("\n".join(lines) + "\n")
    ex.register(csv, "prices")
    assert ex.run().exit_code == 0
    assert not is_acknowledged(tid)[0]
