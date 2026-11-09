"""P3e-a1 acceptance: register a synthetic dataset, profile it, propose, verify, settle the critical questions through
the interview, reach ONBOARDED, see the dataset card in the S1 PDF; then a wrong proposal gets the dataset BLOCKED at S1."""

import itertools
import json
import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qrh.cli import app
from qrh.memory.registry import Registry
from qrh.pipeline.receipt import Stage, Verdict, load_receipt
from qrh.pipeline.stages import Pipeline, RegisterHypothesis, make_receipt, register_new_trial
from qrh.pipeline.stages_s1_s2 import DataCheck
from qrh.reports.inputs import load_stage_data
from qrh.reports.render import render_stage
from qrh.synthetic import instruments as inst

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from onboarding_helpers import proposal_file  # noqa: E402

runner = CliRunner()
SPEC = {
    "text": "a moving average rule", "strategy_family": "trend", "asset_class": "equity", "frequency": "1D",
    "parameter_space_size": 4, "thresholds": {"k": 1}, "difference_statement": "baseline", "datasets": ["ds"],
}
RULES = "rules:\n  - type: schema\n    columns: {close: f}\n  - type: range\n    column: close\n    positive: true\n  - type: timezone\n    zone: UTC\n"


class Stub:
    def __init__(self, stage):
        self.id = stage

    def run(self, ctx):
        return make_receipt(ctx, self.id, Verdict.PASS, [])


@pytest.fixture
def env(tmp_path, home):
    frame, proposal = inst.equity_daily()
    csv, rules = tmp_path / "ds.csv", tmp_path / "rules.yaml"
    inst.write_csv(frame, csv)
    rules.write_text(RULES)

    def cli(*args, code=0):
        res = runner.invoke(app, list(args))
        assert res.exit_code == code, res.output
        return res

    cli("data", "register", str(csv), "--name", "ds", "--rules", str(rules), "--privacy", "public")
    counter = itertools.count(1)
    reg = Registry(home / "registry.sqlite3", id_factory=lambda: f"t{next(counter)}")
    tid = register_new_trial(SPEC, reg, home).trial_id
    yield cli, reg, home, tid, proposal, tmp_path
    reg.close()


def run(reg, home, tid):
    stages = [RegisterHypothesis(), DataCheck()] + [Stub(Stage(f"S{i}")) for i in range(2, 7)]
    return Pipeline(reg, home, stages=stages).run(tid)


def test_correct_proposal_to_onboarded_to_the_s1_card(env):
    cli, reg, home, tid, proposal, tmp = env
    prof = json.loads(cli("data", "profile", "ds", "--json").stdout)
    assert prof["profile"]["time_index"]["inferred_frequency"] == "1B"
    cli("data", "propose", "ds", "--file", str(proposal_file(tmp, proposal)))
    cli("data", "verify", "ds")

    out = run(reg, home, tid)  # critical questions are open: the verdict is PASS · CONDITIONAL, never a block
    assert out.final.verdict is Verdict.PASS_CONDITIONAL
    assert {c["question_id"] for c in out.final.params["conditions"]} == {"onb-adjustments", "onb-timestamp_semantics"}
    s1 = load_receipt(home / "receipts" / tid / "S1.json")
    assert s1.params["onboarding"]["ds"]["state"] == "VERIFIED_WITH_GAPS" and len(s1.params["onboarding"]["ds"]["model_hash"]) == 64

    for qid in ("onb-adjustments", "onb-timestamp_semantics"):  # the human settles them (a mocked terminal)
        cli("interview", "next", "--stage", "S1", "--scope", "dataset:ds", "--json")
        cli("interview", "answer", qid, "--scope", "dataset:ds", "--value", "confirm", "--human")
    assert json.loads(cli("data", "onboarding", "ds", "--json").stdout)["state"] == "ONBOARDED"

    out = run(reg, home, tid)
    assert out.final.verdict is Verdict.PASS and "conditions" not in out.final.params
    s1 = load_receipt(home / "receipts" / tid / "S1.json")
    assert s1.params["onboarding"]["ds"]["state"] == "ONBOARDED"

    render_stage(tid, "S1", home, reg)
    pdf = home / "reports" / tid / "S1.pdf"
    chart = json.loads((home / "reports" / tid / "S1.chart.json").read_text())
    assert pdf.read_bytes().startswith(b"%PDF") and len(re.findall(rb"/Type /Page\b(?!s)", pdf.read_bytes())) == 4
    keys = [f["key"] for f in chart["figures"]]
    assert keys == ["s1_coverage", "s1_rules", "s1_dataset_card_0", "s1_onboarding_0"]
    table = chart["figures"][3]
    assert {r["status"] for r in table["rows"]} <= {"VERIFIED", "CONFIRMED", "UNVERIFIED"} and table["state"] == "ONBOARDED"
    card = chart["figures"][2]
    assert {"Rows", "Inferred frequency"} <= {f["label"] for f in card["facts"]}
    assert load_stage_data(home, tid, "S1").artifacts["onboarding"]["datasets"][0]["privacy"] == "public"


def test_a_wrong_proposal_gets_the_dataset_blocked_at_s1(env):
    cli, reg, home, tid, proposal, tmp = env
    cli("data", "propose", "ds", "--file", str(proposal_file(tmp, inst.with_claim(proposal, "price_units", "fraction"))))
    cli("data", "verify", "ds", code=2)
    out = run(reg, home, tid)
    s1 = out.stage_receipts[Stage.S1]
    assert s1.verdict is Verdict.BLOCKED and any("onboarding: price_units" in r for r in s1.reasons)
    assert s1.params["onboarding"]["ds"]["state"] == "CONTRADICTED"
    assert out.final.verdict is not Verdict.PASS
    cli("data", "propose", "ds", "--file", str(proposal_file(tmp, proposal)))  # fix the claim
    cli("data", "verify", "ds")
    assert run(reg, home, tid).stage_receipts[Stage.S1].verdict is Verdict.PASS


def test_a_wrong_proposal_is_still_blocked_even_if_the_rules_pass_and_nobody_reverifies(env):
    cli, reg, home, tid, proposal, tmp = env
    cli("data", "propose", "ds", "--file", str(proposal_file(tmp, inst.with_claim(proposal, "frequency", "1h"))))
    cli("data", "verify", "ds", code=2)
    cli("data", "check", "ds", code=2)  # the quality rules alone would PASS; the contradiction keeps it blocked
