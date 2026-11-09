"""End-to-end checks for the three methodology fixes found by a simulated-trader test.

A1: look-ahead that is rarely active (np.roll, a named negative shift, a whole-frame read) is caught by S2.
A2: a strategy that decides and fills on the same close fails S2 with no declaration by the user.
A3: a pre-registered single combination can PASS S5 (PBO not applicable, DSR decides).
"""

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qrh.cli import app
from qrh.pipeline.receipt import Stage, Verdict, load_receipt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from lookahead_helpers import prices, source  # noqa: E402

from qrh.synthetic.generator import Regime, SyntheticConfig, generate  # noqa: E402

runner = CliRunner()
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

RULES = """rules:
  - type: schema
    columns: {close: f}
  - type: range
    column: close
    positive: true
  - type: gaps
    freq: D
  - type: timezone
    zone: UTC
  - type: null_rate_drift
    column: close
    window_fraction: 0.2
    max_increase: 0.05
"""

BAND_HYPOTHESIS = """text: Buy when the close falls below a lower band on a synthetic series.
strategy_family: mean_reversion
asset_class: synthetic
frequency: 1D
parameter_space_size: 1
thresholds: {max_drawdown: 0.25}
difference_statement: First band idea in this family; nothing earlier to differ from.
adapter: "adapter:Band"
datasets: [prices]
windows:
  development: ["2020-02-01", "2021-04-30"]
walk_forward: {train: 150, test: 50}
bounds: {max_abs_period_return: 0.5, max_gross_leverage: 1.0, max_position_weight: 1.0}
"""

TREND_HYPOTHESIS = """text: A moving-average spread carries information about the next bar (a single pre-registered rule).
strategy_family: trend
asset_class: synthetic
frequency: 1D
parameter_space_size: 1
thresholds: {max_drawdown: 0.25}
difference_statement: Single baseline trend rule; nothing earlier to differ from.
adapter: "adapter:TrendAdapter"
datasets: [prices]
windows:
  development: ["2020-03-01", "2023-12-31"]
walk_forward: {train: 400, test: 100}
bounds: {max_abs_period_return: 0.5, max_gross_leverage: 1.0, max_position_weight: 1.0}
"""


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    return h


class Work:
    def __init__(self, tmp_path: Path, home: Path, adapter_src: str, hypothesis: str, data) -> None:
        self.home, self.dir = home, tmp_path / "work"
        self.dir.mkdir()
        (self.dir / "adapter.py").write_text(adapter_src)
        (self.dir / "hypothesis.yaml").write_text(hypothesis)
        (self.dir / "rules.yaml").write_text(RULES)
        (self.dir / "data").mkdir()
        data.to_csv(self.dir / "data" / "prices.csv")
        res = runner.invoke(app, ["data", "register", str(self.dir / "data" / "prices.csv"), "--name", "prices",
                                  "--rules", str(self.dir / "rules.yaml")])
        assert res.exit_code == 0, res.output
        res = runner.invoke(app, ["new", str(self.dir / "hypothesis.yaml")])
        assert res.exit_code == 0, res.output
        self.trial_id = res.stdout.strip().splitlines()[-1]

    def run(self):
        return runner.invoke(app, ["run", self.trial_id])

    def receipt(self, stage: str):
        return load_receipt(self.home / "receipts" / self.trial_id / f"{Stage(stage).value}.json")


def band(tmp_path, home, variant):
    return Work(tmp_path, home, source(variant), BAND_HYPOTHESIS, prices())


def text(r):
    return " | ".join(r.reasons)


# --- A1 ---------------------------------------------------------------------------------------------------------


def test_a_clean_band_strategy_passes_s2_with_both_new_checks_recorded(tmp_path, home):
    w = band(tmp_path, home, "clean")
    w.run()
    s2 = w.receipt("S2")
    assert s2.verdict is Verdict.PASS, text(s2)
    assert s2.params["perturbation"]["verdict"] == "PASS" and s2.params["perturbation"]["requested_cutoffs"] == 20
    assert s2.params["same_bar_execution"]["verdict"] == "PASS"
    assert s2.methods["s2.future_perturbation"] == "future_perturbation@1"
    assert s2.methods["s2.same_bar_execution"] == "same_bar@1"


@pytest.mark.parametrize("variant", ["roll", "negvar", "fullframe"])
def test_sparse_lookahead_fails_s2_through_the_future_perturbation_check(tmp_path, home, variant):
    w = band(tmp_path, home, variant)
    w.run()
    s2 = w.receipt("S2")
    assert s2.verdict is Verdict.FAIL, text(s2)
    assert s2.params["perturbation"]["verdict"] == "FAIL"
    assert any(r.startswith("perturbation:") and "first differing timestamp" in r for r in s2.reasons), text(s2)


def test_the_future_perturbation_still_fails_when_the_lint_is_suppressed(tmp_path, home):
    src = source("roll").replace("nxt = np.roll(ret.to_numpy(), -1)",
                                 "nxt = np.roll(ret.to_numpy(), -1)  # qrh: allow LA006 deliberately wrong for the test")
    w = Work(tmp_path, home, src, BAND_HYPOTHESIS, prices())
    w.run()
    s2 = w.receipt("S2")
    assert s2.verdict is Verdict.FAIL and s2.params["lint"]["verdict"] == "PASS"
    assert s2.params["perturbation"]["verdict"] == "FAIL"


def test_a_whole_frame_statistic_fails_s2(tmp_path, home):
    w = band(tmp_path, home, "globalstd")
    w.run()
    assert w.receipt("S2").verdict is Verdict.FAIL


# --- A2 ---------------------------------------------------------------------------------------------------------


def test_decide_and_fill_on_the_same_close_fails_s2_by_default(tmp_path, home):
    w = band(tmp_path, home, "sameday")
    assert "execution" not in (w.dir / "hypothesis.yaml").read_text()  # nothing declared, nothing pinned
    w.run()
    s2 = w.receipt("S2")
    assert s2.verdict is Verdict.FAIL, text(s2)
    assert s2.params["same_bar_execution"]["verdict"] == "FAIL"
    assert any(r.startswith("same_bar_execution:") and "same-bar execution" in r for r in s2.reasons), text(s2)
    assert s2.params["perturbation"]["verdict"] == "PASS"  # the position itself uses no future data


def test_a_strategy_whose_returns_cannot_be_reconstructed_is_never_passed_silently(tmp_path, home):
    src = source("clean").replace("(position.shift(1) * ret)", "(position.shift(1) * ret * 0.37)")
    w = Work(tmp_path, home, src, BAND_HYPOTHESIS, prices())
    w.run()
    s2 = w.receipt("S2")
    assert s2.params["same_bar_execution"]["verdict"] == "INCONCLUSIVE"
    assert s2.verdict is Verdict.INCONCLUSIVE and "cannot verify" in text(s2)


# --- A3 ---------------------------------------------------------------------------------------------------------


def trend_adapter() -> str:
    src = (EXAMPLES / "01_clean_strategy" / "adapter.py").read_text()
    old = '{"fast": [3, 10, 30], "slow": [60, 120, 240]}'
    assert old in src
    return src.replace(old, '{"fast": [10], "slow": [60]}')


def trending_prices():
    cfg = SyntheticConfig(
        seed=11, start="2020-01-01", periods=1500, freq="D", tz="UTC",
        regimes=[Regime(share=1.0, drift=0.0, volatility=0.01)], trend_persistence=0.97, trend_scale=0.004,
    )
    return generate(cfg)


def noise_prices(seed):
    cfg = SyntheticConfig(seed=seed, start="2020-01-01", periods=1500, freq="D", tz="UTC",
                          regimes=[Regime(share=1.0, drift=0.0, volatility=0.01)])
    return generate(cfg)


def test_a_clean_single_combination_trend_can_pass_s5(tmp_path, home):
    w = Work(tmp_path, home, trend_adapter(), TREND_HYPOTHESIS, trending_prices())
    w.run()
    s5 = w.receipt("S5")
    assert s5.verdict is Verdict.PASS, text(s5)
    assert "pbo" not in s5.metrics and s5.metrics["dsr"] >= 0.95
    assert s5.params["pbo"] == {"applicable": False,
                                "note": "not applicable (one combination, no selection among variants)"}
    assert s5.params["n_strategies"] == 1 and s5.params["n_trials_used"] >= 1
    assert not any("PBO needs" in r or "ValueError" in r for r in s5.reasons)


NOISE_SEED = 0  # set below once a seed that reaches S5 is known


def test_a_single_combination_on_noise_still_fails_on_the_deflated_sharpe(tmp_path, home):
    w = Work(tmp_path, home, trend_adapter(), TREND_HYPOTHESIS, noise_prices(NOISE_SEED))
    w.run()
    s5 = w.receipt("S5")
    assert s5.verdict is Verdict.FAIL, text(s5)
    assert any(r.startswith("DSR") and "below min_dsr" in r for r in s5.reasons), text(s5)
    assert not any("exceeds max_pbo" in r for r in s5.reasons)


# --- review follow-ups ------------------------------------------------------------------------------------------

DUMMY_LLM = """llm:
  models:
    - {id: dummy, provider: none, training_cutoff: "1990-01-01", cutoff_source: provider_documented}
  live_retrieval: false
"""


def test_a_dummy_llm_block_does_not_switch_the_perturbation_test_off(tmp_path, home):
    src = source("roll").replace("nxt = np.roll(ret.to_numpy(), -1)",
                                 "nxt = np.roll(ret.to_numpy(), -1)  # qrh: allow LA006 vectorised")
    w = Work(tmp_path, home, src, BAND_HYPOTHESIS + DUMMY_LLM, prices())
    w.run()
    s2 = w.receipt("S2")
    assert s2.params["perturbation"]["verdict"] == "FAIL", text(s2)
    assert s2.verdict is Verdict.FAIL


def test_the_llm_example_with_real_calls_gets_its_perturbation_test(shared_09):
    s2 = shared_09.receipt("S2")
    assert s2.params["perturbation"]["verdict"] == "PASS", text(s2)
    assert s2.params["perturbation"]["n_cutoffs"] > 0


def test_a_perturbation_test_that_could_not_run_makes_s2_inconclusive_never_pass(tmp_path, home, monkeypatch):
    import qrh.pipeline.stages_s1_s2 as stage

    monkeypatch.setattr(stage, "current_session", lambda: None)  # a declared llm with no replay session
    w = Work(tmp_path, home, source("clean"), BAND_HYPOTHESIS + DUMMY_LLM, prices())
    w.run()
    s2 = w.receipt("S2")
    assert s2.params["perturbation"]["verdict"] == "NOT RUN"
    assert s2.verdict is Verdict.INCONCLUSIVE, text(s2)
