"""S5 end to end: the deflated Sharpe variance floor and the parameter stability check (s5.param_stability@1)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_methodology_fixes import TREND_HYPOTHESIS, Work, text, trend_adapter, trending_prices  # noqa: E402

from qrh.memory.registry import Registry  # noqa: E402
from qrh.pipeline.receipt import Verdict  # noqa: E402
from qrh.synthetic.generator import Regime, SyntheticConfig, generate  # noqa: E402

@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    return h


SPIKE_ADAPTER = '''
import numpy as np
import pandas as pd
from qrh.adapters.base import DatasetRef, RunResult


class Spike:
    """Only the exact setting w=10 reads the planted column; any other setting reads an unrelated shifted copy."""

    def parameter_space(self):
        return {"w": [10]}

    def required_data(self):
        return [DatasetRef("prices")]

    def run(self, params, data, start, end):
        frame = data.as_of(end)
        ret = frame["close"].pct_change()
        oracle = frame["oracle"] if params["w"] == 10 else frame["oracle"].shift(params["w"])
        position = np.sign(oracle).fillna(0.0)
        strategy = (position.shift(1) * ret).fillna(0.0)
        window = slice(start, end)
        return RunResult(strategy.loc[window], pd.DataFrame({"qty": position.diff().fillna(position)}).loc[window],
                         position.to_frame("weight").loc[window])
'''

SPIKE_HYPOTHESIS = TREND_HYPOTHESIS.replace("adapter:TrendAdapter", "adapter:Spike").replace("trend", "spike")


def planted_prices():
    cfg = SyntheticConfig(seed=5, start="2020-01-01", periods=1500, freq="D", tz="UTC",
                          regimes=[Regime(share=1.0, drift=0.0, volatility=0.01)])
    df = generate(cfg)
    df["oracle"] = (df["close"].pct_change().shift(-1)).apply(lambda x: 0.0 if x != x else (1.0 if x > 0 else -1.0))
    return df


def s5_of(work):
    work.run()
    return work.receipt("S5")


def test_the_variance_floor_is_recorded_in_the_s5_receipt(tmp_path, home):
    w = Work(tmp_path, home, trend_adapter(), TREND_HYPOTHESIS, trending_prices())
    s5 = s5_of(w)
    assert s5.verdict is Verdict.PASS, text(s5)
    v = s5.params["var_sharpe"]
    assert set(v) == {"grid", "estimator", "used"} and v["grid"] is None and v["used"] == "estimator"
    assert s5.metrics["var_sharpe_estimator"] > 0 and "var_sharpe_grid" not in s5.metrics
    assert s5.metrics["var_sharpe"] == pytest.approx(s5.metrics["var_sharpe_estimator"])
    assert any("Sharpe variance" in r and "estimator" in r for r in s5.reasons)


def test_a_clean_single_combination_has_a_stable_neighbourhood_and_adds_no_trials(tmp_path, home):
    w = Work(tmp_path, home, trend_adapter(), TREND_HYPOTHESIS, trending_prices())
    s5 = s5_of(w)
    ps = s5.params["param_stability"]
    assert ps["status"] == "PASS" and s5.methods["s5.param_stability"] == "param_stability@1"
    assert ps["neighbour"]["n_neighbours"] == 8 and ps["neighbour"]["source"] == "generated"
    assert s5.metrics["param_neighbour_ratio"] >= 0.5
    assert "not counted" in ps["trial_count_note"]
    reg = Registry(home / "registry.sqlite3")
    try:
        assert reg.trial_count(w.trial_id) == 1  # only the pre-registered combination is a trial
        assert s5.params["family_trial_count"] == 1
    finally:
        reg.close()


def test_a_single_combination_that_only_works_at_one_setting_fails_s5_as_a_spike(tmp_path, home):
    w = Work(tmp_path, home, SPIKE_ADAPTER, SPIKE_HYPOTHESIS, planted_prices())
    s5 = s5_of(w)
    assert s5.metrics["dsr"] >= 0.95  # the level is excellent, so only the neighbourhood can fail it
    assert s5.verdict is Verdict.FAIL, text(s5)
    assert s5.params["param_stability"]["status"] == "FAIL"
    assert any(r.startswith("parameter stability") and "neighbour" in r for r in s5.reasons), text(s5)


def test_a_spike_hidden_behind_a_string_typed_parameter_fails_s5(tmp_path, home):
    src = SPIKE_ADAPTER.replace('{"w": [10]}', '{"w": ["10"]}').replace('params["w"]', 'int(params["w"])')
    w = Work(tmp_path, home, src, SPIKE_HYPOTHESIS, planted_prices())
    s5 = s5_of(w)
    assert s5.verdict is Verdict.FAIL, text(s5)
    assert s5.params["param_stability"]["status"] == "FAIL" and s5.params["param_stability"]["skipped"] == []


def test_an_inert_parameter_cannot_make_the_spike_pass(tmp_path, home):
    src = SPIKE_ADAPTER.replace('{"w": [10]}', '{"w": [10], "k": [1.0]}').replace(
        'oracle = frame["oracle"] if', 'oracle = frame["oracle"] * params["k"] if')
    w = Work(tmp_path, home, src, SPIKE_HYPOTHESIS, planted_prices())
    s5 = s5_of(w)
    nb = s5.params["param_stability"]["neighbour"]
    assert s5.verdict is Verdict.FAIL, text(s5)
    assert nb["worst_parameter"] == "w" and nb["pooled_mean_ratio"] > nb["ratio"]


def test_declared_hairline_neighbours_block_s5(tmp_path, home):
    src = SPIKE_ADAPTER.replace('{"w": [10]}', '{"w": [10.0]}').replace(
        'if params["w"] == 10 else frame["oracle"].shift(params["w"])',
        'if abs(params["w"] - 10.0) < 0.01 else frame["oracle"].shift(7)')
    hyp = SPIKE_HYPOTHESIS + "\nstability:\n  neighbours:\n    w: [9.999, 10.001]\n"
    w = Work(tmp_path, home, src, hyp, planted_prices())
    s5 = s5_of(w)
    assert s5.verdict is Verdict.BLOCKED and "too close" in text(s5)


def test_bounds_that_leave_no_room_block_s5_instead_of_passing(tmp_path, home):
    hyp = SPIKE_HYPOTHESIS + "\nstability:\n  bounds:\n    w: [10, 10]\n"
    w = Work(tmp_path, home, SPIKE_ADAPTER, hyp, planted_prices())
    s5 = s5_of(w)
    assert s5.verdict is Verdict.BLOCKED, text(s5)


def test_a_strategy_whose_parameters_are_not_numeric_and_do_not_vary_is_blocked_not_passed(tmp_path, home):
    src = trend_adapter().replace('{"fast": [10], "slow": [60]}', '{"fast": ["a"], "slow": ["b"]}').replace(
        'params["fast"]', '10').replace('params["slow"]', '60')
    w = Work(tmp_path, home, src, TREND_HYPOTHESIS, trending_prices())
    s5 = s5_of(w)
    assert s5.verdict is Verdict.BLOCKED and "none is numeric" in text(s5), text(s5)


def test_a_declared_stability_block_is_listed_in_the_readiness_report(tmp_path, home):
    from qrh.readiness import collect as rc

    hyp = TREND_HYPOTHESIS + "\nstability:\n  neighbours:\n    fast: [7, 13]\n"
    w = Work(tmp_path, home, trend_adapter(), hyp, trending_prices())
    w.run()
    codes = {x.code for x in rc.collect(home, w.trial_id).warnings}
    assert "stability-declared" in codes


def test_old_hypotheses_without_the_stability_thresholds_use_the_defaults_and_say_so(tmp_path, home, monkeypatch):
    import qrh.validation.defaults as defaults

    with monkeypatch.context() as m:  # register as an earlier version of the harness did: no stability thresholds
        m.setattr(defaults, "DEFAULT_THRESHOLDS",
                  {k: v for k, v in defaults.DEFAULT_THRESHOLDS.items() if not k.startswith("stability_")})
        w = Work(tmp_path, home, trend_adapter(), TREND_HYPOTHESIS, trending_prices())
    reg = Registry(home / "registry.sqlite3")
    assert not any(k.startswith("stability_") for k in reg.get_trial(w.trial_id)["thresholds"])
    reg.close()
    s5 = s5_of(w)
    ps = s5.params["param_stability"]
    assert ps["thresholds_source"] == "defaults (not frozen in the hypothesis)" and ps["thresholds"]["steps"] == [0.1, 0.2]
