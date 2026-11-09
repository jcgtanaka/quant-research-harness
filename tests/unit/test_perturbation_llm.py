"""The future-perturbation test for a strategy that asks a model: misses after the cutoff get a stub, others fail."""

import numpy as np
import pandas as pd
import pytest

from qrh.adapters.base import DataView, DatasetRef, RunResult
from qrh.checks.perturbation import run_perturbation
from qrh.llm import complete, llm_session, register_backend
from qrh.llm.cache import LLMCacheMiss
from qrh.llm.runtime import replay_only, stub_misses_after

H = "a" * 64


def backend(model, prompt, params):
    return "UP" if len(prompt) % 2 else "DOWN"


@pytest.fixture
def session(tmp_path):
    register_backend("testprov", backend)
    with llm_session("record", tmp_path / "cache", {"m": "testprov"}, dataset_hash=H, code_hash=H) as s:
        yield s


def frame(n=120):
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"close": 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))}, index=idx)


class Asker:
    def __init__(self, leak=False):
        self.leak = leak

    def parameter_space(self):
        return {}

    def required_data(self):
        return [DatasetRef("prices")]

    def run(self, params, data, start, end):
        px = data.as_of(end)["close"]
        pos = np.zeros(len(px))
        for i in range(5, len(px) - 1):
            seen = px.iloc[i - 4 : i + (2 if self.leak else 1)]  # the leak adds the next close
            answer = complete("m", ",".join(f"{c:.4f}" for c in seen), as_of=px.index[i])
            pos[i] = 1.0 if answer == "UP" else -1.0
        position = pd.Series(pos, index=px.index)
        r = px.pct_change()
        w = slice(start, end)
        return RunResult((position.shift(1) * r).fillna(0.0).loc[w], pd.DataFrame(), position.to_frame("weight").loc[w])


def test_a_miss_after_the_cutoff_gets_a_deterministic_stub_and_an_earlier_one_fails(session):
    complete("m", "p1", as_of="2020-01-01")
    with replay_only(), stub_misses_after(pd.Timestamp("2020-02-01", tz="UTC")):
        first = complete("m", "new prompt", as_of=pd.Timestamp("2020-03-01", tz="UTC"))
        again = complete("m", "other new prompt", as_of=pd.Timestamp("2020-03-02", tz="UTC"))
        assert first == again  # deterministic
        for as_of in (pd.Timestamp("2020-02-01", tz="UTC"), pd.Timestamp("2020-01-15", tz="UTC"), None):
            with pytest.raises(LLMCacheMiss):
                complete("m", f"missing {as_of}", as_of=as_of)


def run(leak):
    df = frame()
    ad = Asker(leak)
    s, e = df.index[10], df.index[-1]
    full = ad.run({}, DataView(df), s, e)
    return run_perturbation(ad, df, {}, s, e, full_result=full, context=lambda cutoff: _ctx(cutoff))


def _ctx(cutoff):
    import contextlib

    @contextlib.contextmanager
    def both():
        with replay_only(), stub_misses_after(cutoff):
            yield

    return both()


def test_a_clean_prompt_strategy_passes(session):
    res = run(False)
    assert res.verdict == "PASS", res.reasons


def test_a_prompt_that_contains_a_later_close_fails(session):
    res = run(True)
    assert res.verdict == "FAIL" and "prompt" in " ".join(res.reasons)
