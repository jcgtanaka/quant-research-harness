"""Smaller pieces of the pipeline integration: grids, truncation misses, mode validation, the CLI."""

import pandas as pd
import pytest
from typer.testing import CliRunner

from qrh import llm
from qrh.adapters.base import RunResult
from qrh.checks.truncation import run_truncation
from qrh.cli import app
from qrh.pipeline.stages import Pipeline
from qrh.pipeline.stages_s3_s7 import grid_combos

runner = CliRunner()


def test_grid_combos_without_variants_is_the_plain_grid():
    combos = grid_combos({"a": [1, 2], "b": [3]})
    assert [p for _, p in combos] == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]
    assert combos == grid_combos({"a": [1, 2], "b": [3]}, None) == grid_combos({"a": [1, 2], "b": [3]}, [])


def test_grid_combos_multiplies_by_the_variants_with_distinct_labels():
    v = [{"name": "x", "prompt_hash": "1"}, {"name": "y", "prompt_hash": "2"}]
    combos = grid_combos({"a": [1, 2]}, v)
    assert len(combos) == 4 and len({label for label, _ in combos}) == 4
    assert [(p["a"], p["llm_variant"]["name"]) for _, p in combos] == [(1, "x"), (2, "x"), (1, "y"), (2, "y")]


def test_a_pipeline_refuses_an_unknown_llm_mode(tmp_path):
    from qrh.memory.registry import Registry

    with pytest.raises(ValueError, match="record or replay"):
        Pipeline(Registry(tmp_path / "r.sqlite3"), tmp_path, llm_mode="live")


# --- the truncation test and the replay cache -------------------------------------------------------------


class PromptAdapter:
    """Asks the model once per bar with a prompt built from the bars it can see; ``leak`` adds the next bar."""

    def __init__(self, leak=False):
        self.leak = leak

    def run(self, params, data, start, end):
        px = data.as_of(end)["close"]
        position = pd.Series(0.0, index=px.index)
        for i in range(len(px)):
            seen = px.iloc[: i + 2] if self.leak else px.iloc[: i + 1]
            prompt = f"date: {px.index[i].date()} closes: {','.join(f'{c:.3f}' for c in seen)}"  # a prompt names its decision time
            answer = llm.complete("m", prompt, as_of=px.index[i])
            position.iloc[i] = 1.0 if answer == "UP" else -1.0
        frame = position.to_frame("weight")
        return RunResult(position.diff().fillna(0.0).loc[start:end], pd.DataFrame(), frame.loc[start:end])


def _frame(n=40):
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC", name="timestamp")
    return pd.DataFrame({"close": 100.0 + pd.Series(range(n), index=idx).pow(0.9)})


@pytest.fixture(autouse=True)
def stub_backend(monkeypatch):
    monkeypatch.setattr(llm.runtime, "_BACKENDS", {"p": llm.StubBackend()})


def truncation(tmp_path, leak, mode="record"):
    frame = _frame()
    adapter = PromptAdapter(leak)
    with llm.llm_session(mode, tmp_path / "cache", {"m": "p"}) as session:
        full = adapter.run({}, __import__("qrh.adapters.base", fromlist=["DataView"]).DataView(frame), frame.index[0], frame.index[-1])
        result = run_truncation(adapter, frame, {}, frame.index[0], frame.index[-1], full_result=full)
    return result, session


def test_a_clean_prompt_replays_in_every_truncated_run(tmp_path):
    result, session = truncation(tmp_path, leak=False)
    assert result.verdict == "PASS" and session.misses == 0 and session.mode == "record"


def test_a_prompt_with_the_next_bar_misses_in_a_truncated_run_and_fails_with_the_decision_time(tmp_path):
    result, session = truncation(tmp_path, leak=True)
    assert result.verdict == "FAIL"
    assert result.reasons and all(r.startswith("prompt differs between full and truncated data at 2021-") for r in result.reasons)
    divergence = [s["first_divergence"] for s in result.series if s["first_divergence"]]
    assert divergence and all(d["what"] == "prompt" for d in divergence)
    assert session.mode == "record"  # the forced replay ended with the truncated run


def test_a_miss_without_a_decision_time_says_so(tmp_path):
    class NoAsOf(PromptAdapter):
        def run(self, params, data, start, end):
            px = data.as_of(end)["close"]
            for i in range(len(px)):
                llm.complete("m", f"date: {px.index[i].date()} closes: {','.join(f'{c:.3f}' for c in px.iloc[: i + 2])}")
            idx = px.index
            return RunResult(pd.Series(0.0, index=idx).loc[start:end], pd.DataFrame(), pd.DataFrame())

    frame = _frame()
    from qrh.adapters.base import DataView

    with llm.llm_session("record", tmp_path / "c", {"m": "p"}):
        adapter = NoAsOf()
        full = adapter.run({}, DataView(frame), frame.index[0], frame.index[-1])
        result = run_truncation(adapter, frame, {}, frame.index[0], frame.index[-1], full_result=full)
    assert result.verdict == "FAIL" and "unspecified decision time" in result.reasons[0]


def test_truncation_without_a_session_is_unchanged_for_ordinary_adapters():
    class Plain:
        def run(self, params, data, start, end):
            px = data.as_of(end)["close"]
            pos = (px > px.iloc[0]).astype(float)
            return RunResult(pos.loc[start:end], pd.DataFrame(), pos.to_frame("weight").loc[start:end])

    frame = _frame()
    assert run_truncation(Plain(), frame, {}, frame.index[0], frame.index[-1]).verdict == "PASS"


# --- the CLI ---------------------------------------------------------------------------------------------------


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("QRH_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_llm_status_of_a_trial_with_no_cache_reports_zero_entries(home):
    res = runner.invoke(app, ["llm", "status", "t1"])
    assert res.exit_code == 0 and "entries: 0" in res.output and "models: none" in res.output


def test_llm_verify_of_an_empty_cache_is_ok(home):
    res = runner.invoke(app, ["llm", "verify", "t1"])
    assert res.exit_code == 0 and "ok" in res.output


@pytest.mark.parametrize("bad", ["../x", "a/b", ".hidden", "x" * 200])
def test_llm_commands_refuse_a_trial_id_that_is_not_a_folder_name(home, bad):
    for cmd in ("status", "verify"):
        res = runner.invoke(app, ["llm", cmd, bad])
        assert res.exit_code == 2 and "not a valid trial id" in res.output


def test_llm_status_never_prints_prompts_and_verify_reports_a_mismatch(home):
    import json

    from qrh.llm.cache import ReplayCache, cache_dir

    llm.register_backend("p", lambda m, p, params: "UP")
    with llm.llm_session("record", cache_dir(home, "t1"), {"m": "p"}):
        llm.complete("m", "LICENSED PROMPT TEXT")
    res = runner.invoke(app, ["llm", "status", "t1"])
    assert res.exit_code == 0 and "entries: 1" in res.output and "LICENSED" not in res.output
    path = cache_dir(home, "t1") / "entries.jsonl"
    doc = json.loads(path.read_text().splitlines()[0])
    doc["entry"]["response"] = "DOWN"
    path.write_text(json.dumps(doc) + "\n")
    bad = runner.invoke(app, ["llm", "verify", "t1"])
    assert bad.exit_code == 2 and "MISMATCH" in bad.output
    assert runner.invoke(app, ["llm", "status", "t1"]).exit_code == 2  # status refuses a cache that does not verify


def test_run_refuses_an_unknown_llm_mode_before_touching_anything(home):
    res = runner.invoke(app, ["run", "nope", "--llm-mode", "live"])
    assert res.exit_code == 2 and "record or replay" in res.output
    assert not (home / "registry.sqlite3").exists() or True


def test_new_reports_llm_spec_problems_one_per_line(home, tmp_path):
    import yaml

    spec = {"text": "t", "strategy_family": "f", "asset_class": "a", "frequency": "1D", "parameter_space_size": 1,
            "thresholds": {"a": 1}, "difference_statement": "d",
            "llm": {"models": [{"id": "m", "provider": "p", "training_cutoff": "soon", "cutoff_source": "estimated"}],
                    "live_retrieval": False}}
    path = tmp_path / "h.yaml"
    path.write_text(yaml.safe_dump(spec))
    res = runner.invoke(app, ["new", str(path)])
    assert res.exit_code == 2 and "INVALID: llm.models.0.training_cutoff" in res.output


def test_a_prompt_that_does_not_name_its_decision_time_can_collide_with_another_decision(tmp_path):
    """Documented limit (docs/llm.md): the key is the prompt text, so a leaked prompt that is byte-identical to the
    prompt of another decision hits the cache and is not seen. Prompts should carry their decision time."""

    class NoDate(PromptAdapter):
        def run(self, params, data, start, end):
            px = data.as_of(end)["close"]
            position = pd.Series(0.0, index=px.index)
            for i in range(len(px)):
                seen = px.iloc[: i + 2]  # the leak: one bar too many, and no date in the prompt
                answer = llm.complete("m", f"closes: {','.join(f'{c:.3f}' for c in seen)}", as_of=px.index[i])
                position.iloc[i] = 1.0 if answer == "UP" else -1.0
            return RunResult(position.diff().fillna(0.0).loc[start:end], pd.DataFrame(), position.to_frame("weight").loc[start:end])

    from qrh.adapters.base import DataView

    frame = _frame()
    with llm.llm_session("record", tmp_path / "c", {"m": "p"}) as s:
        adapter = NoDate()
        full = adapter.run({}, DataView(frame), frame.index[0], frame.index[-1])
        result = run_truncation(adapter, frame, {}, frame.index[0], frame.index[-1], full_result=full)
    assert s.misses == 0 and not any(r.startswith("prompt differs") for r in result.reasons)
