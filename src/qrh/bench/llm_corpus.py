"""The planted-bug corpus for strategies that use a language model (the ``llm-gate`` suite).

Four bug classes plus clean controls, each a recipe materialized per seed into a synthetic frame, the ``llm``
declaration a hypothesis would carry (or none, for undeclared use) and an adapter source that asks a deterministic
stub model through ``qrh.llm.complete``. Nothing here decides a verdict; ``qrh.bench.suites.run_llm_gate`` runs the
S2 detectors over the materialized cases.

- ``llm_cutoff_violation``: the evaluation window starts before the declared cutoff plus the margin;
- ``llm_cutoff_unknown``: a declared model has an unknown cutoff or an unknown cutoff source;
- ``llm_undeclared``: the adapter imports an LLM client library and the hypothesis declares no ``llm``;
- ``llm_prompt_lookahead``: the prompt for a decision is built with a value from after the decision time;
- ``llm_direct_client``: a declared adapter imports a provider client in a source file that does not register a backend
  (LLM002, advisory); a file that calls ``register_backend`` is exempt and is one of the clean controls.

Controls are declared, start after the cutoff plus the margin, and replay cleanly: they must never be flagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from qrh.bench.engine import WARMUP, make_frame
from qrh.llm.stub import StubBackend

LLM_CLASSES = ("llm_cutoff_violation", "llm_cutoff_unknown", "llm_undeclared", "llm_prompt_lookahead",
               "llm_direct_client")
CONTROLS = "controls"
CASE_BARS = 400
MODEL = "bench-model"
PROVIDER = "bench"
BACKEND = StubBackend()
MARGIN_DAYS = 180.0
# The frame starts 2020-01-01 and the development window starts WARMUP bars later (2020-03-01): a cutoff on or before
# 2019-09-02 clears the default margin, one after it does not.
CLEAN_CUTOFF = "2019-06-30"

TEMPLATE = '''import numpy as np
import pandas as pd

from qrh.adapters.base import RunResult
from qrh.llm import complete
@@IMPORTS@@

MODEL = "bench-model"
PARAMS = {"temperature": 0}


class Planted:
    def parameter_space(self):
        return {"lookback": [20]}

    def required_data(self):
        return []

    def run(self, params, data, start, end, cost_multiplier=1.0):
        px = data.as_of(end)["close"]
        ret = px.pct_change()
        lookback = params["lookback"]
        held = np.zeros(len(px))
        for i in range(lookback, len(px)):
@@PROMPT@@
            answer = complete(MODEL, prompt, params=PARAMS, as_of=px.index[i])
            held[i] = @@SIGN@@ * {"UP": 1.0, "DOWN": -1.0}.get(answer.strip(), 0.0)
        position = pd.Series(held, index=px.index)
        strategy = (position.shift(1) * ret).fillna(0.0)
        trades = pd.DataFrame({"qty": position.diff().fillna(0.0)})
        window = slice(start, end)
        return RunResult(strategy.loc[window], trades.loc[window], position.to_frame("weight").loc[window])
'''

_HEAD = '''            closes = ", ".join(f"{c:.4f}" for c in @@SLICE@@)
            prompt = (f"Decision date: {px.index[i].date().isoformat()}\\n"
                      f"Closes, oldest first: {closes}\\n"
                      f"Flat threshold (fraction): 0.001\\n"@@EXTRA@@)'''
PAST = "px.iloc[i - lookback + 1 : i + 1]"


def _prompt(slice_: str = PAST, extra: str = "") -> str:
    extra_line = f'\n                      {extra}' if extra else ""
    return _HEAD.replace("@@SLICE@@", slice_).replace("@@EXTRA@@", extra_line)


def _model(id_: str = MODEL, cutoff: str = CLEAN_CUTOFF, source: str = "provider_documented") -> dict[str, str]:
    return {"id": id_, "provider": PROVIDER, "training_cutoff": cutoff, "cutoff_source": source}


def _spec(models: list[dict[str, str]] | None = None, variants: list[dict[str, Any]] | None = None,
          retrieval: Any = False) -> dict[str, Any]:
    return {
        "models": models or [_model()], "prompt_files": ["prompts/a.txt", "prompts/b.txt"] if variants else [],
        "params": {"temperature": 0}, "live_retrieval": retrieval, "variants": variants or [],
    }


_VARIANTS = lambda model_b: [  # noqa: E731
    {"name": "a", "model": MODEL, "prompt_file": "prompts/a.txt", "params": {}},
    {"name": "b", "model": model_b, "prompt_file": "prompts/b.txt", "params": {}},
]


@dataclass(frozen=True)
class LLMCase:
    bug_class: str
    name: str
    spec: dict[str, Any] | None  # the hypothesis ``llm`` section; None = undeclared
    prompt: str = _prompt()
    imports: str = ""  # extra source lines (the LLM client an undeclared adapter imports)
    sign: float = 1.0
    margin: float = MARGIN_DAYS
    helper: str | None = None  # source of a second module of the adapter (linted, not executed)
    extra_windows: tuple[tuple[str, str, str], ...] = ()  # (name, start, end) of further windows in the spec


@dataclass
class LLMMaterialized:
    case: LLMCase
    frame: pd.DataFrame
    source: str
    spec: dict[str, Any] | None
    windows: dict[str, tuple[str, str]]
    margin: float
    helper: str | None = None

    @property
    def kind(self) -> str:
        return "control" if self.case.bug_class == CONTROLS else "llm"

    def adapter(self) -> Any:
        ns: dict[str, Any] = {}
        exec(compile(self.source, f"<llm-planted:{self.case.bug_class}:{self.case.name}>", "exec"), ns)  # noqa: S102
        return ns["Planted"]()


def _source(case: LLMCase) -> str:
    return (TEMPLATE.replace("@@IMPORTS@@", case.imports).replace("@@PROMPT@@", case.prompt)
            .replace("@@SIGN@@", repr(case.sign)))


def llm_corpus() -> list[LLMCase]:
    cases: list[LLMCase] = []
    V = "llm_cutoff_violation"
    for name, cutoff in (("cutoff_after_window", "2021-06-30"), ("cutoff_inside_window", "2020-06-15"),
                         ("cutoff_equals_window_start", "2020-03-01"), ("cutoff_inside_margin", "2019-09-30"),
                         ("cutoff_one_day_short_of_margin", "2019-09-03")):
        cases.append(LLMCase(V, name, _spec([_model(cutoff=cutoff)])))
    cases.append(LLMCase(V, "tight_margin_365_days", _spec([_model(cutoff=CLEAN_CUTOFF)]), margin=365.0))
    cases.append(LLMCase(V, "two_models_one_late", _spec([_model(cutoff="2019-01-31"), _model("late", "2021-03-31")])))
    cases.append(LLMCase(V, "variant_uses_a_late_model", _spec(
        [_model(cutoff="2019-01-31"), _model("late", "2021-03-31")], _VARIANTS("late"))))
    cases.append(LLMCase(V, "second_window_starts_early", _spec(), extra_windows=(("stress", "2019-10-01", "2019-12-31"),)))
    U = "llm_cutoff_unknown"
    cases.append(LLMCase(U, "cutoff_unknown", _spec([_model(cutoff="unknown")])))
    cases.append(LLMCase(U, "source_unknown", _spec([_model(source="unknown")])))
    cases.append(LLMCase(U, "cutoff_and_source_unknown", _spec([_model(cutoff="unknown", source="unknown")])))
    cases.append(LLMCase(U, "one_of_two_models_unknown", _spec([_model(), _model("second", "unknown")])))
    cases.append(LLMCase(U, "unknown_model_only_in_a_variant", _spec([_model(), _model("other", "unknown", "unknown")],
                                                                      _VARIANTS("other"))))
    N = "llm_undeclared"
    for name, line in (
        ("anthropic_import", "import anthropic"), ("openai_from_import", "from openai import OpenAI"),
        ("google_generativeai", "import google.generativeai as genai"), ("google_genai_from", "from google import genai"),
        ("ollama_import", "import ollama"), ("litellm_from_import", "from litellm import completion"),
        ("langchain_core", "from langchain_core.prompts import PromptTemplate"),
        ("transformers_pipeline", "from transformers import pipeline"), ("llama_cpp_import", "from llama_cpp import Llama"),
        ("vllm_import", "import vllm"), ("aliased_import", "import openai as oa"), ("qrh_llm_import", "import qrh.llm"),
        ("dynamic_import_module", "import importlib\n_m = importlib.import_module('openai')"),
    ):
        cases.append(LLMCase(N, name, None, imports=line))
    L = "llm_prompt_lookahead"
    cases.append(LLMCase(L, "next_close_in_the_slice", _spec(), prompt=_prompt("px.iloc[i - lookback + 1 : i + 2]")))
    cases.append(LLMCase(L, "centered_context_window", _spec(),
                         prompt=_prompt("px.iloc[i - lookback // 2 : i + lookback // 2 + 1]")))
    cases.append(LLMCase(L, "next_return_via_roll", _spec(),
                         prompt=_prompt(extra='f"Next move: {np.roll(ret.to_numpy(), -1)[i]:.5f}\\n"')))
    cases.append(LLMCase(L, "full_sample_mean_in_prompt", _spec(), prompt=_prompt(extra='f"Sample mean: {px.mean():.4f}\\n"')))
    cases.append(LLMCase(L, "highest_close_still_to_come", _spec(),
                         prompt=_prompt(extra='f"High to come: {px.iloc[i:].max():.4f}\\n"')))
    cases.append(LLMCase(L, "next_day_close_by_position", _spec(),
                         prompt=_prompt(extra='f"Tomorrow close: {px.iloc[min(i + 1, len(px) - 1)]:.4f}\\n"')))
    cases.append(LLMCase(L, "shift_minus_one_in_the_prompt", _spec(),
                         prompt=_prompt(extra='f"Next close: {px.shift(-1).iloc[i]:.4f}\\n"')))
    D = "llm_direct_client"
    tolerant = "try:\n    import anthropic\nexcept ImportError:\n    anthropic = None"
    cases.append(LLMCase(D, "client_import_in_the_adapter_module", _spec(), imports=tolerant))
    cases.append(LLMCase(D, "client_call_in_a_helper_module", _spec(), helper=(
        "def ask(prompt):\n    import openai\n    return openai.OpenAI().chat.completions.create(prompt)\n")))
    C = CONTROLS
    cases.append(LLMCase(C, "backend_file_is_exempt", _spec(), imports=(
        tolerant + "\nfrom qrh.llm import register_backend\nregister_backend('bench-exempt', lambda m, p, q: 'UP')"),
        helper="import openai\nfrom qrh.llm import register_backend\nregister_backend('bench-exempt-2', lambda m, p, q: 'UP')\n"))
    cases.append(LLMCase(C, "trend_prompt", _spec()))
    cases.append(LLMCase(C, "mean_reversion_prompt", _spec(), sign=-1.0))
    cases.append(LLMCase(C, "expanding_history_in_the_prompt", _spec(),
                         prompt=_prompt(extra='f"History mean so far: {px.iloc[: i + 1].mean():.4f}\\n"')))
    cases.append(LLMCase(C, "two_declared_models", _spec([_model(), _model("second", "2019-01-31")])))
    cases.append(LLMCase(C, "two_declared_variants", _spec([_model()], _VARIANTS(MODEL))))
    cases.append(LLMCase(C, "estimated_cutoff", _spec([_model(source="estimated")])))
    cases.append(LLMCase(C, "retrieval_with_evidence", _spec(retrieval={"point_in_time_evidence": "snapshots by publication time"})))
    cases.append(LLMCase(C, "cutoff_just_clear_of_the_margin", _spec([_model(cutoff="2019-08-30")])))
    return cases


def materialize_llm(case: LLMCase, seed: int) -> LLMMaterialized:
    frame = make_frame(CASE_BARS, seed, 0.0, 0.01)
    frame["ref_close"] = frame["close"]
    windows = {"development": (frame.index[WARMUP].isoformat(), frame.index[-1].isoformat())}
    windows.update({name: (start, end) for name, start, end in case.extra_windows})
    return LLMMaterialized(case, frame, _source(case), case.spec, windows, case.margin, case.helper)
