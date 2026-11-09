"""LLM001: static detection of an LLM client used by an adapter whose hypothesis declares none."""

import textwrap

import pytest

from qrh.checks.lookahead import BLOCK
from qrh.llm.lint import LLM_RULE, lint_llm_source, lint_llm_use


def hits(src, extra=()):
    return lint_llm_source(textwrap.dedent(src), "adapter.py", extra)


@pytest.mark.parametrize("line", [
    "import anthropic",
    "from anthropic import Anthropic",
    "import openai",
    "from openai import OpenAI",
    "import google.generativeai as genai",
    "from google.generativeai import GenerativeModel",
    "from google import genai",
    "import google.genai",
    "import ollama",
    "import litellm",
    "from litellm import completion",
    "import langchain",
    "from langchain_core.prompts import PromptTemplate",
    "from langchain_openai import ChatOpenAI",
    "import mistralai",
    "import cohere",
    "import together",
    "import groq",
    "from transformers import pipeline",
    "import llama_cpp",
    "from llama_cpp import Llama",
    "import vllm",
    "from vllm import LLM",
    "import qrh.llm",
    "from qrh import llm",
    "from qrh.llm import complete",
    "from qrh.llm.stub import StubBackend",
])
def test_imports_of_llm_clients_are_flagged(line):
    found = hits(f"{line}\n\n\nclass A:\n    pass\n")
    assert [f.rule_id for f in found] == [LLM_RULE] == ["LLM001"]
    f = found[0]
    assert f.severity == BLOCK and not f.suppressed and f.line == 1
    assert f.message == "adapter appears to use an LLM but the hypothesis declares none"
    assert line in f.snippet


@pytest.mark.parametrize("src", [
    "import importlib\nm = importlib.import_module('openai')\n",
    "m = __import__('anthropic')\n",
    "import importlib\nm = importlib.import_module('google.generativeai')\n",
    "import transformers\np = transformers.pipeline('text-generation')\n",
])
def test_calls_that_load_an_llm_client_are_flagged(src):
    assert [f.rule_id for f in hits(src)] == ["LLM001"]


@pytest.mark.parametrize("src", [
    "import pandas as pd\nimport numpy as np\nfrom qrh.adapters.base import RunResult\n",
    "import transformers\n",               # the library alone is not a pipeline
    "from transformers import AutoTokenizer\n",
    "import google.cloud.storage\n",
    "from google import protobuf\n",
    "import importlib\nm = importlib.import_module('pandas')\n",
    "x = 'import openai'\n",                # a string is not an import
    "# import openai\n",
    "import qrhelper\n",
    "import openaix\n",
])
def test_ordinary_code_is_not_flagged(src):
    assert hits(src) == []


def test_function_local_imports_are_found():
    src = "def run():\n    import ollama\n    return ollama\n"
    assert [(f.rule_id, f.line) for f in hits(src)] == [("LLM001", 2)]


def test_a_reasoned_allow_comment_suppresses_on_the_same_or_the_previous_line():
    same = hits("import openai  # qrh: allow LLM001 only the type hints are used, no call is made\n")
    assert same[0].suppressed and same[0].reason == "only the type hints are used, no call is made"
    above = hits("# qrh: allow LLM001 vendored stub, never called\nimport openai\n")
    assert above[0].suppressed and above[0].reason == "vendored stub, never called"


def test_an_allow_comment_without_a_reason_does_not_suppress():
    assert not hits("import openai  # qrh: allow LLM001\n")[0].suppressed


def test_a_pack_can_add_libraries():
    assert hits("import acme_llm\n") == []
    assert [f.rule_id for f in hits("import acme_llm\n", extra=("acme_llm",))] == ["LLM001"]
    assert [f.rule_id for f in hits("from acme_llm.client import Chat\n", extra=("acme_llm",))] == ["LLM001"]


def test_unparseable_source_is_left_to_the_existing_lint():
    assert hits("def (:\n") == []


def test_lint_llm_use_reads_files_and_reports_the_path(tmp_path):
    a = tmp_path / "adapter.py"
    a.write_text("import openai\n")
    b = tmp_path / "other.py"
    b.write_text("import pandas\n")
    found = lint_llm_use([a, b])
    assert [(f.rule_id, f.file) for f in found] == [("LLM001", str(a))]


def test_the_existing_lookahead_lint_does_not_report_llm001():
    from qrh.checks.lookahead import lint_source

    assert lint_source("import openai\n") == []


# --- LLM002: a provider client used outside qrh.llm.complete, in an adapter that declares an llm ---------------

from qrh.checks.lookahead import WARN  # noqa: E402
from qrh.llm.lint import DIRECT_MESSAGE, DIRECT_RULE, lint_llm_direct, lint_llm_direct_source  # noqa: E402


def direct(src, extra=()):
    return lint_llm_direct_source(textwrap.dedent(src), "adapter.py", extra)


def test_the_message_and_rule_are_the_specified_ones():
    assert DIRECT_RULE == "LLM002"
    assert DIRECT_MESSAGE == ("provider client used outside qrh.llm.complete; replay cache and the truncation prompt "
                              "detector cannot see these calls")


@pytest.mark.parametrize("line", [
    "import anthropic", "from openai import OpenAI", "import google.generativeai as genai", "from google import genai",
    "import ollama", "from litellm import completion", "from langchain_core.prompts import PromptTemplate",
    "from transformers import pipeline", "from llama_cpp import Llama", "import vllm",
])
def test_a_direct_client_import_is_a_warn_level_finding(line):
    [f] = direct(f"{line}\n\nclass A:\n    pass\n")
    assert (f.rule_id, f.severity, f.line, f.message, f.suppressed) == ("LLM002", WARN, 1, DIRECT_MESSAGE, False)


def test_a_call_that_loads_a_client_by_name_is_found():
    assert [f.rule_id for f in direct("import importlib\nm = importlib.import_module('openai')\n")] == ["LLM002"]


@pytest.mark.parametrize("src", [
    "import qrh.llm\n", "from qrh import llm\n", "from qrh.llm import complete, read_prompt\n",
    "import pandas\n", "x = 'import openai'\n",
])
def test_the_sanctioned_entry_point_and_ordinary_code_are_not_flagged(src):
    assert direct(src) == []


@pytest.mark.parametrize("register", [
    "register_backend('p', fn)", "llm.register_backend('p', fn)", "qrh.llm.register_backend('p', fn)",
])
def test_a_file_that_registers_a_backend_is_the_intended_place_for_provider_code(register):
    src = f"import anthropic\nfrom qrh import llm\nimport qrh.llm\nfn = lambda m, p, params: ''\n{register}\n"
    assert direct(src) == []


def test_a_mere_mention_of_register_backend_is_not_a_call():
    assert [f.rule_id for f in direct("import anthropic\nx = 'register_backend'\nregister_backend\n")] == ["LLM002"]


def test_llm002_is_suppressed_like_the_other_rules():
    [f] = direct("import anthropic  # qrh: allow LLM002 type hints only\n")
    assert f.suppressed and f.reason == "type hints only"
    [g] = direct("import anthropic  # qrh: allow LLM002\n")
    assert not g.suppressed


def test_a_pack_can_add_libraries_to_llm002_too():
    assert direct("import acme_llm\n") == []
    assert [f.rule_id for f in direct("import acme_llm\n", extra=("acme_llm",))] == ["LLM002"]


def test_lint_llm_direct_checks_each_file_on_its_own(tmp_path):
    adapter, helper, backend = tmp_path / "adapter.py", tmp_path / "helper.py", tmp_path / "backend.py"
    adapter.write_text("import pandas\n")
    helper.write_text("import openai\n")
    backend.write_text("import openai\nfrom qrh.llm import register_backend\nregister_backend('p', lambda m, p, q: '')\n")
    assert [(f.rule_id, f.file) for f in lint_llm_direct([adapter, helper, backend])] == [("LLM002", str(helper))]
