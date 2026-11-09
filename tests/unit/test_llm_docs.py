"""The documentation, skill and README state the LLM rules, the privacy rule and the limits."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(*parts):
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def test_llm_doc_covers_the_rules_the_field_the_cache_the_examples_and_the_limits():
    text = read("docs", "llm.md")
    for item in ("## The `llm` field of the hypothesis", "## Rule 1", "## Rule 2", "## Rule 3", "## Examples",
                 "## Limits, stated honestly", "LLM001", "llm_cutoff_margin_days", "qrh.llm.complete",
                 "register_backend", "--llm-mode record", "qrh llm verify", "LLMCacheMiss", "as_of",
                 "prompt differs between full and truncated data", "S2 LLM contamination gate", "examples/bench/llm_gate",
                 "provider claim", "fine-tuned", "never in the\n  repository", "variants-outside-harness",
                 "examples/09_llm_clean", "examples/08_llm_contamination"):
        assert item in text, item


def test_llm_doc_cites_the_bench_receipt_that_exists():
    import re

    text = read("docs", "llm.md")
    assert "BENCH_" not in text, "placeholders must be filled in"
    folder = ROOT / "examples" / "bench" / "llm_gate"
    assert (folder / "receipt.json").is_file() and (folder / "report.txt").is_file() and (folder / "report.pdf").is_file()
    assert not (folder / "chart.json").exists()
    bench_id = re.search(r"bench id `(bench-[0-9a-f]{12})`", text)[1]
    assert bench_id in (folder / "report.txt").read_text(encoding="utf-8")


def test_security_doc_says_the_cache_is_private_and_write_protected():
    text = read("docs", "security.md")
    assert "llm_cache" in text and "never in the repository" in text and "writing" in text


def test_methods_architecture_readiness_interviews_and_packs_docs_mention_part_b():
    assert "s2.llm_gate" in read("docs", "methods.md") and "llm.replay_cache" in read("docs", "methods.md")
    assert "Rules for strategies that use an LLM" in read("docs", "methods.md")
    assert "## Language-model strategies" in read("docs", "architecture.md")
    assert "llm-live-retrieval" in read("docs", "interviews.md") and "uses_llm" in read("docs", "packs.md")
    assert "llm-recorded-cache" not in read("docs", "readiness.md") and "recorded cache" in read("docs", "readiness.md")
    assert "LLM replay cache" in read("docs", "hooks.md")


def test_readme_marks_part_b_done_and_names_the_next_phase():
    text = read("README.md")
    assert "P3d-b2 part B" in text and "docs/llm.md" in text and "P3e" in text and "the three LLM rules" in text
    assert "LLM-drafted pack lessons" not in text


def test_skill_tells_the_agent_the_llm_rules():
    text = read("skills", "qrh-workflow", "SKILL.md").lower()
    for item in ("declare it", "never invent a cutoff", "unknown", "qrh.llm.complete", "never call a provider directly",
                 "record before replay", "--llm-mode record", "llm_cache"):
        assert item.lower() in text, item


def test_llm002_is_documented_with_its_limits():
    text = read("docs", "llm.md")
    for item in ("LLM002", "register_backend", "llm-direct-client", "best-effort", "llm_direct_client"):
        assert item in text, item
    assert "LLM002" in read("docs", "readiness.md") and "LLM002" in read("docs", "security.md")
    assert "LLM002" in read("skills", "qrh-workflow", "SKILL.md")
