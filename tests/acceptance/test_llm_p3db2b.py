"""P3d-b2 part B end to end: the three rules for strategies that use a language model (examples 08 and 09)."""

import json

import pytest
import yaml
from typer.testing import CliRunner

from qrh.cli import app
from qrh.hashing import sha256_file
from qrh.pipeline.receipt import Verdict

runner = CliRunner()
STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")


def text(r):
    return " | ".join(r.reasons)


def edit_spec(ex, **changes):
    p = ex.dir / "hypothesis.yaml"
    spec = yaml.safe_load(p.read_text())
    for key, value in changes.items():
        if value is None:
            spec.pop(key, None)
        else:
            spec[key] = value
    p.write_text(yaml.safe_dump(spec))


def edit_llm(ex, **changes):
    spec = yaml.safe_load((ex.dir / "hypothesis.yaml").read_text())
    llm = spec["llm"]
    for key, value in changes.items():
        llm[key] = value
    edit_spec(ex, llm=llm)


@pytest.fixture
def ran09(shared_09, monkeypatch):
    """The shared example-09 run, with QRH_HOME pointing at it for the CLI."""
    monkeypatch.setenv("QRH_HOME", str(shared_09.home))
    return shared_09


# --- example 09: declared, window after the cutoff, replayable ------------------------------------------


def test_09_records_then_replays_and_passes_s0_to_s7(shared_09):
    ex = shared_09
    assert ex.record_result.exit_code == 0, ex.record_result.output
    assert ex.replay_result.exit_code == 0, ex.replay_result.output
    for s in STAGES:
        assert ex.first_receipts[s].verdict is Verdict.PASS, f"{s}: {text(ex.first_receipts[s])}"
    s3 = ex.receipt("S3")
    assert s3.metrics["grid_size"] == s3.metrics["evaluated_count"] == 9


def test_09_a_replay_gives_identical_receipt_hashes_to_the_recording_run(shared_09):
    assert shared_09.second == shared_09.first


def test_09_s2_records_the_gate_evidence_in_its_receipt_params_and_an_artifact(shared_09):
    from qrh.pipeline.artifacts import read_artifact

    s2 = shared_09.receipt("S2")
    gate = s2.params["llm_gate"]
    assert gate["verdict"] == "PASS" and gate["margin_days"] == 180
    assert gate["binding_cutoff"] == "2019-06-30" and gate["binding_date"] == "2019-12-27"
    assert gate["models"][0]["id"] == "stub-model-1"
    art = read_artifact(shared_09.home, s2, "llm_gate")
    assert art["windows"][0]["name"] == "development" and art["windows"][0]["status"] == "PASS"
    assert s2.methods["s2.llm_gate"] == "llm_gate@1" and s2.methods["llm.replay_cache"] == "replay_cache@1"
    assert "non_default_methods" not in s2.params or "s2.llm_gate" not in s2.params["non_default_methods"]


def test_09_later_stages_record_the_cache_hash_and_the_method(shared_09):
    hashes = set()
    for s in ("S3", "S4", "S5", "S6"):
        r = shared_09.receipt(s)
        assert r.methods["llm.replay_cache"] == "replay_cache@1", s
        hashes.add(len(r.params["llm"]["cache_hash"]))
        assert r.diagnostics["llm"]["calls"] >= 0 and r.diagnostics["llm"]["mode"] == "replay"
    assert hashes == {64}
    assert shared_09.first_receipts["S3"].diagnostics["llm"]["mode"] == "record"
    assert shared_09.first_receipts["S3"].diagnostics["llm"]["recorded"] > 0
    assert shared_09.receipt("S3").diagnostics["llm"]["recorded"] == 0  # the replay recorded nothing


def test_09_logs_every_combination_with_its_variant_and_prompt_hash(shared_09):
    from qrh.memory.registry import Registry

    reg = Registry(shared_09.home / "registry.sqlite3")
    rows = reg.conn.execute("SELECT variant, prompt_hash FROM trials WHERE trial_id=?", (shared_09.trial_id,)).fetchall()
    reg.close()
    assert len(rows) == 9 and {r["variant"] for r in rows} == {"default"}
    assert {r["prompt_hash"] for r in rows} == {sha256_file(shared_09.dir / "prompts" / "trend.txt")}


def test_09_the_cache_lives_in_qrh_home_only(ran09):
    shared_09 = ran09
    cache = shared_09.home / "llm_cache" / shared_09.trial_id
    assert (cache / "entries.jsonl").is_file() and (cache / "index.json").is_file()
    assert not list(shared_09.dir.rglob("entries.jsonl"))
    res = runner.invoke(app, ["llm", "status", shared_09.trial_id])
    assert res.exit_code == 0 and "stub-model-1" in res.output and "entries" in res.output
    assert "Closes, oldest first" not in res.output  # prompts are never printed
    assert runner.invoke(app, ["llm", "verify", shared_09.trial_id]).exit_code == 0


def test_09_a_report_page_for_the_gate_is_part_of_s2(ran09, monkeypatch):
    shared_09 = ran09
    monkeypatch.delenv("QRH_REPORTS", raising=False)
    assert runner.invoke(app, ["report", shared_09.trial_id, "--stage", "S2"]).exit_code == 0
    doc = json.loads((shared_09.home / "reports" / shared_09.trial_id / "S2.chart.json").read_text())
    keys = [f["key"] for f in doc["figures"]]
    assert "s2_llm_gate" in keys
    page = next(f for f in doc["figures"] if f["key"] == "s2_llm_gate")
    assert page["title"] == "S2 LLM contamination gate"
    assert page["binding_date"] == "2019-12-27" and page["margin_days"] == 180
    assert [r["status"] for r in page["rows"]] == ["PASS"]


def test_09_the_hypothesis_declares_an_llm_so_it_is_stored_and_hashed(shared_09):
    s0 = shared_09.receipt("S0")
    assert s0.params["llm"]["models"][0]["id"] == "stub-model-1"
    assert s0.params["thresholds"]["llm_cutoff_margin_days"] == 180


# --- replay and record discipline -------------------------------------------------------------------------


def fresh_09(example):
    return example("09_llm_clean", "prices")


def test_09_without_a_recorded_cache_the_default_replay_run_is_blocked_with_the_hint(example):
    ex = fresh_09(example)
    res = ex.run()
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.BLOCKED
    assert "this trial has no recorded LLM answer for this data and code version" in text(s2)
    assert f"`qrh run {ex.trial_id} --llm-mode record`" in text(s2) and "may cost money" in text(s2)
    assert res.exit_code == 2 and ex.receipt("S7").params["failing_stage"] == "S2"


def test_09_the_mode_is_an_option_not_an_environment_variable(example, monkeypatch):
    monkeypatch.setenv("QRH_LLM_MODE", "record")
    ex = fresh_09(example)
    assert ex.run().exit_code == 2 and ex.receipt("S2").verdict is Verdict.BLOCKED


def test_09_a_bad_llm_mode_is_refused(example):
    ex = fresh_09(example)
    res = ex.run("--llm-mode", "live")
    assert res.exit_code == 2 and "record" in res.output and "replay" in res.output


PROMPT_EDIT = ("Use only the data below.", "Use only the data below, and nothing else.")


def test_09_editing_the_prompt_changes_the_adapter_code_hash(example):
    ex = fresh_09(example)
    assert ex.run("--llm-mode", "record").exit_code == 0
    before = ex.receipt("S2").params["adapter_code_hash"]
    p = ex.dir / "prompts" / "trend.txt"
    p.write_text(p.read_text().replace(*PROMPT_EDIT))
    ex.new_trial()  # the edited prompt is read by a new trial
    ex.run("--llm-mode", "record")
    assert ex.receipt("S2").params["adapter_code_hash"] != before


def test_09_a_recording_does_not_serve_a_prompt_edited_after_it_was_made(example):
    ex = fresh_09(example)
    assert ex.run("--llm-mode", "record").exit_code == 0
    p = ex.dir / "prompts" / "trend.txt"
    p.write_text(p.read_text().replace(*PROMPT_EDIT))
    ex.run()  # replay, same trial: the rendered prompts differ from the recorded ones
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.BLOCKED and "replay cache miss" in text(s2)


def test_09_a_changed_prompt_in_a_new_trial_prints_a_note_that_it_counts_as_another_trial(example):
    ex = fresh_09(example)
    first_id = ex.trial_id
    p = ex.dir / "prompts" / "trend.txt"
    p.write_text(p.read_text().replace(*PROMPT_EDIT))
    res = runner.invoke(app, ["new", str(ex.dir / "hypothesis.yaml")])
    assert res.exit_code == 0, res.output
    note = f"prompt or model changed since trial {first_id}: counts as another trial; family count is now 18"
    assert note in res.output
    ex.trial_id = res.stdout.strip().splitlines()[-1]
    assert ex.receipt("S0").params["llm"]["notes"] == [note]
    assert ex.receipt("S0").verdict is Verdict.PASS  # a notice, never a block


def test_09_an_unchanged_prompt_prints_no_note(example):
    ex = fresh_09(example)
    res = runner.invoke(app, ["new", str(ex.dir / "hypothesis.yaml")])
    assert res.exit_code == 0 and "prompt or model changed" not in res.output
    ex.trial_id = res.stdout.strip().splitlines()[-1]
    assert ex.receipt("S0").params["llm"]["notes"] == []


def test_09_the_s0_receipt_does_not_drift_when_the_family_grows(example):
    ex = fresh_09(example)
    first_trial = ex.trial_id
    p = ex.dir / "prompts" / "trend.txt"
    p.write_text(p.read_text().replace(*PROMPT_EDIT))
    ex.new_trial()  # a second trial of the family, whose registration counts the first
    ex.trial_id = first_trial
    before = ex.receipt("S0").receipt_hash()
    ex.run()
    assert ex.receipt("S0").receipt_hash() == before


def test_09_a_tampered_cache_blocks_the_run_and_verify_exits_2(example):
    ex = fresh_09(example)
    assert ex.run("--llm-mode", "record").exit_code == 0
    path = ex.home / "llm_cache" / ex.trial_id / "entries.jsonl"
    lines = path.read_text().splitlines()
    doc = json.loads(lines[0])
    doc["entry"]["response"] = "DOWN" if doc["entry"]["response"] != "DOWN" else "UP"
    path.write_text("\n".join([json.dumps(doc), *lines[1:]]) + "\n")
    assert runner.invoke(app, ["llm", "verify", ex.trial_id]).exit_code == 2
    ex.run()
    assert ex.receipt("S2").verdict is Verdict.BLOCKED and "qrh llm verify" in text(ex.receipt("S2"))


def test_09_a_prompt_that_uses_a_future_value_is_caught_by_the_truncation_replay(example):
    ex = fresh_09(example)
    src = (ex.dir / "adapter.py").read_text()
    leaky = src.replace("px.iloc[i - lookback + 1 : i + 1]", "px.iloc[i - lookback + 1 : i + 2]")
    assert leaky != src
    (ex.dir / "adapter.py").write_text(leaky)
    ex.new_trial()
    ex.run("--llm-mode", "record")
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.FAIL
    assert any(r.startswith("truncation: prompt differs between full and truncated data at ") for r in s2.reasons), text(s2)
    assert s2.params["truncation"]["verdict"] == "FAIL"


def test_09_two_variants_multiply_the_grid_and_the_declared_size_must_match(example):
    ex = fresh_09(example)
    other = (ex.dir / "prompts" / "trend.txt").read_text().replace("signal generator", "forecaster")
    (ex.dir / "prompts" / "terse.txt").write_text(other)
    variants = [
        {"name": "trend", "model": "stub-model-1", "prompt_file": "prompts/trend.txt"},
        {"name": "terse", "model": "stub-model-1", "prompt_file": "prompts/terse.txt"},
    ]
    edit_llm(ex, prompt_files=["prompts/trend.txt", "prompts/terse.txt"], variants=variants)
    edit_spec(ex, parameter_space_size=9)
    ex.new_trial()
    ex.run("--llm-mode", "record")
    s3 = ex.receipt("S3")
    assert s3.verdict is Verdict.BLOCKED and "18 combinations" in text(s3) and "declared 9" in text(s3)
    edit_spec(ex, parameter_space_size=18)
    ex.new_trial()
    ex.run("--llm-mode", "record")
    assert ex.receipt("S3").verdict is Verdict.PASS and ex.receipt("S3").metrics["grid_size"] == 18
    from qrh.memory.registry import Registry

    reg = Registry(ex.home / "registry.sqlite3")
    rows = reg.conn.execute("SELECT variant, COUNT(*) AS n FROM trials WHERE trial_id=? GROUP BY variant",
                            (ex.trial_id,)).fetchall()
    reg.close()
    assert {r["variant"]: r["n"] for r in rows} == {"trend": 9, "terse": 9}
    assert ex.receipt("S5").trial_family_count >= 18 + 0  # the dsr counts every variant


def test_09_live_retrieval_with_evidence_passes_but_is_a_high_readiness_warning(example):
    ex = fresh_09(example)
    edit_llm(ex, live_retrieval={"point_in_time_evidence": "the vendor returns snapshots keyed by publication time"})
    ex.new_trial()
    ex.run("--llm-mode", "record")
    assert ex.receipt("S2").verdict is Verdict.PASS
    from qrh.readiness.collect import collect

    warnings = collect(ex.home, ex.trial_id).warnings
    hit = [w for w in warnings if w.code == "llm-live-retrieval"]
    assert len(hit) == 1 and hit[0].severity == "high" and "snapshots keyed by publication time" in hit[0].message


def test_09_live_retrieval_without_evidence_is_blocked_at_s2(example):
    ex = fresh_09(example)
    edit_llm(ex, live_retrieval=True)
    ex.new_trial()
    ex.run("--llm-mode", "record")
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.BLOCKED and "point_in_time_evidence" in text(s2)


# --- example 08: the model was trained on the test period --------------------------------------------------


def fresh_08(example):
    return example("08_llm_contamination", "prices")


def test_08_a_cutoff_after_the_window_start_fails_s2_with_the_evidence(example):
    ex = fresh_08(example)
    res = ex.run()  # no recording needed: the gate decides before any call
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.FAIL and res.exit_code == 2
    reason = text(s2)
    for needle in ("stub-model-2", "2021-06-30", "180 days", "window 'development'", "2020-03-01",
                   "earliest contaminated date 2020-03-01", "2021-12-27"):
        assert needle in reason, needle
    assert ex.receipt("S7").params["failing_stage"] == "S2"
    assert not (ex.home / "receipts" / ex.trial_id / "S3.json").exists()
    assert s2.params["llm_gate"]["verdict"] == "FAIL"


def test_08_the_leak_is_invisible_to_the_truncation_test_which_is_why_the_gate_exists(example):
    ex = fresh_08(example)
    edit_llm(ex, models=[{"id": "stub-model-2", "provider": "stub", "training_cutoff": "2019-01-01",
                          "cutoff_source": "provider_documented"}])  # a false claim: the stub still knows 2020-2021
    ex.new_trial()
    ex.run("--llm-mode", "record")
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.PASS  # documented limit: a cutoff is a provider claim, nothing here can measure it
    assert s2.params["truncation"]["verdict"] == "PASS" and s2.params["llm_gate"]["verdict"] == "PASS"


def test_08_an_unknown_cutoff_blocks_and_names_the_model(example):
    ex = fresh_08(example)
    edit_llm(ex, models=[{"id": "stub-model-2", "provider": "stub", "training_cutoff": "unknown",
                          "cutoff_source": "unknown"}])
    ex.new_trial()
    ex.run()
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.BLOCKED and "model 'stub-model-2'" in text(s2) and "unknown" in text(s2)


def test_08_an_undeclared_llm_is_a_block_level_lint_finding(example):
    ex = fresh_08(example)
    edit_spec(ex, llm=None)
    ex.new_trial()
    ex.run("--llm-mode", "record")
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.FAIL
    assert any(r.startswith("lint LLM001 adapter.py:") and "declares none" in r for r in s2.reasons), text(s2)
    assert s2.params["lint"]["verdict"] == "FAIL"
    assert "s2.llm_gate" in s2.methods


def test_08_an_allow_comment_suppresses_llm001_and_is_carried_into_the_readiness_report(example):
    ex = fresh_08(example)
    edit_spec(ex, llm=None)
    src = (ex.dir / "adapter.py").read_text()
    (ex.dir / "adapter.py").write_text(src.replace(
        "from qrh.llm import complete, read_prompt, register_backend",
        "from qrh.llm import complete, read_prompt, register_backend  # qrh: allow LLM001 test fixture, never a real model"))
    ex.new_trial()
    ex.run()
    s2 = ex.receipt("S2")
    assert "LLM001" in [f["rule"] for f in s2.params["lint"]["suppressed"]]
    from qrh.readiness.collect import collect

    hit = [w for w in collect(ex.home, ex.trial_id).warnings if w.code == "llm-undeclared-allowed"]
    assert len(hit) == 1 and hit[0].severity == "medium" and "test fixture, never a real model" in hit[0].message


def test_non_llm_strategies_are_unaffected(example):
    ex = example("01_clean_strategy", "prices")
    ex.run()
    s2 = ex.receipt("S2")
    assert "llm_gate" not in s2.params and "s2.llm_gate" not in s2.methods and "llm.replay_cache" not in s2.methods
    assert "llm" not in ex.receipt("S3").params and ex.receipt("S7").verdict is Verdict.PASS


def test_09_a_miss_in_a_later_stage_is_blocked_with_the_record_hint(example, home):
    from qrh.memory.registry import Registry
    from qrh.pipeline.receipt import Stage
    from qrh.pipeline.stages import Pipeline

    ex = fresh_09(example)
    reg = Registry(home / "registry.sqlite3")
    record, replay = Pipeline(reg, home, llm_mode="record"), Pipeline(reg, home)
    for stage in (Stage.S0, Stage.S1, Stage.S2):  # record only what S2 needs (one parameter point)
        assert record.run_stage(ex.trial_id, stage).verdict is Verdict.PASS
    s3 = replay.run_stage(ex.trial_id, Stage.S3)
    reg.close()
    assert s3.verdict is Verdict.BLOCKED
    assert "this trial has no recorded LLM answer for this data and code version" in text(s3)
    assert f"`qrh run {ex.trial_id} --llm-mode record`" in text(s3) and "may cost money" in text(s3)


# --- the retail-core pack on an LLM strategy ---------------------------------------------------------------


@pytest.fixture
def user_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "user"))


def with_retail_core(example, name):
    from pathlib import Path

    packs = Path(__file__).resolve().parents[2] / "packs"
    assert runner.invoke(app, ["pack", "add", str(packs / "retail-core")]).exit_code == 0
    return example(name, "prices")


def test_09_with_retail_core_the_llm_questions_are_open_conditions_and_the_llm_lessons_fire(example, user_home):
    ex = with_retail_core(example, "09_llm_clean")
    res = ex.run("--llm-mode", "record")
    assert res.exit_code == 0, res.output
    s7 = ex.receipt("S7")
    assert s7.verdict is Verdict.PASS_CONDITIONAL
    open_ids = {c["question_id"] for c in s7.params["conditions"]}
    assert {"llm-cutoff-source", "llm-live-retrieval", "llm-finetune-or-test-data"} <= open_ids
    lessons = {x["id"] for x in ex.receipt("S0").params["lessons"]}
    assert {"llm-memorization-is-lookahead", "count-every-prompt-model-variant", "record-and-replay-llm-calls"} <= lessons


def test_01_with_retail_core_gets_no_llm_questions_or_lessons(example, user_home):
    ex = with_retail_core(example, "01_clean_strategy")
    ex.run()
    open_ids = {c["question_id"] for c in ex.receipt("S7").params.get("conditions", [])}
    assert not [i for i in open_ids if i.startswith("llm-")]
    assert not [x["id"] for x in ex.receipt("S0").params.get("lessons", []) if "llm" in x["id"] or "prompt" in x["id"]]


def test_a_pack_lint_rule_can_add_a_library_to_the_undeclared_llm_detector(example, user_home, tmp_path):
    ex = example("01_clean_strategy", "prices")
    src = (ex.dir / "adapter.py").read_text()
    (ex.dir / "adapter.py").write_text(src.replace(
        "import numpy as np\n", "import numpy as np\n\ntry:\n    import acme_llm\nexcept ImportError:\n    acme_llm = None\n", 1))
    ex.new_trial()
    ex.run()
    assert ex.receipt("S2").verdict is Verdict.PASS  # an unknown library is not an LLM client...
    pack = tmp_path / "libpack"
    (pack / "lint").mkdir(parents=True)
    (pack / "qrh-pack.yaml").write_text(yaml.safe_dump({"name": "libpack", "version": "1.0.0", "schema_version": 1,
                                                        "visibility": "private"}))
    (pack / "lint" / "r.yaml").write_text(yaml.safe_dump({"rules": [{
        "id": "LLM001", "severity": "BLOCK", "message": "acme client", "ast": {"import": "acme_llm"}}]}))
    assert runner.invoke(app, ["pack", "add", str(pack)]).exit_code == 0
    ex.new_trial()
    ex.run()
    s2 = ex.receipt("S2")  # ...until a pack names it
    assert s2.verdict is Verdict.FAIL
    assert any(r.startswith("lint LLM001 adapter.py:") and "declares none" in r for r in s2.reasons), text(s2)
    assert text(s2).count("LLM001") == 1  # reported once, by the detector (the pack rule only adds its library)


def test_09_the_s2_pdf_has_the_gate_page_in_addition_to_the_usual_ones(ran09, monkeypatch):
    import re

    monkeypatch.delenv("QRH_REPORTS", raising=False)
    assert runner.invoke(app, ["report", ran09.trial_id, "--stage", "S2"]).exit_code == 0
    pdf = ran09.home / "reports" / ran09.trial_id / "S2.pdf"
    pages = len(re.findall(rb"/Type /Page\b(?!s)", pdf.read_bytes()))
    assert pages == 1 + 1 + 3 + 1  # lint, LLM gate, three truncation cutoffs, sanity


# --- LLM002: provider code outside qrh.llm.complete in a declared adapter ----------------------------------------

HELPER = "try:\n    import openai\nexcept ImportError:\n    openai = None\n"


def with_helper(ex, body=HELPER):
    (ex.dir / "helper.py").write_text(body)
    src = (ex.dir / "adapter.py").read_text()
    (ex.dir / "adapter.py").write_text(src.replace("import numpy as np\n", "import numpy as np\n\nimport helper  # noqa: F401\n", 1))
    ex.new_trial()
    ex.run("--llm-mode", "record")


def test_09_a_client_in_a_helper_module_is_a_warning_not_a_block_and_a_medium_readiness_warning(example):
    ex = fresh_09(example)
    with_helper(ex)
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.PASS and ex.receipt("S7").verdict is Verdict.PASS  # advisory: it never blocks
    warned = [w for w in s2.params["lint"]["warnings"] if w["rule"] == "LLM002"]
    assert [w["at"] for w in warned] == ["helper.py:2"]
    from qrh.pipeline.artifacts import read_artifact

    finding = [f for f in read_artifact(ex.home, s2, "lint")["findings"] if f["rule"] == "LLM002"]
    assert finding and finding[0]["severity"] == "WARN" and "outside qrh.llm.complete" in finding[0]["message"]
    from qrh.readiness.collect import collect

    [w] = [x for x in collect(ex.home, ex.trial_id).warnings if x.code == "llm-direct-client"]
    assert w.severity == "medium" and "helper.py" in w.message


def test_09_a_helper_that_registers_the_backend_is_exempt(example):
    ex = fresh_09(example)
    with_helper(ex, HELPER + "from qrh.llm import register_backend\nregister_backend('stub-2', lambda m, p, q: 'UP')\n")
    assert ex.receipt("S2").verdict is Verdict.PASS
    assert not [w for w in ex.receipt("S2").params["lint"]["warnings"] if w["rule"] == "LLM002"]


def test_09_an_allow_comment_suppresses_llm002_and_readiness_lists_it(example):
    ex = fresh_09(example)
    with_helper(ex, "try:\n    import openai  # qrh: allow LLM002 only the type names are used\nexcept ImportError:\n    openai = None\n")
    s2 = ex.receipt("S2")
    assert [f["rule"] for f in s2.params["lint"]["suppressed"]] == ["LLM002"]
    from qrh.readiness.collect import collect

    hit = [w for w in collect(ex.home, ex.trial_id).warnings if w.code == "lint-suppressed" and "LLM002" in w.message]
    assert len(hit) == 1 and "only the type names are used" in hit[0].message


def test_an_undeclared_adapter_gets_llm001_only_never_llm002(example):
    ex = example("08_llm_contamination", "prices")
    edit_spec(ex, llm=None)
    ex.new_trial()
    ex.run()
    assert not [w for w in ex.receipt("S2").params["lint"]["warnings"] if w["rule"] == "LLM002"]
    assert any(r.startswith("lint LLM001") for r in ex.receipt("S2").reasons)
