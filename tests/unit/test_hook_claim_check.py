"""claim-check (Stop): "tests pass / build clean / lint clean / all green" needs a matching, later, successful run."""

import json

import pytest

from hook_helpers import human_approve, payload, run, write_config
from qrh.hooks.claim_check import Claim, extract_claims, is_build, is_lint, is_test


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


# --- claims -------------------------------------------------------------------------------------

def kinds(text):
    return sorted({c.kind for c in extract_claims(text)})


@pytest.mark.parametrize("text,expected", [
    ("All tests pass.", ["tests"]), ("All 42 tests passed.", ["tests"]), ("The test suite passes.", ["tests"]),
    ("Tests are green now.", ["tests"]), ("pytest passes and I'm done.", ["tests"]), ("42 passed, 0 failed.", ["tests"]),
    ("Everything passes.", ["checks"]), ("All checks are green.", ["checks"]), ("CI is green locally.", ["checks"]),
    ("The build succeeds.", ["build"]), ("Build is clean.", ["build"]), ("It compiles cleanly.", ["build"]),
    ("Lint is clean.", ["lint"]), ("No lint errors.", ["lint"]), ("ruff passes with no warnings.", ["lint"]),
    ("Tests pass and the build is clean.", ["build", "tests"]),
    ("ALL TESTS PASS", ["tests"]),
])
def test_claims_are_found(text, expected):
    assert kinds(text) == expected


@pytest.mark.parametrize("text", [
    "Not all tests pass.", "The tests don't pass yet.", "Tests are failing.", "3 tests fail.", "No tests pass.",
    "I could not get the tests to pass.", "The tests should pass now.", "Make sure the tests pass before merging.",
    "Once the tests pass we can merge.", "If the build succeeds, ship it.", "Run the tests until they pass.",
    "The build isn't clean.", "Lint is not clean.", "Would all tests pass? Unknown.", "I will make the tests pass.",
    "Done. The files are updated.", "The test for parsing passes? I didn't run it.", "Tests passed previously on main.",
    "Please check that all tests pass.", "Goal: all tests pass.", "Hopefully all tests pass.",
    "The build was clean before my change.",
])
def test_negated_hedged_or_planned_statements_are_not_claims(text):
    assert extract_claims(text) == []


def test_a_claim_in_a_code_block_is_ignored_and_one_claim_per_kind():
    assert extract_claims("```\nall tests pass\n```\n") == []
    assert len(extract_claims("All tests pass. Yes, the tests pass again.")) == 1


def test_claim_has_its_text():
    (c,) = extract_claims("Good news: all 12 tests pass.")
    assert isinstance(c, Claim) and c.kind == "tests" and "tests pass" in c.text


# --- commands -----------------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", [
    "pytest", "pytest -q tests/", "uv run pytest -x", "python -m pytest", "cd x && pytest", "npm test", "npm run test",
    "pnpm test", "yarn test", "go test ./...", "cargo test", "make test", "qrh test-receipt -- pytest -q",
    "env -u VIRTUAL_ENV uv run pytest", "pytest -q | tail -3", "npx vitest run", "tox",
])
def test_test_commands(cmd):
    assert is_test(cmd)


@pytest.mark.parametrize("cmd", ["ls", "git status", "pytest --collect-only", "cat tests/test_a.py", "echo pytest",
                                 "pytest --help", "pytest || true", "grep pytest pyproject.toml"])
def test_not_test_commands(cmd):
    assert not is_test(cmd)


def test_build_and_lint_commands():
    assert is_build("cargo build") and is_build("npm run build") and is_build("go build ./...") and is_build("tsc --noEmit")
    assert is_lint("ruff check .") and is_lint("uv run ruff check src") and is_lint("eslint .") and is_lint("mypy src")
    assert not is_lint("pytest") and not is_build("ruff check .")


def test_commands_are_configurable(home):
    write_config(home, "claim_check:\n  test_commands: ['^\\./ci/run-all']\n  build_commands: ['make dist']\n  lint_commands: ['checkstyle']\n")
    t = transcript(home.parent, Bash("./ci/run-all --fast", "5 passed"), say("All tests pass."))
    assert stop(home, t).exit_code == 0
    t = transcript(home.parent, Bash("make dist", "ok"), say("The build is clean."))
    assert stop(home, t).exit_code == 0


# --- transcripts --------------------------------------------------------------------------------

_n = [0]


def Bash(cmd, out="", error=False):
    _n[0] += 1
    tid = f"tu{_n[0]}"
    use = {"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": cmd}}]}}
    res = {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tid, "content": out, "is_error": error}]}}
    return [use, res]


def Edit(path="a.py", tool="Edit"):
    _n[0] += 1
    return [{"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": f"tu{_n[0]}", "name": tool, "input": {"file_path": path}}]}}]


def say(text):
    return [{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}]


def transcript(tmp, *groups):
    p = tmp / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for g in groups for r in g) + "\n")
    return str(p)


def stop(home, t, active=False, env=None):
    return run("claim-check", payload("Stop", transcript_path=t, stop_hook_active=active), home, env=env)


PASS = "===== 12 passed in 0.5s ====="


def test_claim_with_a_later_passing_run_is_accepted(home, tmp_path):
    t = transcript(tmp_path, Edit(), Bash("uv run pytest -q", PASS), say("All tests pass."))
    out = stop(home, t)
    assert out.exit_code == 0 and out.stderr == ""


def test_claim_without_any_run_is_blocked_and_listed(home, tmp_path):
    t = transcript(tmp_path, say("All tests pass and lint is clean."))
    out = stop(home, t)
    assert out.exit_code == 2 and "tests" in out.stderr and "lint" in out.stderr and "no matching" in out.stderr


def test_run_before_the_last_edit_does_not_count(home, tmp_path):
    t = transcript(tmp_path, Bash("pytest", PASS), Edit(), say("All tests pass."))
    out = stop(home, t)
    assert out.exit_code == 2 and "edit" in out.stderr.lower()


@pytest.mark.parametrize("tool", ["Write", "MultiEdit", "NotebookEdit"])
def test_every_edit_tool_counts_as_an_edit(home, tmp_path, tool):
    t = transcript(tmp_path, Bash("pytest", PASS), Edit(tool=tool), say("All tests pass."))
    assert stop(home, t).exit_code == 2


def test_edit_then_run_then_claim(home, tmp_path):
    t = transcript(tmp_path, Bash("pytest", PASS), Edit(), Bash("pytest", PASS), say("All tests pass."))
    assert stop(home, t).exit_code == 0


def test_a_failing_run_does_not_support_the_claim(home, tmp_path):
    t = transcript(tmp_path, Edit(), Bash("pytest", "1 failed, 11 passed", error=True), say("All tests pass."))
    out = stop(home, t)
    assert out.exit_code == 2 and "failed" in out.stderr


def test_failure_text_without_the_error_flag_still_fails(home, tmp_path):
    t = transcript(tmp_path, Edit(), Bash("pytest | tail -3", "=== 2 failed, 10 passed in 1s ==="), say("All tests pass."))
    assert stop(home, t).exit_code == 2


def test_zero_failures_summary_is_fine(home, tmp_path):
    t = transcript(tmp_path, Edit(), Bash("npm test", "Tests: 0 failed, 9 passed, 9 total"), say("All tests pass."))
    assert stop(home, t).exit_code == 0


def test_latest_matching_run_decides(home, tmp_path):
    t = transcript(tmp_path, Edit(), Bash("pytest", PASS), Bash("pytest", "1 failed", error=True), say("All tests pass."))
    assert stop(home, t).exit_code == 2
    t = transcript(tmp_path, Edit(), Bash("pytest", "1 failed", error=True), Bash("pytest", PASS), say("All tests pass."))
    assert stop(home, t).exit_code == 0


def test_a_test_run_does_not_support_a_lint_claim(home, tmp_path):
    t = transcript(tmp_path, Edit(), Bash("pytest", PASS), say("Lint is clean."))
    assert stop(home, t).exit_code == 2
    t = transcript(tmp_path, Edit(), Bash("ruff check .", "All checks passed!"), say("Lint is clean."))
    assert stop(home, t).exit_code == 0


def test_all_checks_green_needs_a_passing_run_of_tests_build_and_lint(home, tmp_path):
    ok = transcript(tmp_path, Edit(), Bash("pytest", PASS), Bash("cargo build", "ok"), Bash("ruff check .", "ok"),
                    say("All checks are green."))
    assert stop(home, ok).exit_code == 0
    only_lint = transcript(tmp_path, Edit(), Bash("ruff check .", "ok"), say("All checks are green."))
    assert stop(home, only_lint).exit_code == 2
    failing = transcript(tmp_path, Edit(), Bash("pytest", PASS), Bash("ruff check .", "E501", error=True),
                         say("All checks are green."))
    assert stop(home, failing).exit_code == 2


def test_all_checks_claim_without_any_build_or_lint_run_is_blocked(home, tmp_path):
    t = transcript(tmp_path, Edit(), Bash("pytest", PASS), say("All checks pass."))
    out = stop(home, t)
    assert out.exit_code == 2 and "build" in out.stderr and "lint" in out.stderr and "no matching" in out.stderr


def test_qrh_test_receipt_counts_as_a_test_run(home, tmp_path):
    t = transcript(tmp_path, Edit(), Bash("qrh test-receipt -- pytest -q", "test receipt PASS: exit code 0"),
                   say("All tests pass."))
    assert stop(home, t).exit_code == 0


def test_a_run_whose_result_is_missing_is_unsupported(home, tmp_path):
    use = Bash("pytest", PASS)[:1]
    t = transcript(tmp_path, Edit(), use, say("All tests pass."))
    assert stop(home, t).exit_code == 2


def test_list_shaped_tool_result_content_is_read(home, tmp_path):
    g = Bash("pytest")
    g[1]["message"]["content"][0]["content"] = [{"type": "text", "text": PASS}]
    assert stop(home, transcript(tmp_path, Edit(), g, say("All tests pass."))).exit_code == 0


def test_no_claim_means_no_check(home, tmp_path):
    assert stop(home, transcript(tmp_path, say("Updated the parser."))).exit_code == 0


SUMMARY = "===== 3072 passed, 13 xfailed in 230s ====="


@pytest.mark.parametrize("reply", [
    "3072 passed, 13 xfailed", "3,072 passed, 13 xfailed", "The suite passes: 3072 passed.",
    "All 3,072 tests pass.", "Suite passes, 3072 passed in 230s.",
])
def test_a_stated_count_matching_a_later_run_is_accepted(home, tmp_path, reply):
    t = transcript(tmp_path, Edit(), Bash("uv run pytest -q", SUMMARY), say(reply))
    out = stop(home, t)
    assert out.exit_code == 0 and out.stderr == ""


def test_a_stated_count_with_no_run_after_the_last_edit_is_blocked(home, tmp_path):
    t = transcript(tmp_path, Bash("pytest", SUMMARY), Edit(), say("3072 passed, 13 xfailed"))
    assert stop(home, t).exit_code == 2
    assert stop(home, transcript(tmp_path, say("3,072 passed"))).exit_code == 2


@pytest.mark.parametrize("reply", ["3071 passed", "3,073 passed, 13 xfailed", "All 3071 tests pass."])
def test_a_stated_count_that_differs_from_the_run_is_blocked(home, tmp_path, reply):
    t = transcript(tmp_path, Edit(), Bash("uv run pytest -q", SUMMARY), say(reply))
    out = stop(home, t)
    assert out.exit_code == 2 and "3072" in out.stderr


def test_a_stated_count_must_come_from_a_successful_run(home, tmp_path):
    t = transcript(tmp_path, Edit(), Bash("pytest", "3072 passed, 1 failed", error=True), say("3072 passed"))
    assert stop(home, t).exit_code == 2


def test_an_earlier_run_with_the_stated_count_does_not_count_when_the_latest_differs(home, tmp_path):
    t = transcript(tmp_path, Edit(), Bash("pytest -q", "3072 passed"), Bash("pytest -q", "3071 passed"),
                   say("3072 passed"))
    out = stop(home, t)
    assert out.exit_code == 2 and "3071" in out.stderr


def test_the_latest_run_showing_the_stated_count_passes(home, tmp_path):
    t = transcript(tmp_path, Edit(), Bash("pytest -q", "3071 passed"), Bash("pytest -q", "3072 passed"),
                   say("3072 passed"))
    assert stop(home, t).exit_code == 0


def test_separators_are_matched_against_the_latest_run(home, tmp_path):
    t = transcript(tmp_path, Edit(), Bash("pytest -q", "3071 passed"), Bash("pytest -q", "3072 passed"),
                   say("3,072 passed"))
    assert stop(home, t).exit_code == 0
    t = transcript(tmp_path, Edit(), Bash("pytest -q", "3072 passed"), Bash("pytest -q", "3071 passed"),
                   say("3,072 passed"))
    assert stop(home, t).exit_code == 2


def test_count_claims_are_parsed_with_separators():
    (c,) = extract_claims("3,072 passed, 13 xfailed")
    assert c.counts == (3072,)


def test_a_repeated_stop_still_blocks_an_unsupported_claim(home, tmp_path):
    t = transcript(tmp_path, say("All tests pass."))
    assert stop(home, t).exit_code == 2
    assert stop(home, t, active=True).exit_code == 2


def test_a_repeated_stop_with_a_clean_transcript_passes(home, tmp_path):
    assert stop(home, transcript(tmp_path, say("Done, nothing to report.")), active=True).exit_code == 0


def test_unreadable_transcript_is_open(home, tmp_path):
    assert stop(home, str(tmp_path / "missing.jsonl")).exit_code == 0
    assert stop(home, "").exit_code == 0


def test_disabled_in_config(home, tmp_path):
    write_config(home, "claim_check:\n  enabled: false\n")
    assert stop(home, transcript(tmp_path, say("All tests pass."))).exit_code == 0


@pytest.mark.parametrize("text", ["claim_check:\n  nope: 1\n", "claim_check:\n  test_commands: ['(']\n",
                                  "claim_check:\n  enabled: maybe\n"])
def test_invalid_config_blocks(home, tmp_path, text):
    write_config(home, text)
    assert stop(home, transcript(tmp_path, say("All tests pass."))).exit_code == 2


def test_human_approval_lets_the_same_unsupported_claims_through_once(home, tmp_path):
    t = transcript(tmp_path, say("All tests pass."))
    first = stop(home, t)
    assert first.exit_code == 2 and "qrh approve" in first.stderr
    human_approve(home, first.stderr, "tests were run in CI, linked in the PR")
    assert stop(home, t).exit_code == 0
    assert stop(home, t).exit_code == 2
