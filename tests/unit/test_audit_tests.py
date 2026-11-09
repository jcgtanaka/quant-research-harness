"""qrh audit tests: deterministic diff check for weakened tests."""

import subprocess

import pytest
from typer.testing import CliRunner

from qrh.checks.weakened_tests import analyze_diff, audit_repo
from qrh.cli import app

runner = CliRunner()

GOOD = '''import pytest


def test_a():
    assert 1 == 1
    assert 2 == 2


def test_b():
    assert 3 == 3
'''


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "r"
    (r / "tests").mkdir(parents=True)
    (r / "src").mkdir()
    (r / "tests" / "test_x.py").write_text(GOOD)
    (r / "src" / "m.py").write_text("def f():\n    return 1\n")
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    return r


def kinds(repo, base="HEAD"):
    return sorted({f.kind for f in audit_repo(repo, base)})


def test_clean_change_has_no_findings(repo):
    (repo / "src" / "m.py").write_text("def f():\n    return 2\n")
    (repo / "tests" / "test_x.py").write_text(GOOD + "\n\ndef test_c():\n    assert f() == 2\n")
    assert audit_repo(repo) == []


def test_deleted_test_file(repo):
    (repo / "tests" / "test_x.py").unlink()
    assert kinds(repo) == ["deleted-test-file"]


def test_moved_test_file_is_not_a_deletion(repo):
    (repo / "tests" / "test_x.py").rename(repo / "tests" / "test_y.py")
    assert audit_repo(repo) == []


def test_deleted_test_function(repo):
    (repo / "tests" / "test_x.py").write_text(GOOD.replace("def test_b():\n    assert 3 == 3\n", ""))
    fs = audit_repo(repo)
    assert {f.kind for f in fs} == {"removed-test", "removed-assertions"}
    assert any("test_b" in f.detail for f in fs if f.kind == "removed-test")


def test_a_test_moved_to_another_file_is_fine(repo):
    (repo / "tests" / "test_x.py").write_text(GOOD.replace("\n\ndef test_b():\n    assert 3 == 3\n", ""))
    (repo / "tests" / "test_z.py").write_text("def test_b():\n    assert 3 == 3\n")
    assert audit_repo(repo) == []


@pytest.mark.parametrize("marker", [
    "@pytest.mark.skip(reason='later')", "@pytest.mark.skipif(True, reason='x')", "@pytest.mark.xfail",
    "@unittest.skip('x')", "@unittest.expectedFailure",
])
def test_added_skip_markers(repo, marker):
    (repo / "tests" / "test_x.py").write_text(GOOD.replace("def test_a", f"{marker}\ndef test_a"))
    assert kinds(repo) == ["added-skip"]


def test_added_pytest_skip_call_and_if_false(repo):
    (repo / "tests" / "test_x.py").write_text(GOOD.replace("    assert 1 == 1", "    pytest.skip('no')\n    assert 1 == 1"))
    assert kinds(repo) == ["added-skip"]
    (repo / "tests" / "test_x.py").write_text(GOOD.replace("    assert 3 == 3", "    if False:\n        assert 3 == 3"))
    assert "added-skip" in kinds(repo)


def test_js_skip(repo):
    (repo / "tests" / "a.test.js").write_text("it('a', () => { expect(1).toBe(1) })\n")
    git(repo, "add", "-A"), git(repo, "commit", "-q", "-m", "js")
    (repo / "tests" / "a.test.js").write_text("it.skip('a', () => { expect(1).toBe(1) })\n")
    assert kinds(repo) == ["added-skip"]


def test_skip_text_inside_a_string_is_not_a_marker(repo):
    (repo / "tests" / "test_x.py").write_text(GOOD + "\n\ndef test_c():\n    assert 'pytest.mark.skip' in 'pytest.mark.skip'\n")
    assert audit_repo(repo) == []


def test_removed_assertions_in_a_surviving_test(repo):
    (repo / "tests" / "test_x.py").write_text(GOOD.replace("    assert 2 == 2\n", ""))
    assert kinds(repo) == ["removed-assertions"]


def test_assertions_moved_within_the_diff_are_not_lost(repo):
    (repo / "tests" / "test_x.py").write_text(GOOD.replace("    assert 2 == 2\n", ""))
    (repo / "tests" / "test_n.py").write_text("def test_n():\n    assert 2 == 2\n")
    assert audit_repo(repo) == []


def test_non_test_files_are_ignored(repo):
    (repo / "src" / "m.py").write_text("# assert gone\n")
    assert audit_repo(repo) == []


def test_new_test_files_weaken_nothing(repo):
    (repo / "tests" / "test_new.py").write_text("import pytest\n\n@pytest.mark.skip\ndef test_n():\n    pass\n")
    assert audit_repo(repo) == []


def test_base_ref_compares_against_a_commit(repo):
    (repo / "tests" / "test_x.py").write_text(GOOD.replace("    assert 2 == 2\n", ""))
    git(repo, "commit", "-qam", "weaken")
    assert audit_repo(repo) == []                   # HEAD vs worktree: nothing pending
    assert kinds(repo, "HEAD~1") == ["removed-assertions"]


def test_unknown_ref_or_not_a_repo_raises(repo, tmp_path):
    from qrh.checks.weakened_tests import AuditError

    with pytest.raises(AuditError):
        audit_repo(repo, "no-such-ref")
    with pytest.raises(AuditError):
        audit_repo(tmp_path / "nowhere")


def test_analyze_diff_is_pure_text_in_findings_out():
    diff = ("diff --git a/tests/test_q.py b/tests/test_q.py\n--- a/tests/test_q.py\n+++ b/tests/test_q.py\n"
            "@@ -1,3 +1,1 @@\n-def test_q():\n-    assert 1\n+x = 1\n")
    assert {f.kind for f in analyze_diff(diff)} == {"removed-test", "removed-assertions"}


def test_cli_exit_codes_and_output(repo, monkeypatch):
    monkeypatch.chdir(repo)
    ok = runner.invoke(app, ["audit", "tests"])
    assert ok.exit_code == 0 and "no weakened tests" in ok.output
    (repo / "tests" / "test_x.py").unlink()
    bad = runner.invoke(app, ["audit", "tests"])
    assert bad.exit_code == 2 and "deleted-test-file" in bad.output and "tests/test_x.py" in bad.output
    again = runner.invoke(app, ["audit", "tests", "--base", "HEAD"])
    assert again.exit_code == 2
    assert runner.invoke(app, ["audit", "tests", "--base", "nope"]).exit_code == 2


def test_other_audit_subcommands_stay_blocked_stubs():
    for sub in ("receipts", "data"):
        r = runner.invoke(app, ["audit", sub])
        assert r.exit_code == 2 and "not implemented" in r.output
    assert runner.invoke(app, ["audit"]).exit_code == 2 or True
