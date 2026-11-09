"""Git-level pre-push approval: `qrh publish pre-push <remote_name> <remote_url>` reading git's stdin protocol."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from hook_helpers import ensure_test_keys, human_approve_id
from qrh import approvals, human, prepush, signing, tty
from qrh.cli import app

runner = CliRunner()
URL = "https://example.invalid/team/project.git"
A, B, C = "a" * 40, "b" * 40, "c" * 40
ZERO = "0" * 40
LINE = f"refs/heads/main {A} refs/heads/main {ZERO}\n"
REASON = "reviewed this exact push, fine to go"


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    h.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.chdir(repo)
    return h


@pytest.fixture
def home(env):
    return env


def key_id_of(home):
    try:
        pub = signing.load_public(home)
    except signing.SigningError:
        pub = None
    return pub.key_id if pub else "0" * 16


def push(stdin, name="origin", url=URL, home=None, key_id=None):
    """The hook as the pinned hook file calls it: explicit --home and --key-id, QRH_HOME is only noise."""
    home = Path(home) if home is not None else Path(os.environ["QRH_HOME"])
    kid = key_id if key_id is not None else key_id_of(home)
    return runner.invoke(app, ["publish", "pre-push", "--home", str(home), "--key-id", kid, name, url], input=stdin)


def rid_of(output):
    m = re.search(r"qrh approve ([0-9a-f]{12})", output)
    assert m, output
    return m.group(1)


def blocked_then_approved(home, stdin=LINE, url=URL):
    ensure_test_keys(home)
    res = push(stdin, url=url)
    assert res.exit_code == 2, res.output
    human_approve_id(home, rid_of(res.output), REASON)


# --- the happy path -------------------------------------------------------------------------------

def test_push_is_refused_with_an_id_and_the_human_command_until_approved(home):
    ensure_test_keys(home)
    res = push(LINE)
    assert res.exit_code == 2
    assert "qrh approve" in res.output and "--reason" in res.output and "retry" in res.output.lower()
    assert URL in res.output and "refs/heads/main" in res.output


def test_approved_push_passes_once_then_is_refused_again(home):
    blocked_then_approved(home)
    assert push(LINE).exit_code == 0
    again = push(LINE)
    assert again.exit_code == 2
    assert rid_of(again.output)  # a fresh request, not the consumed one


def test_identical_retry_reuses_the_pending_request(home):
    ensure_test_keys(home)
    first = rid_of(push(LINE).output)
    assert rid_of(push(LINE).output) == first


def test_line_order_and_duplicates_do_not_matter(home):
    two = f"refs/heads/main {A} refs/heads/main {ZERO}\nrefs/tags/v1 {B} refs/tags/v1 {ZERO}\n"
    swapped = f"refs/tags/v1 {B} refs/tags/v1 {ZERO}\nrefs/heads/main {A} refs/heads/main {ZERO}\n{LINE}"
    blocked_then_approved(home, two)
    assert push(swapped).exit_code == 0


def test_remote_sha_is_part_of_the_binding(home):
    """A force push over a remote that moved must not reuse a fast-forward approval."""
    blocked_then_approved(home, f"refs/heads/main {A} refs/heads/main {B}\n")
    assert push(f"refs/heads/main {A} refs/heads/main {C}\n").exit_code == 2
    assert push(f"refs/heads/main {A} refs/heads/main {ZERO}\n").exit_code == 2
    assert push(f"refs/heads/main {A} refs/heads/main {B}\n").exit_code == 0


def test_deletion_is_bound_like_any_update(home):
    dele = f"(delete) {ZERO} refs/heads/old {B}\n"
    blocked_then_approved(home, dele)
    assert push(f"(delete) {ZERO} refs/heads/other {B}\n").exit_code == 2
    assert push(dele).exit_code == 0


def test_sha256_object_names_are_accepted(home):
    line = f"refs/heads/main {'d' * 64} refs/heads/main {'0' * 64}\n"
    blocked_then_approved(home, line)
    assert push(line).exit_code == 0


# --- binding --------------------------------------------------------------------------------------

@pytest.mark.parametrize("changed", [
    f"refs/heads/main {B} refs/heads/main {ZERO}\n",                                    # changed sha
    f"refs/heads/main {A} refs/heads/other {ZERO}\n",                                   # other remote ref
    f"refs/heads/topic {A} refs/heads/main {ZERO}\n",                                   # other local ref
    LINE + f"refs/heads/x {C} refs/heads/x {ZERO}\n",                                   # extra update
    f"(delete) {ZERO} refs/heads/main {A}\n",                                            # delete instead of update
    f"refs/heads/main {A} refs/heads/main {B}\n",                                       # remote moved (force push)
])
def test_any_change_to_the_update_set_needs_a_new_approval(home, changed):
    blocked_then_approved(home)
    assert push(changed).exit_code == 2
    assert push(LINE).exit_code == 0  # the original approval is still intact for the original push


def test_a_subset_of_the_approved_updates_is_not_covered(home):
    two = LINE + f"refs/tags/v1 {B} refs/tags/v1 {ZERO}\n"
    blocked_then_approved(home, two)
    assert push(LINE).exit_code == 2


def test_other_remote_url_needs_a_new_approval(home):
    blocked_then_approved(home)
    assert push(LINE, url="https://example.invalid/other/project.git").exit_code == 2
    assert push(LINE).exit_code == 0


def test_remote_name_alias_does_not_matter_only_the_url(home):
    blocked_then_approved(home)
    assert push(LINE, name="mirror").exit_code == 0


def test_other_repository_needs_a_new_approval(home, tmp_path, monkeypatch):
    blocked_then_approved(home)
    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(["git", "init", "-q", str(other)], check=True)
    monkeypatch.chdir(other)
    assert push(LINE).exit_code == 2


def test_repository_is_bound_by_real_path_through_a_symlink_and_subdirectory(home, tmp_path, monkeypatch):
    blocked_then_approved(home)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "repo")
    sub = tmp_path / "repo" / "sub"
    sub.mkdir()
    monkeypatch.chdir(link / "sub")
    assert push(LINE).exit_code == 0


def stored(home):
    return "".join(p.read_text() for p in (home / "approvals").glob("*.json"))


@pytest.mark.parametrize("secret_url, plain", [
    ("https://user:s3cretvalue@example.invalid/team/project.git", "https://user@example.invalid/team/project.git"),
    ("persistent-https::https://user:s3cretvalue@example.invalid/x.git", "persistent-https::https://user@example.invalid/x.git"),
    ("a::b::https://user:s3cretvalue@example.invalid/x.git", "a::b::https://user@example.invalid/x.git"),
    ("https://example.invalid/x.git?private_token=s3cretvalue", "https://example.invalid/x.git"),
    ("https://example.invalid/x.git#s3cretvalue", "https://example.invalid/x.git"),
    ("persistent-https::https://example.invalid/x.git?access_token=s3cretvalue&a=1", "persistent-https::https://example.invalid/x.git"),
])  # gitleaks:allow
def test_secrets_in_remote_urls_are_never_printed_stored_or_bound(home, secret_url, plain):
    ensure_test_keys(home)
    res = push(LINE, url=secret_url)
    assert res.exit_code == 2 and "s3cretvalue" not in res.output
    assert "s3cretvalue" not in stored(home)
    assert plain in res.output and plain in stored(home)
    human_approve_id(home, rid_of(res.output), REASON)
    assert push(LINE, url=plain).exit_code == 0  # the cleaned URL is what is bound


def test_user_names_stay_distinct_in_the_binding(home):
    blocked_then_approved(home, url="ssh://alice@example.invalid/~/r")
    assert push(LINE, url="ssh://bob@example.invalid/~/r").exit_code == 2
    assert push(LINE, url="ssh://alice@example.invalid/~/r").exit_code == 0


def test_url_with_control_characters_is_refused(home):
    ensure_test_keys(home)
    assert push(LINE, url="https://example.invalid/a\nb").exit_code == 2


def test_not_a_git_repository_is_refused(home, tmp_path, monkeypatch):
    ensure_test_keys(home)
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    res = push(LINE)
    assert res.exit_code == 2 and "repository" in res.output.lower()


# --- fail closed ----------------------------------------------------------------------------------

def one_line(res):
    return res.output.strip().splitlines()


@pytest.mark.parametrize("stdin", [
    "", "\n", "   \n",
    "refs/heads/main\n",
    f"refs/heads/main {A} refs/heads/main\n",
    f"refs/heads/main {A} refs/heads/main {ZERO} extra\n",
    f"refs/heads/main nothex refs/heads/main {ZERO}\n",
    f"refs/heads/main {A[:39]} refs/heads/main {ZERO}\n",
    f"refs/heads/main {A.upper()} refs/heads/main {ZERO}\n",
    f"refs/heads/main {A} refs/heads/main {ZERO}\nbroken line\n",
])
def test_empty_or_malformed_stdin_is_refused_without_a_request(home, stdin):
    ensure_test_keys(home)
    res = push(stdin)
    assert res.exit_code == 2
    assert "qrh approve" not in res.output
    assert not list((home / "approvals").glob("*.json")) if (home / "approvals").exists() else True


def test_undecodable_stdin_is_refused(home):
    ensure_test_keys(home)
    res = runner.invoke(app, ["publish", "pre-push", "origin", URL], input=b"\xff\xfe\x00bad\n")
    assert res.exit_code == 2


def test_oversized_stdin_is_refused(home):
    ensure_test_keys(home)
    res = push(LINE * 20000)
    assert res.exit_code == 2 and "qrh approve" not in res.output


def test_empty_remote_url_is_refused(home):
    ensure_test_keys(home)
    assert push(LINE, url="").exit_code == 2


def test_no_approval_key_is_refused_and_says_how_to_create_one(home):
    res = push(LINE)
    assert res.exit_code == 2 and "approve --init" in res.output
    assert "qrh approve " not in res.output.replace("qrh approve --init", "")


def test_unreadable_public_key_is_refused(home):
    ensure_test_keys(home)
    signing.pub_path(home).write_text("{not json")
    res = push(LINE)
    assert res.exit_code == 2 and "qrh approve" not in res.output.replace("approve --init", "")


def test_key_rotated_after_approval_invalidates_it(home):
    blocked_then_approved(home)
    signing.pub_path(home).unlink()
    signing.key_path(home).unlink()
    signing.init_keys(home, "another passphrase, runtime only", datetime.now(timezone.utc))  # gitleaks:allow
    assert push(LINE).exit_code == 2


def test_tampered_token_is_refused(home):
    blocked_then_approved(home)
    path = next((home / "approvals").glob("*.json"))
    rec = json.loads(path.read_text())
    rec["action"]["parts"][-1] = f"refs/heads/main {B} refs/heads/main"
    path.write_text(json.dumps(rec))
    assert push(f"refs/heads/main {B} refs/heads/main {ZERO}\n").exit_code == 2


def test_forged_signature_is_refused(home):
    blocked_then_approved(home)
    path = next((home / "approvals").glob("*.json"))
    rec = json.loads(path.read_text())
    rec["signature"] = "AAAA" + rec["signature"][4:]
    path.write_text(json.dumps(rec))
    assert push(LINE).exit_code == 2


def test_unsigned_pending_request_does_not_allow(home):
    ensure_test_keys(home)
    assert push(LINE).exit_code == 2
    assert push(LINE).exit_code == 2


def test_consumed_marker_blocks_reuse_even_if_marker_is_the_only_trace(home):
    blocked_then_approved(home)
    assert push(LINE).exit_code == 0
    assert list((home / "approvals").glob("*.used"))
    assert push(LINE).exit_code == 2


def test_expired_approval_is_refused(home, monkeypatch):
    blocked_then_approved(home)
    real = prepush._now
    monkeypatch.setattr(prepush, "_now", lambda: real() + timedelta(hours=2))
    assert push(LINE).exit_code == 2


def test_unusable_home_is_refused(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    res = push(LINE, home=blocker / "sub", key_id="0" * 16)
    assert res.exit_code == 2 and res.output.strip()


def test_unreadable_state_directory_is_refused_not_a_crash(home, monkeypatch):
    ensure_test_keys(home)

    def boom(*a, **k):
        raise OSError("disk on fire")

    monkeypatch.setattr(prepush.approvals, "request", boom)
    res = push(LINE)
    assert res.exit_code == 2 and "Traceback" not in res.output


def test_unexpected_error_still_refuses(home, monkeypatch):
    ensure_test_keys(home)
    monkeypatch.setattr(prepush.approvals, "consume", lambda *a, **k: 1 / 0)
    assert push(LINE).exit_code == 2


def test_nothing_secret_is_printed(home):
    ensure_test_keys(home)
    res = push(LINE)
    pub = json.loads(signing.pub_path(home).read_text())["public_key"]
    assert "BEGIN" not in res.output and pub not in res.output


def test_command_installs_nothing(home, tmp_path):
    ensure_test_keys(home)
    push(LINE)
    hooks = tmp_path / "repo" / ".git" / "hooks"
    assert not (hooks / "pre-push").exists()


# --- B1: the hook does not trust the environment -----------------------------------------------------

def test_agent_made_home_in_the_environment_is_ignored(home, tmp_path, monkeypatch):
    ensure_test_keys(home)
    agent = tmp_path / "agent_home"
    agent.mkdir()
    monkeypatch.setenv("QRH_HOME", str(agent))
    ensure_test_keys(agent)
    rid = rid_of(push(LINE, home=home).output)             # the pinned home gets the request, not QRH_HOME
    assert (home / "approvals" / f"{rid}.json").exists() and not (agent / "approvals").exists()
    assert push(LINE, home=home).exit_code == 2


def test_agent_home_with_its_own_keys_is_refused_for_the_pinned_key_id(home, tmp_path):
    ensure_test_keys(home)
    real_id = key_id_of(home)
    agent = tmp_path / "agent_home"
    agent.mkdir()
    ensure_test_keys(agent)
    res = push(LINE, home=agent, key_id=real_id)
    assert res.exit_code == 2 and "key id" in res.output and not (agent / "approvals").exists()


def test_wrong_key_id_is_refused_even_with_a_valid_approval(home):
    blocked_then_approved(home)
    res = push(LINE, key_id="f" * 16)
    assert res.exit_code == 2 and "key id" in res.output
    assert push(LINE).exit_code == 0


def test_relative_or_missing_home_is_refused(home, tmp_path, monkeypatch):
    ensure_test_keys(home)
    monkeypatch.chdir(home.parent)
    res = push(LINE, home="qrh_home", key_id=key_id_of(home))
    assert res.exit_code == 2 and "absolute" in res.output
    res = push(LINE, home=tmp_path / "nope", key_id=key_id_of(home))
    assert res.exit_code == 2 and not (tmp_path / "nope").exists()


def test_home_and_key_id_options_are_required(home):
    for args in ([], ["--home", str(home)], ["--key-id", "0" * 16]):
        res = runner.invoke(app, ["publish", "pre-push", *args, "origin", URL], input=LINE)
        assert res.exit_code != 0 and "qrh approve" not in res.output


def test_print_hook_prints_the_pinned_hook_and_writes_nothing(home, tmp_path):
    ensure_test_keys(home)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    res = runner.invoke(app, ["publish", "print-hook"])
    assert res.exit_code == 0, res.output
    lines = res.output.splitlines()
    assert lines[0] == "#!/bin/sh"
    exec_line = next(x for x in lines if x.startswith("exec "))
    assert exec_line.startswith("exec env -i PATH=") and " -I -m qrh publish pre-push " in exec_line
    assert f"--home {home}" in exec_line and f"--key-id {key_id_of(home)}" in exec_line
    assert exec_line.endswith('"$@"') and exec_line.split()[4 + 0] == sys.executable
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_print_hook_needs_a_key(home):
    res = runner.invoke(app, ["publish", "print-hook"])
    assert res.exit_code == 2 and "approve --init" in res.output


# --- B2: what the human reads is complete ----------------------------------------------------------

THREE = (f"refs/heads/main {A} refs/heads/main {B}\n"
         f"refs/heads/token=release {C} refs/heads/token=release {ZERO}\n"
         f"(delete) {ZERO} refs/heads/main-old {A}\n")


def test_approve_screen_shows_every_update_unredacted_with_the_remote_sha(home, monkeypatch):
    ensure_test_keys(home)
    rid = rid_of(push(THREE).output)
    shown = []
    monkeypatch.setattr(tty, "has_terminal", lambda: True)
    monkeypatch.setattr(tty, "write", shown.append)
    monkeypatch.setattr(tty, "read_secret", lambda prompt="": "test passphrase, generated keys only")  # gitleaks:allow
    human.approve(home, rid, REASON)
    text = "".join(shown)
    assert "3 update" in text
    assert f"refs/heads/main {A} -> refs/heads/main (was {B})" in text
    assert f"refs/heads/token=release {C} -> refs/heads/token=release (was {ZERO})" in text
    assert f"(delete) {ZERO} -> refs/heads/main-old (was {A})" in text
    assert "..." not in text and "\u2026" not in text and "***" not in text and "REDACTED" not in text.upper()
    assert push(THREE).exit_code == 0


def test_a_long_push_is_shown_in_full(home):
    ensure_test_keys(home)
    many = "".join(f"refs/heads/branch-number-{i:03d} {A} refs/heads/branch-number-{i:03d} {ZERO}\n" for i in range(60))
    rid = rid_of(push(many).output)
    text = approvals.describe(approvals.load_pending(home, rid, prepush._now()))
    assert "60 update" in text and "refs/heads/branch-number-059" in text


def test_request_is_not_created_when_the_shown_text_would_be_cut(home, monkeypatch):
    ensure_test_keys(home)
    monkeypatch.setattr(approvals, "summarize", lambda action: "cut short")
    res = push(LINE)
    assert res.exit_code == 2 and "qrh approve" not in res.output and "shown in full" in res.output
    assert not list((home / "approvals").glob("*.json")) if (home / "approvals").exists() else True


def test_other_action_kinds_keep_their_summary_limit():
    long = approvals.parts_action("some-hook", "x" * 500)
    assert len(approvals.summarize(long)) <= approvals.SUMMARY_LIMIT


def test_a_changed_shown_text_breaks_the_signature(home):
    blocked_then_approved(home)
    path = next((home / "approvals").glob("*.json"))
    rec = json.loads(path.read_text())
    rec["summary"] = rec["summary"].replace("(was", "(wos")
    path.write_text(json.dumps(rec))
    assert push(LINE).exit_code == 2


# --- end to end with real git, using the printed hook file -------------------------------------------

def test_real_git_push_is_gated_by_the_printed_hook_file(home, tmp_path):
    ensure_test_keys(home)
    repo, bare = tmp_path / "repo", tmp_path / "remote.git"
    agent = tmp_path / "agent_home"
    agent.mkdir()

    def git(*a, env=None):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, env={**os.environ, **(env or {})})

    def commit(msg):
        git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "--allow-empty", "-m", msg)

    def approved_push(*refs):
        first = git("push", "origin", *refs)
        assert first.returncode != 0 and "qrh approve" in first.stderr, first.stderr
        human_approve_id(home, rid_of(first.stderr), REASON)
        assert git("push", "origin", *refs).returncode == 0

    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    commit("x")
    git("remote", "add", "origin", str(bare))
    hook = repo / ".git" / "hooks" / "pre-push"
    hook.write_text(runner.invoke(app, ["publish", "print-hook"]).output)
    hook.chmod(0o755)
    # the agent's environment (own home, PATH, HOME) changes nothing: the hook file pins them
    attack = {"QRH_HOME": str(agent), "HOME": str(agent), "PYTHONPATH": str(agent)}
    res = git("push", "origin", "HEAD:refs/heads/main", env=attack)
    assert res.returncode != 0 and "qrh approve" in res.stderr and not (agent / "approvals").exists()
    approved_push("HEAD:refs/heads/main")
    approved_push("HEAD:refs/heads/side")
    commit("y")
    assert git("push", "origin", "HEAD:refs/heads/main").returncode != 0
    gone = git("push", "origin", ":refs/heads/side")                     # not the remote HEAD
    assert gone.returncode != 0 and "qrh approve" in gone.stderr, gone.stderr
    listed = subprocess.run(["git", "--git-dir", str(bare), "show-ref"], capture_output=True, text=True).stdout
    assert "refs/heads/side" in listed
