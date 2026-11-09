"""publish-guard: outward-facing and publication-weakening actions in protected repos need a human decision."""

import os

import pytest

from hook_helpers import bash_payload, hook_log, human_approve, payload, run, write_config
from qrh.banner import BANNER
from qrh.hooks.common import ConfigError, load_config

NAME = "publish-guard"


@pytest.fixture
def work(tmp_path):
    w = tmp_path / "work"
    for repo in ("repo", "repo2", "other"):
        (w / repo / ".git" / "hooks").mkdir(parents=True)
        (w / repo / "scripts").mkdir()
        (w / repo / "src").mkdir()
    return w


@pytest.fixture
def home(tmp_path, work):
    h = tmp_path / "home"
    write_config(h, f"publish_guard:\n  protected_repos:\n    - {work / 'repo'}\n")
    return h


def guard(home, command, cwd):
    return run(NAME, bash_payload(command, cwd=str(cwd)), home)


def tool(home, name, cwd, **ti):
    return run(NAME, payload("PreToolUse", tool_name=name, tool_input=ti, tool_use_id="t", cwd=str(cwd)), home)


def gated(out):
    return out.exit_code == 2 and out.stderr.startswith(BANNER) and "qrh approve" in out.stderr \
        and "BLOCKED by qrh hook publish-guard" in out.stderr


# --- configuration ------------------------------------------------------------------------------------

def test_without_the_section_the_hook_is_disabled(tmp_path, work):
    h = tmp_path / "h2"
    write_config(h, "trigger_terms: [alpha]\n")
    assert guard(h, "git push", work / "repo").exit_code == 0
    assert "disabled" in hook_log(h)[-1]["reason"]
    empty = tmp_path / "h3"
    assert guard(empty, "git push", work / "repo").exit_code == 0


@pytest.mark.parametrize("section", [
    "publish_guard: {}\n", "publish_guard:\n  protected_repos: []\n", "publish_guard:\n  protected_repos: [relative/path]\n",
    "publish_guard:\n  protected_repos: x\n", "publish_guard:\n  protected_repos: [/a]\n  other: 1\n",
    "publish_guard: 3\n", "publish_guard:\n  protected_repos: [1]\n",
])
def test_invalid_sections_are_config_errors_and_enforcement_blocks(tmp_path, section):
    h = tmp_path / "h"
    write_config(h, section)
    with pytest.raises(ConfigError):
        load_config(h)
    assert guard(h, "git status", tmp_path).exit_code == 2


def test_tilde_is_expanded_in_protected_repos(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    h = tmp_path / "h"
    write_config(h, "publish_guard:\n  protected_repos: ['~/proj']\n")
    assert load_config(h).publish_guard.protected_repos == (str(tmp_path / "proj"),)


# --- the verbs ----------------------------------------------------------------------------------------

VERBS = [
    "git push", "git push origin main", "git push --no-verify", "git push -u origin HEAD",
    "git remote add origin https://example.com/x.git", "git remote set-url origin git@example.com:x/y.git",
    "git commit --no-verify -m x", "git commit -n -m x", "git commit -anm x", "git commit -m x --no-verify",
    "git config core.hooksPath /tmp/h", "git config --local core.hookspath /tmp/h",
    "git -c core.hooksPath=/tmp/h commit -m x", "pre-commit uninstall",
    "gh repo create foo", "gh repo edit --visibility public --accept-visibility-change-consequences",
    "gh repo edit --description x", "gh repo fork", "gh repo delete x --yes", "gh repo rename y", "gh repo transfer o/r n",
    "gh release create v1", "gh release upload v1 a.zip", "gh pr create --fill", "gh pr merge 3",
    "gh api -X POST repos/o/r/issues", "gh api --method=PUT repos/o/r/contents/x", "gh api repos/o/r -f name=x",
    "gh api -XDELETE repos/o/r", "gh api -X PATCH /repos/o/r -F private=false", "gh -R o/r pr merge 3",
    "npm publish", "npm publish --access public", "pnpm publish", "yarn publish", "twine upload dist/*", "uv publish",
    "cargo publish", "poetry publish --build",
    "bash -c 'git push'", "sudo git push", "git status\ngit push", "true && git push", "if git push; then echo ok; fi",
    "xargs git push", "env GIT_X=1 git push",
]


@pytest.mark.parametrize("cmd", VERBS)
def test_publication_verbs_are_gated_in_a_protected_repo(home, work, cmd):
    assert gated(guard(home, cmd, work / "repo")), cmd


@pytest.mark.parametrize("cmd", VERBS[:12] + ["npm publish", "gh pr merge 3"])
def test_the_same_verbs_pass_in_an_unprotected_repo(home, work, cmd):
    for cwd in (work / "other", work / "repo2", work):
        assert guard(home, cmd, cwd).exit_code == 0, (cmd, cwd)


def test_cd_and_git_dash_C_move_the_command_into_a_protected_repo(home, work):
    repo = work / "repo"
    assert gated(guard(home, f"cd {repo} && git push", work / "other"))
    assert gated(guard(home, f"git -C {repo} push", work / "other"))
    assert gated(guard(home, f"git -C {repo}/src push", work / "other"))
    assert guard(home, f"cd {work / 'other'} && git push", repo).exit_code == 0
    assert guard(home, f"git -C {work / 'other'} push", repo).exit_code == 0


def test_global_hooks_path_changes_are_gated_wherever_they_run(home, work):
    for cmd in ("git config --global core.hooksPath /tmp/h", "git config --system core.hooksPath /tmp/h"):
        assert gated(guard(home, cmd, work / "other")), cmd


@pytest.mark.parametrize("cmd", [
    "git status", "git diff", "git log --oneline", "git add -A", 'git commit -m "docs: git push and --no-verify are gated"',
    "git commit --amend --no-edit", "git commit -am wip", "git remote -v", "git fetch", "git pull", "git checkout -b x",
    "git config user.name x", "git stash", "gh pr view 3", "gh pr list", "gh repo view", "gh api repos/o/r",
    "gh api -X GET repos/o/r", "gh release list", "npm install", "npm run build", "cargo build", "uv build", "uv sync",
    "pre-commit run --all-files", "pre-commit install", "uv run pytest -q", "cat .pre-commit-config.yaml",
    "ls .git/hooks", "rg git src", "echo git push", "python scripts/denylist_check.py",
])
def test_ordinary_development_stays_allowed_in_a_protected_repo(home, work, cmd):
    assert guard(home, cmd, work / "repo").exit_code == 0, cmd


# --- the publication guard files -------------------------------------------------------------------------

GUARD_FILES = [".git/hooks/pre-commit", ".pre-commit-config.yaml", ".gitleaks.toml", "scripts/denylist_check.py",
               ".github/workflows/ci.yml", ".git/config"]


@pytest.mark.parametrize("rel", GUARD_FILES)
@pytest.mark.parametrize("name,extra", [("Write", {"content": "x"}), ("Edit", {"old_string": "a", "new_string": "b"}),
                                         ("MultiEdit", {"edits": [{"old_string": "a", "new_string": "b"}]})])
def test_file_tools_cannot_touch_the_guard_files_of_a_protected_repo(home, work, rel, name, extra):
    path = work / "repo" / rel
    assert gated(tool(home, name, work / "repo", file_path=str(path), **extra)), (name, rel)
    assert tool(home, name, work / "other", file_path=str(work / "other" / rel), **extra).exit_code == 0


def test_relative_and_dotdot_paths_resolve_against_the_payload_cwd(home, work):
    assert gated(tool(home, "Write", work / "repo", file_path=".pre-commit-config.yaml", content="x"))
    assert gated(tool(home, "Write", work / "repo" / "src", file_path="../.gitleaks.toml", content="x"))
    assert gated(tool(home, "Write", work / "other", file_path=str(work / "repo" / "src" / ".." / ".gitleaks.toml"),
                      content="x"))


@pytest.mark.parametrize("rel", ["src/x.py", "README.md", "scripts/other.py", ".github/dependabot.yml",
                                 "tests/unit/test_x.py", ".gitignore", "docs/security.md"])
def test_other_files_stay_editable(home, work, rel):
    assert tool(home, "Write", work / "repo", file_path=str(work / "repo" / rel), content="x").exit_code == 0


@pytest.mark.parametrize("cmd", [
    "sed -i s/a/b/ .pre-commit-config.yaml", "echo x >> .gitleaks.toml", "echo x | tee scripts/denylist_check.py",
    "rm .github/workflows/ci.yml", "cp x .github/workflows/ci.yml", "git rm .pre-commit-config.yaml",
    "mv x .git/hooks/pre-commit", "echo exit 0 > .git/hooks/pre-commit", "rm -rf .git/hooks", "rm .git/hooks/pre-commit",
    "echo '[core]' >> .git/config", "cp evil .git/hooks/pre-commit",
    "cd .git/hooks && rm pre-commit", "chmod -x .git/hooks/pre-commit", "chmod 644 .git/hooks/pre-commit",
    "chmod a-x .git/hooks/*".replace("/*", "/pre-commit"), "chattr +i .gitleaks.toml", "touch .pre-commit-config.yaml",
    "truncate -s 0 .gitleaks.toml", "ln -sf /dev/null .git/hooks/pre-commit", "install -m 644 x .git/hooks/pre-commit",
    "dd if=/dev/null of=scripts/denylist_check.py", "git update-index --chmod=-x .git/hooks/pre-commit",
    "git update-index --chmod=-x scripts/denylist_check.py", "chmod -R a-x .git/hooks",
])
def test_shell_writes_to_the_guard_files_are_gated(home, work, cmd):
    assert gated(guard(home, cmd, work / "repo")), cmd
    assert guard(home, cmd, work / "other").exit_code == 0, cmd


def test_membership_is_by_real_path_and_whole_components(tmp_path, work):
    h = tmp_path / "h"
    write_config(h, f"publish_guard:\n  protected_repos:\n    - {work / 'repo'}\n")
    assert guard(h, "git push", work / "repo2").exit_code == 0           # name prefix is not membership
    link = tmp_path / "alias"
    os.symlink(work / "repo", link)
    assert gated(guard(h, "git push", link))                              # a symlink resolves to the repo
    parent = tmp_path / "h2"
    write_config(parent, f"publish_guard:\n  protected_repos:\n    - {work}\n")   # a listed parent covers its repos
    assert gated(guard(parent, "git push", work / "other"))
    assert gated(guard(parent, "git push", work))
    plain = tmp_path / "plain"                                            # not a git checkout, under a listed path
    plain.mkdir()
    both = tmp_path / "h4"
    write_config(both, f"publish_guard:\n  protected_repos:\n    - {tmp_path}\n")
    assert gated(guard(both, "git push", plain))


# --- approval flow and registration --------------------------------------------------------------------

def test_an_approved_push_passes_once(home, work):
    first = guard(home, "git push origin main", work / "repo")
    assert gated(first)
    human_approve(home, first.stderr)
    assert guard(home, "git push origin main", work / "repo").exit_code == 0
    assert gated(guard(home, "git push origin main", work / "repo"))
    assert gated(guard(home, "git push origin other", work / "repo"))     # a different action needs its own approval


def test_other_events_and_tools_are_ignored(home, work):
    assert run(NAME, payload("Stop", cwd=str(work / "repo")), home).exit_code == 0
    assert tool(home, "Read", work / "repo", file_path=str(work / "repo" / ".gitleaks.toml")).exit_code == 0


def test_registered_for_bash_and_the_edit_tools():
    import re

    from qrh.hooks.settings import desired_settings
    from qrh.hooks.specs import HOOKS

    spec = HOOKS[NAME]
    assert spec.event == "PreToolUse" and spec.enforcement and spec.approvable and spec.needs_config
    for t in ("Bash", "Write", "Edit", "MultiEdit"):
        assert re.fullmatch(spec.matcher, t), t
    for t in ("Read", "Grep", "mcp__x__y"):
        assert not re.fullmatch(spec.matcher, t), t
    groups = desired_settings()["hooks"]["PreToolUse"]
    assert sum(h["command"] == "qrh hook publish-guard" for g in groups for h in g["hooks"]) == 1
