"""Regression matrix for publish-guard (payloads from an independent review), through the real runner and tmp homes.

Nothing is executed. BLOCK cases must be refused behind a human approval (banner), ALLOW cases are normal development.
GAP cases are accepted limitations of a text filter: the command is only text, and what a variable, a command
substitution, an alias, an environment variable or an interpreter's own code does is not visible to it. They are strict
xfails, so the day one starts being caught the suite says so and the entry moves to BLOCK. The stronger control is a
git-level pre-push check that verifies a signed approval (planned for the publication phase; docs/security.md).
"""

import os

import pytest

from hook_helpers import payload, run, write_config
from qrh.banner import BANNER

B = "Bash"


class Env:
    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "user"))
        self.repo = str(tmp_path / "repo")
        self.else_ = str(tmp_path / "elsewhere")
        os.makedirs(self.else_)
        for d in (".git/hooks", "scripts", "src/qrh", "tests/unit", "docs", ".github/workflows"):
            os.makedirs(os.path.join(self.repo, d))
        self.home = tmp_path / "home"
        write_config(self.home, f"publish_guard:\n  protected_repos:\n    - {self.repo}\n")


def build_cases(REPO, ELSE):
    return [
            ("git push origin main", "BLOCK", B, {"command": "git push origin main"}, REPO),
            ("git push bare", "BLOCK", B, {"command": "git push"}, REPO),
            ("git push --force-with-lease", "BLOCK", B, {"command": "git push --force-with-lease"}, REPO),
            ("git push --tags", "BLOCK", B, {"command": "git push --tags"}, REPO),
            ("git push refspec", "BLOCK", B, {"command": "git push origin HEAD:refs/heads/x"}, REPO),
            ("git remote add", "BLOCK", B, {"command": "git remote add origin https://github.com/x/y.git"}, REPO),
            ("git remote set-url", "BLOCK", B, {"command": "git remote set-url origin https://github.com/x/y.git"}, REPO),
            ("commit --no-verify", "BLOCK", B, {"command": "git commit -m x --no-verify"}, REPO),
            ("commit -nm combined", "BLOCK", B, {"command": "git commit -nm x"}, REPO),
            ("commit -n", "BLOCK", B, {"command": "git commit -n -m x"}, REPO),
            ("git -C from elsewhere", "BLOCK", B, {"command": f"git -C {REPO} push"}, ELSE),
            ("cd repo && push from elsewhere", "BLOCK", B, {"command": f"cd {REPO} && git push"}, ELSE),
            ("sh -c push", "BLOCK", B, {"command": "sh -c 'git push'"}, REPO),
            ("bash -c push", "BLOCK", B, {"command": 'bash -c "git push origin main"'}, REPO),
            ("env git push", "BLOCK", B, {"command": "env git push"}, REPO),
            ("command git push", "BLOCK", B, {"command": "command git push"}, REPO),
            ("nohup git push", "BLOCK", B, {"command": "nohup git push"}, REPO),
            ("time git push", "BLOCK", B, {"command": "time git push"}, REPO),
            ("GIT_SSH_COMMAND push", "BLOCK", B, {"command": "GIT_SSH_COMMAND=ssh git push"}, REPO),
            ("compound stash;push", "BLOCK", B, {"command": "git stash; git push"}, REPO),
            ("pipe-chained push", "BLOCK", B, {"command": "true && git push || true"}, REPO),
            ("gh repo create", "BLOCK", B, {"command": "gh repo create foo --public"}, REPO),
            ("gh repo edit visibility", "BLOCK", B, {"command": "gh repo edit --visibility public"}, REPO),
            ("gh repo delete", "BLOCK", B, {"command": "gh repo delete foo --yes"}, REPO),
            ("gh release create", "BLOCK", B, {"command": "gh release create v1"}, REPO),
            ("gh pr create", "BLOCK", B, {"command": "gh pr create --fill"}, REPO),
            ("gh api POST", "BLOCK", B, {"command": "gh api -X POST repos/x/y/issues -f title=t"}, REPO),
            ("npm publish", "BLOCK", B, {"command": "npm publish"}, REPO),
            ("uv publish", "BLOCK", B, {"command": "uv publish"}, REPO),
            ("twine upload", "BLOCK", B, {"command": "twine upload dist/*"}, REPO),
            ("cargo publish", "BLOCK", B, {"command": "cargo publish"}, REPO),
            ("core.hooksPath", "BLOCK", B, {"command": "git config core.hooksPath /tmp/h"}, REPO),
            ("global core.hooksPath", "BLOCK", B, {"command": "git config --global core.hooksPath /tmp/h"}, REPO),
            ("pre-commit uninstall", "BLOCK", B, {"command": "pre-commit uninstall"}, REPO),
            ("rm git hook", "BLOCK", B, {"command": "rm .git/hooks/pre-commit"}, REPO),
            ("chmod -x hook", "BLOCK", B, {"command": "chmod -x .git/hooks/pre-commit"}, REPO),
            ("redirect into hook", "BLOCK", B, {"command": "echo 'exit 0' > .git/hooks/pre-commit"}, REPO),
            ("sed -i gitleaks.toml", "BLOCK", B, {"command": "sed -i 's/a/b/' .gitleaks.toml"}, REPO),
            ("write .git/config", "BLOCK", "Write", {"file_path": REPO + "/.git/config", "content": "x"}, REPO),
            ("write pre-commit hook", "BLOCK", "Write", {"file_path": REPO + "/.git/hooks/pre-commit", "content": "exit 0"}, REPO),
            ("edit pre-commit-config", "BLOCK", "Edit", {"file_path": REPO + "/.pre-commit-config.yaml", "old_string": "a", "new_string": "b"}, REPO),
            ("edit gitleaks.toml", "BLOCK", "Edit", {"file_path": REPO + "/.gitleaks.toml", "old_string": "a", "new_string": "b"}, REPO),
            ("edit denylist_check", "BLOCK", "Edit", {"file_path": REPO + "/scripts/denylist_check.py", "old_string": "a", "new_string": "b"}, REPO),
            ("write ci workflow", "BLOCK", "Write", {"file_path": REPO + "/.github/workflows/ci.yml", "content": "x"}, REPO),
            ("multiedit workflow", "BLOCK", "MultiEdit", {"file_path": REPO + "/.github/workflows/ci.yml", "edits": [{"old_string": "a", "new_string": "b"}]}, REPO),
            # hard-to-see forms: report only
            ("var indirection G=git", "GAP", B, {"command": "G=git; $G push"}, REPO),
            ("cmd substitution verb", "GAP", B, {"command": "git $(echo push)"}, REPO),
            ("python subprocess push", "GAP", B, {"command": "python3 -c \"import subprocess;subprocess.run(['git','push'])\""}, REPO),
            ("python writes gitleaks", "GAP", B, {"command": "python3 -c \"open('.gitleaks.toml','w').write('')\""}, REPO),
            ("alias push", "GAP", B, {"command": "alias gp='git push'; gp"}, REPO),
            ("GIT_CONFIG env hooksPath", "GAP", B, {"command": "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath GIT_CONFIG_VALUE_0=/tmp git commit -m x"}, REPO),
            ("write ~/.gitconfig", "GAP", "Write", {"file_path": os.path.expanduser("~/.gitconfig"), "content": "x"}, REPO),
            # controls: ALLOW
            ("push in UNPROTECTED cwd", "ALLOW", B, {"command": "git push"}, ELSE),
            ("commit message mentions push", "ALLOW", B, {"command": "git commit -m 'docs: mention git push in the guide'"}, REPO),
            ("echo text about push", "ALLOW", B, {"command": "echo 'git push is gated' > /tmp/qrh_notes.txt"}, REPO),
            ("git status", "ALLOW", B, {"command": "git status"}, REPO),
            ("git log", "ALLOW", B, {"command": "git log --oneline -5"}, REPO),
            ("git diff", "ALLOW", B, {"command": "git diff"}, REPO),
            ("git add", "ALLOW", B, {"command": "git add -A"}, REPO),
            ("git commit normal", "ALLOW", B, {"command": "git commit -m 'feat: x'"}, REPO),
            ("git remote -v", "ALLOW", B, {"command": "git remote -v"}, REPO),
            ("git fetch", "ALLOW", B, {"command": "git fetch"}, REPO),
            ("gh repo view", "ALLOW", B, {"command": "gh repo view"}, REPO),
            ("gh pr list", "ALLOW", B, {"command": "gh pr list"}, REPO),
            ("pytest", "ALLOW", B, {"command": "uv run pytest -q"}, REPO),
            ("edit source file", "ALLOW", "Edit", {"file_path": REPO + "/src/qrh/cli.py", "old_string": "a", "new_string": "b"}, REPO),
            ("edit test file", "ALLOW", "Edit", {"file_path": REPO + "/tests/unit/test_x.py", "old_string": "a", "new_string": "b"}, REPO),
            ("write docs", "ALLOW", "Write", {"file_path": REPO + "/docs/x.md", "content": "x"}, REPO),
    ]


class _S:
    pass


LABELS = {exp: [c[0] for c in build_cases("/r", "/e") if c[1] == exp] for exp in ("BLOCK", "ALLOW", "GAP")}


def outcome(tmp_path, monkeypatch, label):
    env = Env(tmp_path, monkeypatch)
    for name, _exp, tool, tool_input, cwd in build_cases(env.repo, env.else_):
        if name == label:
            data = payload("PreToolUse", session_id="s9", tool_name=tool, tool_input=tool_input, tool_use_id="t", cwd=cwd)
            return run("publish-guard", data, env.home)
    raise KeyError(label)


def test_the_matrix_has_the_expected_shape():
    assert (len(LABELS["BLOCK"]), len(LABELS["ALLOW"]), len(LABELS["GAP"])) == (45, 16, 7)


@pytest.mark.parametrize("label", LABELS["BLOCK"])
def test_publication_actions_wait_for_a_human_decision(tmp_path, monkeypatch, label):
    out = outcome(tmp_path, monkeypatch, label)
    assert out.exit_code == 2 and out.stderr.startswith(BANNER) and "qrh approve" in out.stderr, label


@pytest.mark.parametrize("label", LABELS["ALLOW"])
def test_normal_development_stays_allowed(tmp_path, monkeypatch, label):
    assert outcome(tmp_path, monkeypatch, label).exit_code == 0, label


@pytest.mark.xfail(strict=True, reason="accepted limitation of a text filter: indirection, substitution, aliases, "
                   "environment variables and interpreter code are not visible in the command text")
@pytest.mark.parametrize("label", LABELS["GAP"])
def test_hard_to_see_forms_are_accepted_limitations(tmp_path, monkeypatch, label):
    assert outcome(tmp_path, monkeypatch, label).exit_code == 2, label
