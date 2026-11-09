"""approval-guard: the agent cannot approve its own blocked actions or read/alter the approval state."""

import pytest

from hook_helpers import bash_payload, hook_log, payload, run, write_config

NAME = "approval-guard"


@pytest.fixture
def user_home(tmp_path, monkeypatch):
    u = tmp_path / "user"
    u.mkdir()
    monkeypatch.setenv("HOME", str(u))  # never the real home: ~/.claude and ~/.qrh resolve in here
    return u


@pytest.fixture
def home(user_home):
    h = user_home / ".qrh"
    (h / "keys").mkdir(parents=True)
    (h / "approvals").mkdir()
    (h / "logs").mkdir()
    (h / "keys" / "approval.key").write_text("k" * 64)          # legacy name, still protected
    (h / "keys" / "approval.key.enc").write_text("encrypted")  # placeholder, not key material
    (h / "keys" / "approval.pub").write_text("{}")
    (h / "config.yaml").write_text("x: 1\n")
    (h / "registry.sqlite3").write_bytes(b"")
    (h / "logs" / "hooks.jsonl").write_text("{}\n")
    return h


def guard(home, command, **kw):
    return run(NAME, bash_payload(command, **kw), home)


def tool(home, name, cwd=None, **tool_input):
    extra = {"cwd": cwd} if cwd else {}
    return run(NAME, payload("PreToolUse", tool_name=name, tool_input=tool_input, tool_use_id="t", **extra), home)


def blocked(out):
    return out.exit_code == 2 and "BLOCKED by qrh hook approval-guard" in out.stderr


APPROVE_ATTEMPTS = [
    'qrh approve abcdefabcdef --reason "agent says so, long enough"',
    'uv run qrh approve abcdefabcdef --reason "agent says so, long enough"',
    'cd /work && qrh approve abcdefabcdef --reason "agent says so, long enough"',
    'echo yes | qrh approve abcdefabcdef --reason "agent says so, long enough"',
    'bash -c \'qrh approve abcdefabcdef --reason "agent says so, long enough"\'',
    'QRH_HOME=/tmp/x qrh approve abcdefabcdef --reason "agent says so, long enough"',
    'q"r"h approve abcdefabcdef --reason "agent says so, long enough"',
    "q\\rh approve abcdefabcdef --reason 'agent says so, long enough'",
    'python -m qrh approve abcdefabcdef --reason "agent says so, long enough"',
    '/home/u/.venv/bin/qrh   approve   abcdefabcdef --reason "agent says so, long enough"',
    'qrh --verbose approve abcdefabcdef --reason "agent says so, long enough"',
    'python -c "from qrh.cli import app; app()" approve abcdefabcdef --reason "agent says so, long enough"',
    'python -c "from qrh import approvals; approvals.approve(\'/h\', \'abcdefabcdef\', \'agent says so, long\')"',
    'python -c "import qrh.approvals as a; a.approve(1,2,3)"',
    'ls\nqrh approve abcdefabcdef --reason "agent says so, long enough"',
]


@pytest.mark.parametrize("cmd", APPROVE_ATTEMPTS)
def test_agent_cannot_run_qrh_approve(home, cmd):
    out = guard(home, cmd)
    assert blocked(out), cmd
    assert "--reason" not in out.stderr and "pending" not in out.stderr  # no instructions to bypass


def test_block_is_final_and_creates_no_pending_approval(home):
    out = guard(home, "qrh approve abcdefabcdef --reason 'long enough reason'")
    assert out.exit_code == 2 and "cannot be overridden" in out.stderr
    assert list((home / "approvals").iterdir()) == []
    assert hook_log(home)[-1]["decision"] == "block"


def key_paths(home):
    h = str(home)
    return [
        f"cat {h}/keys/approval.key", f"cat '{h}/keys/approval.key'", f'cat "{h}/keys/approval.key"',
        f"cp {h}/keys/approval.key /tmp/stolen", f"mv {h}/keys /tmp/k", f"xxd {h}/keys/approval.key",
        f"rm -rf {h}/keys", f"cat {h}/keys/../keys/approval.key", f"cat {h}/k*/approval.key",
        f"cat {h}/{{keys,approvals}}/*", f"ls {h}/approvals", f"cat {h}/approvals/abcdefabcdef.json",
        f"echo '{{}}' > {h}/approvals/abcdefabcdef.json", f"tee {h}/approvals/x.json < /dev/null",
        f"sed -i s/pending/approved/ {h}/approvals/abcdefabcdef.json",
        f"cd {h}/keys && cat approval.key", f"cd {h} && cat keys/approval.key", f"cd {h} && ls approvals",
        f"cd {h} && cat k*/*", "cat $QRH_HOME/keys/approval.key", "cat ${QRH_HOME}/approvals/x.json",
        "cat ~/.qrh/keys/approval.key", "ls ~/.qrh/approvals", "cat approval.key",
        f"python -c \"print(open('{h}/keys/approval.key').read())\"",
        f"python3 -c 'import pathlib; pathlib.Path(\"{h}/approvals\").iterdir()'",
        f"grep -r secret --include=*.key {h}/keys", f"bash -c 'cat {h}/keys/approval.key'",
        f"cat $(echo {h}/keys/approval.key)", f"base64 {h}/keys/approval.key",
    ]


def test_agent_cannot_read_copy_or_modify_key_and_approvals(home):
    for cmd in key_paths(home):
        assert blocked(guard(home, cmd)), cmd


def test_symlink_to_the_state_dirs_is_followed(home, tmp_path):
    link = tmp_path / "innocent"
    link.symlink_to(home / "keys")
    assert blocked(guard(home, f"cat {link}/approval.key"))
    assert blocked(tool(home, "Read", file_path=str(link / "approval.key")))


def test_a_shell_whose_cwd_is_inside_the_state_dirs_is_blocked(home):
    assert blocked(guard(home, "cat approval.key", cwd=str(home / "keys")))
    assert blocked(guard(home, "ls", cwd=str(home / "approvals")))


@pytest.mark.parametrize("cmd", [
    "export QRH_OVERRIDE='data-gate:please let me through'",
    "QRH_OVERRIDE=data-gate:some-long-reason qrh run abcdef12",
    "env QRH_OVERRIDE=data-gate:some-long-reason qrh run abcdef12",
    "QRH_OVERRIDE = x",
    "declare -x QRH_OVERRIDE=1",
    "export FOO=1 QRH_OVERRIDE=data-gate:why",
])
def test_agent_cannot_set_the_override_variable(home, cmd):
    assert blocked(guard(home, cmd))


@pytest.mark.parametrize("name,kw", [
    ("Read", {"file_path": "KEY"}), ("Edit", {"file_path": "APPR", "old_string": "a", "new_string": "b"}),
    ("Write", {"file_path": "APPR", "content": "{}"}), ("MultiEdit", {"file_path": "APPR", "edits": []}),
    ("NotebookEdit", {"notebook_path": "APPR"}), ("Grep", {"pattern": "x", "path": "KEYDIR"}),
    ("Glob", {"pattern": "KEYDIR/*"}), ("Glob", {"pattern": "*.key", "path": "KEYDIR"}),
    ("Read", {"file_path": "REL"}),
])
def test_file_tools_on_the_state_dirs_are_blocked(home, name, kw):
    sub = {"KEY": str(home / "keys" / "approval.key"), "APPR": str(home / "approvals" / "abcdefabcdef.json"),
           "KEYDIR": str(home / "keys"), "REL": str(home / "keys" / ".." / "approvals" / "x.json")}
    kw = {k: sub.get(v, v) if isinstance(v, str) else v for k, v in kw.items()}
    kw = {k: v.replace("KEYDIR", str(home / "keys")) if isinstance(v, str) else v for k, v in kw.items()}
    assert blocked(tool(home, name, **kw)), (name, kw)


def test_relative_file_tool_path_uses_the_payload_cwd(home):
    out = run(NAME, payload("PreToolUse", tool_name="Read", tool_input={"file_path": "approval.key"},
                            cwd=str(home / "keys")), home)
    assert blocked(out)


@pytest.mark.parametrize("cmd", [
    "qrh run abc12345", "ls -la", "git status", "qrh test-receipt -- pytest -q", "echo approve the plan",
    "rg approved src", "qrh usage report --by week", "git commit -m 'approve docs'", "cat README.md",
    "qrh status", "uv run pytest tests/unit/test_keys.py", "ls keys_report", "cat my_approvals.txt",
    "python script.py --approve", "git commit -m 'document the qrh hook table'", "qrh hooks print-settings",
])
def test_ordinary_commands_pass(home, cmd):
    assert guard(home, cmd).exit_code == 0, cmd


def test_reads_of_config_and_logs_stay_allowed(home):
    assert guard(home, f"cat {home}/config.yaml").exit_code == 0
    assert guard(home, "tail -n 5 ~/.qrh/logs/hooks.jsonl").exit_code == 0
    assert guard(home, f"ls {home}/logs").exit_code == 0
    assert tool(home, "Read", file_path=str(home / "config.yaml")).exit_code == 0
    assert tool(home, "Read", file_path=str(home / "logs" / "hooks.jsonl")).exit_code == 0
    assert tool(home, "Glob", pattern="**/*.py").exit_code == 0


def test_works_with_a_broken_config_and_ignores_other_events(home):
    write_config(home, "strategy_globs: [unclosed")
    assert blocked(guard(home, f"cat {home}/keys/approval.key"))
    assert guard(home, "ls").exit_code == 0
    assert run(NAME, payload("PostToolUse", tool_name="Bash", tool_input={"command": f"cat {home}/keys/approval.key"}),
               home).exit_code == 0  # PostToolUse is not where it enforces


def test_fails_closed(home):
    assert run(NAME, "{not json", home).exit_code == 2                      # unparsable payload
    out = run(NAME, payload("PreToolUse", tool_name="Bash", tool_input={"command": 123}), home)
    assert out.exit_code == 0                                               # no command text to inspect
    odd = run(NAME, payload("PreToolUse", tool_name="Read", tool_input={"file_path": ["x"]}), home)
    assert odd.exit_code == 2                                               # wrong-typed path: cannot be checked


def test_registered_for_bash_the_file_tools_and_mcp_tools():
    import re

    from qrh.hooks.settings import desired_settings
    from qrh.hooks.specs import HOOKS

    spec = HOOKS[NAME]
    assert spec.event == "PreToolUse" and spec.enforcement and not spec.approvable and not spec.needs_config
    for name in ("Bash", "Read", "Edit", "Write", "MultiEdit", "NotebookEdit", "Grep", "Glob",
                 "mcp__filesystem__read_file", "mcp__github__create_or_update_file"):
        assert re.fullmatch(spec.matcher, name), name
    groups = desired_settings()["hooks"]["PreToolUse"]
    routed = [g for g in groups if any(h["command"] == "qrh hook approval-guard" for h in g["hooks"])]
    assert len(routed) == 1 and re.fullmatch(routed[0]["matcher"], "mcp__filesystem__read_file")


HUMAN_ANSWER_ATTEMPTS = [
    'qrh interview answer rf --scope dataset:prices --value pit --human',
    'uv run qrh interview answer rf --scope dataset:prices --human --value pit',
    'cd /work && qrh --verbose interview   answer rf --human --value x --scope dataset:p',
    'q"r"h interview answer rf --human --value x --scope dataset:p',
    'bash -c \'qrh interview answer rf --scope dataset:p --value x --human\'',
    'python -m qrh interview answer rf --scope dataset:p --value x --human',
    'python -c "from qrh.interview.engine import record_answer; record_answer(None)"',
    'python -c "import qrh.interview.engine as e; e.record_answer(1)"',
]


DATA_HUMAN_ATTEMPTS = [  # P3e-a1: the human's word about a dataset (confirm a claim, propose as the user)
    'qrh data confirm prices adjustments',
    'uv run qrh data confirm prices multiplier --value 50',
    'cd /work && qrh --verbose data   confirm prices timestamp_semantics',
    'q"r"h data conf""irm prices adjustments',
    'bash -c \'qrh data confirm prices adjustments\'',
    'python -m qrh data confirm prices adjustments',
    'qrh data propose prices --file p.json --as-user',
    'uv run qrh data propose --as-user prices --file p.json',
    'env qrh data propose prices --as-user --file p.json',
    'python -c "from qrh.onboarding.workflow import propose; propose(None)"',
    'python -c "import qrh.onboarding.store as s; s.add_version(1)"',
    'alias q=qrh; q data confirm prices adjustments',
    '$(echo qrh) data confirm prices adjustments',
    'bash <<< "qrh data confirm prices adjustments"',
]


@pytest.mark.parametrize("cmd", DATA_HUMAN_ATTEMPTS)
def test_agent_cannot_confirm_or_propose_as_the_user(home, cmd):
    out = guard(home, cmd)
    assert blocked(out), cmd
    assert "person at a terminal" in out.stderr


@pytest.mark.parametrize("cmd", [
    'qrh data propose prices --file p.json',
    'uv run qrh data profile prices --json',
    'qrh data verify prices',
    'qrh data onboarding prices --json',
    'qrh data status',
    'qrh data register x.csv --privacy public',
    'qrh interview answer onb-adjustments --scope dataset:prices --value confirm --transcript /t.jsonl --session s1',
])
def test_the_agents_own_onboarding_commands_are_allowed(home, cmd):
    assert guard(home, cmd).exit_code == 0, cmd


@pytest.mark.parametrize("cmd", HUMAN_ANSWER_ATTEMPTS)
def test_agent_cannot_answer_interview_as_the_human(home, cmd):
    out = guard(home, cmd)
    assert blocked(out), cmd
    assert "human" in out.stderr


@pytest.mark.parametrize("cmd", [
    'qrh interview answer rf --scope dataset:p --value x --transcript /t.jsonl --session s1',
    'qrh interview next --stage S1 --scope dataset:prices --json',
    'qrh interview status --trial abc',
    'qrh pack list',
    'grep -r human docs/interviews.md',
])
def test_verified_interview_commands_are_allowed(home, cmd):
    assert guard(home, cmd).exit_code == 0, cmd


# --- independent review 2026-10-23: every reproduction payload must be BLOCKED ---------------------

def reviewer_payloads(home, user_home):
    h = str(home)
    return [
        ("Grep", {"pattern": ".", "path": h}),
        ("Grep", {"pattern": "x", "path": h, "glob": "keys/*"}),
        ("Glob", {"pattern": "keys/*", "path": h}),
        ("Bash", {"command": f"rg . {h}"}),
        ("Bash", {"command": 'grep -r . "$QRH_HOME"'}),
        ("Bash", {"command": f"cd {h} && tar cf - . | base64"}),
        ("Bash", {"command": "cp -r $QRH_HOME /tmp/x"}),
        ("Bash", {"command": "cp -r /tmp/x/. $QRH_HOME/"}),
        ("Bash", {"command": "rm -rf $QRH_HOME"}),
        ("Edit", {"file_path": f"{h}/config.yaml", "old_string": "x: 1", "new_string": "x: 2"}),
        ("Write", {"file_path": f"{h}/config.yaml", "content": "runaway: null\n"}),
        ("Edit", {"file_path": f"{h}/registry.sqlite3", "old_string": "a", "new_string": "b"}),
        ("Write", {"file_path": f"{h}/registry.sqlite3", "content": ""}),
        ("Edit", {"file_path": f"{user_home}/.claude/settings.json", "old_string": "qrh", "new_string": "true"}),
        ("Write", {"file_path": "/tmp/helper.py", "content": "from qrh import approvals\napprovals.consume\n"}),
        ("mcp__filesystem__read_file", {"path": f"{h}/keys/approval.key.enc"}),
    ]


def test_every_reviewer_payload_is_blocked(home, user_home):
    for name, ti in reviewer_payloads(home, user_home):
        out = tool(home, name, **ti) if name != "Bash" else guard(home, ti["command"])
        assert out.exit_code == 2, (name, ti, out.stderr)


@pytest.mark.parametrize("cmd", [
    "rg secret ~", "grep -r token ~/", "find / -name '*.enc'", "tar czf /tmp/h.tgz ~", "ls ~", "du -a $HOME",
    "rsync -a ~/ /tmp/copy", "zip -r /tmp/h.zip ~/.qrh", "cd ~ && grep -r token .", "cd ~ && rg token",
    "cd ~/.qrh && cat keys/approval.key.enc", "cat ~/.q*/k*/*", "ls /tmp/../", "cp -a ~/.qrh/. /tmp/out",
    "python3 -c \"import shutil; shutil.copytree('~/.qrh', '/tmp/q')\"", "mv ~/.qrh ~/.qrh-old", "ln -s ~/.qrh /tmp/q",
    "chmod -R 777 ~/.qrh", "base64 < ~/.qrh/keys/approval.key.enc", "cat <(cat ~/.qrh/keys/approval.pub)",
    "H=~/.qrh; rg . $H", "xxd $QRH_HOME/keys/approval.key.enc",
])
def test_home_and_its_ancestors_are_off_limits_for_reads_and_writes(home, cmd):
    assert blocked(guard(home, cmd)), cmd


def test_walkers_with_an_ancestor_as_working_directory_are_blocked(home, user_home):
    assert blocked(guard(home, "rg token", cwd=str(user_home)))
    assert blocked(guard(home, "grep -R token", cwd=str(user_home)))
    assert blocked(guard(home, "ls -R", cwd=str(user_home)))
    assert blocked(tool(home, "Grep", cwd=str(user_home), pattern="token"))
    assert blocked(tool(home, "Glob", cwd=str(user_home), pattern="**/*.enc"))
    assert guard(home, "ls", cwd=str(user_home)).exit_code == 0       # names only, no walk


@pytest.mark.parametrize("tmpl", [
    "echo 'runaway: null' > {h}/config.yaml", "echo x >> ~/.qrh/config.yaml", "sed -i s/a/b/ ~/.qrh/config.yaml",
    "tee ~/.qrh/packs.json < /dev/null", "mv /tmp/c $QRH_HOME/config.yaml", "cp /tmp/c ~/.qrh/config.yaml",
    "sqlite3 ~/.qrh/registry.sqlite3 'select 1'", "sqlite3 {h}/registry.sqlite3-wal .dump",
    "python -c \"import sqlite3; sqlite3.connect('{h}/registry.sqlite3')\"", "rm -rf ~/.qrh/parity",
    "touch ~/.qrh/runaway/x.json", "rm ~/.qrh/sessions/abc.json", "ln -sf /tmp/x ~/.qrh/config.yaml",
    "chmod 777 ~/.qrh/config.yaml", "cat > ~/.qrh/config.yaml <<EOF\nrunaway: null\nEOF",
    "cp evil.json ~/.claude/settings.json", "echo '{{}}' > ~/.claude/settings.local.json",
    "echo '{{}}' > .claude/settings.local.json", "rm -rf .claude/hooks", "truncate -s0 ~/.qrh/config.yaml",
    "bash -c 'rm ~/.qrh/config.yaml'", "dd if=/dev/null of=~/.qrh/config.yaml",
])
def test_config_state_and_settings_cannot_be_written_from_bash(home, tmpl):
    assert blocked(guard(home, tmpl.format(h=home))), tmpl


@pytest.mark.parametrize("name,rel,extra", [
    ("Edit", "QRH/config.yaml", {"old_string": "a", "new_string": "b"}),
    ("Write", "QRH/packs.json", {"content": "{}"}),
    ("Write", "QRH/parity/report.json", {"content": "{}"}),
    ("Write", "QRH/runaway/state.json", {"content": "{}"}),
    ("Write", "QRH/sessions/x.json", {"content": "{}"}),
    ("MultiEdit", "QRH/config.yaml", {"edits": []}),
    ("Write", "QRH/registry.sqlite3-journal", {"content": ""}),
    ("Write", "USER/.claude/settings.local.json", {"content": "{}"}),
    ("Edit", "CWD/.claude/settings.json", {"old_string": "a", "new_string": "b"}),
    ("Write", "CWD/.claude/hooks/pre.sh", {"content": "exit 0"}),
])
def test_file_tools_cannot_write_config_state_or_settings(home, user_home, tmp_path, name, rel, extra):
    project = tmp_path / "project"
    project.mkdir()
    path = rel.replace("QRH", str(home)).replace("USER", str(user_home)).replace("CWD", str(project))
    assert blocked(tool(home, name, cwd=str(project), file_path=path, **extra)), path


HELPERS = [
    "from qrh import approvals\n",
    "import qrh.approvals as a\n",
    "from qrh.signing import unlock\n",
    "from qrh.interview.engine import record_answer\nrecord_answer(None)\n",
    "import importlib\nm = importlib.import_module('qrh.' + 'appro' + 'vals')\n",
    "m = __import__('qrh')\n",
    "open(os.path.expanduser('~/.qrh/keys/approval.key.enc')).read()\n",
    "import sqlite3\nsqlite3.connect(home + '/registry.sqlite3')\n",
]


@pytest.mark.parametrize("content", HELPERS)
def test_helper_scripts_that_reach_the_approval_code_are_blocked_with_a_banner(home, content):
    from qrh.banner import BANNER

    for name, ti in (("Write", {"file_path": "/tmp/project/helper.py", "content": content}),
                     ("Edit", {"file_path": "/tmp/project/helper.py", "old_string": "pass", "new_string": content}),
                     ("MultiEdit", {"file_path": "/tmp/project/h.py", "edits": [{"old_string": "a", "new_string": content}]}),
                     ("NotebookEdit", {"notebook_path": "/tmp/project/n.ipynb", "new_source": content})):
        out = tool(home, name, **ti)
        assert out.exit_code == 2 and out.stderr.startswith(BANNER), (name, content)
        assert "best-effort" in out.stderr


@pytest.mark.parametrize("ti", [
    {"path": "KEY"}, {"paths": ["/tmp/a", "KEY"]}, {"nested": {"target": "~/.qrh/keys"}}, {"uri": "file://HOME"},
    {"path": "HOME/config.yaml", "content": "runaway: null"}, {"directory": "USER"},
])
def test_mcp_tools_that_reference_protected_paths_are_blocked(home, user_home, ti):
    import json as _json

    text = _json.dumps(ti).replace("KEY", str(home / "keys" / "approval.key.enc")).replace("HOME", str(home))
    text = text.replace("USER", str(user_home))
    assert tool(home, "mcp__filesystem__do", **_json.loads(text)).exit_code == 2, text


@pytest.mark.parametrize("name,ti", [
    ("Bash", {"command": "rg foo src"}), ("Bash", {"command": "grep -r TODO src/"}),
    ("Bash", {"command": "find . -name '*.py'"}), ("Bash", {"command": "cp a.txt b.txt"}),
    ("Bash", {"command": "echo hi > out.txt"}), ("Bash", {"command": "python script.py"}),
    ("Bash", {"command": "ls -la /tmp/project"}), ("Bash", {"command": "git diff HEAD~1"}),
    ("Grep", {"pattern": "x", "path": "/tmp/project"}), ("Glob", {"pattern": "**/*.py"}),
    ("Write", {"file_path": "/tmp/project/src/x.py", "content": "import numpy as np\nprint(np.pi)\n"}),
    ("Edit", {"file_path": "/tmp/project/.claude/agents/x.md", "old_string": "a", "new_string": "b"}),
    ("mcp__filesystem__read_file", {"path": "/tmp/project/a.py"}),
    ("mcp__github__search", {"query": "approval guard"}),
])
def test_controls_stay_allowed(home, name, ti):
    out = tool(home, name, cwd="/tmp/project", **ti) if name != "Bash" else guard(home, ti["command"], cwd="/tmp/project")
    assert out.exit_code == 0, (name, ti, out.stderr)
