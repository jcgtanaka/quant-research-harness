"""approval-guard hardening: state names in any form, file:// URLs, code-path and persistence writes, installs."""

import pytest

from hook_helpers import bash_payload, human_approve, payload, run
from qrh.banner import BANNER

NAME = "approval-guard"
CWD = "/tmp/project"
SITE = ".venv/lib/python3.12/site-packages"


@pytest.fixture
def user_home(tmp_path, monkeypatch):
    u = tmp_path / "user"
    u.mkdir()
    monkeypatch.setenv("HOME", str(u))
    return u


@pytest.fixture
def home(user_home):
    h = user_home / ".qrh"
    for d in ("keys", "approvals", "logs"):
        (h / d).mkdir(parents=True)
    (h / "keys" / "approval.key.enc").write_text("encrypted")
    (h / "keys" / "approval.pub").write_text("{}")
    (h / "config.yaml").write_text("x: 1\n")
    (h / "registry.sqlite3").write_bytes(b"")
    return h


def guard(home, command, cwd=CWD):
    return run(NAME, bash_payload(command, cwd=cwd), home)


def tool(home, name, cwd=CWD, **ti):
    return run(NAME, payload("PreToolUse", tool_name=name, tool_input=ti, tool_use_id="t", cwd=cwd), home)


def final_block(out):
    return out.exit_code == 2 and "cannot be overridden" in out.stderr


def approvable_block(out):
    return out.exit_code == 2 and out.stderr.startswith(BANNER) and "qrh approve" in out.stderr


# --- (i) interpreters and shells naming the state, in any form ----------------------------------------

STATE_REFERENCES = [
    """python3 -c "import os,sqlite3;sqlite3.connect(os.environ['QRH_HOME']+'/registry.sqlite3')" """,
    """python3 -c "import os;open(os.environ['QRH_HOME']+'/'+'sessions'+'/x','w')" """,
    """python3 -c "import os;p=f'{os.environ[\\"QRH_HOME\\"]}/keys/x'" """,
    """node -e "require('fs').writeFileSync(process.env.QRH_HOME+'/config'+'.yaml','x')" """,
    """H=$QRH_HOME; python3 -c "open('$H/parity/x','w')" """,
    """python3 -c "open(os.path.join(os.environ['QRH_HOME'],'approvals','x'),'w')" """,
    """python3 -c "print(open('/somewhere/else/registry.sqlite3','rb').read())" """,
    """perl -e 'open F, ">$ENV{QRH_HOME}/sessions/x"' """,
    """bash -c 'echo y > ${QRH_HOME}/par""" + """ity/z'""",
]


@pytest.mark.parametrize("cmd", STATE_REFERENCES)
def test_interpreters_naming_the_state_are_blocked_for_good(home, cmd):
    out = guard(home, cmd)
    assert final_block(out), (cmd, out.stderr)


@pytest.mark.parametrize("cmd", [
    "QRH_HOME=/tmp/h uv run pytest -k sessions -q",
    'python3 -c "print({}.keys())"',
    "python script.py --config config.yaml",
    "uv run pytest tests/unit/test_hook_runaway.py -q",
    "python train.py --out model.pth",
    "uv run pytest -q",
])
def test_ordinary_interpreter_commands_stay_allowed(home, cmd):
    assert guard(home, cmd).exit_code == 0, cmd


# --- (ii) file:// URLs ---------------------------------------------------------------------------------

def test_file_urls_to_protected_or_outside_paths_are_blocked_for_good(home, user_home):
    for url in (f"file://{home}/keys/approval.key.enc", f"file://{home}/config.yaml", "file:///etc/passwd",
                f"FILE://{home}/registry.sqlite3", "file://evil-host/tmp/project/a.txt",
                "file:///tmp/project/%2e%2e/%2e%2e/etc/passwd", f"file:{home}/keys/approval.pub",
                f"file://localhost{home}/keys/approval.key.enc", "file:///tmp/project/../../etc/hosts"):
        assert final_block(tool(home, "WebFetch", url=url, prompt="show")), url
        assert final_block(tool(home, "mcp__x__fetch", url=url)), url
        assert final_block(guard(home, f"curl {url}")), url


def test_file_url_inside_the_working_tree_is_allowed_unless_protected(home, user_home):
    assert tool(home, "WebFetch", url="file:///tmp/project/README.md", prompt="x").exit_code == 0
    assert tool(home, "WebFetch", url="https://example.com/a", prompt="x").exit_code == 0
    inside = str(home / "keys" / "approval.pub")
    assert final_block(tool(home, "WebFetch", cwd=str(home), url=f"file://{inside}", prompt="x"))
    assert final_block(tool(home, "WebFetch", cwd="", url="file:///tmp/project/README.md", prompt="x"))


def test_webfetch_is_routed_to_the_guard():
    import re

    from qrh.hooks.specs import HOOKS

    assert re.fullmatch(HOOKS[NAME].matcher, "WebFetch")


# --- (iii) writes that load code at interpreter start -------------------------------------------------

FILE_TOOL_WRITES = [
    ("Write", {"file_path": f"/tmp/project/{SITE}/sitecustomize.py", "content": "pass"}),
    ("Write", {"file_path": f"/tmp/project/{SITE}/x.pth", "content": "import os"}),
    ("Write", {"file_path": "/tmp/project/usercustomize.py", "content": "pass"}),
    ("Edit", {"file_path": f"/tmp/project/{SITE}/pkg/__init__.py", "old_string": "a", "new_string": "b"}),
    ("MultiEdit", {"file_path": f"/tmp/project/{SITE}/evil.pth", "edits": [{"old_string": "a", "new_string": "b"}]}),
    ("Write", {"file_path": "~/.config/systemd/user/x.service", "content": "[Service]"}),
]


@pytest.mark.parametrize("name,ti", FILE_TOOL_WRITES)
def test_file_tool_writes_to_startup_code_are_approvable_blocks(home, name, ti):
    ti = {**ti, "file_path": ti["file_path"].replace("~", str(home.parent))}
    assert approvable_block(tool(home, name, **ti)), (name, ti)


@pytest.mark.parametrize("cmd", [
    f"echo 'import os' > {SITE}/x.pth",
    f"echo pass >> {SITE}/sitecustomize.py",
    f"cp evil.pth {SITE}/",
    f"mv x.pth {SITE}/y.pth",
    "echo pass | tee usercustomize.py",
    f"sed -i s/a/b/ {SITE}/sitecustomize.py",
    f"rm {SITE}/x.pth",
    f"dd if=a of={SITE}/z.pth",
    f"""python3 -c "open('{SITE}/sitecustomize.py','w').write('pass')" """,
])
def test_shell_writes_to_startup_code_are_approvable_blocks(home, cmd):
    assert approvable_block(guard(home, cmd)), cmd


@pytest.mark.parametrize("cmd", [
    "cp model.pth /tmp/x", "echo x > checkpoint.pth", "mv a.pth b.pth", "cp model.pth backup/model.pth",
    "touch ckpt/epoch3.pth",
])
def test_pth_outside_site_directories_is_a_checkpoint_not_start_up_code(home, cmd):
    assert guard(home, cmd).exit_code == 0, cmd
    assert tool(home, "Write", file_path="/tmp/project/checkpoint.pth", content="x").exit_code == 0


@pytest.mark.parametrize("path", [
    "/home/u/.local/lib/python3.12/site-packages/x.pth", "/opt/venv/lib/python3.12/y.pth",
    "/usr/lib/python3/dist-packages/z.pth", ".venv/lib64/python3.11/w.pth",
])
def test_pth_in_site_user_site_or_venv_lib_directories_is_gated(home, path):
    assert approvable_block(guard(home, f"cp evil.pth {path}")), path
    assert approvable_block(tool(home, "Write", file_path=path, content="import os")), path


def test_sitecustomize_is_gated_everywhere(home):
    assert approvable_block(guard(home, "echo pass > sitecustomize.py"))
    assert approvable_block(tool(home, "Write", file_path="/tmp/project/usercustomize.py", content="pass"))


@pytest.mark.parametrize("cmd", [
    f"ls {SITE}", f"cat {SITE}/x.pth", f"rg foo {SITE}", "python train.py --out model.pth",
    "echo hi > notes.txt", "uv run pytest -q",
])
def test_reading_and_ordinary_writes_stay_allowed(home, cmd):
    assert guard(home, cmd).exit_code == 0, cmd


def test_normal_development_file_edits_stay_allowed(home):
    for name, ti in (("Edit", {"file_path": "/tmp/project/src/qrh/signing.py", "old_string": "a", "new_string": "b"}),
                     ("Edit", {"file_path": "/tmp/project/pyproject.toml", "old_string": "a", "new_string": "b"}),
                     ("Write", {"file_path": "/tmp/project/tests/unit/test_x.py", "content": "x = 1\n"}),
                     ("Write", {"file_path": "/tmp/project/docs/a.md", "content": "text\n"})):
        assert tool(home, name, **ti).exit_code == 0, (name, ti)


# --- (iv) installs from places the agent controls ------------------------------------------------------

BLOCKED_INSTALLS = [
    "pip install -e .", "pip install --editable ./pkg", "uv pip install -e /tmp/x", "pip3 install ./local_dir",
    "pip install /tmp/pkg.whl", "pip install git+https://example.com/x.git", "python -m pip install -e .",
    "pip install https://example.com/x.tar.gz", "uv tool install --from ./pkg qrh",
    "uvx --from git+https://example.com/x qrh", "uv add ./local", "uv add git+https://example.com/x",
    "pip install --index-url https://example.com/simple numpy", "pip install -Ue .",
    "PYTHONPATH=/tmp/shadow uv run qrh status", "export PYTHONPATH=x; qrh doctor",
    "env PYTHONPATH=/x python -m qrh status", "uv pip install pkg.tar.gz",
]


@pytest.mark.parametrize("cmd", BLOCKED_INSTALLS)
def test_risky_installs_are_approvable_blocks(home, cmd):
    assert approvable_block(guard(home, cmd)), cmd


@pytest.mark.parametrize("cmd", [
    "uv add numpy", 'uv add "pandas>=2"', "uv sync", "pip install numpy==1.26", "pip install -r requirements.txt",
    "pip install -r reqs/dev.txt", "uv pip install pytest", "PYTHONPATH=src uv run pytest -q", "uv tool install ruff",
    'git commit -m "docs: pip install -e . is gated"', "uv run pytest -q", "git status",
])
def test_plain_installs_and_development_stay_allowed(home, cmd):
    assert guard(home, cmd).exit_code == 0, cmd


# --- (v) persistence ---------------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", [
    "crontab -e", "crontab -l | crontab -", "crontab mycron", "echo x | at now", "at -f job.sh 10:00",
    "systemctl --user enable foo.service", "systemctl --user daemon-reload", "systemctl --user start foo",
    "systemd-run --user sleep 100", "sudo crontab -u me -e",
])
def test_scheduling_commands_are_approvable_blocks(home, cmd):
    assert approvable_block(guard(home, cmd)), cmd


@pytest.mark.parametrize("cmd", [
    "crontab -l", "systemctl --user status foo", "systemctl status ssh", 'echo "meet at noon"', "cat crontab.txt",
    "grep at file.txt",
])
def test_reading_schedules_stays_allowed(home, cmd):
    assert guard(home, cmd).exit_code == 0, cmd


def test_an_approved_block_passes_once_for_the_identical_action(home):
    for out in (guard(home, "pip install -e ."),
                tool(home, "Write", file_path=f"/tmp/project/{SITE}/sitecustomize.py", content="pass")):
        assert approvable_block(out)
    for f in (home / "keys").iterdir():  # swap the placeholder key files for a real test key pair
        f.unlink()
    first = guard(home, "crontab -e")
    human_approve(home, first.stderr)
    assert guard(home, "crontab -e").exit_code == 0
    assert approvable_block(guard(home, "crontab -e"))  # valid once
