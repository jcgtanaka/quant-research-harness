import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "denylist_check.py"


def run(files, denylist=None, env_extra=None):
    import os

    env = {k: v for k, v in os.environ.items() if k != "QRH_DENYLIST"}
    if denylist is not None:
        env["QRH_DENYLIST"] = str(denylist)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, files)], capture_output=True, text=True, env=env
    )


def test_unset_warns_and_passes(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("anything")
    r = run([f])
    assert r.returncode == 0
    assert "warning" in (r.stdout + r.stderr).lower()


def test_missing_file_fails_closed(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert run([f], denylist=tmp_path / "nope.txt").returncode == 1


def test_literal_case_insensitive_hit_and_redaction(tmp_path):
    d = tmp_path / "deny.txt"
    d.write_text("# comment\n\nSecretTerm\n")
    f = tmp_path / "a.txt"
    f.write_text("ok\nthis has secretterm inside\n")
    r = run([f], denylist=d)
    assert r.returncode == 1
    out = r.stdout + r.stderr
    assert f"{f}:2" in out
    assert "Se***" in out
    assert "secretterm" not in out.lower()


def test_regex_hit(tmp_path):
    d = tmp_path / "deny.txt"
    d.write_text("re:acct-\\d{4}\n")
    f = tmp_path / "a.txt"
    f.write_text("id ACCT-1234 here\n")
    r = run([f], denylist=d)
    assert r.returncode == 1
    assert "1234" not in r.stdout + r.stderr


def test_clean_passes(tmp_path):
    d = tmp_path / "deny.txt"
    d.write_text("forbidden\n")
    f = tmp_path / "a.txt"
    f.write_text("all good\n")
    assert run([f], denylist=d).returncode == 0


def test_binary_file_skipped(tmp_path):
    d = tmp_path / "deny.txt"
    d.write_text("forbidden\n")
    f = tmp_path / "a.bin"
    f.write_bytes(b"\x00\xff\xfe forbidden")
    assert run([f], denylist=d).returncode == 0


def test_invalid_regex_fails_closed(tmp_path):
    d = tmp_path / "deny.txt"
    d.write_text("re:(unclosed\n")
    f = tmp_path / "a.txt"
    f.write_text("x\n")
    assert run([f], denylist=d).returncode == 1
