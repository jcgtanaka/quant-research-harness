"""data-guard: an agent reading a registered dataset marked privacy: private sends licensed data to the cloud model.

The block is approvable (banner, pending approval, single-use human approval). Public datasets stay readable.
Every payload is only shown to the hook (JSON on stdin, temporary QRH_HOME); nothing is executed.
"""

import os

import pytest

from hook_helpers import human_approve, hook_log, payload, register_dataset, run
from qrh.banner import BANNER

NAME = "data-guard"


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


@pytest.fixture
def priv(tmp_path, home):
    """A private dataset (the default) at <tmp>/vault/prices.csv and a public one at <tmp>/open/synth.csv."""
    register_dataset(home, tmp_path, "synth", privacy="public", subdir="open")
    return register_dataset(home, tmp_path, "prices", subdir="vault")


@pytest.fixture
def pub(tmp_path, priv):
    return tmp_path / "open" / "synth.csv"


def guard(home, tool, cwd, **tool_input):
    data = payload("PreToolUse", session_id="s1", tool_name=tool, tool_input=tool_input, tool_use_id="t", cwd=str(cwd))
    return run(NAME, data, home, env={"QRH_HOME": str(home)})


def sh(home, command, cwd):
    return guard(home, "Bash", cwd, command=command)


def blocked(out):
    return out.exit_code == 2 and out.stderr.startswith(BANNER) and "qrh approve" in out.stderr


# --- the banner and the approval path --------------------------------------------------------------------

def test_read_of_a_private_dataset_is_an_approvable_banner_block(home, priv, tmp_path):
    out = guard(home, "Read", tmp_path, file_path=str(priv))
    assert blocked(out)
    assert "sends licensed data to the cloud model" in out.stderr
    assert "prices" in out.stderr


def test_an_approved_read_passes_once_for_the_identical_action(home, priv, tmp_path):
    first = guard(home, "Read", tmp_path, file_path=str(priv))
    human_approve(home, first.stderr)
    assert guard(home, "Read", tmp_path, file_path=str(priv)).exit_code == 0
    assert blocked(guard(home, "Read", tmp_path, file_path=str(priv)))      # single use
    assert blocked(guard(home, "Read", tmp_path, file_path=str(priv), limit=5))  # a different action


def test_an_approved_bash_read_passes_once(home, priv, tmp_path):
    cmd = f"head -n 3 {priv}"
    first = sh(home, cmd, tmp_path)
    human_approve(home, first.stderr)
    assert sh(home, cmd, tmp_path).exit_code == 0
    assert blocked(sh(home, cmd, tmp_path))


def test_the_block_is_logged_with_a_pending_approval(home, priv, tmp_path):
    guard(home, "Read", tmp_path, file_path=str(priv))
    rec = hook_log(home)[-1]
    assert rec["hook"] == NAME and rec["decision"] == "block" and rec.get("approval_pending")


# --- public and unregistered data stay readable ----------------------------------------------------------

def test_public_dataset_is_readable_by_every_tool(home, pub, tmp_path):
    assert guard(home, "Read", tmp_path, file_path=str(pub)).exit_code == 0
    for cmd in (f"cat {pub}", f"head -n 2 {pub}", f"awk '{{print $1}}' {pub}", f"jq . {pub}",
                f"python3 -c \"import pandas as pd; print(pd.read_csv('{pub}'))\""):
        assert sh(home, cmd, tmp_path).exit_code == 0, cmd
    assert guard(home, "Grep", tmp_path, pattern="x", path=str(pub), output_mode="content").exit_code == 0


def test_unregistered_files_and_an_empty_catalog_are_readable(home, priv, tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("hello\n")
    assert guard(home, "Read", tmp_path, file_path=str(other)).exit_code == 0
    assert sh(home, f"cat {other}", tmp_path).exit_code == 0
    fresh = tmp_path / "fresh_home"
    assert run(NAME, payload("PreToolUse", tool_name="Read", tool_input={"file_path": str(priv)}, cwd=str(tmp_path)),
               fresh, env={}).exit_code == 0


def test_other_events_and_tools_are_ignored(home, priv, tmp_path):
    assert run(NAME, payload("Stop", cwd=str(tmp_path)), home).exit_code == 0
    assert guard(home, "Glob", tmp_path, pattern="**/*.csv").exit_code == 0
    assert guard(home, "WebFetch", tmp_path, url="https://example.com").exit_code == 0


# --- Read ------------------------------------------------------------------------------------------------

def test_read_path_forms_that_reach_the_private_file_are_blocked(home, priv, tmp_path):
    link = tmp_path / "link.csv"
    link.symlink_to(priv)
    for p in (str(priv), str(tmp_path / "vault" / ".." / "vault" / "prices.csv"), str(link),
              str(tmp_path / "vault" / "." / "prices.csv")):
        assert blocked(guard(home, "Read", tmp_path, file_path=p)), p
    assert blocked(guard(home, "Read", tmp_path / "vault", file_path="prices.csv"))   # relative to the cwd
    assert blocked(guard(home, "Read", tmp_path / "vault", file_path="./prices.csv"))


def test_read_with_an_unreadable_shape_fails_closed(home, priv, tmp_path):
    assert guard(home, "Read", tmp_path, file_path=123).exit_code == 2
    assert guard(home, "Grep", tmp_path, pattern="x", path=["a"], output_mode="content").exit_code == 2
    assert guard(home, "Read", tmp_path).exit_code == 0  # no path given: nothing to read


# --- Bash readers -----------------------------------------------------------------------------------------

READERS = [
    "cat {p}", "head -n 5 {p}", "tail -n 5 {p}", "tail -f {p}", "less {p}", "more {p}", "sed -n 1,5p {p}",
    "sed -n '/x/p' {p}", "awk '{{print $1}}' {p}", "gawk -F, 'NR<5' {p}", "jq . {p}", "jq -c '.[]' {p}",
    "tac {p}", "nl {p}", "cut -d, -f1 {p}", "sort {p}", "uniq {p}", "od -c {p}", "xxd {p}", "strings {p}",
    "column -s, -t {p}", "grep 2020 {p}", "rg 2020 {p}", "zcat {p}", "bat {p}", "base64 {p}", "tee out.txt < {p}",
    "cat < {p}", "jq . < {p}", "cat {p} | head -3", "echo hi && cat {p}", "true; head {p}",
    "sudo cat {p}", "env X=1 cat {p}", "nice -n 5 cat {p}", "command cat {p}", "timeout 5 cat {p}",
    "sh -c 'cat {p}'", "bash -c \"head -2 {p}\"", "eval cat {p}", "cat \"{p}\"", "cat '{p}'", "diff {p} /dev/null",
    "dd if={p}", "xargs cat < {p}", "echo {p} | xargs cat", "$(echo cat) {p}", "echo $(cat {p})", "echo `cat {p}`",
    "cat <(cat {p})", "(cat {p})", "{{ cat {p}; }}", "paste {p}", "head -c 100 {p}",
]


@pytest.mark.parametrize("template", READERS)
def test_shell_readers_of_a_private_dataset_are_blocked(home, priv, tmp_path, template):
    assert blocked(sh(home, template.format(p=priv), tmp_path)), template


@pytest.mark.parametrize("template", [
    "cat {rel}", "cat ./{rel}", "head vault/../{rel}", "cat {tmp}/vault/prices.csv", "cat {tmp}//vault///prices.csv",
    "cat {tmp}/vault/./prices.csv", "cat {tmp}/va*/prices.csv", "cat {tmp}/vault/pric?s.csv", "cat {tmp}/vault/*.csv",
    "cat {tmp}/vault/prices.{{csv,txt}}", "cat {tmp}/{{vault,x}}/prices.csv", "cat {tmp}/vault/prices.c[s]v",
])
def test_path_spellings_resolve_to_the_private_dataset(home, priv, tmp_path, template):
    cmd = template.format(rel="prices.csv", tmp=tmp_path)
    cwd = tmp_path / "vault" if "{rel}" in template else tmp_path
    assert blocked(sh(home, cmd, cwd)), cmd


def test_relative_path_follows_cd_and_symlinks(home, priv, tmp_path):
    link = tmp_path / "lnk"
    link.symlink_to(priv)
    assert blocked(sh(home, "cd vault && cat prices.csv", tmp_path))
    assert blocked(sh(home, f"cd {tmp_path}/vault; head prices.csv", tmp_path))
    assert blocked(sh(home, "pushd vault >/dev/null && tail prices.csv", tmp_path))
    assert blocked(sh(home, "cat lnk", tmp_path))
    assert sh(home, "cd open && cat synth.csv", tmp_path).exit_code == 0
    assert sh(home, "cd open && cat prices.csv", tmp_path).exit_code == 0     # no such file there, not the private one


def test_paths_through_variables_are_resolved_or_block(home, priv, tmp_path):
    assert blocked(sh(home, f"F={priv}; cat $F", tmp_path))
    assert blocked(sh(home, f"D={tmp_path}/vault; cat $D/prices.csv", tmp_path))
    assert blocked(sh(home, f"D={tmp_path}/vault && cat ${{D}}/prices.csv", tmp_path))
    assert blocked(sh(home, f"export F={priv} && head $F", tmp_path))
    assert blocked(sh(home, "cat $UNSET_VARIABLE", tmp_path))                  # cannot be resolved: conservative block
    assert blocked(sh(home, "cat $UNSET_DIR/prices.csv", tmp_path))
    assert blocked(sh(home, "cat $(echo prices).csv", tmp_path / "vault"))
    assert sh(home, "cat $UNSET_DIR/README.md", tmp_path).exit_code == 0       # the rest can not name a private file
    assert sh(home, "cat $(git rev-parse --show-toplevel)/README.md", tmp_path).exit_code == 0


def test_home_and_tilde_expand(home, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    register_dataset(home, tmp_path, "prices", subdir="data")
    assert blocked(sh(home, "cat ~/data/prices.csv", tmp_path / "elsewhere"))
    assert blocked(sh(home, "cat $HOME/data/prices.csv", tmp_path / "elsewhere"))
    assert blocked(sh(home, "cat ${HOME}/data/prices.csv", tmp_path / "elsewhere"))


def test_walking_a_directory_that_contains_a_private_dataset(home, priv, tmp_path):
    assert blocked(sh(home, f"grep -r 2020 {tmp_path}/vault", tmp_path))
    assert blocked(sh(home, f"rg 2020 {tmp_path}", tmp_path))
    assert blocked(sh(home, "rg 2020 .", tmp_path))
    assert blocked(sh(home, f"find {tmp_path}/vault -name '*.csv' -exec head {{}} +", tmp_path))
    assert blocked(sh(home, f"find {tmp_path} -name '*.csv' | xargs cat", tmp_path))
    assert sh(home, f"rg -l 2020 {tmp_path}/vault", tmp_path).exit_code == 0    # names only, no content
    assert sh(home, f"rg --files {tmp_path}/vault", tmp_path).exit_code == 0
    assert sh(home, f"grep -rc 2020 {tmp_path}/vault", tmp_path).exit_code == 0
    assert sh(home, f"rg 2020 {tmp_path}/open", tmp_path).exit_code == 0       # public only


@pytest.mark.parametrize("template", [
    "python3 -c \"import pandas as pd; print(pd.read_csv('{p}').head())\"",
    "python -c 'import pandas as pd; df = pd.read_parquet(\"{p}\"); print(df)'",
    "python3 -c \"print(open('{p}').read())\"",
    "python3 -c \"from pathlib import Path; print(Path('{p}').read_text())\"",
    "python3 -c \"import numpy as np; print(np.loadtxt('{p}'))\"",
    "uv run python -c \"import polars as pl; print(pl.read_csv('{p}'))\"",
    "uv run --with pandas python -c \"import pandas as pd; print(pd.read_csv('{p}'))\"",
    "python3 - <<'EOF'\nimport pandas as pd\nprint(pd.read_csv('{p}'))\nEOF",
    "python3 <<EOF\nprint(open('{p}').read())\nEOF",
    "echo \"import pandas as pd; print(pd.read_csv('{p}'))\" | python3",
    "ipython -c \"import pandas as pd; pd.read_csv('{p}')\"",
    "python3 -c \"import pandas as pd; print(pd.read_csv('prices.csv'))\"",
    "python3 -c \"import pandas as pd, os; print(pd.read_csv(os.path.join('/somewhere', 'prices.csv')))\"",
    "Rscript -e \"print(read.csv('{p}'))\"",
    "node -e \"console.log(require('fs').readFileSync('{p}', 'utf8'))\"",
    "perl -e 'open F, \"{p}\"; print <F>'",
    "ruby -e 'puts File.read(\"{p}\")'",
    "duckdb -c \"select * from '{p}'\"",
    "sh -c 'python3 -c \"print(open(\\\"{p}\\\").read())\"'",
])
def test_python_and_other_inline_reads_of_a_private_dataset_are_blocked(home, priv, tmp_path, template):
    assert blocked(sh(home, template.format(p=priv), tmp_path)), template


def test_inline_python_with_variable_and_cwd_forms(home, priv, tmp_path):
    cmd = "python3 -c \"import pandas as pd; print(pd.read_csv('prices.csv'))\""
    assert blocked(sh(home, cmd, tmp_path / "vault"))
    assert blocked(sh(home, f"P={priv}; python3 -c \"import os, pandas as pd; pd.read_csv(os.environ['P'])\"; echo {priv}",
                      tmp_path))


# --- what is NOT caught, and what must stay allowed ------------------------------------------------------

def test_a_script_that_prints_rows_is_not_caught(home, priv, tmp_path):
    """Documented limitation: the hook sees the command, not what a script does with a path it is handed."""
    assert sh(home, f"python3 run_backtest.py --data {priv}", tmp_path).exit_code == 0
    assert sh(home, f"uv run python scripts/show_rows.py {priv}", tmp_path).exit_code == 0
    assert sh(home, f"python3 show.py \"{priv}\"", tmp_path).exit_code == 0
    assert sh(home, f"bash show_rows.sh {priv}", tmp_path).exit_code == 0


def test_the_hook_documents_its_limit():
    from qrh.hooks import data_guard

    doc = data_guard.__doc__ or ""
    assert "script that prints rows" in doc and "NOT caught" in doc
    assert "cloud model" in doc


@pytest.mark.parametrize("template", [
    "wc -l {p}", "ls -l {p}", "stat {p}", "file {p}", "du -h {p}", "sha256sum {p}", "md5sum {p}", "test -f {p}",
    "[ -s {p} ]", "cp {p} {p}.bak", "mv {p} {p}.old", "rm {p}", "touch {p}", "chmod 600 {p}", "wc -l < {p}",
    "git add {p}", "qrh data check prices", "qrh data profile prices --json --no-values",
    "qrh data register {p} --name prices", "uv run qrh run abcdef0123456789", "ls vault",
    "echo {p}", "echo prices.csv", "basename {p}", "dirname {p}", "ls -la {p}", "readlink -f {p}", "cd vault",
])
def test_commands_that_do_not_show_the_content_stay_allowed(home, priv, tmp_path, template):
    assert sh(home, template.format(p=priv), tmp_path).exit_code == 0, template


def test_ordinary_development_commands_stay_allowed(home, priv, tmp_path):
    for cmd in ("ls -la", "git status", "uv run pytest -q", "rg foo src", "cat README.md", "python3 -c 'print(1)'",
                "python3 -c \"import pandas as pd; print(pd.DataFrame({'a': [1]}))\"", "head -n 3 notes.txt",
                "sed -n 1,5p src/qrh/cli.py", "jq . package.json", "awk '{print}' log.txt"):
        assert sh(home, cmd, tmp_path).exit_code == 0, cmd


def test_unrelated_dataset_with_the_same_extension_is_not_blocked(home, priv, tmp_path):
    other = tmp_path / "results.csv"
    other.write_text("a,b\n1,2\n")
    assert sh(home, f"cat {other}", tmp_path).exit_code == 0
    assert guard(home, "Read", tmp_path, file_path=str(other)).exit_code == 0


# --- Grep -------------------------------------------------------------------------------------------------

def test_grep_tool_content_mode_over_a_private_dataset_is_blocked(home, priv, tmp_path):
    assert blocked(guard(home, "Grep", tmp_path, pattern="2020", path=str(priv), output_mode="content"))
    assert blocked(guard(home, "Grep", tmp_path, pattern="2020", path=str(tmp_path / "vault"), output_mode="content"))
    assert blocked(guard(home, "Grep", tmp_path, pattern="2020", output_mode="content"))            # cwd is above it
    assert blocked(guard(home, "Grep", tmp_path / "vault", pattern="2020", output_mode="content", glob="*.csv"))
    assert blocked(guard(home, "Grep", tmp_path, pattern="2020", path=str(tmp_path), glob="**/prices.csv",
                         output_mode="content"))


def test_grep_tool_that_cannot_return_rows_or_cannot_reach_the_file_is_allowed(home, priv, tmp_path):
    base = dict(pattern="2020", path=str(tmp_path / "vault"))
    assert guard(home, "Grep", tmp_path, **base).exit_code == 0                                      # default: file names
    assert guard(home, "Grep", tmp_path, output_mode="files_with_matches", **base).exit_code == 0
    assert guard(home, "Grep", tmp_path, output_mode="count", **base).exit_code == 0
    assert guard(home, "Grep", tmp_path, output_mode="content", glob="*.py", **base).exit_code == 0
    assert guard(home, "Grep", tmp_path, output_mode="content", type="py", **base).exit_code == 0
    assert guard(home, "Grep", tmp_path, output_mode="content", pattern="x", path=str(tmp_path / "open")).exit_code == 0


# --- review fixes: relative registration, pattern blow-up, per-command redirects, unknown cwd, reader rule -----

def _register_relative(home, tmp_path, monkeypatch):
    from hook_helpers import DB
    from qrh.memory.catalog import Catalog

    vault = tmp_path / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    (vault / "prices.csv").write_text("timestamp,close\n2020-01-01T00:00:00Z,1\n", encoding="utf-8")
    (tmp_path / "other").mkdir(exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(vault)
    cat = Catalog(home / DB)
    try:
        cat.register_dataset("prices.csv", name="prices")
        return cat.get_dataset("prices")["path"]
    finally:
        cat.close()


def test_a_relative_registration_is_stored_absolute_and_found_from_another_cwd(home, tmp_path, monkeypatch):
    stored = _register_relative(home, tmp_path, monkeypatch)
    assert os.path.isabs(stored) and stored == str((tmp_path / "vault" / "prices.csv").resolve())
    other = tmp_path / "other"
    monkeypatch.chdir(other)
    assert blocked(sh(home, "cat ../vault/prices.csv", other))
    assert blocked(sh(home, f"head -3 {tmp_path}/vault/prices.csv", other))
    assert blocked(guard(home, "Read", other, file_path=str(tmp_path / "vault" / "prices.csv")))
    assert sh(home, "cat prices.csv", other).exit_code == 0       # a different, unregistered file in this cwd


def test_a_legacy_relative_row_fails_closed_on_the_basename(home, tmp_path):
    register_dataset(home, tmp_path, "prices", subdir="vault")
    from hook_helpers import DB
    from qrh.memory.catalog import Catalog

    cat = Catalog(home / DB)
    try:
        cat.conn.execute("UPDATE datasets SET path=? WHERE name=?", ("vault/prices.csv", "prices"))
    finally:
        cat.close()
    out = sh(home, "cat /somewhere/else/prices.csv", tmp_path)
    assert blocked(out) and "re-register" in out.stderr
    assert blocked(guard(home, "Read", tmp_path, file_path=str(tmp_path / "x" / "prices.csv")))
    assert sh(home, "cat /somewhere/else/notes.csv", tmp_path).exit_code == 0


def test_many_adjacent_unresolved_expansions_do_not_blow_up(home, priv, tmp_path):
    import time

    words = "".join(f"${chr(65 + i)}" for i in range(12))
    t0 = time.monotonic()
    assert blocked(sh(home, f"cat {words}/prices.csv", tmp_path))
    assert blocked(sh(home, "cat " + "".join(f"$(echo {i})" for i in range(12)) + "/prices.csv", tmp_path))
    sh(home, f"cat {words}/zzz.csv", tmp_path)                    # a non-match must also finish fast
    assert time.monotonic() - t0 < 2.0


def test_the_step_budget_blocks_instead_of_running_into_the_hook_timeout(home, priv, tmp_path, monkeypatch):
    from qrh.hooks import data_guard

    monkeypatch.setattr(data_guard, "BUDGET_SECONDS", -1.0)
    out = sh(home, "cat notes.txt", tmp_path)
    assert blocked(out) and "too complex" in out.stderr


def test_a_non_reader_in_the_same_piece_does_not_hide_an_input_redirect(home, priv, tmp_path):
    assert blocked(sh(home, f"[ -f {priv} ] && cat < {priv}", tmp_path))
    assert blocked(sh(home, f"true; head -5 < {priv}", tmp_path))
    assert blocked(sh(home, f"test -s {priv} && jq . < {priv}", tmp_path))
    assert sh(home, f"[ -f {priv} ] && wc -l < {priv}", tmp_path).exit_code == 0
    assert sh(home, f"true; wc -l < {priv}", tmp_path).exit_code == 0


def test_a_cd_to_an_unresolved_or_previous_directory_matches_relative_words_loosely(home, priv, tmp_path):
    assert blocked(sh(home, 'cd "$(git rev-parse --show-toplevel)" && head -5 vault/prices.csv', tmp_path))
    assert blocked(sh(home, "cd vault; cd -; cat vault/prices.csv", tmp_path))
    assert blocked(sh(home, "cd $ELSEWHERE; cat vault/prices.csv", tmp_path))
    assert blocked(sh(home, "cd; cat vault/prices.csv", tmp_path))
    assert sh(home, 'cd "$(git rev-parse --show-toplevel)" && head -5 README.md', tmp_path).exit_code == 0


def test_an_interpreter_given_a_script_file_is_not_a_reader(home, priv, tmp_path):
    assert sh(home, f"python run.py --data {priv} --order sort", tmp_path).exit_code == 0
    assert sh(home, "python run.py --data vault/prices.csv --order sort", tmp_path).exit_code == 0
    assert sh(home, f"python -E run.py --data {priv}", tmp_path).exit_code == 0
    assert sh(home, f"python run.py -c config.toml --data {priv}", tmp_path).exit_code == 0
    assert sh(home, f"uv run python run.py --data {priv} --order cat", tmp_path).exit_code == 0
    assert sh(home, f"cat <<EOF > notes.txt\nsee {priv}\nEOF\npython run.py", tmp_path).exit_code == 0


def test_inline_code_and_real_readers_still_block_under_the_reader_rule(home, priv, tmp_path):
    assert blocked(sh(home, "python -c \"import pandas as pd; pd.read_csv('vault/prices.csv')\"", tmp_path))
    assert blocked(sh(home, f"python -Ec \"print(open('{priv}').read())\"", tmp_path))
    assert blocked(sh(home, f"cat {priv}", tmp_path))
    assert blocked(sh(home, f"timeout 5 cat {priv}", tmp_path))
    assert blocked(sh(home, f"ionice -c 2 cat {priv}", tmp_path))
    assert blocked(sh(home, f"watch -n 1 cat {priv}", tmp_path))
    assert blocked(sh(home, f"uv run cat {priv}", tmp_path))
    assert blocked(sh(home, f"sudo -n cat {priv}", tmp_path))
    assert blocked(sh(home, f"npx --yes cat {priv}", tmp_path))
    assert blocked(sh(home, f"exec -a x cat {priv}", tmp_path))
    assert blocked(sh(home, f"perl -ne 'print' {priv}", tmp_path))
    assert blocked(sh(home, f"python3 <<EOF\nprint(open('{priv}').read())\nEOF", tmp_path))


# --- registration ----------------------------------------------------------------------------------------

def test_registered_as_an_enforcement_pretooluse_hook_for_read_bash_and_grep():
    from qrh.hooks.settings import desired_settings
    from qrh.hooks.specs import HOOKS

    spec = HOOKS[NAME]
    assert spec.event == "PreToolUse" and spec.matcher == "Read|Bash|Grep" and spec.enforcement and spec.approvable
    groups = [g for g in desired_settings()["hooks"]["PreToolUse"] if g.get("matcher") == "Read|Bash|Grep"]
    assert len(groups) == 1 and [h["command"] for h in groups[0]["hooks"]] == ["qrh hook data-guard"]
    assert "licensed data" in spec.risk


def test_a_broken_registry_fails_closed(home, tmp_path):
    home.mkdir(parents=True)
    (home / "registry.sqlite3").write_bytes(b"this is not a database" * 50)
    out = guard(home, "Read", tmp_path, file_path=str(tmp_path / "x.csv"))
    assert out.exit_code == 2


def test_the_doc_states_the_limit():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    text = open(os.path.join(root, "docs", "hooks.md"), encoding="utf-8").read()
    assert "`data-guard`" in text and "script that prints rows" in text
