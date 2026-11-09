import pandas as pd
import pytest
from typer.testing import CliRunner

import qrh
from qrh.cli import app

runner = CliRunner()


@pytest.mark.parametrize("cmd", ["audit"])
def test_stubs_fail_closed(cmd):
    result = runner.invoke(app, [cmd])
    assert result.exit_code == 2
    assert f"BLOCKED: {cmd} is not implemented yet (phase P0)" in result.output


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert qrh.__version__ in result.output


def test_synth_writes_file(tmp_path):
    out = tmp_path / "s.csv"
    result = runner.invoke(app, ["synth", "--seed", "3", "--periods", "50", "--out", str(out)])
    assert result.exit_code == 0
    df = pd.read_csv(out, index_col=0)
    assert len(df) == 50 and "close" in df.columns


def test_synth_deterministic(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    for p in (a, b):
        assert runner.invoke(app, ["synth", "--seed", "3", "--periods", "30", "--out", str(p)]).exit_code == 0
    assert a.read_bytes() == b.read_bytes()
