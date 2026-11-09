from pathlib import Path

import pytest

from qrh import config


def test_env_var_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("QRH_HOME", str(tmp_path / "h"))
    assert config.qrh_home() == tmp_path / "h"


def test_default_is_dot_qrh_in_home(monkeypatch):
    monkeypatch.delenv("QRH_HOME", raising=False)
    assert config.qrh_home() == Path.home() / ".qrh"


def test_ensure_home_creates(monkeypatch, tmp_path):
    monkeypatch.setenv("QRH_HOME", str(tmp_path / "new" / "home"))
    home = config.ensure_home()
    assert home.is_dir()


def test_ensure_home_unwritable_raises_clear_error(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("QRH_HOME", str(blocker / "home"))
    with pytest.raises(config.QrhHomeError, match="QRH_HOME"):
        config.ensure_home()
