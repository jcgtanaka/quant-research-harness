import importlib.util
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qrh.cli import app
from qrh.pipeline.receipt import Stage, load_receipt

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
runner = CliRunner()


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    return h


class Example:
    """A copy of an example directory inside tmp, with its data built and registered."""

    def __init__(self, name: str, tmp_path: Path, home: Path):
        self.home = home
        self.dir = tmp_path / name
        shutil.copytree(EXAMPLES / name, self.dir, ignore=shutil.ignore_patterns("__pycache__"))
        self.trial_id: str | None = None

    def build_data(self, **kwargs) -> Path:
        spec = importlib.util.spec_from_file_location(f"make_data_{self.dir.name}", self.dir / "make_data.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.make(self.dir / "data", **kwargs)

    def register(self, data: Path, name: str) -> None:
        res = runner.invoke(app, ["data", "register", str(data), "--name", name, "--rules", str(self.dir / "rules.yaml")])
        assert res.exit_code == 0, res.output

    def new_trial(self) -> str:
        res = runner.invoke(app, ["new", str(self.dir / "hypothesis.yaml")])
        assert res.exit_code == 0, res.output
        self.trial_id = res.stdout.strip().splitlines()[-1]
        return self.trial_id

    def run(self, *extra: str):
        assert self.trial_id
        return runner.invoke(app, ["run", self.trial_id, *extra])

    def receipt(self, stage: str):
        return load_receipt(self.home / "receipts" / str(self.trial_id) / f"{Stage(stage).value}.json")


@pytest.fixture
def example(tmp_path, home):
    def build(name: str, dataset: str, **data_kwargs) -> Example:
        ex = Example(name, tmp_path, home)
        ex.register(ex.build_data(**data_kwargs), dataset)
        ex.new_trial()
        return ex

    return build


@pytest.fixture
def fresh_example(tmp_path, home):
    """An example copy with nothing built or registered yet."""
    return lambda name: Example(name, tmp_path, home)


@pytest.fixture(scope="module")
def shared_01(tmp_path_factory):
    """Example 01 run once for a whole module (read-mostly tests). Automatic reports are on."""
    with pytest.MonkeyPatch.context() as mp:
        tmp = tmp_path_factory.mktemp("shared01")
        home = tmp / "qrh_home"
        mp.setenv("QRH_HOME", str(home))
        mp.delenv("QRH_REPORTS", raising=False)
        ex = Example("01_clean_strategy", tmp, home)
        ex.register(ex.build_data(), "prices")
        ex.new_trial()
        assert ex.run().exit_code == 0
        yield ex


@pytest.fixture(scope="module")
def shared_09(tmp_path_factory):
    """Example 09 (language-model strategy) recorded once and replayed once for a whole module.

    ``first`` and ``second`` map each stage to its receipt hash after the record run and after the replay run."""
    with pytest.MonkeyPatch.context() as mp:
        tmp = tmp_path_factory.mktemp("shared09")
        home = tmp / "qrh_home"
        mp.setenv("QRH_HOME", str(home))
        mp.setenv("QRH_REPORTS", "0")
        ex = Example("09_llm_clean", tmp, home)
        ex.register(ex.build_data(), "prices")
        ex.new_trial()
        stages = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
        ex.record_result = ex.run("--llm-mode", "record")
        ex.first = {s: ex.receipt(s).receipt_hash() for s in stages}
        ex.first_receipts = {s: ex.receipt(s) for s in stages}
        ex.replay_result = ex.run()
        ex.second = {s: ex.receipt(s).receipt_hash() for s in stages}
        yield ex
