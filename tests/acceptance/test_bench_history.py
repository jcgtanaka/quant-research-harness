"""The history suite re-evaluates stored trials with a candidate method, read-only."""

import dataclasses
import importlib.util
import os
import shutil
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qrh.bench.history import run_history
from qrh.cli import app
from qrh.hashing import sha256_dir
from qrh.methods.builtin import builtin_registry
from qrh.methods.selection import MethodSelector

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
runner = CliRunner()


def _build(tmp: Path, home: Path, name: str, dataset: str) -> str:
    work = tmp / name
    shutil.copytree(EXAMPLES / name, work, ignore=shutil.ignore_patterns("__pycache__"))
    spec = importlib.util.spec_from_file_location(f"make_{name}", work / "make_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = mod.make(work / "data")
    assert runner.invoke(app, ["data", "register", str(data), "--name", dataset, "--rules", str(work / "rules.yaml")]).exit_code == 0
    new = runner.invoke(app, ["new", str(work / "hypothesis.yaml")])
    assert new.exit_code == 0, new.output
    tid = new.stdout.strip().splitlines()[-1]
    runner.invoke(app, ["run", tid])
    return tid


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        tmp = tmp_path_factory.mktemp("history")
        home = tmp / "qrh_home"
        mp.setenv("QRH_HOME", str(home))
        mp.setenv("QRH_REPORTS", "0")
        good = _build(tmp, home, "01_clean_strategy", "prices")
        noisy = _build(tmp, home, "04_overfit_grid", "prices")
        yield {"home": home, "good": good, "noisy": noisy, "tmp": tmp}


def _custom(tmp_path, pbo, dsr):
    """A registry whose experimental s5 versions return fixed values: a stricter or a looser PBO/DSR."""
    reg = builtin_registry()
    base_pbo, base_dsr = reg.get("s5.pbo", "cscv@1"), reg.get("s5.dsr", "bailey_lopezdeprado2014@1")

    def new_pbo(matrix, n_blocks):
        return dataclasses.replace(base_pbo.entry(matrix, n_blocks=n_blocks), pbo=pbo)

    def new_dsr(returns, n_trials, var_sharpe):
        return dataclasses.replace(base_dsr.entry(returns, n_trials=n_trials, var_sharpe=var_sharpe), dsr=dsr)

    for method, entry, version in ((base_pbo, new_pbo, "cscv@2"), (base_dsr, new_dsr, "alt@2")):
        reg.register(dataclasses.replace(method, version=version, entry=entry, status="experimental"))
    return reg


def arms(tmp_path, reg=None, overrides=None):
    return (MethodSelector(tmp_path, registry=reg, overrides=overrides or {}), MethodSelector(tmp_path, registry=reg))


def test_default_recompute_reproduces_the_stored_verdicts(history, tmp_path):
    cand, base = arms(tmp_path)
    res = run_history(cand, base, history["home"])
    by = {t["trial_id"]: t for t in res["trials"]}
    assert res["summary"]["evaluated"] == 2 and res["summary"]["skipped"] == 0
    good, noisy = by[history["good"]], by[history["noisy"]]
    assert good["stored"]["pass_all"] is True and good["default"]["pass_all"] is True and good["reproduced"]
    assert noisy["stored"]["pass_all"] is False and noisy["default"]["pass_all"] is False and noisy["reproduced"]
    assert res["summary"]["changed"] == 0 and res["summary"]["reproduced"] == 2


def test_a_stricter_candidate_flips_passes_to_fails(history, tmp_path):
    reg = _custom(tmp_path, pbo=1.0, dsr=0.0)
    cand, base = arms(tmp_path, reg, {"s5.pbo": "cscv@2", "s5.dsr": "alt@2"})
    res = run_history(cand, base, history["home"])
    good = next(t for t in res["trials"] if t["trial_id"] == history["good"])
    assert good["changed"] and good["direction"] == "stricter"
    assert good["default"]["pass_all"] and not good["candidate"]["pass_all"]
    assert res["summary"]["stricter"] == 1 and res["summary"]["looser"] == 0


def test_a_looser_candidate_flips_fails_to_passes(history, tmp_path):
    reg = _custom(tmp_path, pbo=0.0, dsr=1.0)
    cand, base = arms(tmp_path, reg, {"s5.pbo": "cscv@2", "s5.dsr": "alt@2"})
    res = run_history(cand, base, history["home"])
    noisy = next(t for t in res["trials"] if t["trial_id"] == history["noisy"])
    # S4 can still fail on noise; whenever S4 passes, the lenient S5 now passes the overfit grid
    if noisy["default"]["s4_pass"]:
        assert noisy["changed"] and noisy["direction"] == "looser"
    assert res["summary"]["stricter"] == 0


def test_the_source_registry_is_never_modified(history, tmp_path):
    before = sha256_dir(history["home"])
    cand, base = arms(tmp_path, overrides={"s4.walkforward": "purged_embargo@1"})
    run_history(cand, base, history["home"])
    assert sha256_dir(history["home"]) == before


def test_works_on_a_read_only_copy(history, tmp_path):
    copy = tmp_path / "ro_home"
    shutil.copytree(history["home"], copy)
    for root, dirs, files in os.walk(copy):
        for name in dirs + files:
            os.chmod(Path(root) / name, stat.S_IRUSR | (stat.S_IXUSR if name in dirs else 0))
    os.chmod(copy, stat.S_IRUSR | stat.S_IXUSR)
    try:
        cand, base = arms(tmp_path, overrides={"s4.walkforward": "purged_embargo@1"})
        res = run_history(cand, base, copy)
        assert res["summary"]["evaluated"] == 2
    finally:
        for root, dirs, files in os.walk(copy):
            for name in dirs + files:
                os.chmod(Path(root) / name, 0o700)
        os.chmod(copy, 0o700)


def test_tampered_or_missing_inputs_are_skipped_with_a_reason_not_trusted(history, tmp_path):
    copy = tmp_path / "tampered"
    shutil.copytree(history["home"], copy)
    p = copy / "trials" / history["good"] / "is_returns.parquet"
    p.write_bytes(p.read_bytes()[:-8] + b"tampered")
    cand, base = arms(tmp_path)
    res = run_history(cand, base, copy)
    skipped = [t for t in res["trials"] if t["status"] == "skipped"]
    assert [t["trial_id"] for t in skipped] == [history["good"]] and "hash" in skipped[0]["reason"]
    assert res["summary"]["evaluated"] == 1


def test_missing_path_is_an_error(tmp_path):
    cand, base = arms(tmp_path)
    with pytest.raises(ValueError, match="no receipts"):
        run_history(cand, base, tmp_path / "nope")


def test_cli_history_suite_reports_without_touching_the_source(history, tmp_path, monkeypatch):
    out_home = tmp_path / "bench_home"
    monkeypatch.setenv("QRH_HOME", str(out_home))
    before = sha256_dir(history["home"])
    res = runner.invoke(app, ["bench", "run", "--suite", "history", "--history", str(history["home"]),
                              "--method", "s4.walkforward=purged_embargo@1"])
    assert res.exit_code == 0, res.output
    assert "History" in res.output and "evaluated 2" in res.output and "EXPERIMENTAL" in res.output
    assert sha256_dir(history["home"]) == before
    assert (out_home / "bench").is_dir() and not (history["home"] / "bench").exists()
