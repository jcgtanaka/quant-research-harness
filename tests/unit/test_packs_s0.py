"""S0 with packs: layered thresholds, lessons, account caps, pack record in receipts, imported attempts."""

import itertools

import pytest
from typer.testing import CliRunner

from packs_helpers import lesson, make_pack, write_yaml
from hook_helpers import human_approve_id
from qrh.cli import app
from qrh.memory.registry import Registry
from qrh.pipeline.receipt import Stage, Verdict, load_receipt
from qrh.pipeline.stages import register_new_trial

runner = CliRunner()
SPEC = {
    "text": "a moving average rule on synthetic bars", "strategy_family": "trend", "asset_class": "equity",
    "frequency": "1D", "parameter_space_size": 4, "thresholds": {"k": 1}, "difference_statement": "baseline",
    "bounds": {"max_gross_leverage": 3.0},
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("QRH_HOME", str(home))
    home.mkdir()
    counter = itertools.count(1)
    reg = Registry(home / "registry.sqlite3", id_factory=lambda: f"t{next(counter)}")
    yield home, reg, tmp_path
    reg.close()


def add(path):
    res = runner.invoke(app, ["pack", "add", str(path)])
    assert res.exit_code == 0, res.output


def s0(env, spec=None):
    home, reg, _ = env
    res = register_new_trial(spec or SPEC, reg, home)
    return res, load_receipt(home / "receipts" / res.trial_id / "S0.json")


def test_without_packs_nothing_changes(env):
    res, r = s0(env)
    assert res.reasons == [] and r.packs == {}
    assert "pack_layering" not in r.params and "lessons" not in r.params


def test_pack_set_is_recorded_with_version_and_hash(env):
    add(make_pack(env[2], "alpha", version="2.1.0"))
    _, r = s0(env)
    assert set(r.packs) == {"alpha"} and r.packs["alpha"]["version"] == "2.1.0" and len(r.packs["alpha"]["hash"]) == 64


def test_changing_a_pack_changes_later_receipts_not_old_ones(env):
    d = make_pack(env[2], "alpha")
    add(d)
    _, old = s0(env)
    write_yaml(d / "lessons" / "new.yaml", lesson("NEW"))
    add(d)
    _, new = s0(env)
    assert old.packs["alpha"]["hash"] != new.packs["alpha"]["hash"]
    again = load_receipt(env[0] / "receipts" / "t1" / "S0.json")
    assert again.packs["alpha"]["hash"] == old.packs["alpha"]["hash"]


def test_thresholds_are_layered_and_recorded(env):
    add(make_pack(env[2], "alpha", thresholds={"all": {"max_pbo": 0.1}, "asset_class": {"equity": {"min_dsr": 0.97}}}))
    _, r = s0(env)
    assert r.params["thresholds"]["max_pbo"] == 0.1 and r.params["thresholds"]["min_dsr"] == 0.97
    assert r.params["pack_layering"]["tightened"]["max_pbo"]["pack"] == "alpha"


def test_hypothesis_looser_than_pack_is_recorded(env):
    add(make_pack(env[2], "alpha", thresholds={"all": {"max_pbo": 0.1}}))
    _, r = s0(env, {**SPEC, "thresholds": {"max_pbo": 0.19}})
    assert r.params["thresholds"]["max_pbo"] == 0.19
    assert r.params["pack_layering"]["hypothesis_looser_than_pack"] == ["max_pbo"]


def test_unapproved_loosening_blocks_s0(env):
    d = make_pack(env[2], "alpha", thresholds={"all": {"max_pbo": 0.3}})
    res = runner.invoke(app, ["pack", "add", str(d)])
    rid = res.output.split("qrh approve ")[1][:12]
    human_approve_id(env[0], rid, "owner accepts a looser pbo here")
    add(d)
    _, r = s0(env)
    assert r.params["thresholds"]["max_pbo"] == 0.3
    assert r.packs["alpha"]["approved_loosening"] == ["alpha:max_pbo=0.3"]
    assert r.params["pack_layering"]["approved_loosening"] == ["alpha:max_pbo=0.3"]
    write_yaml(d / "thresholds.yaml", {"all": {"max_pbo": 0.5}})  # edited after approval
    res = register_new_trial(SPEC, env[1], env[0])
    assert res.trial_id is None and "loosen" in " ".join(res.reasons)


def test_warn_lessons_are_recorded_when_triggered(env):
    add(make_pack(env[2], "alpha", lessons=[
        lesson("hit", trigger={"keywords": ["moving average"]}, statement="Count every variant."),
        lesson("miss", trigger={"keywords": ["unrelated"]}),
        lesson("other-asset", applies_to={"asset_class": ["fx"]}),
    ]))
    _, r = s0(env)
    assert [x["id"] for x in r.params["lessons"]] == ["hit"]
    assert r.params["lessons"][0]["action"] == "warn" and r.params["lessons"][0]["pack"] == "alpha"
    assert r.verdict is Verdict.PASS


def test_require_threshold_floor_fills_or_fails(env):
    add(make_pack(env[2], "alpha", lessons=[lesson("floor", action="require", severity="critical",
                                                  requirement={"threshold_floor": {"name": "min_oos_sharpe", "value": 0.5}})]))
    _, r = s0(env)
    assert r.verdict is Verdict.PASS and r.params["thresholds"]["min_oos_sharpe"] == 0.5
    res = register_new_trial({**SPEC, "thresholds": {"min_oos_sharpe": 0.1}}, env[1], env[0])
    assert res.trial_id is None and "floor" in " ".join(res.reasons) and "min_oos_sharpe" in " ".join(res.reasons)


def test_require_method_must_be_pinned(env):
    add(make_pack(env[2], "alpha", lessons=[lesson("m", action="require", requirement={"method": "s5.pbo=cscv@1"})]))
    _, r = s0(env)
    assert r.verdict is Verdict.PASS  # the default of that day is cscv@1
    add(make_pack(env[2], "beta", lessons=[lesson("m2", action="require", requirement={"method": "s5.pbo=nonexistent@9"})]))
    res = register_new_trial(SPEC, env[1], env[0])
    assert res.trial_id is None and "s5.pbo" in " ".join(res.reasons)


def test_account_caps_tighten_bounds(env):
    add(make_pack(env[2], "alpha", account={"capital": 5000, "base_currency": "USD", "caps": {"max_gross_leverage": 1.0}}))
    _, r = s0(env)
    assert r.params["bounds"]["max_gross_leverage"] == 1.0
    assert r.params["account"]["pack"] == "alpha" and r.params["account"]["capital"] == 5000
    assert r.params["pack_layering"]["bounds_tightened"] == {"max_gross_leverage": {"from": 3.0, "to": 1.0}}


def test_attempts_are_imported_once_and_counted(env):
    home, reg, tmp = env
    add(make_pack(tmp, "alpha", attempts=[{
        "text": "earlier idea", "strategy_family": "trend", "asset_class": "equity", "frequency": "1D",
        "variants_tried": 11, "verdict": "FAIL", "date": "2020-01-01", "notes": ""}]))
    s0(env)
    s0(env)
    assert reg.family_trial_count("trend|equity|1D") == 11
