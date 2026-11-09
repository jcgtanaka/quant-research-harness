"""qrh pack: init, add, list, remove, validate, draft, approve-lesson."""

import json
import re

import pytest
import yaml
from typer.testing import CliRunner

from packs_helpers import lesson, make_pack, write_yaml
from hook_helpers import human_approve_id
from qrh import approvals  # noqa: F401
from qrh.cli import app
from qrh.packs.store import active_packs, load_index

runner = CliRunner()


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    monkeypatch.delenv("QRH_DENYLIST", raising=False)
    return h


def invoke(*args):
    return runner.invoke(app, ["pack", *args])


def approval_id(output):
    m = re.search(r"qrh approve ([0-9a-f]{12})", output)
    assert m, output
    return m.group(1)


def test_init_scaffolds_a_valid_private_pack(tmp_path):
    d = tmp_path / "mine"
    res = invoke("init", str(d))
    assert res.exit_code == 0, res.output
    assert yaml.safe_load((d / "qrh-pack.yaml").read_text())["visibility"] == "private"
    assert invoke("validate", str(d)).exit_code == 0
    assert invoke("init", str(d)).exit_code == 2  # never overwrites


def test_add_list_remove_roundtrip_records_path_and_hash(tmp_path, home):
    d = make_pack(tmp_path, lessons=[lesson()])
    res = invoke("add", str(d))
    assert res.exit_code == 0, res.output
    idx = load_index(home)
    assert idx["packs"][0]["name"] == "demo" and len(idx["packs"][0]["hash"]) == 64
    assert idx["packs"][0]["path"] == str(d.resolve())
    assert not (home / "packs" / "demo").exists()  # registered by path, not copied
    out = invoke("list").output
    assert "demo" in out and "1.0.0" in out and "public" in out
    assert invoke("remove", "demo").exit_code == 0
    assert load_index(home)["packs"] == []
    assert invoke("remove", "demo").exit_code == 2


def test_add_invalid_pack_exits_2(tmp_path):
    d = make_pack(tmp_path, version="x")
    res = invoke("add", str(d))
    assert res.exit_code == 2 and "version" in res.output


def test_active_order_is_public_then_private(tmp_path, home):
    invoke("add", str(make_pack(tmp_path, "zeta", visibility="private")))
    invoke("add", str(make_pack(tmp_path, "alpha", visibility="public")))
    assert [p.name for p in active_packs(home)] == ["alpha", "zeta"]


def test_loosening_pack_needs_a_human_approval(tmp_path, home):
    d = make_pack(tmp_path, "loose", thresholds={"all": {"max_pbo": 0.3}})
    res = invoke("add", str(d))
    assert res.exit_code == 2 and "loosen" in res.output and "max_pbo" in res.output
    rid = approval_id(res.output)
    assert load_index(home)["packs"] == []
    human_approve_id(home, rid, "owner accepts a looser pbo for this test")
    res = invoke("add", str(d))
    assert res.exit_code == 0, res.output
    entry = load_index(home)["packs"][0]
    assert entry["approved_loosening"] == ["loose:max_pbo=0.3"]
    assert active_packs(home)[0].name == "loose"


def test_editing_an_approved_pack_to_loosen_more_blocks_loading(tmp_path, home):
    d = make_pack(tmp_path, "loose", thresholds={"all": {"max_pbo": 0.3}})
    rid = approval_id(invoke("add", str(d)).output)
    human_approve_id(home, rid, "owner accepts a looser pbo for this test")
    invoke("add", str(d))
    write_yaml(d / "thresholds.yaml", {"all": {"max_pbo": 0.5}})
    from qrh.packs.model import PackError
    with pytest.raises(PackError, match="loosen"):
        active_packs(home)


def test_validate_public_pack_runs_the_safety_lint(tmp_path):
    d = make_pack(tmp_path, lessons=[lesson(statement="Buy $ZZZQ when it is 12% down on 2020-03-01 for $5000.")])
    res = invoke("validate", str(d))
    assert res.exit_code == 2
    assert "ticker" in res.output and "dollar" in res.output and "percent" in res.output and "date" in res.output


def test_validate_private_pack_skips_the_safety_lint(tmp_path):
    d = make_pack(tmp_path, visibility="private", lessons=[lesson(statement="Sized at $5000 on 2020-03-01.")])
    assert invoke("validate", str(d)).exit_code == 0


def test_validate_detects_emails_paths_and_account_numbers(tmp_path):
    d = make_pack(tmp_path, lessons=[lesson(statement="mail me at someone@example.org, see /home/someone/x, acct DU1234567")])
    out = invoke("validate", str(d)).output
    assert "email" in out and "home path" in out and "account" in out


def test_validate_runs_the_denylist_when_configured(tmp_path, monkeypatch):
    d = make_pack(tmp_path, lessons=[lesson(statement="mentions the forbidden-widget project")])
    deny = tmp_path / "deny.txt"
    deny.write_text("forbidden-widget\n")
    monkeypatch.setenv("QRH_DENYLIST", str(deny))
    res = invoke("validate", str(d))
    assert res.exit_code == 2 and "denylist" in res.output
    assert "forbidden-widget" not in res.output


def test_draft_creates_skeletons_and_never_public(tmp_path):
    src = tmp_path / "notes.md"
    src.write_text("# Costs\n\n- always check fees\n- check slippage\n\nPlain paragraph here.\n\n# Data\n\n- use filing dates\n")
    out = tmp_path / "drafts"
    res = invoke("draft", "--from", str(src), "--out", str(out))
    assert res.exit_code == 0, res.output
    files = sorted((out / "lessons").glob("*.yaml"))
    assert len(files) == 3
    docs = [yaml.safe_load(f.read_text()) for f in files]
    assert all(d["status"] == "draft" and len(d["statement"]) <= 300 and len(d["source_ref"]) == 64 for d in docs)
    assert yaml.safe_load((out / "qrh-pack.yaml").read_text())["visibility"] == "private"
    again = invoke("draft", "--from", str(src), "--out", str(out))
    assert again.exit_code == 0 and len(list((out / "lessons").glob("*.yaml"))) == 3  # deterministic, no duplicates


def test_draft_excerpt_is_capped(tmp_path):
    src = tmp_path / "n.md"
    src.write_text("# H\n\n" + "word " * 200 + "\n")
    out = tmp_path / "d"
    invoke("draft", "--from", str(src), "--out", str(out))
    doc = yaml.safe_load(next((out / "lessons").glob("*.yaml")).read_text())
    assert len(doc["statement"]) <= 300


def test_drafts_are_not_loaded_until_approved_by_a_human(tmp_path, home):
    src = tmp_path / "n.md"
    src.write_text("# Costs\n\n- always check fees\n")
    out = tmp_path / "d"
    invoke("draft", "--from", str(src), "--out", str(out))
    assert invoke("add", str(out)).exit_code == 0
    assert active_packs(home)[0].lessons == [] and active_packs(home)[0].draft_count == 1
    f = next((out / "lessons").glob("*.yaml"))
    res = invoke("approve-lesson", str(f))
    assert res.exit_code == 2 and "qrh approve" in res.output
    assert yaml.safe_load(f.read_text())["status"] == "draft"
    human_approve_id(home, approval_id(res.output), "reviewed this draft lesson myself")
    res = invoke("approve-lesson", str(f))
    assert res.exit_code == 0, res.output
    assert yaml.safe_load(f.read_text())["status"] == "approved"
    invoke("add", str(out))
    assert len(active_packs(home)[0].lessons) == 1


def test_approval_for_one_lesson_does_not_cover_another(tmp_path, home):
    out = tmp_path / "d"
    for name in ("a", "b"):
        make_pack(tmp_path, "d", lessons=[])
    write_yaml(out / "lessons" / "a.yaml", lesson("A", status="draft"))
    write_yaml(out / "lessons" / "b.yaml", lesson("B", status="draft", statement="another"))
    res = invoke("approve-lesson", str(out / "lessons" / "a.yaml"))
    human_approve_id(home, approval_id(res.output), "reviewed this draft lesson myself")
    res = invoke("approve-lesson", str(out / "lessons" / "b.yaml"))
    assert res.exit_code == 2


def test_approve_lesson_rejects_non_drafts(tmp_path):
    d = make_pack(tmp_path, lessons=[lesson()])
    assert invoke("approve-lesson", str(next((d / "lessons").glob("*.yaml")))).exit_code == 2


def test_pack_gate_prints_the_decision_banner(tmp_path):
    from qrh.banner import BANNER, RELAY

    res = invoke("add", str(make_pack(tmp_path, "loose", thresholds={"all": {"max_pbo": 0.3}})))
    assert res.exit_code == 2 and res.output.startswith(BANNER) and RELAY.strip() in res.output
    assert "pack-loosen" in res.output and "max_pbo" in res.output
