"""The two new methods: default status without promotion, no invented references, shown by the methods commands."""

from typer.testing import CliRunner

from qrh.cli import app
from qrh.methods.builtin import BUILTIN
from qrh.methods.selection import MethodSelector

runner = CliRunner()


def test_both_methods_are_default_with_no_predecessor_and_no_invented_reference():
    for mid, version in (("s2.llm_gate", "llm_gate@1"), ("llm.replay_cache", "replay_cache@1")):
        versions = BUILTIN.versions(mid)
        assert [m.version for m in versions] == [version]  # new checks: no earlier version to beat
        m = versions[0]
        assert m.status == "default" and m.stage == "S2"
        assert all(not r.verified or (r.doi or r.arxiv or r.url) for r in m.references)  # never verified without an id
        assert m.references == ()  # none was checked against a publisher record


def test_a_selector_hands_them_out_as_the_default_and_records_them_unflagged(tmp_path):
    sel = MethodSelector(tmp_path)
    assert sel.get("s2.llm_gate").version == "llm_gate@1" and sel.get("llm.replay_cache").version == "replay_cache@1"
    assert sel.non_default() == {}
    assert sel.full_map()["s2.llm_gate"] == "llm_gate@1"


def test_methods_list_and_show_name_them(tmp_path, monkeypatch):
    monkeypatch.setenv("QRH_HOME", str(tmp_path / "h"))
    listing = runner.invoke(app, ["methods", "list", "--stage", "S2"])
    assert listing.exit_code == 0 and "s2.llm_gate" in listing.output and "llm.replay_cache" in listing.output
    shown = runner.invoke(app, ["methods", "show", "s2.llm_gate@llm_gate@1"])
    assert shown.exit_code == 0 and "llm_cutoff_margin_days" in shown.output and "LLM001" in shown.output
