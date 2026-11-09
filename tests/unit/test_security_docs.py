"""The security documentation and the workflow skill state the rules the agent must follow."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_security_doc_covers_setup_guarantees_and_recovery():
    text = (ROOT / "docs" / "security.md").read_text(encoding="utf-8")
    for item in ("qrh approve --init", "qrh config sign", "Threat model", "does not guarantee", "passphrase",
                 "lost", "--rotate", "monkeypatch"):
        assert item in text, item


def test_security_doc_has_deployment_hardening_and_no_overclaims():
    text = (ROOT / "docs" / "security.md").read_text(encoding="utf-8")
    assert "## Deployment hardening (recommended)" in text
    for item in ("non-editable", "out of band", "separate OS user", "base64", "strict xfail", "informational",
                 "for that\n  key", "key replacement", "code editing", "only while a baseline exists and\n  is intact"):
        assert item in text, item
    assert "no valid approval can be made.\n" not in text and "(only a warning follows)" not in text


def test_hooks_doc_documents_the_publish_guard_and_its_config():
    text = (ROOT / "docs" / "hooks.md").read_text(encoding="utf-8")
    assert "## Publish guard" in text and "publish_guard:" in text and "protected_repos" in text
    assert "thirteen" in text and "eleven" not in text


def test_interviews_doc_states_the_matching_rules_and_links_security():
    text = (ROOT / "docs" / "interviews.md").read_text(encoding="utf-8")
    assert "security.md" in text and "whole word" in text and "being hardened" not in text


def test_skill_relays_banners_verbatim_and_never_approves():
    text = (ROOT / "skills" / "qrh-workflow" / "SKILL.md").read_text(encoding="utf-8")
    for item in ("WARNING: DECISION TO BE MADE BY YOU!", "verbatim", "one question at a time",
                 "never answer for the user", "qrh interview next", "qrh config sign"):
        assert item.lower() in text.lower(), item


def test_readme_marks_p3d_b1_done():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "P3d-b1" in text and "docs/security.md" in text


def test_architecture_has_packs_and_interviews():
    text = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert "## Packs and interviews" in text and "## Approvals and integrity" in text


def test_readiness_docs_skill_readme_and_architecture_cover_p3d_b2_part_a():
    doc = (ROOT / "docs" / "readiness.md").read_text(encoding="utf-8")
    for item in ("qrh readiness ack", "readiness-ack", "report hash", "PASS · CONDITIONAL", "guard.require_readiness",
                 "WARNING: DECISION TO BE MADE BY YOU!", "methods.md"):
        assert item in doc, item
    skill = (ROOT / "skills" / "qrh-workflow" / "SKILL.md").read_text(encoding="utf-8")
    assert "never run it" in skill and "qrh readiness ack" in skill and "verbatim" in skill
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "P3d-b2 part A" in readme and "the three LLM rules" in readme
    assert "## Deployment readiness" in (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    hooks = (ROOT / "docs" / "hooks.md").read_text(encoding="utf-8")
    assert "require_readiness" in hooks and "readiness ack" in hooks
