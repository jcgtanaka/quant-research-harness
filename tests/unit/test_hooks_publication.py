from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_denylist_check_is_wired_in_pre_commit():
    cfg = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hooks = {h["id"]: h for repo in cfg["repos"] for h in repo["hooks"]}
    assert "gitleaks" in hooks
    deny = hooks["private-denylist"]
    assert "scripts/denylist_check.py" in deny["entry"]
    assert deny["language"] == "system" and deny["pass_filenames"] is True
    assert (ROOT / "scripts" / "denylist_check.py").is_file()
