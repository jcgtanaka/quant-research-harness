"""Declarative lint rules from packs: regex lines and AST descriptors, with the usual reasoned suppression."""

from pathlib import Path

import pytest

from packs_helpers import make_pack
from qrh.packs.lint import lint_with_rules
from qrh.packs.model import LintRule, load_pack

SRC = '''import pandas as pd
import forbidden_lib

def f(df):
    a = df["x"].rolling(5, center=True).mean()
    b = df["x"].shift(-2)
    c = df["x"].shift(1)  # fine
    d = df["x"].resample("1D").last()  # qrh: allow PK_REGEX needed for the demo sample
    return a, b, c, d
'''


def rules(*docs):
    return [LintRule.model_validate(d) for d in docs]


@pytest.fixture
def src(tmp_path):
    p = tmp_path / "strat.py"
    p.write_text(SRC)
    return p


def test_regex_rule_flags_matching_lines(src):
    found = lint_with_rules([src], rules({"id": "PK_REGEX", "severity": "WARN", "message": "avoid resample().last()",
                                          "regex": r"resample\(.*\)\.last\(\)"}))
    assert len(found) == 1 and found[0].line == 8 and found[0].suppressed and found[0].reason == "needed for the demo sample"
    assert found[0].severity == "WARN" and found[0].rule_id == "PK_REGEX"


def test_ast_call_with_keyword(src):
    found = lint_with_rules([src], rules({"id": "PK_CENTER", "severity": "BLOCK", "message": "centered window",
                                          "ast": {"call": "rolling", "keyword": {"center": True}}}))
    assert [(f.line, f.severity, f.suppressed) for f in found] == [(5, "BLOCK", False)]


def test_ast_negative_first_argument(src):
    found = lint_with_rules([src], rules({"id": "PK_SHIFT", "severity": "BLOCK", "message": "negative shift",
                                          "ast": {"call": "shift", "negative_first_arg": True}}))
    assert [f.line for f in found] == [6]


def test_ast_import(src):
    found = lint_with_rules([src], rules({"id": "PK_IMPORT", "severity": "WARN", "message": "banned import",
                                          "ast": {"import": "forbidden_lib"}}))
    assert [f.line for f in found] == [2]


def test_unparseable_source_is_ignored_by_ast_rules_but_regex_still_runs(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("def (:\n  resample('1D').last()\n")
    found = lint_with_rules([p], rules({"id": "R", "severity": "WARN", "message": "m", "regex": "resample"},
                                       {"id": "A", "severity": "WARN", "message": "m", "ast": {"call": "x"}}))
    assert [f.rule_id for f in found] == ["R"]


def test_rules_load_from_a_pack(tmp_path):
    d = make_pack(tmp_path, lint=[{"rules": [{"id": "R1", "severity": "WARN", "message": "m", "regex": "x"}]}])
    assert [r.id for r in load_pack(d).lint_rules] == ["R1"]
