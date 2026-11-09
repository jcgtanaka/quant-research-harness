from qrh.checks.lookahead import lint_file, lint_source


def rules(src):
    return sorted((f.rule_id, f.severity) for f in lint_source(src, "x.py") if not f.suppressed)


def test_negative_shift_literal_and_keyword():
    assert ("LA001", "BLOCK") in rules("y = x.shift(-1)\n")
    assert ("LA001", "BLOCK") in rules("y = x.shift(periods=-3)\n")
    assert rules("y = x.shift(1)\n") == []
    assert rules("y = x.shift(n)\n") == [("LA007", "WARN")]  # was silent before the non-literal shift warning


def test_rolling_center():
    assert ("LA002", "BLOCK") in rules("y = x.rolling(5, center=True).mean()\n")
    assert rules("y = x.rolling(5).mean()\n") == []


def test_backfill_variants():
    for src in ('x.bfill()', 'x.backfill()', 'x.fillna(method="bfill")', "x.fillna(method='backfill')"):
        assert ("LA003", "BLOCK") in rules(src + "\n"), src
    assert rules("x.ffill()\nx.fillna(0)\n") == []


def test_iloc_minus_one_in_loops_only():
    assert ("LA004", "BLOCK") in rules("for i in r:\n    v = x.iloc[-1]\n")
    assert ("LA004", "BLOCK") in rules("while c:\n    v = x.iloc[-1]\n")
    assert rules("v = x.iloc[-1]\n") == []
    assert rules("for i in r:\n    v = x.iloc[i]\n") == []


def test_full_series_stat_is_warn_only():
    f = rules("m = x.mean()\ny = z - m\n")
    assert ("LA005", "WARN") in f and all(s == "WARN" for _, s in f)
    assert ("LA005", "WARN") in rules("y = (x - x.mean()) / x.std()\n")
    assert rules("m = x.rolling(5).mean()\ny = z - m\n") == []
    assert rules("m = x.expanding().mean()\ny = z / m\n") == []


def test_finding_fields():
    (f,) = lint_source("a = 1\ny = x.shift(-2)\n", "mod.py")
    assert (f.file, f.line, f.rule_id, f.severity) == ("mod.py", 2, "LA001", "BLOCK")
    assert f.snippet == "y = x.shift(-2)"


def test_suppression_same_line_and_previous_line_with_reason():
    src = "y = x.shift(-1)  # qrh: allow LA001 label for evaluation only\n"
    (f,) = lint_source(src, "x.py")
    assert f.suppressed and f.reason == "label for evaluation only"
    src2 = "# qrh: allow LA001 documented target column\ny = x.shift(-1)\n"
    (g,) = lint_source(src2, "x.py")
    assert g.suppressed and g.reason == "documented target column"


def test_suppression_without_reason_or_wrong_rule_does_not_suppress():
    (f,) = lint_source("y = x.shift(-1)  # qrh: allow LA001\n", "x.py")
    assert not f.suppressed
    (g,) = lint_source("y = x.shift(-1)  # qrh: allow LA002 some reason\n", "x.py")
    assert not g.suppressed


def test_syntax_error_is_block():
    (f,) = lint_source("def (:\n", "x.py")
    assert f.severity == "BLOCK" and f.rule_id == "LA000"


def test_lint_file_missing_is_block(tmp_path):
    (f,) = lint_file(tmp_path / "nope.py")
    assert f.severity == "BLOCK" and f.rule_id == "LA000"


def test_lint_file(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("y = x.shift(-1)\n")
    assert [f.rule_id for f in lint_file(p)] == ["LA001"]


# --- disguised look-ahead: np.roll, named constants, non-literal amounts --------------------------------------------


def test_np_roll_with_a_negative_literal_blocks():
    for src in ("y = np.roll(a, -1)\n", "y = numpy.roll(a, -3)\n", "y = np.roll(a, shift=-2)\n",
                "y = np.roll(a, (-1, 0), axis=(0, 1))\n"):
        assert ("LA006", "BLOCK") in rules(src), src


def test_np_roll_with_a_positive_literal_is_clean_and_a_non_literal_warns():
    assert rules("y = np.roll(a, 1)\n") == []
    assert rules("y = np.roll(a, k)\n") == [("LA007", "WARN")]
    assert rules("y = np.roll(a, shift=k)\n") == [("LA007", "WARN")]


def test_np_roll_under_an_import_alias_is_found():
    assert ("LA006", "BLOCK") in rules("import numpy as nump\ny = nump.roll(a, -1)\n")


def test_a_named_constant_that_is_negative_is_resolved():
    assert ("LA001", "BLOCK") in rules("NEG = -1\ny = x.shift(NEG)\n")
    assert ("LA006", "BLOCK") in rules("NEG = -1\ny = np.roll(a, NEG)\n")
    assert rules("LAG = 1\ny = x.shift(LAG)\n") == []


def test_a_name_that_is_reassigned_or_a_parameter_is_not_resolved_and_warns():
    assert rules("NEG = -1\nNEG = other()\ny = x.shift(NEG)\n") == [("LA007", "WARN")]
    assert rules("def f(x, lag):\n    return x.shift(lag)\n") == [("LA007", "WARN")]
    assert rules("y = x.shift(params['lag'])\n") == [("LA007", "WARN")]


def test_shift_without_an_amount_or_with_a_frequency_is_clean():
    assert rules("y = x.shift()\n") == []
    assert rules("y = x.shift(freq='D')\n") == []


def test_the_new_warning_can_be_suppressed_with_a_reason():
    (f,) = lint_source("y = x.shift(k)  # qrh: allow LA007 k is a positive lag from the grid\n", "x.py")
    assert f.rule_id == "LA007" and f.suppressed
