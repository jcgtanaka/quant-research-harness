import numpy as np
import pandas as pd
import pytest

from qrh.checks.data_quality import (
    RulesError,
    check_dataset,
    check_dataset_report,
    coverage_summary,
    load_rules,
    run_rules,
)
from qrh.memory.catalog import Catalog


def frame(n=40, freq="D", tz="UTC"):
    idx = pd.date_range("2020-01-01", periods=n, freq=freq, tz=tz, name="timestamp")
    base = np.linspace(100.0, 120.0, n)
    return pd.DataFrame({"a": base, "b": base * 1.001, "name": ["x"] * n}, index=idx)


def one(df, rule):
    res = run_rules(df, [rule])
    assert len(res.results) == 1
    return res, res.results[0]


def test_schema_pass_and_fail():
    df = frame()
    assert one(df, {"type": "schema", "columns": {"a": "f", "name": "O"}})[1].passed
    _, r = one(df, {"type": "schema", "columns": {"a": "i", "missing": "f"}})
    assert not r.passed and any("missing" in x for x in r.reasons) and any("'a'" in x for x in r.reasons)


def test_range_min_max_and_positive_with_offenders():
    df = frame()
    df.iloc[3, 0] = -1.0
    df.iloc[7, 0] = 1e9
    _, r = one(df, {"type": "range", "column": "a", "positive": True, "max": 1000})
    assert not r.passed
    text = " ".join(r.reasons)
    assert "2020-01-04" in text and "2020-01-08" in text
    assert one(frame(), {"type": "range", "column": "a", "min": 0, "max": 200})[1].passed


def test_range_reports_max_five_timestamps():
    df = frame()
    df["a"] = -1.0
    _, r = one(df, {"type": "range", "column": "a", "positive": True})
    assert sum("2020-01-" in x for x in r.reasons[0].split(",")) <= 5
    assert "40" in " ".join(r.reasons)  # offender count


def test_relationship_tolerance_and_nulls():
    df = frame()
    assert one(df, {"type": "relationship", "column": "b", "reference": "a", "rel_tol": 0.01})[1].passed
    df.iloc[10:15, 1] = df.iloc[10:15, 1] * 2
    _, r = one(df, {"type": "relationship", "column": "b", "reference": "a", "rel_tol": 0.01})
    assert not r.passed and "2020-01-11" in " ".join(r.reasons)
    df2 = frame()
    df2.iloc[2, 1] = np.nan
    rule = {"type": "relationship", "column": "b", "reference": "a", "rel_tol": 0.01}
    assert not one(df2, rule)[1].passed
    assert one(df2, {**rule, "only_where_both_non_null": True})[1].passed


def test_null_rate_drift_fraction_and_rows():
    df = frame(100)
    df.iloc[-20:, 0] = np.nan
    rule = {"type": "null_rate_drift", "column": "a", "window_fraction": 0.2, "max_increase": 0.1}
    assert not one(df, rule)[1].passed
    assert not one(df, {"type": "null_rate_drift", "column": "a", "window_rows": 20, "max_increase": 0.1})[1].passed
    assert one(frame(100), rule)[1].passed
    # window covering everything cannot be compared with history: fail closed
    assert not one(df, {**rule, "window_fraction": 1.0})[1].passed


def test_gaps_and_duplicates():
    df = frame()
    rule = {"type": "gaps", "freq": "D"}
    assert one(df, rule)[1].passed
    holes = df.drop(df.index[[5, 6]])
    _, r = one(holes, rule)
    assert not r.passed and "2020-01-06" in " ".join(r.reasons)
    dup = pd.concat([df, df.iloc[[3]]]).sort_index()
    _, r = one(dup, rule)
    assert not r.passed and "duplicate" in " ".join(r.reasons)


def test_gaps_freq_inferred_when_not_declared_and_intraday():
    df = frame(30, freq="15min")
    assert one(df, {"type": "gaps"})[1].passed
    assert not one(df.drop(df.index[10]), {"type": "gaps"})[1].passed


def test_timezone():
    assert one(frame(), {"type": "timezone", "zone": "UTC"})[1].passed
    assert not one(frame(tz="America/New_York"), {"type": "timezone", "zone": "UTC"})[1].passed
    naive = frame().tz_localize(None)
    _, r = one(naive, {"type": "timezone", "zone": "UTC"})
    assert not r.passed and "aware" in " ".join(r.reasons)


def test_unknown_rule_type_and_empty_rules_blocked():
    res = run_rules(frame(), [{"type": "nonsense"}])
    assert res.verdict == "BLOCKED"
    assert run_rules(frame(), []).verdict == "BLOCKED"


def test_report_verdicts():
    ok = run_rules(frame(), [{"type": "gaps", "freq": "D"}])
    assert ok.verdict == "PASS"
    bad = run_rules(frame().drop(frame().index[3]), [{"type": "gaps", "freq": "D"}])
    assert bad.verdict == "FAIL" and bad.reasons


def test_load_rules(tmp_path):
    p = tmp_path / "d.rules.yaml"
    p.write_text("rules:\n  - type: gaps\n    freq: D\n")
    assert load_rules(p) == [{"type": "gaps", "freq": "D"}]
    p.write_text("not: [valid")
    with pytest.raises(RulesError):
        load_rules(p)
    p.write_text("rules: 5\n")
    with pytest.raises(RulesError):
        load_rules(p)
    with pytest.raises(RulesError):
        load_rules(tmp_path / "missing.yaml")


@pytest.fixture
def cat(tmp_path):
    c = Catalog(tmp_path / "r.sqlite3")
    yield c
    c.close()


def make(tmp_path, cat, rules_text=None, df=None):
    p = tmp_path / "d.csv"
    (frame() if df is None else df).to_csv(p)
    r = None
    if rules_text is not None:
        r = tmp_path / "d.rules.yaml"
        r.write_text(rules_text)
    cat.register_dataset(p, rules=r)
    return p


def test_check_dataset_no_rules_stays_unchecked(cat, tmp_path):
    make(tmp_path, cat)
    status, reasons = check_dataset(cat, "d")
    assert status == "UNCHECKED" and any("no quality rules declared" in r for r in reasons)
    assert cat.get_dataset("d")["status"] == "UNCHECKED"


def test_check_dataset_pass_and_fail_update_catalog(cat, tmp_path):
    make(tmp_path, cat, "rules:\n  - type: gaps\n    freq: D\n  - type: timezone\n    zone: UTC\n")
    assert check_dataset(cat, "d")[0] == "PASS"
    assert cat.get_dataset("d")["status"] == "PASS"
    df = frame().drop(frame().index[4])
    make(tmp_path, cat, "rules:\n  - type: gaps\n    freq: D\n", df=df)
    status, reasons = check_dataset(cat, "d")
    assert status == "BLOCKED" and reasons
    assert cat.get_dataset("d")["status"] == "BLOCKED"


def test_check_dataset_changed_file_blocked(cat, tmp_path):
    p = make(tmp_path, cat, "rules:\n  - type: gaps\n    freq: D\n")
    frame(41).to_csv(p)
    status, reasons = check_dataset(cat, "d")
    assert status == "BLOCKED" and any("changed" in r for r in reasons)


def test_check_dataset_unknown_and_bad_rules(cat, tmp_path):
    assert check_dataset(cat, "nope")[0] == "BLOCKED"
    make(tmp_path, cat, "rules: [{type: wat}]\n")
    assert check_dataset(cat, "d")[0] == "BLOCKED"


# --- evidence for reports: offenders, coverage, per-rule results ------------------------------


def test_rule_results_keep_up_to_fifty_offending_timestamps_and_the_rule_type():
    df = frame(120)
    df["a"] = -1.0  # every row is out of range
    _, r = one(df, {"type": "range", "column": "a", "min": 0, "name": "a_non_negative"})
    assert r.type == "range" and r.rule == "a_non_negative"
    assert len(r.offenders) == 50 and r.offenders[0] == "2020-01-01T00:00:00+00:00"
    assert len(r.reasons) == 1 and r.reasons[0].count("T00:00:00") == 5  # the message still lists five


def test_passing_rules_have_no_offenders_and_gap_rule_lists_missing_stamps():
    _, ok = one(frame(), {"type": "gaps", "freq": "D"})
    assert ok.offenders == []
    _, bad = one(frame().drop(frame().index[[3, 9]]), {"type": "gaps", "freq": "D"})
    assert bad.offenders == ["2020-01-04T00:00:00+00:00", "2020-01-10T00:00:00+00:00"]


def test_coverage_summary_reports_span_rows_and_gaps():
    df = frame(30).drop(frame(30).index[[10, 11, 12]])
    cov = coverage_summary(df)
    assert cov["start"] == "2020-01-01T00:00:00+00:00" and cov["end"] == "2020-01-30T00:00:00+00:00"
    assert cov["rows"] == 27 and cov["gap_count"] == 1 and cov["step_seconds"] == 86400.0
    assert cov["gaps"] == [{"after": "2020-01-10T00:00:00+00:00", "before": "2020-01-14T00:00:00+00:00", "missing": 3}]


def test_coverage_summary_caps_the_stored_gaps_but_counts_all():
    idx = pd.date_range("2020-01-01", periods=1000, freq="D", tz="UTC")
    df = pd.DataFrame({"a": 1.0}, index=idx.delete(list(range(3, 1000, 4))))  # 250 deleted, the last is the end so 249 holes
    cov = coverage_summary(df)
    assert cov["gap_count"] == 249 and len(cov["gaps"]) == 200


def test_coverage_summary_of_a_non_datetime_index_is_empty_of_gaps():
    cov = coverage_summary(pd.DataFrame({"a": [1.0, 2.0]}))
    assert cov["rows"] == 2 and cov["start"] is None and cov["gaps"] == []


def test_check_dataset_report_returns_per_rule_results(cat, tmp_path):
    make(tmp_path, cat, "rules:\n  - type: gaps\n    freq: D\n  - type: timezone\n    zone: UTC\n")
    status, reasons, report, cov = check_dataset_report(cat, "d")
    assert status == "PASS" and reasons == []
    assert [r.type for r in report.results] == ["gaps", "timezone"] and all(r.passed for r in report.results)
    assert cov["rows"] == 40
