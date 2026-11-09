import numpy as np
import pandas as pd
import pytest

from qrh.adapters.base import DataView
from qrh.bench.corpus import BUG_CLASSES, CONTROLS, Case, corpus, materialize
from qrh.bench.engine import WARMUP


def test_corpus_covers_the_required_bug_classes_and_controls():
    cases = corpus()
    classes = {c.bug_class for c in cases}
    assert classes == set(BUG_CLASSES) | {CONTROLS}
    assert set(BUG_CLASSES) == {
        "lookahead_shift", "centered_rolling", "backward_fill", "full_sample_normalization",
        "same_bar_execution", "percent_units", "relationship_corruption", "gaps_duplicates",
    }
    names = [(c.bug_class, c.name) for c in cases]
    assert len(names) == len(set(names))
    shifts = [c for c in cases if c.bug_class == "lookahead_shift"]
    assert len(shifts) >= 4  # several k, literal and through a variable
    rel = [c for c in cases if c.bug_class == "relationship_corruption"]
    assert len(rel) >= 6  # several magnitudes and lengths
    same_bar = [c for c in cases if c.bug_class == "same_bar_execution"]
    assert len(same_bar) >= 9 and len({c.volatility for c in same_bar}) >= 3  # volatilities x smoothing
    assert len({c.name.rsplit("_", 1)[-1] for c in same_bar}) >= 3
    pct = [c for c in cases if c.bug_class == "percent_units"]
    assert len(pct) >= 6 and len({c.volatility for c in pct}) >= 3
    assert len([c for c in cases if c.bug_class == CONTROLS]) >= 5


def test_materialize_is_deterministic_and_seed_dependent():
    for case in corpus():
        a, b, c = materialize(case, 11), materialize(case, 11), materialize(case, 12)
        assert a.frame.equals(b.frame) and a.source == b.source
        assert not a.frame.equals(c.frame), case.name


@pytest.mark.parametrize("case", [c for c in corpus() if c.kind in ("code", "control")], ids=lambda c: f"{c.bug_class}:{c.name}")
def test_every_code_case_compiles_and_runs(case):
    m = materialize(case, 3)
    assert m.source and "class Planted" in m.source
    adapter = m.adapter()
    start, end = m.frame.index[WARMUP], m.frame.index[-1]
    res = adapter.run({"w": 10}, DataView(m.frame), start, end)
    assert len(res.returns) == len(m.frame.loc[start:end]) and np.isfinite(res.returns.to_numpy()).all()


def test_code_cases_carry_the_pattern_they_claim():
    by = {(c.bug_class, c.name): materialize(c, 1).source for c in corpus() if c.source_template}
    assert any("shift(-1)" in s for (k, _), s in by.items() if k == "lookahead_shift")
    assert any("center=True" in s for (k, _), s in by.items() if k == "centered_rolling")
    assert any(".bfill(" in s for (k, _), s in by.items() if k == "backward_fill")
    assert all("shift(-" not in s and "center=True" not in s and "bfill" not in s for (k, _), s in by.items() if k == CONTROLS)


def test_relationship_corruption_changes_only_the_planted_rows():
    for case in (c for c in corpus() if c.bug_class == "relationship_corruption"):
        m = materialize(case, 5)
        clean = m.frame["ref_close"]
        off = (m.frame["close"] - clean).abs() > 0
        assert off.sum() == case.params["length"]
        rel = ((m.frame["close"] / clean) - 1).abs()[off]
        assert np.allclose(rel, case.params["magnitude"])


def test_gaps_and_duplicates_change_the_index_as_declared():
    cases = {c.name: c for c in corpus() if c.bug_class == "gaps_duplicates"}
    base_len = len(materialize(next(c for c in corpus() if c.bug_class == CONTROLS), 4).frame)
    for name, case in cases.items():
        m = materialize(case, 4)
        n = case.params["count"]
        if case.params["what"] == "gap":
            assert len(m.frame) == base_len - n and not m.frame.index.has_duplicates
        else:
            assert len(m.frame) == base_len + n and m.frame.index.has_duplicates and m.frame.index.is_monotonic_increasing


def test_percent_unit_cases_scale_strategy_returns_by_100():
    case = next(c for c in corpus() if c.bug_class == "percent_units")
    m = materialize(case, 2)
    assert "* 100.0" in m.source
    assert isinstance(case, Case) and m.frame["close"].gt(0).all() and isinstance(m.frame.index, pd.DatetimeIndex)


def test_variants_below_the_rule_tolerance_are_expected_undetectable():
    from qrh.bench.corpus import RULES

    tol = next(r["rel_tol"] for r in RULES if r["type"] == "relationship")
    rel = [c for c in corpus() if c.bug_class == "relationship_corruption"]
    assert {c.expected_undetectable for c in rel} == {True, False}
    for c in rel:
        assert c.expected_undetectable == (c.params["magnitude"] <= tol), c.name
    assert not any(c.expected_undetectable for c in corpus() if c.bug_class != "relationship_corruption")


def test_same_bar_cases_execute_in_the_same_bar_and_percent_cases_scale_by_100():
    for c in corpus():
        src = materialize(c, 1).source or ""
        if c.bug_class == "same_bar_execution":
            assert "strategy = (position * ret)" in src and "shift(1) * ret" not in src
        if c.bug_class == "percent_units":
            assert "* 100.0" in src
