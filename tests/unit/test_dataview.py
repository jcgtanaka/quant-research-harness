import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qrh.adapters.base import DataView, DatasetRef


def _frame(n=5, tz="UTC"):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz=tz)
    return pd.DataFrame({"x": range(n)}, index=idx)


def test_dataset_ref_is_frozen():
    ref = DatasetRef(name="a", version="1")
    with pytest.raises(Exception):
        ref.name = "b"  # type: ignore[misc]


def test_rejects_naive_index():
    with pytest.raises(ValueError):
        DataView(_frame(tz=None))


def test_rejects_non_datetime_index():
    with pytest.raises(ValueError):
        DataView(pd.DataFrame({"x": [1, 2]}))


def test_index_based_no_lookahead():
    view = DataView(_frame())
    out = view.as_of(pd.Timestamp("2024-01-03", tz="UTC"))
    assert list(out["x"]) == [0, 1, 2]


def test_publication_hides_late_published_row():
    df = _frame()
    df["pub"] = df.index
    df.loc[df.index[1], "pub"] = df.index[1] + pd.Timedelta(days=3)
    view = DataView(df, publication_column="pub")
    out = view.as_of(pd.Timestamp("2024-01-03", tz="UTC"))
    assert 1 not in set(out["x"])
    assert set(out["x"]) == {0, 2}
    later = view.as_of(pd.Timestamp("2024-01-05", tz="UTC"))
    assert 1 in set(later["x"])


def test_rejects_naive_decision_time():
    with pytest.raises(ValueError):
        DataView(_frame()).as_of(pd.Timestamp("2024-01-03"))


def test_rejects_naive_publication_column():
    df = _frame()
    df["pub"] = df.index.tz_localize(None)
    with pytest.raises(ValueError):
        DataView(df, publication_column="pub")


def test_missing_publication_column():
    with pytest.raises(ValueError):
        DataView(_frame(), publication_column="nope")


def test_copy_isolation():
    view = DataView(_frame())
    out = view.as_of(pd.Timestamp("2024-01-05", tz="UTC"))
    out.iloc[0, 0] = 999
    again = view.as_of(pd.Timestamp("2024-01-05", tz="UTC"))
    assert again.iloc[0, 0] == 0


def test_source_mutation_does_not_leak_in():
    df = _frame()
    view = DataView(df)
    df.iloc[0, 0] = 999
    assert view.as_of(pd.Timestamp("2024-01-05", tz="UTC")).iloc[0, 0] == 0


@settings(max_examples=100, deadline=None)
@given(
    lags=st.lists(st.integers(min_value=-5, max_value=10), min_size=1, max_size=30),
    decision_offset=st.integers(min_value=-3, max_value=40),
    use_pub=st.booleans(),
)
def test_as_of_never_returns_unavailable_rows(lags, decision_offset, use_pub):
    n = len(lags)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"x": range(n)}, index=idx)
    if use_pub:
        df["pub"] = [idx[i] + pd.Timedelta(hours=lags[i]) for i in range(n)]
        view = DataView(df, publication_column="pub")
    else:
        view = DataView(df)
    t = idx[0] + pd.Timedelta(hours=decision_offset)
    out = view.as_of(t)
    avail = out["pub"] if use_pub else out.index.to_series()
    assert (avail <= t).all()
    expected = (df["pub"] if use_pub else df.index.to_series()) <= t
    assert len(out) == int(expected.sum())
