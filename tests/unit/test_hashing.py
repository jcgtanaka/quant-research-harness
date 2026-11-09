import math

import pytest

from qrh import hashing


def test_bytes_known_vector():
    assert hashing.sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_file_matches_bytes_and_streams(tmp_path):
    p = tmp_path / "f.bin"
    data = b"x" * (3 * 1024 * 1024 + 5)
    p.write_bytes(data)
    assert hashing.sha256_file(p) == hashing.sha256_bytes(data)


def test_dir_hash_deterministic_and_content_sensitive(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        (root / "sub").mkdir(parents=True)
    (a / "x.txt").write_text("1")
    (a / "sub" / "y.txt").write_text("2")
    # Different creation order, same content.
    (b / "sub" / "y.txt").write_text("2")
    (b / "x.txt").write_text("1")
    assert hashing.sha256_dir(a) == hashing.sha256_dir(b)
    (b / "x.txt").write_text("changed")
    assert hashing.sha256_dir(a) != hashing.sha256_dir(b)


def test_dir_hash_sensitive_to_names(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "one").write_text("same")
    (b / "two").write_text("same")
    assert hashing.sha256_dir(a) != hashing.sha256_dir(b)


def test_canonical_json_sorted_compact():
    assert hashing.canonical_json({"b": 1, "a": [1.5, "x"]}) == '{"a":[1.5,"x"],"b":1}'
    assert hashing.hash_json({"a": 1, "b": 2}) == hashing.hash_json({"b": 2, "a": 1})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        hashing.canonical_json({"v": [bad]})


def test_code_hash_order_independent_and_content_sensitive(tmp_path):
    f1, f2 = tmp_path / "a.py", tmp_path / "b.py"
    f1.write_text("a = 1")
    f2.write_text("b = 2")
    h = hashing.code_hash([f1, f2])
    assert h == hashing.code_hash([f2, f1])
    f2.write_text("b = 3")
    assert h != hashing.code_hash([f1, f2])


def test_code_hash_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        hashing.code_hash([tmp_path / "nope.py"])
