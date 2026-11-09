"""The shared 'equal to or under a root' rule used by publish-guard, approval-guard and data-guard."""

from __future__ import annotations

import os

from qrh.hooks.paths import is_under, real_path, under_any, under_root


def test_equal_path_matches(tmp_path):
    root = tmp_path / "a"
    root.mkdir()
    assert under_root(str(root), str(root))
    assert is_under(str(root), str(root))


def test_child_matches(tmp_path):
    root = tmp_path / "a"
    (root / "x").mkdir(parents=True)
    assert under_root(str(root / "x" / "f.txt"), str(root))


def test_sibling_sharing_a_prefix_does_not_match(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "bc").mkdir()
    assert not under_root(str(tmp_path / "bc"), str(tmp_path / "b"))
    assert not is_under("/a/bc", "/a/b")
    assert not is_under("/a/bc/x", "/a/b")


def test_trailing_separator_on_root_is_ignored():
    assert is_under("/a/b/c", "/a/b/")
    assert is_under("/anything", "/")


def test_symlink_into_the_root_is_resolved(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "f.txt").write_text("x")
    link = tmp_path / "link"
    os.symlink(root, link)
    assert under_root(str(link / "f.txt"), str(root))
    assert real_path(str(link)) == os.path.realpath(root)


def test_symlink_out_of_the_root_does_not_match(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, root / "escape")
    assert not under_root(str(root / "escape" / "f"), str(root))


def test_missing_path_is_handled_lexically(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert under_root(str(root / "not" / "there.txt"), str(root))
    assert not under_root(str(tmp_path / "rootless" / "x"), str(root))


def test_home_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert real_path("~/x") == os.path.realpath(tmp_path / "x")


def test_under_any_checks_every_root():
    assert under_any("/a/b/c", ["/z", "/a/b"])
    assert not under_any("/a/bc", ["/z", "/a/b"])
    assert not under_any("/a/b", [])
