import os
import py_compile
import sys
import textwrap

import pytest

from qrh.adapters.loader import AdapterLoadError, load_adapter
from qrh.hashing import code_hash

CODE = textwrap.dedent(
    """
    from qrh.adapters.base import DatasetRef
    class A:
        def parameter_space(self): return {"k": [1, 2]}
        def required_data(self): return [DatasetRef("px")]
        def run(self, params, data, start, end): raise NotImplementedError
    """
)


def write(d, name="adapter.py", code=CODE):
    d.mkdir(exist_ok=True)
    (d / name).write_text(code)


def test_loads_relative_to_base_dir_and_returns_source_files(tmp_path):
    write(tmp_path / "one")
    loaded = load_adapter("adapter:A", tmp_path / "one")
    assert loaded.adapter.parameter_space() == {"k": [1, 2]}
    assert [p.name for p in loaded.source_files] == ["adapter.py"]


def test_same_module_name_in_two_dirs_does_not_collide(tmp_path):
    write(tmp_path / "one")
    write(tmp_path / "two", code=CODE.replace("[1, 2]", "[9]"))
    assert load_adapter("adapter:A", tmp_path / "one").adapter.parameter_space() == {"k": [1, 2]}
    assert load_adapter("adapter:A", tmp_path / "two").adapter.parameter_space() == {"k": [9]}


def test_sibling_imports_are_tracked(tmp_path):
    write(tmp_path / "d", "helper.py", "X = 1\n")
    write(tmp_path / "d", "adapter.py", "import helper\n" + CODE)
    loaded = load_adapter("adapter:A", tmp_path / "d")
    assert sorted(p.name for p in loaded.source_files) == ["adapter.py", "helper.py"]


def test_dotted_module_from_importable_path():
    loaded = load_adapter("qrh.adapters.base:DataView", None, instantiate=False)
    assert loaded.adapter.__name__ == "DataView"


@pytest.mark.parametrize("ref", ["nope:A", "adapter:Missing", "bad"])
def test_failures_raise_adapter_load_error(tmp_path, ref):
    write(tmp_path / "d")
    with pytest.raises(AdapterLoadError):
        load_adapter(ref, tmp_path / "d")


def test_not_an_adapter(tmp_path):
    write(tmp_path / "d", code="class A:\n    pass\n")
    with pytest.raises(AdapterLoadError, match="StrategyAdapter"):
        load_adapter("adapter:A", tmp_path / "d")


def test_import_error_inside_module(tmp_path):
    write(tmp_path / "d", code="import definitely_not_a_module\n")
    with pytest.raises(AdapterLoadError, match="definitely_not_a_module"):
        load_adapter("adapter:A", tmp_path / "d")


# --- receipt integrity: loaded code is exactly the source bytes that were hashed -----------------


def _rewrite_same_mtime(path, text):
    """Replace a file's content with same-size text while keeping its exact mtime."""
    st = path.stat()
    assert len(text.encode()) == st.st_size, "test needs a same-size edit"
    path.write_text(text)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def test_same_size_same_mtime_edit_runs_the_new_adapter_source(tmp_path):
    d = tmp_path / "d"
    write(d)
    v1 = load_adapter("adapter:A", d)
    assert v1.adapter.parameter_space() == {"k": [1, 2]}
    _rewrite_same_mtime(d / "adapter.py", CODE.replace("[1, 2]", "[1, 3]"))
    v2 = load_adapter("adapter:A", d)
    assert v2.adapter.parameter_space() == {"k": [1, 3]}
    assert v2.code_hash == code_hash([d / "adapter.py"]) != v1.code_hash


def test_stale_cached_bytecode_is_never_used(tmp_path):
    d = tmp_path / "d"
    write(d)
    py_compile.compile(str(d / "adapter.py"))  # a pyc of v1 exists, as after an earlier import
    _rewrite_same_mtime(d / "adapter.py", CODE.replace("[1, 2]", "[1, 3]"))
    assert load_adapter("adapter:A", d).adapter.parameter_space() == {"k": [1, 3]}


def test_same_size_same_mtime_edit_of_a_sibling_module_runs_the_new_source(tmp_path):
    d = tmp_path / "d"
    write(d, "helper.py", "X = 1\n")
    write(d, "adapter.py", "import helper\n" + CODE.replace('[1, 2]', "[helper.X]"))
    assert load_adapter("adapter:A", d).adapter.parameter_space() == {"k": [1]}
    _rewrite_same_mtime(d / "helper.py", "X = 2\n")
    v2 = load_adapter("adapter:A", d)
    assert v2.adapter.parameter_space() == {"k": [2]}
    assert v2.code_hash == code_hash([d / "adapter.py", d / "helper.py"])


def test_sibling_package_modules_are_compiled_from_source_too(tmp_path):
    d = tmp_path / "d"
    (d / "pkg").mkdir(parents=True)
    (d / "pkg" / "__init__.py").write_text("")
    (d / "pkg" / "util.py").write_text("X = 1\n")
    write(d, "adapter.py", "from pkg import util\n" + CODE.replace('[1, 2]', "[util.X]"))
    assert load_adapter("adapter:A", d).adapter.parameter_space() == {"k": [1]}
    _rewrite_same_mtime(d / "pkg" / "util.py", "X = 5\n")
    assert load_adapter("adapter:A", d).adapter.parameter_space() == {"k": [5]}


def test_no_bytecode_is_written_and_the_flag_is_restored(tmp_path):
    d = tmp_path / "d"
    write(d, "helper.py", "X = 1\n")
    write(d, "adapter.py", "import helper\n" + CODE)
    before = sys.dont_write_bytecode
    load_adapter("adapter:A", d)
    assert not list(d.rglob("*.pyc")) and not (d / "__pycache__").exists()
    assert sys.dont_write_bytecode is before


def test_same_sibling_name_in_two_directories_does_not_leak(tmp_path):
    for name, val in (("one", 1), ("two", 2)):
        write(tmp_path / name, "helper.py", f"X = {val}\n")
        write(tmp_path / name, "adapter.py", "import helper\n" + CODE.replace("[1, 2]", "[helper.X]"))
    assert load_adapter("adapter:A", tmp_path / "one").adapter.parameter_space() == {"k": [1]}
    two = load_adapter("adapter:A", tmp_path / "two")
    assert two.adapter.parameter_space() == {"k": [2]}
    assert sorted(p.parent.name for p in two.source_files) == ["two", "two"]


def test_code_hash_is_taken_from_the_bytes_that_were_compiled(tmp_path):
    d = tmp_path / "d"
    write(d)
    loaded = load_adapter("adapter:A", d)
    assert loaded.code_hash == code_hash([d / "adapter.py"])
