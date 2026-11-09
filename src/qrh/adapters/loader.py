"""Load a strategy adapter from a ``module.path:ClassName`` reference.

The module is looked up first as a file relative to the hypothesis directory
(loaded under a unique name, so two hypotheses can both use ``adapter.py``),
then through the normal import system. Every failure is an AdapterLoadError,
which the pipeline turns into BLOCKED.

Receipt integrity: every module loaded from the hypothesis directory (the
adapter and any sibling it imports) is compiled from the exact source bytes
read at load time, never from cached bytecode, and no bytecode is written. The
loader hashes those same bytes, so the recorded code hash is the hash of the
code that actually ran, even after a same-size edit within the same second.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qrh.adapters.base import StrategyAdapter
from qrh.hashing import hash_json, sha256_bytes


class AdapterLoadError(Exception):
    """The adapter cannot be imported, instantiated or does not satisfy the interface."""


@dataclass
class LoadedAdapter:
    adapter: Any
    source_files: list[Path]
    code_hash: str = ""  # same scheme as ``qrh.hashing.code_hash``, over the bytes actually compiled
    dependency_files: list[Path] = field(default_factory=list)  # prompt files: hashed with the code, never linted


# Names of modules this loader has put into sys.modules; evicted before the next load so a
# changed or same-named sibling in another directory is never served from a stale entry.
_LOADED_NAMES: set[str] = set()


def _source_loader(compiled: dict[Path, str]) -> type[importlib.machinery.SourceFileLoader]:
    """A loader class that compiles from source bytes only and records their hashes."""

    class SourceOnlyLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname: str) -> Any:
            path = self.get_filename(fullname)
            data = self.get_data(path)  # read once: hash and compile the same bytes
            compiled[Path(path).resolve()] = sha256_bytes(data)
            return self.source_to_code(data, path)

    return SourceOnlyLoader


class _DirectoryFinder(importlib.abc.MetaPathFinder):
    """Resolves imports that live under ``base`` with the source-only loader."""

    def __init__(self, base: Path, loader_cls: type[importlib.machinery.SourceFileLoader]):
        self.base = base
        self.loader_cls = loader_cls

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        roots = [str(self.base)] if path is None else [
            p for p in path if Path(p).resolve().is_relative_to(self.base)
        ]
        for root in roots:
            finder = importlib.machinery.FileFinder(
                root, (self.loader_cls, importlib.machinery.SOURCE_SUFFIXES)
            )  # a fresh finder per lookup: no cached directory listing
            spec = finder.find_spec(fullname, target)
            if spec is not None and spec.loader is not None:
                _LOADED_NAMES.add(fullname)
                return spec
        return None


def _module_file(module: str, base_dir: Path) -> Path | None:
    rel = Path(*module.split("."))
    for cand in (base_dir / f"{rel}.py", base_dir / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _import_local(module: str, file: Path, base_dir: Path, loader_cls: type[importlib.machinery.SourceFileLoader]) -> Any:
    unique = "qrh_user_" + hashlib.sha256(str(file.resolve()).encode()).hexdigest()[:12] + "_" + module.replace(".", "_")
    spec = importlib.util.spec_from_file_location(unique, file, loader=loader_cls(unique, str(file)))
    if spec is None or spec.loader is None:
        raise AdapterLoadError(f"cannot load module file {str(file)!r}")
    mod = importlib.util.module_from_spec(spec)
    _LOADED_NAMES.add(unique)
    sys.modules[unique] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(unique, None)
        raise
    return mod


def read_dependencies(base_dir: str | Path | None, relative: Sequence[str]) -> dict[str, tuple[Path, bytes]]:
    """Read prompt files (paths relative to the hypothesis folder) once: {relative: (resolved path, bytes)}.

    A missing file, or one that resolves outside the folder (``..``, an absolute path, a symlink), is an error.
    """
    out: dict[str, tuple[Path, bytes]] = {}
    if not relative:
        return out
    if base_dir is None:
        raise AdapterLoadError("prompt files need the hypothesis folder (base_dir), which this hypothesis has not recorded")
    root = Path(base_dir).resolve()
    for rel in relative:
        path = (root / rel).resolve()
        if Path(rel).is_absolute() or not path.is_relative_to(root):
            raise AdapterLoadError(f"prompt file {rel!r} is outside the hypothesis folder")
        try:
            out[rel] = (path, path.read_bytes())
        except OSError as exc:
            raise AdapterLoadError(f"prompt file {rel!r} cannot be read: {exc}") from exc
    return out


def load_adapter(
    ref: str, base_dir: str | Path | None, instantiate: bool = True, dependencies: Sequence[str] = (),
) -> LoadedAdapter:
    """Load the adapter. ``dependencies`` are prompt files (relative to ``base_dir``) hashed into the code hash."""
    module_name, sep, class_name = ref.partition(":")
    if not sep or not module_name or not class_name:
        raise AdapterLoadError(f"adapter reference {ref!r} must look like 'module.path:ClassName'")
    base = Path(base_dir).resolve() if base_dir else None
    extra = read_dependencies(base_dir, dependencies)  # fail before anything is imported
    for name in list(_LOADED_NAMES):
        sys.modules.pop(name, None)
    _LOADED_NAMES.clear()
    before = set(sys.modules)
    inserted = False
    compiled: dict[Path, str] = {}
    finder: _DirectoryFinder | None = None
    previous_flag = sys.dont_write_bytecode
    if base is not None:
        sys.dont_write_bytecode = True
        loader_cls = _source_loader(compiled)
        finder = _DirectoryFinder(base, loader_cls)
        at = next((i for i, f in enumerate(sys.meta_path) if f is importlib.machinery.PathFinder), len(sys.meta_path))
        sys.meta_path.insert(at, finder)
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
            inserted = True
    try:
        file = _module_file(module_name, base) if base else None
        try:
            mod = _import_local(module_name, file, base, loader_cls) if file and base else importlib.import_module(module_name)
        except Exception as exc:
            raise AdapterLoadError(f"cannot import {module_name!r}: {type(exc).__name__}: {exc}") from exc
        cls = getattr(mod, class_name, None)
        if cls is None:
            raise AdapterLoadError(f"module {module_name!r} has no attribute {class_name!r}")
        files: set[Path] = set()
        if getattr(mod, "__file__", None):
            files.add(Path(mod.__file__).resolve())
        if base is not None:
            for name in set(sys.modules) - before:
                f = getattr(sys.modules[name], "__file__", None)
                if f and Path(f).resolve().is_relative_to(base) and str(f).endswith(".py"):
                    files.add(Path(f).resolve())
        obj: Any = cls
        if instantiate:
            try:
                obj = cls()
            except Exception as exc:
                raise AdapterLoadError(f"cannot instantiate {class_name}: {type(exc).__name__}: {exc}") from exc
            if not isinstance(obj, StrategyAdapter):
                raise AdapterLoadError(f"{class_name} does not implement the StrategyAdapter interface")
        local = sorted(f for f in files if base is not None and f.is_relative_to(base))
        unseen = [f for f in local if f not in compiled]
        if unseen:  # fail closed: code we cannot vouch for must not be hashed from disk
            raise AdapterLoadError(f"module file(s) were not loaded from source: {[str(f) for f in unseen]}")
        hashes = [compiled[f] for f in local] + [sha256_bytes(data) for _p, data in extra.values()]
        digest = hash_json(sorted(hashes)) if hashes else ""
        return LoadedAdapter(obj, sorted(files), digest, sorted(p for p, _d in extra.values()))
    finally:
        sys.dont_write_bytecode = previous_flag
        if finder is not None:
            try:
                sys.meta_path.remove(finder)
            except ValueError:
                pass
        if inserted:
            try:
                sys.path.remove(str(base))
            except ValueError:
                pass
