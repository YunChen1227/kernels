"""Register extracted kernel trees under the original sglang.* import paths."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent


def _ensure_pkg(name: str, path: Optional[Path] = None) -> types.ModuleType:
    if name in sys.modules:
        mod = sys.modules[name]
    else:
        mod = types.ModuleType(name)
        mod.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = mod
    if path is not None:
        path_str = str(path)
        existing = list(getattr(mod, "__path__", []))
        if path_str not in existing:
            existing.append(path_str)
            mod.__path__ = existing  # type: ignore[attr-defined]
        if not getattr(mod, "__file__", None):
            init = path / "__init__.py"
            if init.exists():
                mod.__file__ = str(init)
    return mod


def _load_file_as(name: str, file_path: Path, package: Optional[str] = None) -> types.ModuleType:
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(file_path):
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {file_path} as {name}")
    mod = importlib.util.module_from_spec(spec)
    if package is not None:
        mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Mapping: (sglang module name, local relative path under kernels/)
# Directories become packages; .py files become modules.
_ALIAS_MAP: List[Tuple[str, str]] = [
    # QSA layers package
    ("sglang.srt.layers.attention.qsa", "qsa/layers"),
    ("sglang.srt.layers.attention.qwen_sparse_attn_backend", "qsa/backend/qwen_sparse_attn_backend.py"),
    ("sglang.kernels.ops.attention.qsa_indexer", "qsa/ops/qsa_indexer.py"),
    ("sglang.kernels.kda_kernels.qwen38_qsa_sm121", "qsa/sm121"),
    # DSV4 layers / backends / ops
    ("sglang.srt.layers.attention.dsv4", "dsv4/layers"),
    ("sglang.srt.layers.attention.deepseek_v4_backend", "dsv4/backend/deepseek_v4_backend.py"),
    ("sglang.srt.layers.attention.deepseek_v4_trtllm_backend", "dsv4/backend/deepseek_v4_trtllm_backend.py"),
    ("sglang.srt.layers.attention.deepseek_v4_backend_hip_radix", "dsv4/backend/deepseek_v4_backend_hip_radix.py"),
    ("sglang.kernels.ops.attention.dsv4", "dsv4/ops"),
    ("sglang.kernels.ops.attention.deepseek_v4_rope", "dsv4/ops/deepseek_v4_rope.py"),
    ("sglang.kernels.ops.attention.dsv4_attn_metadata_kernels", "dsv4/ops/dsv4_attn_metadata_kernels.py"),
    ("sglang.kernels.ops.attention.flash_mla_sm120", "dsv4/sm120/flash_mla_sm120.py"),
    ("sglang.kernels.ops.attention.flash_mla_sm120_triton", "dsv4/sm120/flash_mla_sm120_triton.py"),
    ("sglang.kernels.ops.attention.flash_attention_v4_sm120", "dsv4/sm120/flash_attention_v4_sm120.py"),
    ("sglang.srt.layers.moe.moe_runner.deep_gemm_sm120", "dsv4/sm120/deep_gemm_sm120.py"),
    # DSA
    ("sglang.srt.layers.attention.dsa", "dsa/layers"),
    ("sglang.srt.layers.attention.dsa_backend", "dsa/backend/dsa_backend.py"),
    ("sglang.kernels.ops.attention.dsa", "dsa/ops/dsa"),
    ("sglang.kernels.ops.attention.nsa_triton_decode", "dsa/ops/nsa_triton_decode"),
    ("sglang.kernels.ops.attention.dsa_kpool_metadata", "dsa/ops/dsa_kpool_metadata"),
    ("sglang.kernels.ops.attention.dsa_metadata", "dsa/ops/dsa_metadata.py"),
]


def _parent_chain(name: str) -> Iterable[str]:
    parts = name.split(".")
    for i in range(1, len(parts)):
        yield ".".join(parts[:i])


def register_sglang_aliases(root: Optional[Path] = None) -> Dict[str, str]:
    """Create package shells and load extracted modules under sglang.* names.

    Returns a dict of {module_name: local_path} that were registered.
    Heavy modules that need GPU deps are registered as path packages only
    (not eagerly executed) so CPU demos can import pure-torch leaves.
    """
    root = root or _ROOT
    registered: Dict[str, str] = {}

    # Always ensure the top-level package chain exists.
    for pkg in (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.attention",
        "sglang.srt.layers.moe",
        "sglang.srt.layers.moe.moe_runner",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.attention",
        "sglang.kernels.kda_kernels",
    ):
        _ensure_pkg(pkg)

    for mod_name, rel in _ALIAS_MAP:
        target = root / rel
        if not target.exists():
            continue
        for parent in _parent_chain(mod_name):
            _ensure_pkg(parent)
        if target.is_dir():
            _ensure_pkg(mod_name, target)
            registered[mod_name] = str(target)
        else:
            # Register a lazy loader: put a ModuleSpec without executing yet
            # for heavy backends; for small leaf modules we load on demand via
            # a package __getattr__ pattern by creating a placeholder with
            # __spec__ pointing at the file.
            parent_name = mod_name.rsplit(".", 1)[0]
            leaf = mod_name.rsplit(".", 1)[-1]
            parent_mod = _ensure_pkg(parent_name)

            def _make_loader(full: str, path: Path, pkg: str):
                def _load() -> types.ModuleType:
                    return _load_file_as(full, path, package=pkg)

                return _load

            # Store loader on parent for optional lazy use; also pre-register
            # the file module only when it looks like a pure-torch leaf.
            # Eager load is deferred: demos import specific symbols explicitly.
            if not hasattr(parent_mod, "__getattr__"):
                parent_mod._lazy_loaders = {}  # type: ignore[attr-defined]

                def __getattr__(attr: str, _parent=parent_mod, _pname=parent_name):
                    loaders = getattr(_parent, "_lazy_loaders", {})
                    if attr in loaders:
                        return loaders[attr]()
                    raise AttributeError(attr)

                parent_mod.__getattr__ = __getattr__  # type: ignore[attr-defined]
            parent_mod._lazy_loaders[leaf] = _make_loader(  # type: ignore[attr-defined]
                mod_name, target, parent_name
            )
            # Also register a module finder entry so `import sglang....X` works.
            # We install a thin ModuleSpec in sys.modules only on first import
            # via a custom finder below; for simplicity load now if the file
            # is known-safe (qsa kernels).
            registered[mod_name] = str(target)

    # Install a PathFinder-like meta path for remaining file aliases.
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder(root, _ALIAS_MAP))

    return registered


class _AliasFinder:
    """Meta-path finder that resolves sglang.* names to extracted files."""

    def __init__(self, root: Path, alias_map: List[Tuple[str, str]]) -> None:
        self.root = root
        self.map = {name: root / rel for name, rel in alias_map}

    def find_spec(self, fullname: str, path=None, target=None):
        target_path = self.map.get(fullname)
        if target_path is None or not target_path.exists():
            return None
        if target_path.is_dir():
            init = target_path / "__init__.py"
            if init.exists():
                return importlib.util.spec_from_file_location(
                    fullname,
                    init,
                    submodule_search_locations=[str(target_path)],
                )
            # Namespace package
            spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
            spec.submodule_search_locations = [str(target_path)]
            return spec
        return importlib.util.spec_from_file_location(fullname, target_path)

    def invalidate_caches(self) -> None:
        return None
