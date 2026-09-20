"""Bootstrap extracted kernels for dual-backend demos.

Usage::

    from _compat import bootstrap
    bootstrap()
"""

from __future__ import annotations

from typing import Optional

from . import env as _env
from .sglang_alias import register_sglang_aliases
from .triton_stub import install_triton_stub

_BOOTSTRAPPED = False


def bootstrap(*, force: bool = False) -> _env.EnvInfo:
    """Install stubs / aliases and refresh environment probes.

    - If real triton is missing, install a stub so QSA/DSA/DSV4 modules can
      import and expose their torch reference functions.
    - Register extracted trees under the original ``sglang.*`` import paths.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED and not force:
        return _env.get_env()

    info = _env.refresh()
    if not info.has_triton:
        install_triton_stub()
        # Re-probe so HAS_TRITON reflects the stub (still False — stub is not
        # a real triton; we keep HAS_TRITON=False intentionally).
    register_sglang_aliases()
    _BOOTSTRAPPED = True
    return info


__all__ = [
    "bootstrap",
    "env",
]
