"""Minimal triton stub so QSA/DSA/DSV4 modules can import on CPU-only hosts.

Only module-level usages are supported: @triton.jit, tl.constexpr annotations,
triton.cdiv, and triton.next_power_of_2. Launching a jitted kernel raises.
"""

from __future__ import annotations

import math
import sys
import types
from typing import Any, Callable, Optional


class _ConstExpr:
    """Stand-in for tl.constexpr used only as a type annotation / default."""

    def __init__(self, value: Any = None) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"constexpr({self.value!r})"


class _LanguageModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("triton.language")
        self.constexpr = _ConstExpr

    def __getattr__(self, name: str) -> Any:
        # Attribute access during decoration / annotation should not fail.
        return _ConstExpr(name)


class _JitKernel:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", "stub_kernel")
        self.__doc__ = getattr(fn, "__doc__", None)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"triton kernel '{self.__name__}' cannot run under the CPU stub. "
            "Use the torch reference path (cpu_ref backend), or install a real "
            "triton on a CUDA-capable GPU (sm120 for 5070 Ti)."
        )

    def __getitem__(self, grid: Any) -> "_JitKernel":
        # Support kernel[grid](...) launch syntax; still fails on call.
        return self


def jit(
    fn: Optional[Callable[..., Any]] = None, **_kwargs: Any
) -> Callable[..., Any] | _JitKernel:
    def decorator(f: Callable[..., Any]) -> _JitKernel:
        return _JitKernel(f)

    if fn is None:
        return decorator
    return decorator(fn)


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def autotune(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return decorator


def heuristics(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return decorator


class Config:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def install_triton_stub() -> None:
    """Register stub modules in sys.modules if real triton is absent."""
    try:
        import triton  # noqa: F401

        return
    except ImportError:
        pass

    triton_mod = types.ModuleType("triton")
    triton_mod.jit = jit  # type: ignore[attr-defined]
    triton_mod.cdiv = cdiv  # type: ignore[attr-defined]
    triton_mod.next_power_of_2 = next_power_of_2  # type: ignore[attr-defined]
    triton_mod.autotune = autotune  # type: ignore[attr-defined]
    triton_mod.heuristics = heuristics  # type: ignore[attr-defined]
    triton_mod.Config = Config  # type: ignore[attr-defined]
    triton_mod.__version__ = "0.0.0-stub"  # type: ignore[attr-defined]
    triton_mod.math = math  # type: ignore[attr-defined]

    lang = _LanguageModule()
    triton_mod.language = lang  # type: ignore[attr-defined]

    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = lang
