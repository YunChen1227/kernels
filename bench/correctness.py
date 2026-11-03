"""Correctness against the fp32 reference (flashinfer-bench semantics).

An element fails only when it exceeds *both* ``atol`` and ``rtol``; the run
passes when the fraction of passing elements is >= ``matched_ratio``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from bench.ops.base import DEFAULT_TOL, Problem, Runner, Tol
from bench.schema import Correctness, Status


def tolerance(p: Problem, runner: Runner, override: Optional[Tol] = None) -> Tol:
    if override is not None:
        return override
    if runner.tol is not None:
        return runner.tol
    tol = DEFAULT_TOL.get(p.ctx.dtype, Tol(1e-2, 1e-2))
    if p.ctx.op in _MATCHED_999 and p.ctx.dtype != torch.float32:
        tol = Tol(tol.atol, tol.rtol, 0.999)
    return tol


# Half-precision rounding (rope cancellation to ~0; long-K GEMM accumulation and
# bf16 intermediates) leaves a handful of elements just above atol. Layout / weight
# order bugs fail nearly every element, so 99.9% still catches them.
_MATCHED_999 = ("rope", "gemm_qkv", "gemm_o", "mlp", "moe")


def check(
    p: Problem, runner: Runner, out: object, tol: Tol
) -> Tuple[str, Optional[Correctness], str]:
    """Returns (status, correctness, message)."""
    if runner.to_ref_layout is not None:
        out = runner.to_ref_layout(out)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        return Status.RUNTIME_ERROR, None, f"backend returned {type(out).__name__}, not a tensor"
    ref = p.ref()
    if tuple(out.shape) != tuple(ref.shape):
        if out.numel() == ref.numel():
            out = out.reshape(ref.shape)
        else:
            return (
                Status.INCORRECT_SHAPE,
                None,
                f"shape {tuple(out.shape)} != reference {tuple(ref.shape)}",
            )
    if out.dtype != p.ctx.dtype:
        return Status.INCORRECT_DTYPE, None, f"dtype {out.dtype} != {p.ctx.dtype}"
    x = out.float()
    y = ref.float()
    if not torch.isfinite(x).all():
        c = Correctness(float("nan"), float("nan"), 0.0, tol.atol, tol.rtol)
        return Status.INCORRECT_NUMERICAL, c, "output contains NaN/Inf"
    abs_err = (x - y).abs()
    rel_err = abs_err / (y.abs() + 1e-8)
    bad = (abs_err > tol.atol) & (rel_err > tol.rtol)
    n = max(1, abs_err.numel())
    matched = 1.0 - float(bad.sum().item()) / n
    c = Correctness(
        max_abs_error=float(abs_err.max().item()) if abs_err.numel() else 0.0,
        max_rel_error=float(rel_err.max().item()) if rel_err.numel() else 0.0,
        matched_ratio=matched,
        atol=tol.atol,
        rtol=tol.rtol,
    )
    if matched < tol.matched_ratio:
        return (
            Status.INCORRECT_NUMERICAL,
            c,
            f"matched {matched:.4%} < required {tol.matched_ratio:.2%} "
            f"(max_abs {c.max_abs_error:.3e}, max_rel {c.max_rel_error:.3e})",
        )
    return Status.PASSED, c, ""
