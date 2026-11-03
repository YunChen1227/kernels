"""Attention backend dispatch: cpu_ref (golden) vs sm120 (5070 Ti)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

BackendName = Literal["cpu_ref", "sm120"]


@dataclass
class BackendCaps:
    name: BackendName
    device: torch.device
    dtype: torch.dtype
    notes: str


def resolve_backend(
    name: Optional[BackendName] = None,
    *,
    prefer_auto: bool = True,
) -> BackendCaps:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from _compat import bootstrap
    from _compat import env as envmod

    bootstrap()
    info = envmod.get_env()

    if name is None and prefer_auto:
        name = "sm120" if (info.is_sm120 and info.has_triton) else "cpu_ref"
    if name is None:
        name = "cpu_ref"

    if name == "cpu_ref":
        return BackendCaps(
            name="cpu_ref",
            device=torch.device("cpu"),
            dtype=torch.float32,
            notes="pure-torch reference / golden",
        )

    if name == "sm120":
        envmod.require_sm120()
        return BackendCaps(
            name="sm120",
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            notes="SM120 kernels: flash_mla_sm120 / flashinfer trtllm / FA2",
        )

    raise ValueError(f"unknown backend {name!r}; expected 'cpu_ref' or 'sm120'")


_IMPL_SEEN: dict = {}


def _note_impl(op: str, impl: str) -> None:
    """Print which implementation served ``op`` the first time (or when it changes)."""
    if _IMPL_SEEN.get(op) != impl:
        _IMPL_SEEN[op] = impl
        print(f"  [sm120] {op} -> {impl}")


# ---------------------------------------------------------------------------
# QSA sparse attention
# ---------------------------------------------------------------------------


def qsa_sparse_attention(
    backend: BackendCaps,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Dispatch sparse GQA.

    cpu_ref -> qsa_sparse_attention_reference
    sm120   -> try flashinfer / FA2; fall back to reference on CUDA with warning
    """
    if backend.name == "cpu_ref":
        from sglang.srt.layers.attention.qsa.kernel import (
            qsa_sparse_attention_reference,
        )

        return qsa_sparse_attention_reference(
            q, k_cache, v_cache, token_slots, softmax_scale
        )

    from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention_reference

    scale = softmax_scale or q.shape[-1] ** -0.5
    try:
        from bench.ops.direct import sparse_gqa_decode

        out, impl = sparse_gqa_decode(q, k_cache, v_cache, token_slots, scale)
        _note_impl("qsa_sparse_attention", impl)
        return out
    except Exception as exc:
        _note_impl("qsa_sparse_attention", f"torch reference (kernels failed: {exc})")
    return qsa_sparse_attention_reference(q, k_cache, v_cache, token_slots, softmax_scale)


def qsa_mqa_logits_prefill(
    backend: BackendCaps,
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    from sglang.srt.layers.attention.qsa.mqa import torch_qsa_mqa_prefill

    # Same math on both backends for the demo; production SM120 uses TileLang /
    # fused indexer kernels when available.
    return torch_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)


def qsa_topk(
    backend: BackendCaps,
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk

    return qsa_fast_topk(logits, row_starts, row_ends, topk)


# ---------------------------------------------------------------------------
# DSV4 combined attention
# ---------------------------------------------------------------------------


def dsv4_combined_attention(
    backend: BackendCaps,
    q: torch.Tensor,
    keys: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Combined SWA ∪ compressed attention for one layer.

    cpu_ref: pure torch einsum + softmax (+ optional sink).
    sm120: try flash_mla_sm120; fall back to torch on CUDA for the demo.
    """
    from ref_ops import sparse_mla_with_sink

    if backend.name == "sm120":
        # Production DSV4 uses flash_mla_with_kvcache_sm120 on FP8 pages; the
        # demo holds bf16 gathered keys, so run the same math on the triton /
        # FlashInfer MLA decode kernels. bench/ops/attention_mla.py covers the
        # FP8 FlashMLA port.
        scale = softmax_scale or q.shape[-1] ** -0.5
        try:
            from bench.ops.direct import gathered_mqa_decode

            out, impl = gathered_mqa_decode(q, keys, scale, attn_sink)
            _note_impl("dsv4_combined_attention", impl)
            return out
        except Exception as exc:
            _note_impl("dsv4_combined_attention", f"torch reference (kernels failed: {exc})")

    return sparse_mla_with_sink(
        q, keys, softmax_scale=softmax_scale, attn_sink=attn_sink
    )
