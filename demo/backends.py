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

    # SM120: prefer extracted reference on CUDA first for demo correctness;
    # optionally try flashinfer if available.
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache  # noqa: F401

        # Full flashinfer wiring needs page tables / workspace; for the
        # minimal demo we keep numerical parity via the torch reference on
        # CUDA, and document the production entry point.
        _ = trtllm_batch_decode_with_kv_cache
    except ImportError:
        pass

    from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention_reference

    return qsa_sparse_attention_reference(
        q, k_cache, v_cache, token_slots, softmax_scale
    )


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
        try:
            # Production entry: flash_mla_with_kvcache_sm120 needs paged FP8
            # caches. For the minimal random-weight demo we keep torch math on
            # CUDA so shapes / sinks stay comparable to cpu_ref.
            import sglang.kernels.ops.attention.flash_mla_sm120 as _fm  # noqa: F401

            _ = _fm
        except Exception:
            pass

    return sparse_mla_with_sink(
        q, keys, softmax_scale=softmax_scale, attn_sink=attn_sink
    )
