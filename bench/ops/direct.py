"""One-shot kernel calls for code that is not driven by a ``Problem`` (the demos).

Each helper tries real kernels in order and returns ``(out, impl_name)``;
callers fall back to their torch reference when every kernel fails.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from bench.ops.attention_flashinfer import workspace
from bench.ops.attention_triton import MAX_KV_SPLITS, _decode_fn
from bench.ops.base import call_compat


def _row_indices(slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """slots [T, K] with -1 padding -> (indptr [T+1], indices) int32."""
    valid = slots >= 0
    lens = valid.sum(-1).to(torch.int32)
    indptr = torch.zeros(slots.shape[0] + 1, dtype=torch.int32, device=slots.device)
    indptr[1:] = torch.cumsum(lens, 0)
    return indptr, slots[valid].to(torch.int32)


def _triton(q, k_buf, v_buf, indptr, indices, scale, sinks, has_mla) -> torch.Tensor:
    fwd = _decode_fn()
    t, h = q.shape[:2]
    dv = v_buf.shape[-1]
    o = torch.empty((t, h, dv), dtype=q.dtype, device=q.device)
    logits = torch.empty((t, h, MAX_KV_SPLITS, dv), dtype=torch.float32, device=q.device)
    lse = torch.empty((t, h, MAX_KV_SPLITS), dtype=torch.float32, device=q.device)
    splits = torch.full((t,), MAX_KV_SPLITS, dtype=torch.int32, device=q.device)
    call_compat(
        fwd, q, k_buf, v_buf, o, indptr, indices, logits, lse, splits, MAX_KV_SPLITS,
        sm_scale=scale, k_scale=1.0, v_scale=1.0, sinks=sinks, has_mla=has_mla,
    )
    return o


def sparse_gqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, str]:
    """q [T, Hq, D], k/v [S, Hkv, D], token_slots [T, K] (-1 = pad); one request per row."""
    indptr, indices = _row_indices(token_slots)
    errors: List[str] = []
    try:
        import flashinfer

        t, hq, d = q.shape
        hkv = k_cache.shape[1]
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace(q.device), "NHD", use_tensor_cores=hq // hkv >= 4
        )
        last = torch.ones(t, dtype=torch.int32, device=q.device)
        call_compat(
            w.plan, indptr, indices, last, hq, hkv, d, 1,
            q_data_type=q.dtype, kv_data_type=k_cache.dtype, sm_scale=scale,
        )
        return w.run(q, (k_cache.unsqueeze(1), v_cache.unsqueeze(1))), "flashinfer.decode(page=1)"
    except Exception as e:
        errors.append(f"flashinfer: {e}")
    try:
        return _triton(q, k_cache, v_cache, indptr, indices, scale, None, False), "sglang.triton.decode"
    except Exception as e:
        errors.append(f"triton: {e}")
    raise RuntimeError("; ".join(errors))


def gathered_mqa_decode(
    q: torch.Tensor,
    keys: torch.Tensor,
    scale: float,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, str]:
    """Single-latent MQA over per-query gathered keys (V = K): q [T,H,D], keys [T,K,D]."""
    t, k, d = keys.shape
    flat = keys.reshape(t * k, 1, d)
    slots = torch.arange(t * k, device=q.device, dtype=torch.int32).view(t, k)
    indptr, indices = _row_indices(slots)
    errors: List[str] = []
    try:
        return (
            _triton(q, flat, flat, indptr, indices, scale, attn_sink, True),
            "sglang.triton.decode(mla)",
        )
    except Exception as e:
        errors.append(f"triton: {e}")
    if attn_sink is None and d == 512:
        try:
            from flashinfer.mla import BatchMLAPagedAttentionWrapper

            w = BatchMLAPagedAttentionWrapper(workspace(q.device))
            qo = torch.arange(t + 1, dtype=torch.int32, device=q.device)
            kv_len = torch.full((t,), k, dtype=torch.int32, device=q.device)
            w.plan(qo, indptr, indices, kv_len, q.shape[1], 512, 64, 1, False, scale, q.dtype, q.dtype)
            kpe = torch.zeros((t * k, 1, 64), dtype=q.dtype, device=q.device)
            q_pe = torch.zeros(q.shape[:-1] + (64,), dtype=q.dtype, device=q.device)
            return w.run(q, q_pe, flat, kpe), "flashinfer.mla(page=1)"
        except Exception as e:
            errors.append(f"flashinfer mla: {e}")
    raise RuntimeError("; ".join(errors))
