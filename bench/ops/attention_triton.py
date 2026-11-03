"""SGLang Triton decode attention (split-KV, token-level indices)."""

from __future__ import annotations

import importlib
from typing import Optional

import torch

from bench.ops.base import (
    Ctx,
    OpAdapter,
    Problem,
    Runner,
    call_compat,
    register,
    token_index_lists,
)

_MODULES = (
    "sglang.kernels.ops.attention.decode_attention",
    "sglang.srt.layers.attention.triton_ops.decode_attention",
)
MAX_KV_SPLITS = 8


def _decode_fn():
    err = None
    for name in _MODULES:
        try:
            return importlib.import_module(name).decode_attention_fwd
        except Exception as e:  # noqa: PERF203
            err = e
    raise ImportError(f"sglang triton decode_attention not importable: {err}")


@register
class TritonDecodeAttention(OpAdapter):
    """Window and sparse selection are expressed as per-request token index
    lists, exactly how sglang's triton backend handles SWA and NSA decode."""

    backend = "triton"
    ops = ("attn_decode",)
    phases = ("decode",)
    features = frozenset({"sliding_window", "sink", "softcap", "sparse"})
    sm_min = (8, 0)
    decode_q_len_1_only = True

    def extra_check(self, ctx: Ctx) -> Optional[str]:
        try:
            _decode_fn()
        except Exception as e:
            return str(e)
        return None

    def prepare(self, p: Problem, graph: bool) -> Runner:
        fwd = _decode_fn()
        g = p.geom
        dev, dt = p.ctx.device, p.ctx.dtype
        B = len(p.meta["seq_lens"])
        indptr, indices = token_index_lists(p)
        if g.kind == "mla":
            k_buf = p.inputs["kv_cache"].reshape(-1, 1, g.qk_dim)
            v_buf = k_buf[..., : g.v_dim]
        else:
            k_buf = p.inputs["k_cache"].reshape(-1, g.num_kv_heads, g.qk_dim)
            v_buf = p.inputs["v_cache"].reshape(-1, g.num_kv_heads, g.v_dim)
        o = torch.empty((B, g.num_q_heads, g.v_dim), dtype=dt, device=dev)
        logits = torch.empty(
            (B, g.num_q_heads, MAX_KV_SPLITS, g.v_dim), dtype=torch.float32, device=dev
        )
        lse = torch.empty((B, g.num_q_heads, MAX_KV_SPLITS), dtype=torch.float32, device=dev)
        splits = torch.full((B,), MAX_KV_SPLITS, dtype=torch.int32, device=dev)
        sinks = p.inputs.get("sinks")
        q = p.inputs["q"]

        def fn() -> torch.Tensor:
            call_compat(
                fwd,
                q,
                k_buf,
                v_buf,
                o,
                indptr,
                indices,
                logits,
                lse,
                splits,
                MAX_KV_SPLITS,
                sm_scale=g.scale,
                k_scale=1.0,
                v_scale=1.0,
                logit_cap=g.softcap or 0.0,
                sinks=sinks,
                has_mla=g.kind == "mla",
            )
            return o

        return Runner(fn=fn, meta={"kernel": fwd.__module__, "max_kv_splits": MAX_KV_SPLITS})
