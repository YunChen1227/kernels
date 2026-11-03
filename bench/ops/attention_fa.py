"""FlashAttention adapters: FA2 (``flash_attn``) and FA3 (``sgl_kernel`` / ``flash_attn_interface``)."""

from __future__ import annotations

import inspect
from typing import Callable, Optional

import torch

from bench.ops.base import (
    Ctx,
    OpAdapter,
    Problem,
    Runner,
    decode_key_positions,
    gather_tokens,
    register,
    repage,
    slots_for_positions,
    token_slots,
)


def _window(g) -> tuple:
    return (g.window - 1, 0) if g.window is not None else (-1, -1)


@register
class FlashAttn2(OpAdapter):
    """FA2 paged KV needs page (block) size divisible by 256, so the cache is
    re-paged to 256 at prepare time; varlen prefill uses gathered contiguous KV."""

    backend = "fa2"
    ops = ("attn_decode", "attn_prefill")
    kinds = ("gqa",)
    features = frozenset({"sliding_window", "softcap"})
    requires = ("flash_attn",)
    sm_min = (8, 0)
    kv_dtypes = ("bfloat16", "float16")

    def extra_check(self, ctx: Ctx) -> Optional[str]:
        g = ctx.geom
        if g is not None and g.qk_dim != g.v_dim:
            return "FA2 needs head_dim_qk == head_dim_v"
        if g is not None and g.qk_dim > 256:
            return "FA2 supports head_dim <= 256"
        return None

    def prepare(self, p: Problem, graph: bool) -> Runner:
        import flash_attn

        g = p.geom
        q_all = p.inputs["q"]
        q_lens = p.meta["q_lens"]
        B = len(q_lens)
        softcap = g.softcap or 0.0
        if len(set(q_lens)) == 1:
            (kc, vc), table = repage(p, [p.inputs["k_cache"], p.inputs["v_cache"]], 256)
            seq_lens = p.inputs["seq_lens"]
            ql = q_lens[0]

            def fn() -> torch.Tensor:
                o = flash_attn.flash_attn_with_kvcache(
                    q_all.view(B, ql, g.num_q_heads, g.qk_dim),
                    kc,
                    vc,
                    cache_seqlens=seq_lens,
                    block_table=table,
                    softmax_scale=g.scale,
                    causal=True,
                    window_size=_window(g),
                    softcap=softcap,
                )
                return o.reshape(-1, g.num_q_heads, g.v_dim)

            return Runner(fn=fn, meta={"kernel": "flash_attn_with_kvcache", "page_size": 256})

        slots = torch.cat([token_slots(p, b) for b in range(B)])
        k = gather_tokens(p.inputs["k_cache"], slots).contiguous()
        v = gather_tokens(p.inputs["v_cache"], slots).contiguous()
        cu_k = torch.zeros(B + 1, dtype=torch.int32, device=q_all.device)
        cu_k[1:] = torch.cumsum(p.inputs["seq_lens"], 0)
        cu_q = p.inputs["qo_indptr"]
        max_q, max_k = p.meta["max_q_len"], p.meta["max_seq_len"]

        def fn_varlen() -> torch.Tensor:
            return flash_attn.flash_attn_varlen_func(
                q_all,
                k,
                v,
                cu_q,
                cu_k,
                max_q,
                max_k,
                softmax_scale=g.scale,
                causal=True,
                window_size=_window(g),
                softcap=softcap,
            )

        return Runner(fn=fn_varlen, meta={"kernel": "flash_attn_varlen_func", "kv": "contiguous"})


def _fa3_kvcache() -> Optional[Callable]:
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        return flash_attn_with_kvcache
    except Exception:
        pass
    try:
        from flash_attn_interface import flash_attn_with_kvcache

        return flash_attn_with_kvcache
    except Exception:
        return None


@register
class FlashAttn3(OpAdapter):
    """FA3 (Hopper). Uses sgl_kernel's build when present (supports sinks), else
    the upstream ``flash_attn_interface``. Sparse decode = page_size 1 page_table
    of selected slots; MLA decode uses the ``qv`` path like sglang's fa3 backend."""

    backend = "fa3"
    ops = ("attn_decode", "attn_prefill")
    kinds = ("gqa", "mla")
    features = frozenset({"sliding_window", "softcap", "sink", "sparse"})
    sm_allow = ((9, 0),)
    kv_dtypes = ("bfloat16", "float16")

    def extra_check(self, ctx: Ctx) -> Optional[str]:
        fn = _fa3_kvcache()
        if fn is None:
            return "neither sgl_kernel.flash_attn nor flash_attn_interface is installed"
        g = ctx.geom
        if g is not None and g.sink and "sinks" not in inspect.signature(fn).parameters:
            return "installed FA3 build has no sinks argument"
        if g is not None and g.kind == "mla" and g.rope_dim == 0:
            return "fa3 MLA (qv) path here needs a rope part (qk_rope_dim > 0)"
        if g is not None and g.sparse_topk is not None and ctx.wl.q_len != 1:
            return "sparse decode via page_size=1 page_table needs q_len == 1"
        return None

    def prepare(self, p: Problem, graph: bool) -> Runner:
        fa = _fa3_kvcache()
        g = p.geom
        B = len(p.meta["seq_lens"])
        dev = p.ctx.device
        q_all = p.inputs["q"]
        if g.sparse_topk is not None:
            rows = [slots_for_positions(p, b, decode_key_positions(p, b)) for b in range(B)]
            width = max(len(r) for r in rows)
            page_table = torch.zeros((B, width), dtype=torch.int32, device=dev)
            for b, r in enumerate(rows):
                page_table[b, : len(r)] = r
            cache_seqlens = torch.tensor([len(r) for r in rows], dtype=torch.int32, device=dev)
            flat = True
            window = (-1, -1)
        else:
            page_table = p.inputs["page_table"]
            cache_seqlens = p.inputs["seq_lens"]
            flat = False
            window = _window(g)

        def pages(c: torch.Tensor) -> torch.Tensor:
            return c.reshape(-1, 1, *c.shape[2:]) if flat else c

        kw = dict(
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            cu_seqlens_q=p.inputs["qo_indptr"],
            max_seqlen_q=p.meta["max_q_len"],
            softmax_scale=g.scale,
            causal=True,
            window_size=window,
            softcap=g.softcap or 0.0,
        )
        if g.sink:
            kw["sinks"] = p.inputs["sinks"]

        if g.kind == "mla":
            cache = p.inputs["kv_cache"]
            ckv = pages(cache[..., : g.v_dim]).unsqueeze(-2)
            kpe = pages(cache[..., g.v_dim :]).unsqueeze(-2)

            def fn_mla() -> torch.Tensor:
                q_nope = q_all[..., : g.v_dim]
                q_pe = q_all[..., g.v_dim :]
                return fa(q_pe, kpe, ckv, qv=q_nope, **kw).reshape(-1, g.num_q_heads, g.v_dim)

            return Runner(fn=fn_mla, meta={"kernel": "fa3 kvcache (qv MLA)", "module": fa.__module__})

        kc, vc = pages(p.inputs["k_cache"]), pages(p.inputs["v_cache"])

        def fn() -> torch.Tensor:
            return fa(q_all, kc, vc, **kw)

        return Runner(fn=fn, meta={"kernel": "fa3 kvcache", "module": fa.__module__})
