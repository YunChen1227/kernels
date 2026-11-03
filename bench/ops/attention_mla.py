"""MLA-specific decode kernels: FlashMLA (sm_90/sm_100) and the SM120 FlashMLA port.

``flashmla_sm120`` is the DSV4 sparse decode entry in ``dsv4/sm120/flash_mla_sm120.py``.
It reads an FP8 page layout (448 fp8 nope + 64 bf16 rope per token, per-64
e8m0 scales in a page footer), so prepare packs the canonical bf16 cache into
that layout and the correctness check uses an FP8-sized tolerance.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from bench.ops.base import (
    Ctx,
    OpAdapter,
    Problem,
    Runner,
    Tol,
    decode_key_positions,
    register,
    repage,
)


@register
class FlashMLA(OpAdapter):
    backend = "flashmla"
    ops = ("attn_decode",)
    phases = ("decode",)
    kinds = ("mla",)
    features = frozenset()
    requires = ("flash_mla",)
    sm_allow = ((9, 0), (10, 0), (10, 3))
    kv_dtypes = ("bfloat16",)
    dtypes = ("bfloat16",)

    def extra_check(self, ctx: Ctx) -> Optional[str]:
        g = ctx.geom
        if g is not None and (g.v_dim != 512 or g.rope_dim != 64):
            return "FlashMLA dense decode needs kv_lora_rank 512 and rope 64"
        return None

    def prepare(self, p: Problem, graph: bool) -> Runner:
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata

        g = p.geom
        B = len(p.meta["seq_lens"])
        ql = p.meta["q_lens"][0]
        (cache,), table = repage(p, [p.inputs["kv_cache"]], 64)
        kcache = cache.unsqueeze(2)  # [P, 64, 1, 576]
        seq_lens = p.inputs["seq_lens"]
        sched, splits = get_mla_metadata(seq_lens, ql * g.num_q_heads, 1)
        q = p.inputs["q"]

        def fn() -> torch.Tensor:
            out, _ = flash_mla_with_kvcache(
                q.view(B, ql, g.num_q_heads, g.qk_dim),
                kcache,
                table,
                seq_lens,
                g.v_dim,
                sched,
                splits,
                softmax_scale=g.scale,
                causal=True,
            )
            return out.reshape(-1, g.num_q_heads, g.v_dim)

        return Runner(fn=fn, meta={"kernel": "flash_mla_with_kvcache", "page_size": 64})


_NOPE, _ROPE, _TILE = 448, 64, 64
_STRIDE = _NOPE + 2 * _ROPE  # 576 bytes per token (fp8 nope + bf16 rope)
_SCALE_STRIDE = _NOPE // _TILE + 1  # 7 scales + 1 pad


def pack_dsv4_fp8_pages(cache: torch.Tensor) -> torch.Tensor:
    """bf16 [P, ps, 512] -> FP8 footer-format pages viewed as [P, ps, 1, 576]."""
    P, ps, d = cache.shape
    assert d == _NOPE + _ROPE
    page_bytes = math.ceil((ps * _STRIDE + ps * _SCALE_STRIDE) / _STRIDE) * _STRIDE
    raw = torch.zeros((P, page_bytes), dtype=torch.uint8, device=cache.device)
    nope = cache[..., :_NOPE].float().view(P, ps, _NOPE // _TILE, _TILE)
    amax = nope.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    nope_fp8 = (nope / scale).to(torch.float8_e4m3fn).view(torch.uint8).view(P, ps, _NOPE)
    rope = cache[..., _NOPE:].to(torch.bfloat16).contiguous().view(torch.uint8)
    raw[:, : ps * _STRIDE] = torch.cat([nope_fp8, rope], dim=-1).reshape(P, -1)
    sc = torch.zeros((P, ps, _SCALE_STRIDE), dtype=torch.uint8, device=cache.device)
    sc[..., : _NOPE // _TILE] = scale.squeeze(-1).to(torch.float8_e8m0fnu).view(torch.uint8)
    raw[:, ps * _STRIDE : ps * (_STRIDE + _SCALE_STRIDE)] = sc.reshape(P, -1)
    fp8 = raw.view(torch.float8_e4m3fn)
    return fp8[:, : ps * _STRIDE].view(P, ps, 1, _STRIDE)


@register
class FlashMLASm120(OpAdapter):
    backend = "flashmla_sm120"
    ops = ("attn_decode",)
    phases = ("decode",)
    kinds = ("mla",)
    features = frozenset({"sink", "sparse", "sliding_window"})
    sm_allow = ((12, 0), (12, 1))
    dtypes = ("bfloat16",)
    kv_dtypes = ("bfloat16", "fp8")
    decode_q_len_1_only = True

    def extra_check(self, ctx: Ctx) -> Optional[str]:
        g = ctx.geom
        if g is not None and (g.qk_dim != 512 or g.rope_dim != 0):
            return "SM120 FlashMLA port is DSV4-only (512-dim latent, 448 nope + 64 rope inside)"
        if not hasattr(torch, "float8_e8m0fnu"):
            return "torch build lacks float8_e8m0fnu (needed to pack FP8 scales)"
        try:
            self._mod()
        except Exception as e:  # import needs triton + sglang aliases
            return f"cannot import flash_mla_sm120: {e}"
        return None

    @staticmethod
    def _mod():
        from _compat import bootstrap

        bootstrap()
        import sglang.kernels.ops.attention.flash_mla_sm120 as m

        return m

    def prepare(self, p: Problem, graph: bool) -> Runner:
        m = self._mod()
        g = p.geom
        B = len(p.meta["seq_lens"])
        dev = p.ctx.device
        (cache,), table = repage(p, [p.inputs["kv_cache"]], 256)
        kcache = pack_dsv4_fp8_pages(cache)
        rows = []
        for b in range(B):
            pos = decode_key_positions(p, b)
            # repage() lays request b's pages out contiguously from table[b, 0].
            rows.append((table[b, 0].long() * 256 + pos).to(torch.int32))
        width = max(len(r) for r in rows)
        indices = torch.full((B, 1, width), -1, dtype=torch.int32, device=dev)
        for b, r in enumerate(rows):
            indices[b, 0, : len(r)] = r
        topk_len = torch.tensor([len(r) for r in rows], dtype=torch.int32, device=dev)
        sinks = p.inputs.get("sinks")
        q = p.inputs["q"]

        def fn() -> torch.Tensor:
            out = m.flash_mla_with_kvcache_sm120(
                q=q.view(B, 1, g.num_q_heads, g.qk_dim),
                k_cache=kcache,
                indices=indices,
                topk_length=topk_len,
                attn_sink=sinks,
                head_dim_v=g.v_dim,
                softmax_scale=g.scale,
            )[0]
            return out.reshape(-1, g.num_q_heads, g.v_dim)

        return Runner(
            fn=fn,
            tol=Tol(5e-2, 5e-2, 0.99),
            meta={
                "kernel": "flash_mla_with_kvcache_sm120",
                "backend_env": getattr(m, "_sm120_default_backend", "?"),
                "kv": "fp8 nope + bf16 rope, page 256",
            },
        )
