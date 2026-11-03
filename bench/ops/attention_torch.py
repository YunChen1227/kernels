"""Pure-PyTorch attention: ``torch_ref`` (fp32 golden) and ``sdpa``."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from bench.ops.base import (
    Ctx,
    OpAdapter,
    Problem,
    Runner,
    gather_tokens,
    register,
    token_slots,
)
from bench.ops.problems import attention_reference

ALL_FEATURES = frozenset({"sliding_window", "sink", "softcap", "sparse"})


@register
class TorchRefAttention(OpAdapter):
    """The fp32 reference itself, timed as the baseline (like flashinfer-bench)."""

    backend = "torch_ref"
    ops = ("attn_decode", "attn_prefill")
    features = ALL_FEATURES
    needs_cuda = False
    dtypes = ("bfloat16", "float16", "float32")
    kv_dtypes = ("bfloat16", "float16", "float32", "fp8")
    is_reference = True

    def prepare(self, p: Problem, graph: bool) -> Runner:
        dt = p.ctx.dtype
        return Runner(
            fn=lambda: attention_reference(p).to(dt),
            graph_safe=True,
            meta={"impl": "fp32 einsum per request"},
        )


@register
class SdpaAttention(OpAdapter):
    """torch.nn.functional.scaled_dot_product_attention over gathered KV.

    Uniform batches gather once into [B, S, H, D] and make one SDPA call;
    varlen batches loop per request. The KV gather is part of the timed path,
    which is what a paged cache costs a backend without native paging.
    """

    backend = "sdpa"
    ops = ("attn_decode", "attn_prefill")
    features = frozenset({"sliding_window", "sparse"})
    needs_cuda = False
    dtypes = ("bfloat16", "float16", "float32")
    kv_dtypes = ("bfloat16", "float16", "float32", "fp8")

    def prepare(self, p: Problem, graph: bool) -> Runner:
        g = p.geom
        seq_lens, q_lens = p.meta["seq_lens"], p.meta["q_lens"]
        B = len(seq_lens)
        dev = p.ctx.device
        uniform = len(set(seq_lens)) == 1 and len(set(q_lens)) == 1
        sel = p.inputs.get("sel_pos")
        # Static per-request slot lists (host work, untimed).
        slots = []
        for b in range(B):
            s = token_slots(p, b)
            slots.append(s[sel[b].long()] if sel is not None else s)
        k_len = [len(s) for s in slots]
        slot_mat = torch.stack(slots) if uniform else None

        def mask_for(b: int) -> Optional[torch.Tensor]:
            s, ql = seq_lens[b], q_lens[b]
            k_pos = sel[b].long() if sel is not None else torch.arange(s, device=dev)
            q_pos = torch.arange(s - ql, s, device=dev).view(-1, 1)
            m = k_pos.view(1, -1) <= q_pos
            if g.window is not None:
                m &= (q_pos - k_pos.view(1, -1)) < g.window
            if bool(m.all()):
                return None
            return m

        masks = [mask_for(b) for b in range(B)]
        mask_mat = None
        if uniform and any(m is not None for m in masks):
            full = torch.ones((q_lens[0], k_len[0]), dtype=torch.bool, device=dev)
            mask_mat = torch.stack([full if m is None else m for m in masks]).unsqueeze(1)
        q_all = p.inputs["q"]
        hq, hkv = g.num_q_heads, g.num_kv_heads

        def kv_for(slot_idx: torch.Tensor):
            if g.kind == "mla":
                kv = gather_tokens(p.inputs["kv_cache"], slot_idx.reshape(-1))
                k = kv.view(*slot_idx.shape, 1, g.qk_dim)
                v = k[..., : g.v_dim]
            else:
                k = gather_tokens(p.inputs["k_cache"], slot_idx.reshape(-1)).view(
                    *slot_idx.shape, hkv, g.qk_dim
                )
                v = gather_tokens(p.inputs["v_cache"], slot_idx.reshape(-1)).view(
                    *slot_idx.shape, hkv, g.v_dim
                )
            return k, v

        def sdpa(q, k, v, mask):
            # q [.., Hq, L, D], k/v [.., Hkv, S, D]
            if hkv != hq:
                k = k.repeat_interleave(hq // hkv, dim=-3)
                v = v.repeat_interleave(hq // hkv, dim=-3)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=g.scale)

        if uniform:
            ql = q_lens[0]

            def fn() -> torch.Tensor:
                k, v = kv_for(slot_mat)  # [B, S, H, D]
                q = q_all.view(B, ql, hq, g.qk_dim).transpose(1, 2)
                o = sdpa(q, k.transpose(1, 2), v.transpose(1, 2), mask_mat)
                return o.transpose(1, 2).reshape(B * ql, hq, g.v_dim)

        else:
            offs = [0]
            for ql in q_lens:
                offs.append(offs[-1] + ql)

            def fn() -> torch.Tensor:
                outs = []
                for b in range(B):
                    k, v = kv_for(slots[b])
                    q = q_all[offs[b] : offs[b + 1]].transpose(0, 1)
                    m = masks[b]
                    o = sdpa(q, k.transpose(0, 1), v.transpose(0, 1), m)
                    outs.append(o.transpose(0, 1))
                return torch.cat(outs)

        return Runner(fn=fn, meta={"uniform": uniform, "keys_per_req": max(k_len)})
