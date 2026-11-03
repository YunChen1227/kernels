"""Decoder-layer (block) benchmark: N decode layers sharing one set of weights.

A single large kernel hides launch overhead; a decode layer is ~15-30 small
kernels, which is where CUDA graphs pay off. The attention step is the chosen
attention adapter's ``Runner``; the other sub-ops use the fastest available
implementation (overridable per sub-op) and are recorded in trace meta.

Simplifications (recorded in meta): random weights, the new token's K/V is
written to the canonical cache (backends that re-paged at prepare time read
their own copy), and correctness is a graph-vs-eager consistency check since
every sub-op is already validated at op level.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from bench.ops import dense
from bench.ops.base import Ctx, OpAdapter, Problem, Runner, module_available, slots_for_positions
from bench.ops.problems import build_attention, rope_cos_sin_cache
from bench.workloads import num_experts

SUB_IMPLS: Dict[str, Dict[str, Callable]] = {
    "rmsnorm": {
        "flashinfer": dense.rmsnorm_flashinfer,
        "sgl_kernel": dense.rmsnorm_sgl,
        "torch": dense.rmsnorm_torch,
    },
    "rope": {"flashinfer": dense.rope_flashinfer, "torch": dense.rope_torch},
    "act": {
        "flashinfer": dense.silu_mul_flashinfer,
        "sgl_kernel": dense.silu_mul_sgl,
        "torch": dense.silu_mul_torch,
    },
    "moe": {
        "sglang_triton": dense.moe_sglang_triton,
        "flashinfer_cutlass": None,  # needs swapped weights; wired below
        "torch": dense.moe_torch_loop,
    },
}

_REQUIRES = {"flashinfer": "flashinfer", "sgl_kernel": "sgl_kernel", "sglang_triton": "triton",
             "flashinfer_cutlass": "flashinfer"}


def pick_impls(device: torch.device, overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    overrides = overrides or {}
    out = {}
    for sub, impls in SUB_IMPLS.items():
        if sub in overrides:
            out[sub] = overrides[sub]
            continue
        choice = "torch"
        if device.type == "cuda":
            for name in impls:
                if name == "torch":
                    break
                if module_available(_REQUIRES.get(name, name)):
                    if sub == "moe" and name == "sglang_triton":
                        try:
                            dense._sglang_fused_experts()
                        except Exception:
                            continue
                    choice = name
                    break
        out[sub] = choice
    return out


def _w(shape, gen, device, dtype, fan_in: int) -> torch.Tensor:
    return (torch.randn(shape, generator=gen, device=device) / math.sqrt(fan_in)).to(dtype)


def build_block(
    ctx: Ctx, attn: OpAdapter, graph: bool, layers: int, impl_overrides: Optional[Dict[str, str]],
    seed: int = 0, attn_problem: Optional[Problem] = None,
) -> Tuple[Problem, Runner, Dict]:
    """Returns (attention problem, block runner, meta). ``attn_problem`` may be shared."""
    m = ctx.model
    dev, dt = ctx.device, ctx.dtype
    actx = Ctx(model=m, variant=ctx.variant, op="attn_decode", wl=ctx.wl, device=dev, sm=ctx.sm,
               geom=ctx.geom, options=ctx.options)
    p = attn_problem or build_attention(actx, seed)
    g = p.geom
    attn_runner = attn.prepare(p, graph)
    impls = pick_impls(dev, impl_overrides)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed + 1)
    h = m.hidden_size
    B = len(p.meta["seq_lens"])
    T = p.inputs["q"].shape[0]
    q_len = T // B
    seq_lens = p.meta["seq_lens"]
    # Positions / cache slots of the newest q_len tokens of every request.
    pos_list, slot_list = [], []
    for b in range(B):
        pos = torch.arange(seq_lens[b] - q_len, seq_lens[b], device=dev)
        pos_list.append(pos)
        slot_list.append(slots_for_positions(p, b, pos).long())
    positions = torch.cat(pos_list)
    new_slots = torch.cat(slot_list)
    rot = m.rope_dim
    cos_sin = rope_cos_sin_cache(max(seq_lens) + 1, rot, 10000.0, dev) if rot else None

    w = {
        "norm_in": torch.ones(h, device=dev, dtype=dt),
        "norm_post": torch.ones(h, device=dev, dtype=dt),
        "o": _w((h, m.num_q_heads * (m.v_head_dim or m.vo_head_dim)), gen, dev, dt,
                m.num_q_heads * m.vo_head_dim),
    }
    if m.is_mla:
        nope, rope_d, lora = m.qk_nope_dim, m.qk_rope_dim, m.kv_lora_rank
        v_head = m.v_head_dim or lora
        q_in = m.q_lora_rank or h
        if m.q_lora_rank:
            w["q_a"] = _w((m.q_lora_rank, h), gen, dev, dt, h)
            w["q_a_norm"] = torch.ones(m.q_lora_rank, device=dev, dtype=dt)
        w["q_b"] = _w((m.num_q_heads * (nope + rope_d), q_in), gen, dev, dt, q_in)
        w["kv_a"] = _w((lora + rope_d, h), gen, dev, dt, h)
        w["kv_norm"] = torch.ones(lora, device=dev, dtype=dt)
        w["w_uk"] = _w((m.num_q_heads, nope, lora), gen, dev, dt, nope)
        if m.v_head_dim:
            w["w_uv"] = _w((m.num_q_heads, lora, v_head), gen, dev, dt, lora)
            w["o"] = _w((h, m.num_q_heads * v_head), gen, dev, dt, m.num_q_heads * v_head)
        else:
            w["o"] = _w((h, m.num_q_heads * lora), gen, dev, dt, m.num_q_heads * lora)
    else:
        d = m.head_dim
        w["qkv"] = _w(((m.num_q_heads + 2 * m.num_kv_heads) * d, h), gen, dev, dt, h)
        w["o"] = _w((h, m.num_q_heads * d), gen, dev, dt, m.num_q_heads * d)
        if m.qk_norm:
            w["q_norm"] = torch.ones(d, device=dev, dtype=dt)
            w["k_norm"] = torch.ones(d, device=dev, dtype=dt)
    moe_fn = None
    if m.moe:
        e = num_experts(m, ctx.options.get("moe_expert_cap", 0))
        i = m.moe.intermediate_size
        w["router"] = _w((e, h), gen, dev, dt, h)
        w["w13"] = _w((e, 2 * i, h), gen, dev, dt, h)
        w["w2"] = _w((e, h, i), gen, dev, dt, i)
        topk = min(m.moe.topk, e)
        if impls["moe"] == "flashinfer_cutlass":
            w31 = dense.swap_w13(w["w13"])

            def moe_fn(x, ids, wts, logits):
                return dense.moe_flashinfer_cutlass(x, w31, w["w2"], ids, wts)

        else:
            base = SUB_IMPLS["moe"][impls["moe"]]

            def moe_fn(x, ids, wts, logits):
                return base(x, w["w13"], w["w2"], ids, wts, router_logits=logits)

        if m.moe.num_shared_experts:
            si = i * m.moe.num_shared_experts
            w["sh13"] = _w((2 * si, h), gen, dev, dt, h)
            w["sh2"] = _w((h, si), gen, dev, dt, si)
    else:
        i = m.intermediate_size
        w["w13"] = _w((2 * i, h), gen, dev, dt, h)
        w["w2"] = _w((h, i), gen, dev, dt, i)

    rms = SUB_IMPLS["rmsnorm"][impls["rmsnorm"]]
    rope = SUB_IMPLS["rope"][impls["rope"]]
    act = SUB_IMPLS["act"][impls["act"]]
    attn_q = p.inputs["q"]
    x_in = torch.randn((T, h), generator=gen, device=dev).to(dt)

    def ffn(x: torch.Tensor) -> torch.Tensor:
        if not m.moe:
            return F.linear(act(F.linear(x, w["w13"])), w["w2"])
        logits = F.linear(x, w["router"]).float()
        wts, ids = torch.softmax(logits, -1).topk(topk, dim=-1)
        wts = wts / wts.sum(-1, keepdim=True)
        y = moe_fn(x, ids.to(torch.int32), wts, logits)
        if "sh13" in w:
            y = y + F.linear(act(F.linear(x, w["sh13"])), w["sh2"])
        return y

    def attention_gqa(hn: torch.Tensor) -> torch.Tensor:
        d = m.head_dim
        qkv = F.linear(hn, w["qkv"])
        q, k, v = qkv.split([m.num_q_heads * d, m.num_kv_heads * d, m.num_kv_heads * d], -1)
        q = q.reshape(T, m.num_q_heads, d)
        k = k.reshape(T, m.num_kv_heads, d)
        v = v.reshape(T, m.num_kv_heads, d)
        if m.qk_norm:
            q = rms(q.reshape(-1, d), w["q_norm"]).view(T, m.num_q_heads, d)
            k = rms(k.reshape(-1, d), w["k_norm"]).view(T, m.num_kv_heads, d)
        q, k = rope(q, k, positions, cos_sin, rot)
        kc = p.inputs["k_cache"].view(-1, m.num_kv_heads, d)
        vc = p.inputs["v_cache"].view(-1, m.num_kv_heads, g.v_dim)
        kc.index_copy_(0, new_slots, k.to(kc.dtype))
        vc.index_copy_(0, new_slots, v.to(vc.dtype))
        attn_q.copy_(q)
        o = attn_runner.fn()
        return F.linear(o.reshape(T, -1), w["o"])

    def attention_mla(hn: torch.Tensor) -> torch.Tensor:
        nope, rope_d, lora = m.qk_nope_dim, m.qk_rope_dim, m.kv_lora_rank
        qa = hn
        if m.q_lora_rank:
            qa = rms(F.linear(hn, w["q_a"]), w["q_a_norm"])
        q = F.linear(qa, w["q_b"]).view(T, m.num_q_heads, nope + rope_d)
        q_nope, q_pe = q[..., :nope], q[..., nope:]
        q_lat = torch.bmm(q_nope.transpose(0, 1), w["w_uk"]).transpose(0, 1)
        kv = F.linear(hn, w["kv_a"])
        lat = rms(kv[:, :lora].contiguous(), w["kv_norm"])
        if rope_d:
            q_pe, k_pe = rope(q_pe.contiguous(), kv[:, None, lora:].contiguous(), positions,
                              cos_sin, rot)
            entry = torch.cat([lat, k_pe.reshape(T, rope_d)], -1)
        else:
            entry = lat
        p.inputs["kv_cache"].view(-1, g.qk_dim).index_copy_(0, new_slots, entry.to(dt))
        attn_q[..., :lora].copy_(q_lat)
        if rope_d:
            attn_q[..., lora:].copy_(q_pe)
        o = attn_runner.fn()  # [T, H, lora]
        if "w_uv" in w:
            o = torch.bmm(o.transpose(0, 1), w["w_uv"]).transpose(0, 1)
        return F.linear(o.reshape(T, -1), w["o"])

    attention = attention_mla if m.is_mla else attention_gqa

    def layer(x: torch.Tensor) -> torch.Tensor:
        x = x + attention(rms(x, w["norm_in"]))
        return x + ffn(rms(x, w["norm_post"]))

    def fn() -> torch.Tensor:
        x = x_in
        for _ in range(layers):
            x = layer(x)
        return x

    moe_loop = m.moe is not None and impls["moe"] == "torch"
    graph_safe = attn_runner.graph_safe and not moe_loop
    reason = attn_runner.graph_reason or ("torch MoE loop has data-dependent shapes" if moe_loop else "")
    meta = {
        "block_impl": impls if m.moe else {k: v for k, v in impls.items() if k != "moe"},
        "layers": layers,
        "attn": attn_runner.meta,
        "notes": "random weights; correctness = graph vs eager consistency",
    }
    if m.moe:
        meta["experts"] = num_experts(m, ctx.options.get("moe_expert_cap", 0))
    return p, Runner(fn=fn, graph_safe=graph_safe, graph_reason=reason, meta=meta), meta
