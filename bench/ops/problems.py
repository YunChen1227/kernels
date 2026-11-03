"""Build canonical inputs + fp32 reference outputs for every op.

Canonical attention layout (adapters convert at prepare time):

* GQA: ``k_cache [P, page, Hkv, Dqk]``, ``v_cache [P, page, Hkv, Dv]``
* MLA: ``kv_cache [P, page, kv_lora + rope]`` (V = first kv_lora dims)
* pages are allocated in a shuffled order, like a real allocator after churn
* ``q [total_q, Hq, Dqk]`` packed by request (``qo_indptr``)
* sparse variants carry ``sel_pos [B, K]`` sorted key positions per request
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List

import torch

from bench.ops.base import (
    Ctx,
    Problem,
    attn_geom,
    gather_tokens,
    kv_is_fp8,
    token_slots,
)
from bench.workloads import num_experts, qkv_out_dim


def _gen(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def _randn(shape, gen, device, dtype, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(shape, generator=gen, device=device, dtype=torch.float32) * scale).to(dtype)


def _seq_lens(ctx: Ctx, gen: torch.Generator) -> List[int]:
    wl = ctx.wl
    if not wl.varlen or wl.batch_size == 1:
        return [wl.kv_len] * wl.batch_size
    lo = max(wl.q_len, wl.kv_len // 2)
    r = torch.randint(lo, wl.kv_len + 1, (wl.batch_size,), generator=gen, device=ctx.device)
    out = [int(x) for x in r.tolist()]
    out[-1] = wl.kv_len
    return out


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def build_attention(ctx: Ctx, seed: int = 0) -> Problem:
    phase = ctx.wl.phase
    g = ctx.geom or attn_geom(ctx.model, ctx.variant, phase)
    ctx.geom = g
    dev, dt = ctx.device, ctx.dtype
    gen = _gen(seed, dev)
    seq_lens = _seq_lens(ctx, gen)
    B = len(seq_lens)
    ps = ctx.wl.page_size
    q_lens = list(seq_lens) if phase == "prefill" else [ctx.wl.q_len] * B
    pages_per = [(s + ps - 1) // ps for s in seq_lens]
    total_pages = sum(pages_per)
    perm = torch.randperm(total_pages, generator=gen, device=dev).to(torch.int32)
    page_table = torch.zeros((B, max(pages_per)), dtype=torch.int32, device=dev)
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=dev)
    off = 0
    for b, n in enumerate(pages_per):
        page_table[b, :n] = perm[off : off + n]
        off += n
        kv_indptr[b + 1] = off
    kv_indices = perm.clone()
    last_page_len = torch.tensor(
        [s - (n - 1) * ps for s, n in zip(seq_lens, pages_per)], dtype=torch.int32, device=dev
    )
    qo_indptr = torch.zeros(B + 1, dtype=torch.int32, device=dev)
    qo_indptr[1:] = torch.cumsum(torch.tensor(q_lens, dtype=torch.int32, device=dev), 0)
    total_q = sum(q_lens)

    inputs: Dict[str, torch.Tensor] = {
        "q": _randn((total_q, g.num_q_heads, g.qk_dim), gen, dev, dt),
        "page_table": page_table,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "last_page_len": last_page_len,
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int32, device=dev),
        "qo_indptr": qo_indptr,
    }

    def cache(shape):
        c = _randn(shape, gen, dev, dt)
        if kv_is_fp8(ctx.wl):
            # Round through fp8 so the bf16 reference matches what fp8 kernels read.
            c = c.to(torch.float8_e4m3fn).to(dt)
        return c

    if g.kind == "mla":
        inputs["kv_cache"] = cache((total_pages, ps, g.qk_dim))
    else:
        inputs["k_cache"] = cache((total_pages, ps, g.num_kv_heads, g.qk_dim))
        inputs["v_cache"] = cache((total_pages, ps, g.num_kv_heads, g.v_dim))
    if g.sink:
        inputs["sinks"] = _randn((g.num_q_heads,), gen, dev, torch.float32)
    if g.sparse_topk is not None:
        k = min(g.sparse_topk, min(seq_lens))
        sel = []
        for s in seq_lens:
            # Always keep the newest token (the query's own position), like real indexers.
            rest = torch.randperm(s - 1, generator=gen, device=dev)[: k - 1]
            pos = torch.cat([rest, torch.tensor([s - 1], device=dev)]).sort().values
            sel.append(pos.to(torch.int32))
        inputs["sel_pos"] = torch.stack(sel)

    meta = dict(
        seq_lens=seq_lens,
        q_lens=q_lens,
        page_size=ps,
        total_pages=total_pages,
        max_seq_len=max(seq_lens),
        max_q_len=max(q_lens),
    )
    return Problem(ctx=ctx, inputs=inputs, meta=meta, reference=attention_reference)


def attention_reference(p: Problem) -> torch.Tensor:
    """Exact fp32 attention over the canonical paged cache (chunked over queries)."""
    g = p.geom
    seq_lens: List[int] = p.meta["seq_lens"]
    q_lens: List[int] = p.meta["q_lens"]
    q_all = p.inputs["q"]
    dev = q_all.device
    out = torch.empty((q_all.shape[0], g.num_q_heads, g.v_dim), dtype=torch.float32, device=dev)
    sinks = p.inputs.get("sinks")
    qo = 0
    for b, (s, ql) in enumerate(zip(seq_lens, q_lens)):
        slots = token_slots(p, b)
        if g.kind == "mla":
            kv = gather_tokens(p.inputs["kv_cache"], slots).float()
            k = kv.unsqueeze(1)
            v = kv[:, : g.v_dim].unsqueeze(1)
        else:
            k = gather_tokens(p.inputs["k_cache"], slots).float()
            v = gather_tokens(p.inputs["v_cache"], slots).float()
        k_pos = torch.arange(s, device=dev)
        if "sel_pos" in p.inputs:
            k_pos = p.inputs["sel_pos"][b].long()
            k, v = k[k_pos], v[k_pos]
        q_b = q_all[qo : qo + ql].float()
        q_pos = torch.arange(s - ql, s, device=dev)
        hkv, grp = g.num_kv_heads, g.group
        chunk = max(1, (1 << 27) // max(1, g.num_q_heads * k.shape[0]))
        for c0 in range(0, ql, chunk):
            c1 = min(ql, c0 + chunk)
            qc = q_b[c0:c1].view(c1 - c0, hkv, grp, g.qk_dim)
            scores = torch.einsum("chgd,shd->chgs", qc, k) * g.scale
            if g.softcap is not None:
                scores = g.softcap * torch.tanh(scores / g.softcap)
            qp = q_pos[c0:c1].view(-1, 1)
            mask = k_pos.view(1, -1) <= qp
            if g.window is not None:
                mask &= (qp - k_pos.view(1, -1)) < g.window
            scores = scores.masked_fill(~mask.view(c1 - c0, 1, 1, -1), float("-inf"))
            if sinks is not None:
                sk = sinks.view(1, hkv, grp, 1).expand(c1 - c0, hkv, grp, 1)
                probs = torch.softmax(torch.cat([scores, sk], dim=-1), dim=-1)[..., :-1]
            else:
                probs = torch.softmax(scores, dim=-1)
            o = torch.einsum("chgs,shd->chgd", probs, v)
            out[qo + c0 : qo + c1] = o.reshape(c1 - c0, g.num_q_heads, g.v_dim)
        qo += ql
    return out


# ---------------------------------------------------------------------------
# Dense ops
# ---------------------------------------------------------------------------


def _tokens(ctx: Ctx) -> int:
    wl = ctx.wl
    return wl.batch_size * (wl.input_len if wl.phase == "prefill" else wl.q_len)


def build_rmsnorm(ctx: Ctx, seed: int = 0) -> Problem:
    gen = _gen(seed, ctx.device)
    t, h = _tokens(ctx), ctx.model.hidden_size
    inputs = {
        "x": _randn((t, h), gen, ctx.device, ctx.dtype),
        "w": _randn((h,), gen, ctx.device, ctx.dtype, 0.1) + 1.0,
    }

    def ref(p: Problem) -> torch.Tensor:
        x = p.inputs["x"].float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * p.inputs["w"].float()

    return Problem(ctx, inputs, {"eps": 1e-6}, ref)


def rope_cos_sin_cache(max_pos: int, rot: int, theta: float, device) -> torch.Tensor:
    """[max_pos, rot] fp32: first half cos, second half sin (FlashInfer/sglang layout)."""
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2, device=device, dtype=torch.float32) / rot))
    f = torch.outer(torch.arange(max_pos, device=device, dtype=torch.float32), inv)
    return torch.cat([f.cos(), f.sin()], dim=-1)


def rope_neox_ref(x: torch.Tensor, cos_sin: torch.Tensor, rot: int) -> torch.Tensor:
    """x [T, H, D] fp32, cos_sin [T, rot]; rotate first ``rot`` dims, NeoX style."""
    half = rot // 2
    cos, sin = cos_sin[:, None, :half], cos_sin[:, None, half:]
    x1, x2 = x[..., :half], x[..., half:rot]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., rot:]], dim=-1)


def build_rope(ctx: Ctx, seed: int = 0) -> Problem:
    m = ctx.model
    gen = _gen(seed, ctx.device)
    t = _tokens(ctx)
    if m.is_mla:
        hq, hk, d, rot = m.num_q_heads, 1, m.qk_rope_dim, m.qk_rope_dim
    else:
        hq, hk, d, rot = m.num_q_heads, m.num_kv_heads, m.head_dim, m.rope_dim
    max_pos = max(ctx.wl.kv_len, ctx.wl.input_len, 1) + 8
    inputs = {
        "x": _randn((t, hq, d), gen, ctx.device, ctx.dtype),  # q
        "k": _randn((t, hk, d), gen, ctx.device, ctx.dtype),
        "positions": torch.randint(0, max_pos, (t,), generator=gen, device=ctx.device),
        "cos_sin": rope_cos_sin_cache(max_pos, rot, 10000.0, ctx.device),
    }

    def ref(p: Problem) -> torch.Tensor:
        cs = p.inputs["cos_sin"][p.inputs["positions"]]
        q = rope_neox_ref(p.inputs["x"].float(), cs, rot)
        k = rope_neox_ref(p.inputs["k"].float(), cs, rot)
        return torch.cat([q.flatten(1), k.flatten(1)], dim=1)

    return Problem(ctx, inputs, {"rot": rot, "head_dim": d}, ref)


def _linear_problem(ctx: Ctx, k: int, n: int, seed: int) -> Problem:
    gen = _gen(seed, ctx.device)
    t = _tokens(ctx)
    inputs = {
        "x": _randn((t, k), gen, ctx.device, ctx.dtype),
        "w": _randn((n, k), gen, ctx.device, ctx.dtype, 1.0 / math.sqrt(k)),
    }

    def ref(p: Problem) -> torch.Tensor:
        return p.inputs["x"].float() @ p.inputs["w"].float().t()

    return Problem(ctx, inputs, {"m": t, "k": k, "n": n}, ref)


def build_gemm_qkv(ctx: Ctx, seed: int = 0) -> Problem:
    return _linear_problem(ctx, ctx.model.hidden_size, qkv_out_dim(ctx.model), seed)


def build_gemm_o(ctx: Ctx, seed: int = 0) -> Problem:
    m = ctx.model
    return _linear_problem(ctx, m.num_q_heads * m.vo_head_dim, m.hidden_size, seed)


def build_mlp(ctx: Ctx, seed: int = 0) -> Problem:
    m = ctx.model
    gen = _gen(seed, ctx.device)
    t, h, i = _tokens(ctx), m.hidden_size, m.intermediate_size
    inputs = {
        "x": _randn((t, h), gen, ctx.device, ctx.dtype),
        "w13": _randn((2 * i, h), gen, ctx.device, ctx.dtype, 1.0 / math.sqrt(h)),
        "w2": _randn((h, i), gen, ctx.device, ctx.dtype, 1.0 / math.sqrt(i)),
    }

    def ref(p: Problem) -> torch.Tensor:
        gu = p.inputs["x"].float() @ p.inputs["w13"].float().t()
        a = torch.nn.functional.silu(gu[:, :i]) * gu[:, i:]
        # implementations store the activation in the model dtype between the two GEMMs
        a = a.to(ctx.dtype).float()
        return a @ p.inputs["w2"].float().t()

    return Problem(ctx, inputs, {"intermediate": i}, ref)


def build_moe(ctx: Ctx, seed: int = 0) -> Problem:
    m = ctx.model
    assert m.moe is not None
    gen = _gen(seed, ctx.device)
    t, h, i = _tokens(ctx), m.hidden_size, m.moe.intermediate_size
    e = num_experts(m, ctx.options.get("moe_expert_cap", 0))
    topk = min(m.moe.topk, e)
    logits = torch.randn((t, e), generator=gen, device=ctx.device)
    w, ids = torch.softmax(logits, -1).topk(topk, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    inputs = {
        "x": _randn((t, h), gen, ctx.device, ctx.dtype),
        "w13": _randn((e, 2 * i, h), gen, ctx.device, ctx.dtype, 1.0 / math.sqrt(h)),
        "w2": _randn((e, h, i), gen, ctx.device, ctx.dtype, 1.0 / math.sqrt(i)),
        "router_logits": logits,
        "topk_ids": ids.to(torch.int32),
        "topk_weights": w.float(),
    }

    def ref(p: Problem) -> torch.Tensor:
        x = p.inputs["x"].float()
        out = torch.zeros_like(x)
        ids_, wts = p.inputs["topk_ids"].long(), p.inputs["topk_weights"]
        for ex in ids_.unique().tolist():
            tok, slot = (ids_ == ex).nonzero(as_tuple=True)
            gu = x[tok] @ p.inputs["w13"][ex].float().t()
            a = (torch.nn.functional.silu(gu[:, :i]) * gu[:, i:]).to(ctx.dtype).float()
            out.index_add_(0, tok, (a @ p.inputs["w2"][ex].float().t()) * wts[tok, slot, None])
        return out

    return Problem(ctx, inputs, {"num_experts": e, "topk": topk, "intermediate": i}, ref)


BUILDERS: Dict[str, Callable[[Ctx, int], Problem]] = {
    "attn_decode": build_attention,
    "attn_prefill": build_attention,
    "rmsnorm": build_rmsnorm,
    "rope": build_rope,
    "gemm_qkv": build_gemm_qkv,
    "gemm_o": build_gemm_o,
    "mlp": build_mlp,
    "moe": build_moe,
}


def build(ctx: Ctx, seed: int = 0) -> Problem:
    if ctx.op in ("attn_decode", "attn_prefill") and ctx.geom is None:
        ctx.geom = attn_geom(ctx.model, ctx.variant, ctx.wl.phase)
    return BUILDERS[ctx.op](ctx, seed)
