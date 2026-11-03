"""FlashInfer attention adapters: paged decode/prefill, ragged prefill, MLA, trtllm-gen.

CUDA graph mode builds wrappers with ``use_cuda_graph=True`` and fixed index
buffers, then calls ``plan()`` once at prepare time (outside capture), the
same split sglang's ``cuda_graph_runner`` uses.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from bench.ops.base import call_compat as _call
from bench.ops.base import (
    Ctx,
    OpAdapter,
    Problem,
    Runner,
    gather_tokens,
    kv_is_fp8,
    register,
    repage,
    token_index_lists,
    token_slots,
)

_WS: Dict[str, torch.Tensor] = {}


def workspace(device: torch.device, nbytes: int = 256 << 20, name: str = "fi") -> torch.Tensor:
    key = f"{name}:{device}:{nbytes}"
    if key not in _WS:
        _WS[key] = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    return _WS[key]


def _kv_dtype(p: Problem) -> torch.dtype:
    return torch.float8_e4m3fn if kv_is_fp8(p.ctx.wl) else p.ctx.dtype


def _maybe_fp8(t: torch.Tensor, p: Problem) -> torch.Tensor:
    return t.to(torch.float8_e4m3fn) if kv_is_fp8(p.ctx.wl) else t


@register
class FlashInferAttention(OpAdapter):
    backend = "flashinfer"
    ops = ("attn_decode", "attn_prefill")
    features = frozenset({"sliding_window", "softcap", "sparse"})
    requires = ("flashinfer",)
    sm_min = (8, 0)
    kv_dtypes = ("bfloat16", "float16", "fp8")

    def extra_check(self, ctx: Ctx) -> Optional[str]:
        g = ctx.geom
        if g is None:
            return None
        if g.kind == "mla":
            if g.v_dim != 512 or g.rope_dim not in (0, 64):
                return "FlashInfer MLA kernel needs kv_lora_rank 512 and rope 64"
            if g.window is not None or g.softcap is not None:
                return "FlashInfer MLA kernel has no window/softcap"
            return None
        if g.qk_dim not in (64, 128, 256):
            return f"FlashInfer paged kernels need head_dim in 64/128/256, got {g.qk_dim}"
        if g.sparse_topk is not None and ctx.wl.q_len != 1:
            return "sparse decode via page_size=1 indices needs q_len == 1"
        return None

    def prepare(self, p: Problem, graph: bool) -> Runner:
        if p.geom.kind == "mla":
            return self._mla(p, graph)
        if p.ctx.wl.phase == "decode" and p.ctx.wl.q_len == 1:
            return self._decode(p, graph)
        if p.geom.qk_dim != p.geom.v_dim:
            return self._ragged_prefill(p, graph)
        return self._paged_prefill(p, graph)

    # -- GQA decode ---------------------------------------------------------
    def _decode(self, p: Problem, graph: bool) -> Runner:
        import flashinfer

        g, dev, dt = p.geom, p.ctx.device, p.ctx.dtype
        k_cache, v_cache = p.inputs["k_cache"], p.inputs["v_cache"]
        page_size = p.meta["page_size"]
        if g.sparse_topk is not None:
            indptr, indices = token_index_lists(p)
            last = torch.ones(len(p.meta["seq_lens"]), dtype=torch.int32, device=dev)
            k_cache = k_cache.reshape(-1, 1, g.num_kv_heads, g.qk_dim)
            v_cache = v_cache.reshape(-1, 1, g.num_kv_heads, g.v_dim)
            page_size, window_left = 1, -1  # window already folded into indices
        else:
            indptr = p.inputs["kv_indptr"].clone()
            indices = p.inputs["kv_indices"].clone()
            last = p.inputs["last_page_len"].clone()
            window_left = g.window - 1 if g.window is not None else -1
        k_cache, v_cache = _maybe_fp8(k_cache, p), _maybe_fp8(v_cache, p)
        use_tc = g.group >= 4
        kw = dict(use_cuda_graph=graph, use_tensor_cores=use_tc)
        if graph:
            kw.update(
                paged_kv_indptr_buffer=indptr,
                paged_kv_indices_buffer=indices,
                paged_kv_last_page_len_buffer=last,
            )
        w = _call(flashinfer.BatchDecodeWithPagedKVCacheWrapper, workspace(dev), "NHD", **kw)
        _call(
            w.plan,
            indptr,
            indices,
            last,
            g.num_q_heads,
            g.num_kv_heads,
            g.qk_dim,
            page_size,
            window_left=window_left,
            logits_soft_cap=g.softcap,
            q_data_type=dt,
            kv_data_type=_kv_dtype(p),
            sm_scale=g.scale,
        )
        q = p.inputs["q"]
        return Runner(
            fn=lambda: w.run(q, (k_cache, v_cache)),
            meta={"wrapper": "BatchDecodeWithPagedKVCacheWrapper", "tensor_cores": use_tc,
                  "page_size": page_size},
        )

    # -- GQA paged prefill (also decode with q_len > 1) -----------------------
    def _paged_prefill(self, p: Problem, graph: bool) -> Runner:
        import flashinfer

        g, dev, dt = p.geom, p.ctx.device, p.ctx.dtype
        qo = p.inputs["qo_indptr"].clone()
        indptr = p.inputs["kv_indptr"].clone()
        indices = p.inputs["kv_indices"].clone()
        last = p.inputs["last_page_len"].clone()
        kw = dict(use_cuda_graph=graph)
        if graph:
            kw.update(
                qo_indptr_buf=qo,
                paged_kv_indptr_buf=indptr,
                paged_kv_indices_buf=indices,
                paged_kv_last_page_len_buf=last,
            )
        w = _call(flashinfer.BatchPrefillWithPagedKVCacheWrapper, workspace(dev), "NHD", **kw)
        _call(
            w.plan,
            qo,
            indptr,
            indices,
            last,
            g.num_q_heads,
            g.num_kv_heads,
            g.qk_dim,
            p.meta["page_size"],
            causal=True,
            window_left=g.window - 1 if g.window is not None else -1,
            logits_soft_cap=g.softcap,
            sm_scale=g.scale,
            q_data_type=dt,
            kv_data_type=_kv_dtype(p),
        )
        kc = _maybe_fp8(p.inputs["k_cache"], p)
        vc = _maybe_fp8(p.inputs["v_cache"], p)
        q = p.inputs["q"]
        return Runner(
            fn=lambda: w.run(q, (kc, vc)),
            meta={"wrapper": "BatchPrefillWithPagedKVCacheWrapper"},
        )

    # -- Ragged prefill (head_dim_qk != head_dim_vo, e.g. DeepSeek 192/128) ---
    def _ragged_prefill(self, p: Problem, graph: bool) -> Runner:
        import flashinfer

        g, dev, dt = p.geom, p.ctx.device, p.ctx.dtype
        B = len(p.meta["seq_lens"])
        slots = torch.cat([token_slots(p, b) for b in range(B)])
        k = gather_tokens(p.inputs["k_cache"], slots).contiguous()
        v = gather_tokens(p.inputs["v_cache"], slots).contiguous()
        kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=dev)
        kv_indptr[1:] = torch.cumsum(p.inputs["seq_lens"], 0)
        qo = p.inputs["qo_indptr"].clone()
        kw = dict(use_cuda_graph=graph)
        if graph:
            kw.update(qo_indptr_buf=qo, kv_indptr_buf=kv_indptr)
        w = _call(flashinfer.BatchPrefillWithRaggedKVCacheWrapper, workspace(dev), "NHD", **kw)
        _call(
            w.plan,
            qo,
            kv_indptr,
            g.num_q_heads,
            g.num_kv_heads,
            g.qk_dim,
            head_dim_vo=g.v_dim,
            causal=True,
            window_left=g.window - 1 if g.window is not None else -1,
            logits_soft_cap=g.softcap,
            sm_scale=g.scale,
            q_data_type=dt,
            kv_data_type=dt,
        )
        q = p.inputs["q"]
        return Runner(
            fn=lambda: w.run(q, k, v),
            meta={"wrapper": "BatchPrefillWithRaggedKVCacheWrapper", "kv": "ragged, gathered"},
        )

    # -- MLA ----------------------------------------------------------------
    def _mla(self, p: Problem, graph: bool) -> Runner:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        g, dev, dt = p.geom, p.ctx.device, p.ctx.dtype
        cache = p.inputs["kv_cache"]
        page_size = p.meta["page_size"]
        if g.sparse_topk is not None:
            kv_indptr, kv_indices = token_index_lists(p)
            kv_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
            cache = cache.reshape(-1, 1, g.qk_dim)
            page_size = 1
        else:
            kv_indptr = p.inputs["kv_indptr"].clone()
            kv_indices = p.inputs["kv_indices"].clone()
            kv_len = p.inputs["seq_lens"].clone()
        ckv = cache[..., : g.v_dim]
        if g.rope_dim:
            kpe = cache[..., g.v_dim :]
            q_pe_static = None
        else:
            # Single-latent MQA (DSV4): pad a zero rope part so q.k is unchanged.
            kpe = torch.zeros(cache.shape[:-1] + (64,), dtype=dt, device=dev)
            q_pe_static = torch.zeros(p.inputs["q"].shape[:-1] + (64,), dtype=dt, device=dev)
        qo = p.inputs["qo_indptr"].clone()
        kw = dict(use_cuda_graph=graph)
        if graph:
            kw.update(qo_indptr=qo, kv_indptr=kv_indptr, kv_indices=kv_indices, kv_len_arr=kv_len)
        w = _call(BatchMLAPagedAttentionWrapper, workspace(dev), **kw)
        w.plan(
            qo,
            kv_indptr,
            kv_indices,
            kv_len,
            g.num_q_heads,
            512,
            64,
            page_size,
            True,
            g.scale,
            dt,
            dt,
        )
        q = p.inputs["q"]

        def fn() -> torch.Tensor:
            q_nope = q[..., : g.v_dim]
            q_pe = q[..., g.v_dim :] if q_pe_static is None else q_pe_static
            return w.run(q_nope, q_pe, ckv, kpe)

        return Runner(
            fn=fn,
            meta={"wrapper": "BatchMLAPagedAttentionWrapper", "page_size": page_size},
        )


@register
class TrtllmGenAttention(OpAdapter):
    """FlashInfer trtllm-gen decode (Blackwell): GQA on SM100/SM120, MLA on SM100."""

    backend = "trtllm"
    ops = ("attn_decode",)
    phases = ("decode",)
    features = frozenset({"sliding_window", "sink"})
    requires = ("flashinfer",)
    kv_dtypes = ("bfloat16", "float16", "fp8")
    decode_q_len_1_only = True

    def extra_check(self, ctx: Ctx) -> Optional[str]:
        if ctx.sm is None or ctx.sm[0] not in (10, 12):
            return "trtllm-gen kernels need Blackwell (sm_100 / sm_120)"
        g = ctx.geom
        if g is not None and g.kind == "mla":
            if ctx.sm[0] != 10:
                return "trtllm-gen MLA decode needs sm_100"
            if g.v_dim != 512 or g.rope_dim != 64:
                return "trtllm-gen MLA needs kv_lora_rank 512 and rope 64"
            if g.window is not None or g.sink:
                return "trtllm-gen MLA path here has no window/sink"
        return None

    def prepare(self, p: Problem, graph: bool) -> Runner:
        import flashinfer.decode as fd

        g, dev = p.geom, p.ctx.device
        ws = workspace(dev, 512 << 20, name="trtllm")
        ps = 64 if p.meta["page_size"] not in (16, 32, 64) else p.meta["page_size"]
        q = p.inputs["q"]
        if g.kind == "mla":
            (cache,), table = repage(p, [p.inputs["kv_cache"]], 64)
            kv = cache.unsqueeze(1)  # [P, 1, page, 576]
            seq_lens = p.inputs["seq_lens"]
            max_len = p.meta["max_seq_len"]
            B = len(p.meta["seq_lens"])

            def fn_mla() -> torch.Tensor:
                out = _call(
                    fd.trtllm_batch_decode_with_kv_cache_mla,
                    query=q.view(B, 1, g.num_q_heads, g.qk_dim),
                    kv_cache=kv,
                    workspace_buffer=ws,
                    qk_nope_head_dim=p.ctx.model.qk_nope_dim,
                    kv_lora_rank=g.v_dim,
                    qk_rope_head_dim=g.rope_dim,
                    block_tables=table,
                    seq_lens=seq_lens,
                    max_seq_len=max_len,
                    bmm1_scale=g.scale,
                    bmm2_scale=1.0,
                )
                return out.view(-1, g.num_q_heads, g.v_dim)

            return Runner(fn=fn_mla, meta={"kernel": "trtllm_batch_decode_with_kv_cache_mla"})

        (kc, vc), table = repage(p, [p.inputs["k_cache"], p.inputs["v_cache"]], ps)
        # trtllm-gen wants HND pages: [P, Hkv, page, D]
        kc = _maybe_fp8(kc.transpose(1, 2).contiguous(), p)
        vc = _maybe_fp8(vc.transpose(1, 2).contiguous(), p)
        sinks = p.inputs.get("sinks")
        seq_lens = p.inputs["seq_lens"]
        max_len = p.meta["max_seq_len"]
        window_left = g.window - 1 if g.window is not None else -1

        def fn() -> torch.Tensor:
            return _call(
                fd.trtllm_batch_decode_with_kv_cache,
                query=q,
                kv_cache=(kc, vc),
                workspace_buffer=ws,
                block_tables=table,
                seq_lens=seq_lens,
                max_seq_len=max_len,
                bmm1_scale=g.scale,
                bmm2_scale=1.0,
                window_left=window_left,
                sinks=sinks,
                kv_layout="HND",
            )

        return Runner(fn=fn, meta={"kernel": "trtllm_batch_decode_with_kv_cache", "page_size": ps})
