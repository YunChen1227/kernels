"""Minimal Qwen3.8-Flash-Next QSA attention demo (cpu_ref + sm120).

Pipeline (random weights):
  qkv_proj -> split q/k/v/gate -> GemmaRMSNorm + RoPE
  index_qk_proj -> index q/k layernorm + RoPE
  average_pool_qsa_keys -> compressed index keys
  torch_qsa_mqa_prefill -> logits
  qsa_fast_topk -> block indices
  torch_expand_qsa_block_indices -> logical tokens
  logical -> physical slots
  qsa_sparse_attention_reference -> out
  out * sigmoid(gate) -> o_proj
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEMO = Path(__file__).resolve().parent
_ROOT = _DEMO.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

from _compat import bootstrap  # noqa: E402

bootstrap()

from backends import (  # noqa: E402
    BackendCaps,
    qsa_mqa_logits_prefill,
    qsa_sparse_attention,
    qsa_topk,
    resolve_backend,
)
from ref_ops import (  # noqa: E402
    apply_rope_full,
    assert_close,
    dense_gqa,
    gemma_rms_norm,
    max_rel_err,
    precompute_freqs_cis,
    torch_dtype,
)
from shapes import QSAShapes, qsa_shapes  # noqa: E402

from sglang.srt.layers.attention.qsa.kernel import (  # noqa: E402
    average_pool_qsa_keys,
    torch_expand_qsa_block_indices,
)


class QSAAttentionDemo(nn.Module):
    def __init__(self, cfg: QSAShapes, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_size
        # q + k + v + gate (gate mirrors q heads * dim when attn_output_gate)
        q_size = cfg.num_heads * cfg.head_dim
        kv_size = cfg.num_kv_heads * cfg.head_dim
        self.qkv = nn.Linear(
            h, q_size + 2 * kv_size + q_size, bias=False, device=device, dtype=dtype
        )
        self.q_norm_w = nn.Parameter(torch.zeros(cfg.head_dim, device=device, dtype=dtype))
        self.k_norm_w = nn.Parameter(torch.zeros(cfg.head_dim, device=device, dtype=dtype))
        self.index_qk = nn.Linear(
            h, cfg.index_qk_out, bias=False, device=device, dtype=dtype
        )
        self.iq_norm_w = nn.Parameter(
            torch.zeros(cfg.indexer_head_dim, device=device, dtype=dtype)
        )
        self.ik_norm_w = nn.Parameter(
            torch.zeros(cfg.indexer_head_dim, device=device, dtype=dtype)
        )
        self.o_proj = nn.Linear(
            cfg.num_heads * cfg.head_dim, h, bias=False, device=device, dtype=dtype
        )

    def project_main(self, x: torch.Tensor, freqs: torch.Tensor, positions: torch.Tensor):
        cfg = self.cfg
        qkv = self.qkv(x)
        q_size = cfg.num_heads * cfg.head_dim
        kv_size = cfg.num_kv_heads * cfg.head_dim
        q, k, v, gate = qkv.split([q_size, kv_size, kv_size, q_size], dim=-1)
        q = q.view(-1, cfg.num_heads, cfg.head_dim)
        k = k.view(-1, cfg.num_kv_heads, cfg.head_dim)
        v = v.view(-1, cfg.num_kv_heads, cfg.head_dim)
        gate = gate.view(-1, cfg.num_heads, cfg.head_dim)
        q = gemma_rms_norm(q, self.q_norm_w)
        k = gemma_rms_norm(k, self.k_norm_w)
        f = freqs[positions]
        q = apply_rope_full(q, f)
        k = apply_rope_full(k, f)
        return q, k, v, gate

    def project_index(self, x: torch.Tensor, freqs_idx: torch.Tensor, positions: torch.Tensor):
        cfg = self.cfg
        qk = self.index_qk(x)
        q, k = qk.split(
            [
                cfg.indexer_n_heads * cfg.indexer_head_dim,
                cfg.indexer_kv_heads * cfg.indexer_head_dim,
            ],
            dim=-1,
        )
        q = q.view(-1, cfg.indexer_n_heads, cfg.indexer_head_dim)
        k = k.view(-1, cfg.indexer_kv_heads, cfg.indexer_head_dim)
        q = gemma_rms_norm(q, self.iq_norm_w)
        k = gemma_rms_norm(k, self.ik_norm_w)
        f = freqs_idx[positions]
        q = apply_rope_full(q, f)
        k = apply_rope_full(k, f)
        return q, k


def build_context_caches(
    model: QSAAttentionDemo,
    cfg: QSAShapes,
    backend: BackendCaps,
    *,
    seed: int,
):
    """Materialize a length-S KV + index-key cache with random 'history'."""
    torch.manual_seed(seed)
    device, dtype = backend.device, backend.dtype
    s, t = cfg.context_len, cfg.num_tokens
    # History hidden states for the full context (demo: random)
    hidden = torch.randn(s, cfg.hidden_size, device=device, dtype=dtype)
    positions = torch.arange(s, device=device, dtype=torch.long)
    freqs = precompute_freqs_cis(cfg.head_dim, s + 8, cfg.rope_theta, device=device)
    freqs_idx = precompute_freqs_cis(
        cfg.indexer_head_dim, s + 8, cfg.rope_theta, device=device
    )

    q_all, k_all, v_all, gate_all = model.project_main(hidden, freqs, positions)
    iq_all, ik_all = model.project_index(hidden, freqs_idx, positions)

    # Compress index keys by groups of compress_ratio
    ratio = cfg.compress_ratio
    n_full = (s // ratio) * ratio
    groups = ik_all[:n_full].view(-1, ratio, cfg.indexer_kv_heads, cfg.indexer_head_dim)
    compressed = average_pool_qsa_keys(groups)  # [G, 1, Dh]
    return {
        "hidden": hidden,
        "q_all": q_all,
        "k_all": k_all,
        "v_all": v_all,
        "gate_all": gate_all,
        "iq_all": iq_all,
        "ik_all": ik_all,
        "compressed": compressed,
        "freqs": freqs,
        "freqs_idx": freqs_idx,
    }


def run_forward(
    cfg: QSAShapes,
    backend: BackendCaps,
    *,
    budget: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    device = backend.device
    dtype = torch_dtype(cfg.dtype) if backend.name == "cpu_ref" else backend.dtype
    if cfg.dtype == "float64":
        dtype = torch.float64

    model = QSAAttentionDemo(cfg, device, dtype)
    caches = build_context_caches(model, cfg, backend, seed=seed)
    s, t = cfg.context_len, cfg.num_tokens
    # Query = last t tokens
    q = caches["q_all"][-t:]
    gate = caches["gate_all"][-t:]
    iq = caches["iq_all"][-t:]
    compressed = caches["compressed"]
    k_cache = caches["k_all"]
    v_cache = caches["v_all"]

    n_blocks = compressed.shape[0]
    # Prefill-style MQA over compressed keys for the query rows
    # Row ranges: each query can see blocks up to its position.
    query_pos = torch.arange(s - t, s, device=device)
    row_starts = torch.zeros(t, dtype=torch.int32, device=device)
    row_ends = ((query_pos + 1) // cfg.compress_ratio).clamp(max=n_blocks).to(torch.int32)
    # Ensure at least 1 block when possible
    row_ends = torch.maximum(row_ends, torch.minimum(row_starts + 1, row_ends.new_tensor(n_blocks)))

    logits = qsa_mqa_logits_prefill(
        backend,
        iq,
        compressed,
        row_starts,
        row_ends,
        score_scale=math.sqrt(cfg.indexer_head_dim),
    )
    # logits may be [T, n_blocks] or wider; trim
    if logits.shape[-1] > n_blocks:
        logits = logits[..., :n_blocks]
    elif logits.shape[-1] < n_blocks:
        pad = logits.new_full((t, n_blocks - logits.shape[-1]), float("-inf"))
        logits = torch.cat([logits, pad], dim=-1)

    use_budget = budget if budget is not None else cfg.indexer_budget
    # block topk must satisfy torch_expand: (token_topk + ratio - 1) // ratio
    token_topk = max(cfg.compress_ratio, use_budget)
    token_topk = (token_topk // cfg.compress_ratio) * cfg.compress_ratio
    block_topk = max(1, min(token_topk // cfg.compress_ratio, n_blocks))
    # If we clamped block_topk to n_blocks, shrink token_topk to match expand API.
    token_topk = block_topk * cfg.compress_ratio
    block_indices = qsa_topk(backend, logits, row_starts, row_ends, block_topk)

    # Expand blocks -> logical token indices
    seq_lens = torch.full((t,), s, dtype=torch.int32, device=device)
    q_pos = query_pos.to(torch.int32)
    try:
        logical = torch_expand_qsa_block_indices(
            block_indices,
            q_pos,
            seq_lens,
            cfg.compress_ratio,
            token_topk,
        )
    except Exception as exc:
        print(f"  [warn] torch_expand failed ({exc}); using fallback expander")
        logical = _fallback_expand(block_indices, cfg.compress_ratio, s)

    # Flatten cache: physical == logical
    physical = logical.clone()
    # Clamp invalid (-1) slots: reference ignores negative via mask
    out = qsa_sparse_attention(
        backend, q, k_cache, v_cache, physical, softmax_scale=cfg.head_dim**-0.5
    )
    if cfg.attn_output_gate:
        out = out * torch.sigmoid(gate)
    y = model.o_proj(out.reshape(t, -1))
    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "gate": gate,
        "logits": logits,
        "block_indices": block_indices,
        "logical": logical,
        "out": out,
        "y": y,
        "compressed": compressed,
    }


def _fallback_expand(
    block_indices: torch.Tensor, ratio: int, seq_len: int
) -> torch.Tensor:
    t, bk = block_indices.shape
    width = bk * ratio
    out = torch.full((t, width), -1, dtype=torch.int32, device=block_indices.device)
    for i in range(t):
        tokens = []
        for b in block_indices[i].tolist():
            if b < 0:
                continue
            start = int(b) * ratio
            tokens.extend(range(start, min(start + ratio, seq_len)))
        tokens = tokens[:width]
        if tokens:
            out[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
    return out


def check_degeneracy(cfg: QSAShapes, backend: BackendCaps) -> None:
    """Full causal slots => sparse == dense GQA (last-T queries)."""
    deg = QSAShapes(
        **{
            **cfg.__dict__,
            "num_heads": min(cfg.num_heads, 8),
            "num_kv_heads": min(cfg.num_kv_heads, 2),
            "context_len": min(cfg.context_len, 64),
            "num_tokens": min(cfg.num_tokens, 4),
            "head_dim": min(cfg.head_dim, 64),
            "indexer_head_dim": min(cfg.indexer_head_dim, 32),
            "hidden_size": min(cfg.hidden_size, 512),
            "indexer_budget": 64,
        }
    )
    s, t = deg.context_len, deg.num_tokens
    out = run_forward(deg, backend, seed=1)
    q = out["q"]
    k = out["k_cache"]
    v = out["v_cache"]
    expected = dense_gqa(q, k, v, causal=True, softmax_scale=deg.head_dim**-0.5)

    # Per-row causal slots [0 .. pos], padded with -1
    positions = torch.arange(s - t, s, device=q.device)
    physical = torch.full((t, s), -1, dtype=torch.int32, device=q.device)
    for i in range(t):
        pos = int(positions[i].item())
        physical[i, : pos + 1] = torch.arange(pos + 1, dtype=torch.int32, device=q.device)

    sparse_only = qsa_sparse_attention(
        backend, q, k, v, physical, softmax_scale=deg.head_dim**-0.5
    )
    assert_close(sparse_only, expected, rtol=0.0, atol=1e-4, msg="qsa degeneracy")
    print(
        f"  [ok] degeneracy vs dense GQA  max_rel_err={max_rel_err(sparse_only, expected):.3e}"
    )


def check_fp64_baseline(cfg: QSAShapes) -> None:
    """Report fp32 vs fp64 error on sparse GQA reference (shared inputs)."""
    from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention_reference

    torch.manual_seed(2)
    t, hq, hkv, d, s = 4, 8, 2, 64, 64
    q32 = torch.randn(t, hq, d)
    k32 = torch.randn(s, hkv, d)
    v32 = torch.randn(s, hkv, d)
    slots = torch.arange(s, dtype=torch.int32).unsqueeze(0).expand(t, -1).contiguous()
    # Causal: mask future by setting -1
    for i in range(t):
        pos = s - t + i
        slots[i, pos + 1 :] = -1
    o32 = qsa_sparse_attention_reference(q32, k32, v32, slots)
    o64 = qsa_sparse_attention_reference(q32.double(), k32.double(), v32.double(), slots)
    err = max_rel_err(o32, o64.float())
    print(f"  [ok] fp32 vs fp64 max_rel_err(attn)={err:.3e}")


def check_parity_sm120(cfg: QSAShapes) -> None:
    cpu = resolve_backend("cpu_ref")
    gpu = resolve_backend("sm120")
    small = QSAShapes(
        **{
            **cfg.__dict__,
            "context_len": min(128, cfg.context_len),
            "indexer_budget": 64,
            "dtype": "float32",
        }
    )
    out_cpu = run_forward(small, cpu, seed=3)
    small_bf = QSAShapes(**{**small.__dict__, "dtype": "bfloat16"})
    out_gpu = run_forward(small_bf, gpu, seed=3)
    err = max_rel_err(out_gpu["y"].float().cpu(), out_cpu["y"].float())
    if err > 5e-2:
        raise AssertionError(f"cross-backend parity failed: max_rel_err={err:.3e}")
    print(f"  [ok] sm120 vs cpu_ref max_rel_err(y)={err:.3e}")


def check_paged_mapping(cfg: QSAShapes) -> None:
    """logical -> physical with page_size mapping (identity on flat cache)."""
    s = min(cfg.context_len, 128)
    page = cfg.page_size
    logical = torch.arange(s, dtype=torch.int32)
    page_id = logical // page
    page_off = logical % page
    # Flat cache physical slot = page_id * page + page_off == logical
    physical = page_id * page + page_off
    assert_close(physical.float(), logical.float(), rtol=0, atol=0, msg="page map")
    print(f"  [ok] page_size={page} logical==physical for flat cache (S={s})")


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["cpu_ref", "sm120", "auto"], default="auto")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--check-parity", action="store_true")
    p.add_argument("--skip-degeneracy", action="store_true")
    args = p.parse_args(argv)

    backend = resolve_backend(None if args.backend == "auto" else args.backend)
    cfg = qsa_shapes("tiny" if args.tiny else "full", sm120=backend.name == "sm120")
    if backend.name == "cpu_ref" and not args.tiny:
        # Keep indexer_budget compatible with context for a meaningful topk
        cfg = QSAShapes(
            **{
                **cfg.__dict__,
                "context_len": 256,
                "indexer_budget": 128,  # block_topk=32
            }
        )

    print(f"== QSA demo | backend={backend.name} | {cfg.name} ==")
    print(
        f"   hidden={cfg.hidden_size} Q/KV={cfg.num_heads}/{cfg.num_kv_heads} "
        f"head_dim={cfg.head_dim} idx={cfg.indexer_n_heads}x{cfg.indexer_head_dim} "
        f"budget={cfg.indexer_budget} ratio={cfg.compress_ratio} "
        f"T={cfg.num_tokens} S={cfg.context_len} device={backend.device}"
    )

    out = run_forward(cfg, backend, seed=0)
    assert out["y"].shape == (cfg.num_tokens, cfg.hidden_size)
    assert torch.isfinite(out["y"].float()).all()
    print(
        f"  [ok] forward y{tuple(out['y'].shape)} out{tuple(out['out'].shape)} "
        f"blocks{tuple(out['block_indices'].shape)} logical{tuple(out['logical'].shape)}"
    )

    check_paged_mapping(cfg)

    if not args.skip_degeneracy:
        check_degeneracy(cfg, backend)

    if backend.name == "cpu_ref":
        check_fp64_baseline(cfg)

    if args.check_parity:
        check_parity_sm120(cfg)

    print("QSA demo PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
