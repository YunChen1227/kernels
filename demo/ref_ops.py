"""Shared pure-torch operators for demos (RMSNorm, RoPE, dense baselines)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standard RMSNorm (DeepSeek-style, no +1)."""
    orig = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    x_f = x_f * torch.rsqrt(var + eps)
    return (x_f * weight.float()).to(orig)


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """GemmaRMSNorm: y = x / rms * (1 + weight)."""
    orig = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    x_f = x_f * torch.rsqrt(var + eps)
    return (x_f * (1.0 + weight.float())).to(orig)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Complex freqs [end, dim // 2] for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(end, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_tail(
    x: torch.Tensor, freqs: torch.Tensor, rope_dim: int, inverse: bool = False
) -> torch.Tensor:
    """Rotate the last ``rope_dim`` features of x with complex freqs [T, rope_dim // 2].

    Matches ``dsv41_sparse.rope_tail``.
    """
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    tc = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)).contiguous())
    f = freqs.conj() if inverse else freqs
    # Broadcast freqs over head dims: freqs is [T, rope_dim//2]
    while f.ndim < tc.ndim:
        f = f.unsqueeze(1)
    rotated = torch.view_as_real(tc * f).flatten(-2).to(x.dtype)
    return torch.cat([head, rotated], dim=-1)


def apply_rope_full(
    x: torch.Tensor, freqs: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply RoPE to the entire last dimension of x [..., D] with D even."""
    d = x.shape[-1]
    assert d % 2 == 0
    tc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)).contiguous())
    f = freqs.conj() if inverse else freqs
    while f.ndim < tc.ndim:
        f = f.unsqueeze(1)
    return torch.view_as_real(tc * f).flatten(-2).to(x.dtype)


def pool_pairs(kv2: torch.Tensor, score2: torch.Tensor) -> torch.Tensor:
    """V4.1 ratio-2 compressor: kv2, score2 [n, 2, D] -> [n, D]."""
    return (kv2 * score2.softmax(dim=1)).sum(dim=1)


def dense_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Dense GQA: q [T, Hq, D], k/v [S, Hkv, D] -> [T, Hq, D]."""
    t, hq, d = q.shape
    s, hkv, _ = k.shape
    assert hq % hkv == 0
    repeats = hq // hkv
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    k_r = k.repeat_interleave(repeats, dim=1)  # [S, Hq, D]
    v_r = v.repeat_interleave(repeats, dim=1)
    # scores [T, Hq, S]
    scores = torch.einsum("thd,shd->ths", q.float(), k_r.float()) * scale
    if causal:
        # Assume q aligns to the last T positions of a length-S cache.
        q_pos = torch.arange(s - t, s, device=q.device).view(t, 1, 1)
        k_pos = torch.arange(s, device=q.device).view(1, 1, s)
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("ths,shd->thd", probs, v_r.float())
    return out.to(q.dtype)


def dense_mla_causal(
    q: torch.Tensor,
    kv: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dense causal MLA with shared KV latent: q [T,H,D], kv [S,D] (MQA).

    Equivalent to treating kv as both K and V with a single KV head.
    """
    t, h, d = q.shape
    s = kv.shape[0]
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    # Expand kv to [S, H, D] by broadcast in einsum
    scores = torch.einsum("thd,sd->ths", q.float(), kv.float()) * scale
    q_pos = torch.arange(s - t, s, device=q.device).view(t, 1, 1)  # [T,1,1]
    k_pos = torch.arange(s, device=q.device).view(1, 1, s)  # [1,1,S]
    scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
    if attn_sink is not None:
        sink = attn_sink.view(1, h, 1).to(scores.dtype).expand(t, h, 1)
        scores = torch.cat([scores, sink], dim=-1)
        probs = torch.softmax(scores, dim=-1)[..., :-1]
    else:
        probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("ths,sd->thd", probs, kv.float())
    return out.to(q.dtype)


def sparse_mla_with_sink(
    q: torch.Tensor,
    keys: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One-query row attention over an explicit key set (already gathered).

    q [T,H,D], keys [T,K,D] (per-query selected keys; V = K for MLA).
    """
    t, h, d = q.shape
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    scores = torch.einsum("thd,tkd->thk", q.float(), keys.float()) * scale
    if attn_sink is not None:
        sink = attn_sink.view(1, h, 1).to(scores.dtype).expand(t, h, 1)
        scores = torch.cat([scores, sink], dim=-1)
        probs = torch.softmax(scores, dim=-1)[..., :-1]
    else:
        probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("thk,tkd->thd", probs, keys.float())
    return out.to(q.dtype)


def max_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f, b_f = a.float(), b.float()
    denom = b_f.abs().clamp_min(1e-8)
    return float(((a_f - b_f).abs() / denom).max().item())


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    msg: str = "",
) -> None:
    if not torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol):
        err = max_rel_err(actual, expected)
        raise AssertionError(
            f"{msg} tensors differ: max_rel_err={err:.3e}, rtol={rtol}, atol={atol}, "
            f"shapes={tuple(actual.shape)}/{tuple(expected.shape)}"
        )
