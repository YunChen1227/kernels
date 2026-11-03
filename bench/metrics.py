"""Achieved throughput (TFLOP/s, GB/s) from problem sizes and measured latency."""

from __future__ import annotations

from typing import Dict

from bench.ops.base import Problem, decode_key_positions

_ELEM = {"bfloat16": 2, "float16": 2, "float32": 4, "fp8": 1}


def _causal_pairs(s: int, ql: int, window) -> int:
    """Number of (query, key) pairs for the last ``ql`` queries of a length-``s`` sequence."""
    first = s - ql
    if window is None:
        return (s * (s + 1) - first * (first + 1)) // 2
    return sum(min(p + 1, window) for p in range(first, s))


def compute(p: Problem, latency_ms: float) -> Dict[str, float]:
    if latency_ms <= 0:
        return {}
    op = p.ctx.op
    wl = p.ctx.wl
    e = _ELEM.get(wl.dtype, 2)
    kv_e = _ELEM.get(wl.kv_dtype, 2)
    flops = 0.0
    nbytes = 0.0
    if op in ("attn_decode", "attn_prefill", "decoder_layer"):
        g = p.geom
        pairs = 0
        keys_read = 0
        for b, (s, ql) in enumerate(zip(p.meta["seq_lens"], p.meta["q_lens"])):
            if wl.phase == "decode" and ql == 1:
                k = len(decode_key_positions(p, b))
                pairs += k
                keys_read += k
            else:
                pairs += _causal_pairs(s, ql, g.window)
                keys_read += s if g.window is None else min(s, g.window + ql)
        flops = 2.0 * g.num_q_heads * (g.qk_dim + g.v_dim) * pairs
        per_key = g.qk_dim if g.kind == "mla" else g.num_kv_heads * (g.qk_dim + g.v_dim)
        tq = p.inputs["q"].shape[0]
        nbytes = keys_read * per_key * kv_e + tq * g.num_q_heads * (g.qk_dim + g.v_dim) * e
    elif op in ("gemm_qkv", "gemm_o"):
        m, k, n = p.meta["m"], p.meta["k"], p.meta["n"]
        flops = 2.0 * m * n * k
        nbytes = (m * k + n * k + m * n) * e
    elif op == "mlp":
        t, h = p.inputs["x"].shape
        i = p.meta["intermediate"]
        flops = 2.0 * t * h * 3 * i
        nbytes = (3 * h * i + t * (2 * h + 3 * i)) * e
    elif op == "moe":
        t, h = p.inputs["x"].shape
        i, k = p.meta["intermediate"], p.meta["topk"]
        touched = int(p.inputs["topk_ids"].unique().numel())
        flops = 2.0 * t * k * h * 3 * i
        nbytes = touched * 3 * h * i * e + t * 2 * h * e
    elif op in ("rmsnorm", "rope"):
        nbytes = 2.0 * sum(v.numel() * v.element_size() for k, v in p.inputs.items() if k in ("x", "k"))
    s = latency_ms * 1e-3
    out = {}
    if flops:
        out["tflops"] = flops / s / 1e12
    if nbytes:
        out["gbps"] = nbytes / s / 1e9
    return out
