"""Canonical QKV shapes for DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class DSV4Shapes:
    name: str = "DeepSeek-V4.1-Flash"
    hidden_size: int = 4096
    num_heads: int = 64
    q_lora_rank: int = 1280
    qk_nope_head_dim: int = 448
    qk_rope_head_dim: int = 64
    v_head_dim: int = 512
    kv_lora_rank: int = 512  # MQA latent = head_dim
    o_lora_rank: int = 1024
    o_groups: int = 8
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 512
    window_size: int = 128
    compress_ratio: int = 2
    rope_theta: float = 10000.0
    # Demo batching (not model config)
    num_tokens: int = 4
    context_len: int = 256  # keep CPU demos light; use larger on SM120
    dtype: str = "float32"

    @property
    def head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


@dataclass(frozen=True)
class QSAShapes:
    name: str = "Qwen3.8-Flash-Next"
    hidden_size: int = 2560
    num_heads: int = 24
    num_kv_heads: int = 2
    head_dim: int = 256
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    compress_ratio: int = 4
    page_size: int = 64
    rope_theta: float = 10000.0
    attn_output_gate: bool = True
    num_tokens: int = 4
    context_len: int = 256
    dtype: str = "float32"

    @property
    def block_topk(self) -> int:
        return self.indexer_budget // self.compress_ratio

    @property
    def expand_width(self) -> int:
        # budget + compress_ratio - 1 (tail of incomplete block)
        return self.indexer_budget + self.compress_ratio - 1

    @property
    def index_qk_out(self) -> int:
        return (self.indexer_n_heads + self.indexer_kv_heads) * self.indexer_head_dim


PresetName = Literal["full", "tiny"]


def dsv4_shapes(preset: PresetName = "full", *, sm120: bool = False) -> DSV4Shapes:
    if preset == "tiny":
        return DSV4Shapes(
            name="DeepSeek-V4.1-Flash-tiny",
            hidden_size=512,
            num_heads=8,
            q_lora_rank=128,
            qk_nope_head_dim=56,
            qk_rope_head_dim=8,
            v_head_dim=64,
            kv_lora_rank=64,
            o_lora_rank=128,
            o_groups=2,
            index_head_dim=32,
            index_n_heads=8,
            index_topk=16,
            window_size=32,
            compress_ratio=2,
            num_tokens=4,
            context_len=64,
        )
    base = DSV4Shapes()
    if sm120:
        return replace(base, context_len=4096, dtype="bfloat16")
    return base


def qsa_shapes(preset: PresetName = "full", *, sm120: bool = False) -> QSAShapes:
    if preset == "tiny":
        return QSAShapes(
            name="Qwen3.8-Flash-Next-tiny",
            hidden_size=512,
            num_heads=8,
            num_kv_heads=2,
            head_dim=64,
            indexer_n_heads=4,
            indexer_kv_heads=1,
            indexer_head_dim=32,
            indexer_budget=64,
            compress_ratio=4,
            page_size=16,
            num_tokens=4,
            context_len=64,
        )
    base = QSAShapes()
    if sm120:
        return replace(base, context_len=4096, dtype="bfloat16")
    return base
