"""Model architecture specs: just enough of each HF config to size every op."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AttnVariant:
    """One distinct attention-layer flavour inside a model.

    ``window``: sliding window size in keys (query attends to keys with
    ``q_pos - k_pos < window``), ``None`` = full causal.
    ``sparse_topk``: decode attends to only this many selected tokens
    (DSA/DSV4/QSA style); the selection is modelled as random sorted indices.
    """

    name: str
    num_layers: int
    window: Optional[int] = None
    sink: bool = False
    softcap: Optional[float] = None
    sparse_topk: Optional[int] = None

    @property
    def features(self) -> Tuple[str, ...]:
        f = []
        if self.window is not None:
            f.append("sliding_window")
        if self.sink:
            f.append("sink")
        if self.softcap is not None:
            f.append("softcap")
        if self.sparse_topk is not None:
            f.append("sparse")
        return tuple(f)


@dataclass(frozen=True)
class MoESpec:
    num_experts: int
    topk: int
    intermediate_size: int
    num_shared_experts: int = 0


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    hidden_size: int
    num_layers: int
    num_q_heads: int
    attn_type: str = "gqa"  # "gqa" | "mla"
    num_kv_heads: int = 1
    head_dim: int = 128
    v_head_dim: Optional[int] = None
    # MLA (absorbed decode form): q/k dim = kv_lora_rank + qk_rope_dim, v dim = kv_lora_rank
    kv_lora_rank: int = 0
    qk_rope_dim: int = 0
    qk_nope_dim: int = 0
    q_lora_rank: int = 0
    rotary_dim: Optional[int] = None
    qk_norm: bool = False
    intermediate_size: int = 0
    moe: Optional[MoESpec] = None
    variants: Tuple[AttnVariant, ...] = ()
    vocab_size: int = 0
    notes: str = ""
    hf_id: str = ""

    @property
    def is_mla(self) -> bool:
        return self.attn_type == "mla"

    @property
    def qk_head_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_dim if self.is_mla else self.head_dim

    @property
    def vo_head_dim(self) -> int:
        if self.is_mla:
            return self.kv_lora_rank
        return self.v_head_dim or self.head_dim

    @property
    def kv_heads(self) -> int:
        return 1 if self.is_mla else self.num_kv_heads

    @property
    def softmax_scale(self) -> float:
        if self.is_mla:
            return (self.qk_nope_dim + self.qk_rope_dim) ** -0.5
        return self.head_dim**-0.5

    @property
    def rope_dim(self) -> int:
        if self.is_mla:
            return self.qk_rope_dim
        return self.rotary_dim or self.head_dim

    def attn_variants(self) -> Tuple[AttnVariant, ...]:
        return self.variants or (AttnVariant("full", self.num_layers),)

    def kv_bytes_per_token(self, elem_bytes: int = 2) -> int:
        """KV cache bytes for one token in one layer."""
        if self.is_mla:
            return self.qk_head_dim * elem_bytes
        return 2 * self.num_kv_heads * self.head_dim * elem_bytes

    def ops(self) -> List[str]:
        out = ["attn_decode", "attn_prefill", "rmsnorm", "rope", "gemm_qkv", "gemm_o"]
        out += ["moe"] if self.moe else ["mlp"]
        return out

    def block_ops(self) -> List[str]:
        """Op order inside one decoder layer (used by block-level bench + NVTX)."""
        ffn = "moe" if self.moe else "mlp"
        return [
            "rmsnorm",
            "gemm_qkv",
            "rope",
            "kv_store",
            "attn",
            "gemm_o",
            "residual",
            "rmsnorm",
            ffn,
            "residual",
        ]

    def with_overrides(self, **kw: Any) -> "ModelSpec":
        return replace(self, **kw)


def from_hf_config(path: str, name: Optional[str] = None) -> ModelSpec:
    """Build a spec from a HuggingFace ``config.json`` (best effort, common fields only)."""
    p = Path(path)
    if p.is_dir():
        p = p / "config.json"
    cfg: Dict[str, Any] = json.loads(p.read_text())
    if "text_config" in cfg:
        cfg = {**cfg, **cfg["text_config"]}
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    layers = cfg["num_hidden_layers"]
    moe = None
    n_exp = cfg.get("n_routed_experts") or cfg.get("num_experts") or cfg.get("num_local_experts")
    if n_exp:
        moe = MoESpec(
            num_experts=n_exp,
            topk=cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_token") or 8,
            intermediate_size=cfg.get("moe_intermediate_size") or cfg.get("intermediate_size"),
            num_shared_experts=cfg.get("n_shared_experts") or 0,
        )
    common = dict(
        name=name or cfg.get("_name_or_path") or p.parent.name,
        family=cfg.get("model_type", "unknown"),
        hidden_size=hidden,
        num_layers=layers,
        num_q_heads=heads,
        intermediate_size=cfg.get("intermediate_size", 0),
        moe=moe,
        vocab_size=cfg.get("vocab_size", 0),
    )
    if cfg.get("kv_lora_rank"):
        return ModelSpec(
            attn_type="mla",
            kv_lora_rank=cfg["kv_lora_rank"],
            qk_rope_dim=cfg["qk_rope_head_dim"],
            qk_nope_dim=cfg["qk_nope_head_dim"],
            q_lora_rank=cfg.get("q_lora_rank") or 0,
            head_dim=cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"],
            v_head_dim=cfg.get("v_head_dim"),
            **common,
        )
    head_dim = cfg.get("head_dim") or hidden // heads
    variants: Tuple[AttnVariant, ...] = ()
    window = cfg.get("sliding_window")
    layer_types = cfg.get("layer_types")
    if window and layer_types:
        n_local = sum(1 for t in layer_types if "sliding" in t)
        variants = (
            AttnVariant("local", n_local, window=window),
            AttnVariant("global", layers - n_local),
        )
    rotary = None
    if cfg.get("partial_rotary_factor"):
        rotary = int(head_dim * cfg["partial_rotary_factor"])
    return ModelSpec(
        attn_type="gqa",
        num_kv_heads=cfg.get("num_key_value_heads", heads),
        head_dim=head_dim,
        rotary_dim=rotary,
        qk_norm=bool(cfg.get("use_qk_norm"))
        or cfg.get("model_type", "") in ("qwen3", "qwen3_moe", "gemma3_text", "gemma3"),
        variants=variants,
        **common,
    )


@dataclass
class ModelRegistry:
    specs: Dict[str, ModelSpec] = field(default_factory=dict)

    def add(self, spec: ModelSpec) -> ModelSpec:
        self.specs[spec.name] = spec
        return spec

    def get(self, name: str) -> ModelSpec:
        if name in self.specs:
            return self.specs[name]
        if Path(name).exists():
            return from_hf_config(name)
        raise KeyError(f"unknown model {name!r}; known: {sorted(self.specs)}")

    def names(self) -> List[str]:
        return sorted(self.specs)
