"""Workload grid: phase x batch_size x input_len x output_len (+ derived kv_len).

* prefill: q_len = kv_len = input_len per request; output_len is irrelevant.
* decode: q_len = 1 (``spec_tokens`` for MTP / speculative verify) and the KV
  length grows from input_len to input_len + output_len, so each
  (input_len, output_len) pair is sampled at 3 KV points instead of every step.
* non-attention ops only care about the token count, so kv_len collapses to 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bench.models.base import AttnVariant, ModelSpec
from bench.schema import WorkloadRecord

ATTN_OPS = ("attn_decode", "attn_prefill")
TOKEN_OPS = ("rmsnorm", "rope", "gemm_qkv", "gemm_o", "mlp", "moe")
BLOCK_OP = "decoder_layer"


@dataclass
class Grid:
    name: str = "smoke"
    phases: List[str] = field(default_factory=lambda: ["decode", "prefill"])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 16])
    input_lens: List[int] = field(default_factory=lambda: [2048])
    output_lens: List[int] = field(default_factory=lambda: [256])
    decode_kv_points: str = "start_mid_end"  # or "end" / "start"
    spec_tokens: int = 1
    dtype: str = "bfloat16"
    kv_dtype: str = "bfloat16"
    page_size: int = 16
    varlen: bool = False
    # Prefill token budget: skip prefill workloads with batch*input_len above this.
    max_prefill_tokens: int = 65536
    # Fraction of free device memory a single workload may use.
    mem_fraction: float = 0.6
    graph_phases: List[str] = field(default_factory=lambda: ["decode"])

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Grid":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for k in ("phases", "batch_sizes", "input_lens", "output_lens", "graph_phases"):
            v = known.get(k)
            if v is not None and not isinstance(v, (list, tuple)):
                known[k] = [v]
        return cls(**known)


BUILTIN_GRIDS: Dict[str, Dict[str, Any]] = {
    "tiny": dict(
        name="tiny",
        batch_sizes=[1, 3],
        input_lens=[64],
        output_lens=[16],
        page_size=4,
        dtype="float32",
        kv_dtype="float32",
        max_prefill_tokens=4096,
    ),
    "smoke": dict(name="smoke"),
    "decode": dict(
        name="decode",
        phases=["decode"],
        batch_sizes=[1, 4, 16, 64, 256],
        input_lens=[512, 2048, 8192],
        output_lens=[1024],
    ),
    "full": dict(
        name="full",
        batch_sizes=[1, 4, 16, 64, 256],
        input_lens=[512, 2048, 8192, 32768],
        output_lens=[128, 1024],
    ),
}


def load_grid(name_or_path: str, overrides: Optional[Dict[str, Any]] = None) -> Grid:
    if name_or_path in BUILTIN_GRIDS:
        d = dict(BUILTIN_GRIDS[name_or_path])
    else:
        d = load_config_file(Path(name_or_path)).get("grid", {})
        d.setdefault("name", Path(name_or_path).stem)
    if overrides:
        d.update({k: v for k, v in overrides.items() if v is not None})
    return Grid.from_dict(d)


def load_config_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError("pyyaml is required for YAML configs (pip install pyyaml)") from e
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _kv_points(grid: Grid, input_len: int, output_len: int) -> List[int]:
    end = input_len + output_len
    if grid.decode_kv_points == "end":
        pts = [end]
    elif grid.decode_kv_points == "start":
        pts = [input_len]
    else:
        pts = [input_len, input_len + output_len // 2, end]
    return sorted(set(pts))


def expand(
    grid: Grid, op: str, *, phase_filter: Optional[List[str]] = None
) -> List[WorkloadRecord]:
    """All workloads for one op, de-duplicated on the fields the op depends on."""
    out: Dict[str, WorkloadRecord] = {}
    phases = phase_filter or grid.phases
    common = dict(
        dtype=grid.dtype, kv_dtype=grid.kv_dtype, page_size=grid.page_size, varlen=grid.varlen
    )
    for phase in phases:
        if op == "attn_decode" and phase != "decode":
            continue
        if op == "attn_prefill" and phase != "prefill":
            continue
        if op == BLOCK_OP and phase != "decode":
            continue
        for b in grid.batch_sizes:
            for il in grid.input_lens:
                if phase == "prefill":
                    if b * il > grid.max_prefill_tokens:
                        continue
                    kv = il if op in ATTN_OPS else 0
                    wl = WorkloadRecord(
                        phase="prefill",
                        batch_size=b,
                        input_len=il,
                        output_len=0,
                        kv_len=kv,
                        q_len=il,
                        **common,
                    )
                    out[wl.key()] = wl
                    continue
                for ol in grid.output_lens:
                    if op in TOKEN_OPS:
                        wl = WorkloadRecord(
                            phase="decode",
                            batch_size=b,
                            input_len=0,
                            output_len=0,
                            kv_len=0,
                            q_len=grid.spec_tokens,
                            **common,
                        )
                        out[wl.key()] = wl
                        continue
                    for kv in _kv_points(grid, il, ol):
                        wl = WorkloadRecord(
                            phase="decode",
                            batch_size=b,
                            input_len=il,
                            output_len=ol,
                            kv_len=kv,
                            q_len=grid.spec_tokens,
                            **common,
                        )
                        out[wl.key()] = wl
    return list(out.values())


def num_tokens(wl: WorkloadRecord) -> int:
    return wl.batch_size * wl.q_len


def effective_kv_len(variant: AttnVariant, wl: WorkloadRecord) -> int:
    """Keys actually read per query (window / sparse selection caps it)."""
    kv = wl.kv_len
    if variant.sparse_topk is not None:
        kv = min(kv, variant.sparse_topk)
    if variant.window is not None and wl.phase == "decode":
        kv = min(kv, variant.window + wl.q_len - 1)
    return kv


_ELEM = {"bfloat16": 2, "float16": 2, "float32": 4, "fp8": 1, "float8_e4m3fn": 1}


def estimate_bytes(
    model: ModelSpec, variant: AttnVariant, op: str, wl: WorkloadRecord, moe_expert_cap: int = 0
) -> int:
    e = _ELEM.get(wl.dtype, 2)
    kv_e = _ELEM.get(wl.kv_dtype, 2)
    t = num_tokens(wl)
    h = model.hidden_size
    if op in ATTN_OPS or op == BLOCK_OP:
        kv = wl.batch_size * wl.kv_len * model.kv_bytes_per_token(kv_e)
        qo = t * model.num_q_heads * (model.qk_head_dim + model.vo_head_dim) * e
        # reference + backend layouts keep ~2 copies of the cache
        total = int(2.2 * kv + 4 * qo)
        if op == BLOCK_OP:
            total += layer_weight_bytes(model, e, moe_expert_cap)
        return total
    if op in ("rmsnorm", "rope"):
        return 6 * t * max(h, model.num_q_heads * model.qk_head_dim) * 4
    if op == "gemm_qkv":
        n = qkv_out_dim(model)
        return (h * n + t * (h + n)) * e + t * n * 4
    if op == "gemm_o":
        k = model.num_q_heads * model.vo_head_dim
        return (h * k + t * (h + k)) * e + t * h * 4
    if op == "mlp":
        i = model.intermediate_size
        return (3 * h * i + t * (h + 3 * i)) * e + t * h * 4
    if op == "moe":
        return layer_weight_bytes(model, e, moe_expert_cap, ffn_only=True) + t * h * 16
    return 0


def qkv_out_dim(model: ModelSpec) -> int:
    if model.is_mla:
        q = model.num_q_heads * model.qk_head_dim
        return q + model.kv_lora_rank + model.qk_rope_dim
    return (model.num_q_heads + 2 * model.num_kv_heads) * model.head_dim


def num_experts(model: ModelSpec, cap: int = 0) -> int:
    if model.moe is None:
        return 0
    return min(model.moe.num_experts, cap) if cap else model.moe.num_experts


def layer_weight_bytes(model: ModelSpec, e: int, moe_expert_cap: int = 0, ffn_only: bool = False) -> int:
    h = model.hidden_size
    if model.moe:
        ffn = num_experts(model, moe_expert_cap) * 3 * h * model.moe.intermediate_size * e
        ffn += h * model.moe.num_experts * e
    else:
        ffn = 3 * h * model.intermediate_size * e
    if ffn_only:
        return ffn
    attn = h * qkv_out_dim(model) * e + model.num_q_heads * model.vo_head_dim * h * e
    return ffn + attn


def describe(grid: Grid) -> str:
    return (
        f"grid={grid.name} phases={grid.phases} batch={grid.batch_sizes} "
        f"input_len={grid.input_lens} output_len={grid.output_lens} "
        f"kv_points={grid.decode_kv_points} dtype={grid.dtype} page={grid.page_size}"
    )


def fits(bytes_needed: int, free_bytes: int, frac: float) -> Tuple[bool, str]:
    if bytes_needed <= free_bytes * frac:
        return True, ""
    return False, (
        f"estimated {bytes_needed / 2**30:.2f} GiB > {frac:.0%} of free "
        f"{free_bytes / 2**30:.2f} GiB"
    )
