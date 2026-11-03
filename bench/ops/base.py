"""Operator adapter interface, problem container, and adapter registry.

Contract for adapters:

* ``check(ctx)`` is cheap and runs before any tensor is allocated; it returns
  a reason string when the (model, variant, workload, hardware) combination
  is unsupported, which is recorded as a ``SKIPPED`` trace.
* ``prepare(problem, graph)`` does all host-side work (layout conversion,
  FlashInfer ``plan()``, metadata) and is never timed.
* ``Runner.fn()`` launches only device work and must read the problem's
  primary input tensor (``problem.inputs["q"]`` for attention, ``"x"`` for
  dense ops) by reference, so block-level runs can ``copy_`` new data in and
  CUDA graphs replay against the same buffers. Any transform of that input
  (padding, reshaping copies) happens inside ``fn`` and is therefore timed.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

import torch

from bench.models.base import AttnVariant, ModelSpec
from bench.schema import WorkloadRecord


@dataclass(frozen=True)
class Tol:
    atol: float
    rtol: float
    matched_ratio: float = 1.0


DEFAULT_TOL = {
    torch.float32: Tol(1e-4, 1e-4),
    torch.float16: Tol(1e-2, 1e-2),
    torch.bfloat16: Tol(1e-2, 1e-2),
}


@dataclass(frozen=True)
class AttnGeom:
    kind: str  # "gqa" | "mla"
    num_q_heads: int
    num_kv_heads: int
    qk_dim: int
    v_dim: int
    scale: float
    rope_dim: int = 0  # MLA: trailing rope part of qk_dim
    window: Optional[int] = None
    sink: bool = False
    softcap: Optional[float] = None
    sparse_topk: Optional[int] = None

    @property
    def group(self) -> int:
        return self.num_q_heads // self.num_kv_heads

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


def attn_geom(model: ModelSpec, variant: AttnVariant, phase: str) -> AttnGeom:
    common = dict(
        window=variant.window,
        sink=variant.sink,
        softcap=variant.softcap,
        sparse_topk=variant.sparse_topk if phase == "decode" else None,
    )
    if model.is_mla:
        if phase == "prefill" and model.v_head_dim:
            # DeepSeek-style MLA prefill runs as un-absorbed MHA (qk 192 / v 128).
            return AttnGeom(
                kind="gqa",
                num_q_heads=model.num_q_heads,
                num_kv_heads=model.num_q_heads,
                qk_dim=model.qk_nope_dim + model.qk_rope_dim,
                v_dim=model.v_head_dim,
                scale=model.softmax_scale,
                **common,
            )
        return AttnGeom(
            kind="mla",
            num_q_heads=model.num_q_heads,
            num_kv_heads=1,
            qk_dim=model.kv_lora_rank + model.qk_rope_dim,
            v_dim=model.kv_lora_rank,
            rope_dim=model.qk_rope_dim,
            scale=model.softmax_scale,
            **common,
        )
    return AttnGeom(
        kind="gqa",
        num_q_heads=model.num_q_heads,
        num_kv_heads=model.num_kv_heads,
        qk_dim=model.head_dim,
        v_dim=model.v_head_dim or model.head_dim,
        scale=model.softmax_scale,
        **common,
    )


@dataclass
class Ctx:
    model: ModelSpec
    variant: AttnVariant
    op: str
    wl: WorkloadRecord
    device: torch.device
    sm: Optional[Tuple[int, int]]
    geom: Optional[AttnGeom] = None
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def dtype(self) -> torch.dtype:
        return torch_dtype(self.wl.dtype)


@dataclass
class Problem:
    ctx: Ctx
    inputs: Dict[str, torch.Tensor]
    meta: Dict[str, Any] = field(default_factory=dict)
    reference: Optional[Callable[["Problem"], torch.Tensor]] = None
    _ref_out: Optional[torch.Tensor] = None

    @property
    def geom(self) -> AttnGeom:
        assert self.ctx.geom is not None
        return self.ctx.geom

    def ref(self) -> torch.Tensor:
        if self._ref_out is None:
            assert self.reference is not None
            with torch.no_grad():
                self._ref_out = self.reference(self)
        return self._ref_out


@dataclass
class Runner:
    fn: Callable[[], torch.Tensor]
    graph_safe: bool = True
    graph_reason: str = ""
    tol: Optional[Tol] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    # Optional: map raw output to the reference layout for correctness checks.
    to_ref_layout: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


class OpAdapter:
    backend: str = ""
    ops: Tuple[str, ...] = ()
    kinds: Tuple[str, ...] = ("gqa", "mla")
    features: FrozenSet[str] = frozenset()
    requires: Tuple[str, ...] = ()
    needs_cuda: bool = True
    sm_allow: Optional[Tuple[Tuple[int, int], ...]] = None  # exact (major, minor) list
    sm_min: Optional[Tuple[int, int]] = None
    dtypes: Tuple[str, ...] = ("bfloat16", "float16")
    kv_dtypes: Tuple[str, ...] = ("bfloat16", "float16", "float32")
    decode_q_len_1_only: bool = False
    phases: Tuple[str, ...] = ("decode", "prefill")
    is_reference: bool = False

    def check(self, ctx: Ctx) -> Optional[str]:
        if ctx.op not in self.ops and not (ctx.op == "decoder_layer" and "attn_decode" in self.ops):
            return f"{self.backend} does not implement {ctx.op}"
        if self.needs_cuda and ctx.device.type != "cuda":
            return f"{self.backend} needs CUDA"
        if ctx.wl.phase not in self.phases:
            return f"{self.backend} has no {ctx.wl.phase} kernel"
        if ctx.wl.dtype not in self.dtypes:
            return f"{self.backend} does not support dtype {ctx.wl.dtype}"
        if ctx.wl.kv_dtype not in self.kv_dtypes:
            return f"{self.backend} does not support kv dtype {ctx.wl.kv_dtype}"
        if self.sm_allow is not None and ctx.sm not in self.sm_allow:
            allowed = ", ".join(f"sm_{a}{b}" for a, b in self.sm_allow)
            return f"{self.backend} only runs on {allowed}"
        if self.sm_min is not None and (ctx.sm is None or ctx.sm < self.sm_min):
            return f"{self.backend} needs sm_{self.sm_min[0]}{self.sm_min[1]}+"
        for mod in self.requires:
            if not module_available(mod):
                return f"python module {mod!r} not installed"
        g = ctx.geom
        if g is not None:
            if g.kind not in self.kinds:
                return f"{self.backend} has no {g.kind.upper()} kernel"
            missing = [f for f in g.features if f not in self.features]
            if missing:
                return f"{self.backend} lacks {', '.join(missing)}"
            if self.decode_q_len_1_only and ctx.wl.phase == "decode" and ctx.wl.q_len != 1:
                return f"{self.backend} decode kernel needs q_len == 1"
        return self.extra_check(ctx)

    def extra_check(self, ctx: Ctx) -> Optional[str]:
        return None

    def prepare(self, p: Problem, graph: bool) -> Runner:
        raise NotImplementedError


_ADAPTERS: List[OpAdapter] = []


def register(cls: type) -> type:
    _ADAPTERS.append(cls())
    return cls


def adapters_for(op: str, backends: Optional[List[str]] = None) -> List[OpAdapter]:
    target = "attn_decode" if op == "decoder_layer" else op
    out = [a for a in _ADAPTERS if target in a.ops]
    if backends:
        out = [a for a in out if a.backend in backends]
    return out


def all_adapters() -> List[OpAdapter]:
    return list(_ADAPTERS)


def call_compat(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` dropping kwargs its signature does not accept.

    Kernel libraries rename/add keyword arguments between releases; this keeps
    one adapter working across versions. Positional args are passed as-is.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(v.kind == v.VAR_KEYWORD for v in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


_MOD_CACHE: Dict[str, bool] = {}


def module_available(name: str) -> bool:
    if name not in _MOD_CACHE:
        try:
            importlib.import_module(name)
            _MOD_CACHE[name] = True
        except Exception:
            _MOD_CACHE[name] = False
    return _MOD_CACHE[name]


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp8": torch.bfloat16,
        "float8_e4m3fn": torch.bfloat16,
    }[name]


def kv_is_fp8(wl: WorkloadRecord) -> bool:
    return wl.kv_dtype in ("fp8", "float8_e4m3fn")


# ---------------------------------------------------------------------------
# Paged-KV helpers shared by attention adapters
# ---------------------------------------------------------------------------


def token_slots(p: Problem, b: int) -> torch.Tensor:
    """Flat slot ids (page * page_size + offset) of request ``b``'s tokens, in order."""
    ps = p.meta["page_size"]
    s = p.meta["seq_lens"][b]
    pos = torch.arange(s, device=p.ctx.device)
    pages = p.inputs["page_table"][b, (pos // ps)].long()
    return pages * ps + pos % ps


def gather_tokens(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """cache [P, ps, ...] -> [len(slots), ...]."""
    return cache.reshape(-1, *cache.shape[2:]).index_select(0, slots)


def selected_positions(p: Problem, b: int) -> torch.Tensor:
    """Key positions request ``b`` attends to (sparse selection or all)."""
    if "sel_pos" in p.inputs:
        return p.inputs["sel_pos"][b].long()
    return torch.arange(p.meta["seq_lens"][b], device=p.ctx.device)


def decode_key_positions(p: Problem, b: int) -> torch.Tensor:
    """Positions read by a q_len==1 decode query: sparse set, window, or all."""
    s = p.meta["seq_lens"][b]
    g = p.geom
    pos = selected_positions(p, b)
    if g.window is not None:
        pos = pos[pos >= s - g.window]
    return pos


def repage(
    p: Problem, caches: List[torch.Tensor], page_size: int
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Copy caches into a fresh contiguous page layout with ``page_size``.

    Returns (new caches, block_table [B, max_pages] int32). Runs at prepare time.
    """
    seq_lens = p.meta["seq_lens"]
    B = len(seq_lens)
    pages_per = [(s + page_size - 1) // page_size for s in seq_lens]
    max_pages = max(pages_per)
    total = sum(pages_per)
    dev = p.ctx.device
    new = [
        torch.zeros((total, page_size) + tuple(c.shape[2:]), dtype=c.dtype, device=dev)
        for c in caches
    ]
    table = torch.zeros((B, max_pages), dtype=torch.int32, device=dev)
    nxt = 0
    for b in range(B):
        slots = token_slots(p, b)
        n = pages_per[b]
        table[b, :n] = torch.arange(nxt, nxt + n, dtype=torch.int32, device=dev)
        for src, dst in zip(caches, new):
            flat = dst[nxt : nxt + n].reshape(n * page_size, *dst.shape[2:])
            flat[: seq_lens[b]] = gather_tokens(src, slots)
        nxt += n
    return new, table


def slots_for_positions(p: Problem, b: int, pos: torch.Tensor) -> torch.Tensor:
    ps = p.meta["page_size"]
    pages = p.inputs["page_table"][b, (pos // ps)].long()
    return (pages * ps + pos % ps).to(torch.int32)


def token_index_lists(p: Problem) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-request flat slot lists for q_len==1 decode as (indptr, indices) int32.

    Encodes sparse selection and sliding window, so any backend that accepts
    page_size=1 indices gets exactly the keys the reference attends to.
    """
    B = len(p.meta["seq_lens"])
    idx = [slots_for_positions(p, b, decode_key_positions(p, b)) for b in range(B)]
    lens = torch.tensor([len(i) for i in idx], dtype=torch.int32, device=p.ctx.device)
    indptr = torch.zeros(B + 1, dtype=torch.int32, device=p.ctx.device)
    indptr[1:] = torch.cumsum(lens, 0)
    return indptr, torch.cat(idx)
