"""Non-attention ops: rmsnorm, rope, gemm_qkv, gemm_o, mlp, moe.

Each backend is a plain function over the problem inputs so the block-level
decoder layer (``bench/ops/block.py``) can reuse the exact same kernels.
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from bench.ops.base import Ctx, OpAdapter, Problem, Runner, call_compat, register

# ---------------------------------------------------------------------------
# Functional kernels: name -> fn(**tensors) -> Tensor
# ---------------------------------------------------------------------------


def rmsnorm_torch(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), w, eps)
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def rmsnorm_flashinfer(x, w, eps: float = 1e-6):
    import flashinfer.norm

    return flashinfer.norm.rmsnorm(x, w, eps)


def rmsnorm_sgl(x, w, eps: float = 1e-6):
    import sgl_kernel

    return sgl_kernel.rmsnorm(x, w, eps)


def silu_mul_torch(gu: torch.Tensor) -> torch.Tensor:
    i = gu.shape[-1] // 2
    return F.silu(gu[..., :i]) * gu[..., i:]


def silu_mul_flashinfer(gu):
    import flashinfer.activation

    return flashinfer.activation.silu_and_mul(gu)


def silu_mul_sgl(gu):
    import sgl_kernel

    return sgl_kernel.silu_and_mul(gu)


def rope_torch(x, k, positions, cos_sin, rot: int):
    """NeoX RoPE on the first ``rot`` dims, computed in the input dtype."""
    half = rot // 2
    cs = cos_sin.index_select(0, positions).to(x.dtype)
    cos, sin = cs[:, None, :half], cs[:, None, half:]

    def rot_(t):
        t1, t2 = t[..., :half], t[..., half:rot]
        return torch.cat([t1 * cos - t2 * sin, t2 * cos + t1 * sin, t[..., rot:]], dim=-1)

    return rot_(x), rot_(k)


def rope_flashinfer(x, k, positions, cos_sin, rot: int):
    import flashinfer.rope

    t, hq, d = x.shape
    q2, k2 = flashinfer.rope.apply_rope_with_cos_sin_cache(
        positions, x.reshape(t, hq * d), k.reshape(t, -1), d, cos_sin, is_neox=True
    )
    return q2.view(t, hq, d), k2.view(t, -1, d)


def linear_torch(x, w):
    return F.linear(x, w)


def mlp_with(act: Callable) -> Callable:
    def mlp(x, w13, w2):
        return F.linear(act(F.linear(x, w13)), w2)

    return mlp


# ---------------------------------------------------------------------------
# MoE kernels
# ---------------------------------------------------------------------------


def moe_torch_loop(x, w13, w2, topk_ids, topk_weights, **_):
    """Per-expert loop with data-dependent shapes (not CUDA-graph capturable)."""
    out = torch.zeros_like(x, dtype=torch.float32)
    i = w2.shape[-1]
    ids = topk_ids.long()
    for ex in ids.unique().tolist():
        tok, slot = (ids == ex).nonzero(as_tuple=True)
        gu = F.linear(x[tok], w13[ex])
        a = F.silu(gu[:, :i]) * gu[:, i:]
        out.index_add_(0, tok, F.linear(a, w2[ex]).float() * topk_weights[tok, slot, None])
    return out.to(x.dtype)


def _sglang_fused_experts() -> Tuple[Callable, str]:
    try:
        mod = importlib.import_module("sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe")
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        fe = mod.fused_experts

        def run_new(x, w13, w2, topk_ids, topk_weights, router_logits=None, **_):
            topk = StandardTopKOutput(topk_weights, topk_ids, router_logits)
            e = w13.shape[0]
            cfg = call_compat(
                MoeRunnerConfig,
                num_experts=e,
                num_local_experts=e,
                top_k=topk_ids.shape[-1],
                inplace=False,
                gate_up_interleaved=False,
            )
            return fe(x, w13, w2, topk, cfg)

        return run_new, "sglang.moe_runner.triton_utils.fused_experts"
    except Exception:
        pass
    mod = importlib.import_module("sglang.srt.layers.moe.fused_moe_triton.fused_moe")
    fe = mod.fused_experts

    def run_old(x, w13, w2, topk_ids, topk_weights, **_):
        return call_compat(fe, x, w13, w2, topk_weights, topk_ids, inplace=False)

    return run_old, "sglang.fused_moe_triton.fused_experts (legacy)"


def moe_sglang_triton(x, w13, w2, topk_ids, topk_weights, **kw):
    fn, _ = _sglang_fused_experts()
    return fn(x, w13, w2, topk_ids, topk_weights, **kw)


def moe_flashinfer_cutlass(x, w31, w2, topk_ids, topk_weights, **_):
    from flashinfer.fused_moe import cutlass_fused_moe

    out = cutlass_fused_moe(
        input=x,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        fc1_expert_weights=w31,
        fc2_expert_weights=w2,
        output_dtype=x.dtype,
        quant_scales=[],
    )
    return out[0] if isinstance(out, (list, tuple)) else out


def swap_w13(w13: torch.Tensor) -> torch.Tensor:
    """[E, (gate; up), H] -> [E, (up; gate), H] (TRT-LLM cutlass gated layout)."""
    e, two_i, h = w13.shape
    return w13.view(e, 2, two_i // 2, h).flip(1).reshape(e, two_i, h).contiguous()


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class DenseAdapter(OpAdapter):
    needs_cuda = True
    dtypes = ("bfloat16", "float16")

    def run_fn(self, p: Problem) -> Callable[[], object]:
        raise NotImplementedError

    def prepare(self, p: Problem, graph: bool) -> Runner:
        fn = self.run_fn(p)
        return Runner(fn=fn, **self.runner_kwargs(p))

    def runner_kwargs(self, p: Problem) -> Dict:
        return {}


def _rope_check(ctx: Ctx) -> Optional[str]:
    if ctx.model.rope_dim == 0:
        return "model has no separate rope part"
    return None


def _rope_layout(out) -> torch.Tensor:
    q, k = out
    return torch.cat([q.flatten(1), k.flatten(1)], dim=1)


def _make(op: str, backend: str, fn_builder: Callable[[Problem], Callable], *,
          requires: Tuple[str, ...] = (), needs_cuda: bool = True,
          dtypes: Tuple[str, ...] = ("bfloat16", "float16"),
          check: Optional[Callable[[Ctx], Optional[str]]] = None,
          graph_safe: bool = True, graph_reason: str = "",
          to_ref_layout: Optional[Callable] = None, is_reference: bool = False,
          sm_min: Optional[Tuple[int, int]] = None) -> None:
    attrs = dict(
        backend=backend,
        ops=(op,),
        requires=requires,
        needs_cuda=needs_cuda,
        dtypes=dtypes,
        is_reference=is_reference,
        sm_min=sm_min,
        run_fn=lambda self, p: fn_builder(p),
        runner_kwargs=lambda self, p: dict(
            graph_safe=graph_safe, graph_reason=graph_reason, to_ref_layout=to_ref_layout
        ),
    )
    if check is not None:
        attrs["extra_check"] = lambda self, ctx: check(ctx)
    cls = type(f"{op}_{backend}", (DenseAdapter,), attrs)
    register(cls)


ALL_DT = ("bfloat16", "float16", "float32")


def _ref_runner(p: Problem) -> Callable:
    dt = p.ctx.dtype
    if p.ctx.op == "rope":
        d = p.meta["head_dim"]

        def fn():
            out = p.reference(p).to(dt)
            t = out.shape[0]
            nq = p.inputs["x"].shape[1] * d
            return out[:, :nq].view(t, -1, d), out[:, nq:].view(t, -1, d)

        return fn
    return lambda: p.reference(p).to(dt)


for _op in ("rmsnorm", "rope", "gemm_qkv", "gemm_o", "mlp", "moe"):
    _make(
        _op,
        "torch_ref",
        _ref_runner,
        needs_cuda=False,
        dtypes=ALL_DT,
        is_reference=True,
        check=_rope_check if _op == "rope" else None,
        to_ref_layout=_rope_layout if _op == "rope" else None,
        graph_safe=_op != "moe",
        graph_reason="data-dependent expert loop" if _op == "moe" else "",
    )

# rmsnorm
_make("rmsnorm", "torch", lambda p: lambda: rmsnorm_torch(p.inputs["x"], p.inputs["w"]),
      needs_cuda=False, dtypes=ALL_DT)
_make("rmsnorm", "flashinfer", lambda p: lambda: rmsnorm_flashinfer(p.inputs["x"], p.inputs["w"]),
      requires=("flashinfer",))
_make("rmsnorm", "sgl_kernel", lambda p: lambda: rmsnorm_sgl(p.inputs["x"], p.inputs["w"]),
      requires=("sgl_kernel",))

# rope
_ROPE_ARGS = lambda p: (p.inputs["x"], p.inputs["k"], p.inputs["positions"], p.inputs["cos_sin"],
                        p.meta["rot"])
_make("rope", "torch", lambda p: lambda: rope_torch(*_ROPE_ARGS(p)), needs_cuda=False,
      dtypes=ALL_DT, check=_rope_check, to_ref_layout=_rope_layout)
_make("rope", "flashinfer", lambda p: lambda: rope_flashinfer(*_ROPE_ARGS(p)),
      requires=("flashinfer",), check=_rope_check, to_ref_layout=_rope_layout)

# gemms
for _op in ("gemm_qkv", "gemm_o"):
    _make(_op, "torch", lambda p: lambda: linear_torch(p.inputs["x"], p.inputs["w"]),
          needs_cuda=False, dtypes=ALL_DT)

# mlp
_make("mlp", "torch",
      lambda p: lambda: mlp_with(silu_mul_torch)(p.inputs["x"], p.inputs["w13"], p.inputs["w2"]),
      needs_cuda=False, dtypes=ALL_DT)
_make("mlp", "flashinfer",
      lambda p: lambda: mlp_with(silu_mul_flashinfer)(p.inputs["x"], p.inputs["w13"], p.inputs["w2"]),
      requires=("flashinfer",))
_make("mlp", "sgl_kernel",
      lambda p: lambda: mlp_with(silu_mul_sgl)(p.inputs["x"], p.inputs["w13"], p.inputs["w2"]),
      requires=("sgl_kernel",))


# moe
def _moe_inputs(p: Problem) -> Dict:
    return {k: p.inputs[k] for k in ("x", "w13", "w2", "topk_ids", "topk_weights")} | {
        "router_logits": p.inputs["router_logits"]
    }


def _sglang_moe_check(ctx: Ctx) -> Optional[str]:
    try:
        _sglang_fused_experts()
    except Exception as e:
        return f"sglang fused_experts not importable: {e}"
    return None


_make("moe", "torch", lambda p: lambda: moe_torch_loop(**_moe_inputs(p)), needs_cuda=False,
      dtypes=ALL_DT, graph_safe=False, graph_reason="data-dependent expert loop")
_make("moe", "sglang_triton", lambda p: lambda: moe_sglang_triton(**_moe_inputs(p)),
      requires=("triton",), check=_sglang_moe_check)


def _cutlass_fn(p: Problem) -> Callable:
    w31 = swap_w13(p.inputs["w13"])
    x, w2 = p.inputs["x"], p.inputs["w2"]
    ids, wts = p.inputs["topk_ids"], p.inputs["topk_weights"]
    return lambda: moe_flashinfer_cutlass(x, w31, w2, ids, wts)


_make("moe", "flashinfer_cutlass", _cutlass_fn, requires=("flashinfer",), sm_min=(9, 0))
