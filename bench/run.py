"""Run the model x op x variant x workload x backend x mode matrix, appending traces."""

from __future__ import annotations

import gc
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch

from bench import correctness, hardware, metrics
from bench.models import REGISTRY
from bench.models.base import AttnVariant, ModelSpec
from bench.ops import Ctx, OpAdapter, Problem, Runner, Tol, adapters_for, attn_geom
from bench.ops import problems as P
from bench.ops.block import build_block
from bench.schema import Level, Mode, Status, Trace, append_traces
from bench.timing import GraphCaptureError, TimingConfig, capture_graph, gpu_time, host_overhead
from bench.workloads import BLOCK_OP, Grid, estimate_bytes, expand, fits

ALL_OPS = (
    "attn_decode",
    "attn_prefill",
    "rmsnorm",
    "rope",
    "gemm_qkv",
    "gemm_o",
    "mlp",
    "moe",
    BLOCK_OP,
)
ATTN_LIKE = ("attn_decode", "attn_prefill", BLOCK_OP)


@dataclass
class RunConfig:
    models: List[str]
    ops: List[str]
    grid: Grid
    out: Path
    backends: Optional[List[str]] = None
    modes: List[str] = field(default_factory=lambda: list(Mode.ALL))
    variants: Optional[List[str]] = None
    timing: TimingConfig = field(default_factory=TimingConfig)
    device: str = "cuda"
    seed: int = 0
    moe_expert_cap: int = 0
    block_layers: int = 1
    block_impl: Dict[str, str] = field(default_factory=dict)
    tol: Optional[Tol] = None
    dry_run: bool = False
    tag: str = ""


def ops_for(model: ModelSpec, requested: List[str]) -> List[str]:
    out = []
    for op in requested:
        if op == "moe" and model.moe is None:
            continue
        if op == "mlp" and model.moe is not None:
            continue
        if op == "rope" and model.rope_dim == 0:
            continue
        out.append(op)
    return out


def _variants(model: ModelSpec, op: str, only: Optional[List[str]]) -> List[AttnVariant]:
    if op not in ATTN_LIKE:
        return [AttnVariant("-", model.num_layers)]
    vs = list(model.attn_variants())
    if only:
        vs = [v for v in vs if v.name in only] or vs
    return vs


class Matrix:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.hw = hardware.probe()
        self.libs = hardware.lib_versions()
        self.sm = hardware.sm()
        self.counts: Dict[str, int] = {}

    # -- trace helpers --------------------------------------------------------
    def _trace(self, ctx: Ctx, backend: str, mode: str, status: str, **kw) -> Trace:
        level = Level.BLOCK if ctx.op == BLOCK_OP else Level.OP
        meta = dict(kw.pop("meta", {}) or {})
        if self.cfg.tag:
            meta["tag"] = self.cfg.tag
        if ctx.geom is not None:
            meta.setdefault("geom", {"kind": ctx.geom.kind, "features": sorted(ctx.geom.features)})
        return Trace(
            level=level,
            model=ctx.model.name,
            op=ctx.op,
            variant=ctx.variant.name,
            backend=backend,
            mode=mode,
            workload=ctx.wl,
            status=status,
            hardware=self.hw,
            libs=self.libs,
            meta=meta,
            **kw,
        )

    def emit(self, t: Trace) -> None:
        append_traces(self.cfg.out, [t])
        self.counts[t.status] = self.counts.get(t.status, 0) + 1
        lat = f"{t.latency.median_ms:9.4f} ms" if t.latency and t.latency.n else " " * 12
        extra = ""
        if t.metrics.get("graph_speedup"):
            extra = f"  graph x{t.metrics['graph_speedup']:.2f}"
        msg = f"  ({t.message})" if t.message and t.status != Status.PASSED else ""
        print(
            f"[{t.status:<19}] {t.model:<22} {t.op:<13} {t.variant:<12} {t.backend:<16} "
            f"{t.mode:<9} {t.workload.key():<52} {lat}{extra}{msg[:160]}",
            flush=True,
        )

    # -- matrix ---------------------------------------------------------------
    def cases(self) -> Iterator[Tuple[Ctx, List[str]]]:
        cfg = self.cfg
        for name in cfg.models:
            model = REGISTRY.get(name)
            for op in ops_for(model, cfg.ops):
                for variant in _variants(model, op, cfg.variants):
                    for wl in expand(cfg.grid, op):
                        ctx = make_ctx(model, variant, op, wl, self.device, cfg.moe_expert_cap)
                        modes = [Mode.EAGER] if Mode.EAGER in cfg.modes else []
                        if (
                            Mode.CUDAGRAPH in cfg.modes
                            and self.device.type == "cuda"
                            and wl.phase in cfg.grid.graph_phases
                        ):
                            modes.append(Mode.CUDAGRAPH)
                        if modes:
                            yield ctx, modes

    def run(self) -> Dict[str, int]:
        for ctx, modes in self.cases():
            if self.cfg.dry_run:
                names = [a.backend for a in adapters_for(ctx.op, self.cfg.backends)]
                print(
                    f"{ctx.model.name:<22} {ctx.op:<13} {ctx.variant.name:<12} "
                    f"{ctx.wl.key():<52} modes={modes} backends={names}"
                )
                continue
            try:
                self._run_case(ctx, modes)
            finally:
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        return self.counts

    def _run_case(self, ctx: Ctx, modes: List[str]) -> None:
        cfg = self.cfg
        adapters = adapters_for(ctx.op, cfg.backends)
        if not adapters:
            return
        need = estimate_bytes(ctx.model, ctx.variant, ctx.op, ctx.wl, cfg.moe_expert_cap)
        ok, why = fits(need, hardware.free_mem_bytes(), cfg.grid.mem_fraction)
        if not ok:
            for a in adapters:
                for mode in modes:
                    self.emit(self._trace(ctx, a.backend, mode, Status.SKIPPED, message=why))
            return
        problem: Optional[Problem] = None
        for a in adapters:
            reason = a.check(ctx)
            if reason:
                for mode in modes:
                    self.emit(self._trace(ctx, a.backend, mode, Status.SKIPPED, message=reason))
                continue
            if problem is None:
                try:
                    problem = self._build(ctx)
                except Exception as e:
                    status = Status.OOM if _is_oom(e) else Status.RUNTIME_ERROR
                    for mode in modes:
                        self.emit(
                            self._trace(ctx, a.backend, mode, status, message=f"build: {_short(e)}")
                        )
                    return
            if ctx.op == BLOCK_OP:
                self._run_block(ctx, a, problem, modes)
            else:
                self._run_op(ctx, a, problem, modes)

    def _build(self, ctx: Ctx) -> Problem:
        return build_problem(ctx, self.cfg.seed)

    # -- op level -------------------------------------------------------------
    def _run_op(self, ctx: Ctx, a: OpAdapter, p: Problem, modes: List[str]) -> None:
        eager_lat = None
        eager_ok = False
        if Mode.EAGER in modes:
            eager_ok, eager_lat = self._eager(ctx, a, p)
        if Mode.CUDAGRAPH in modes:
            if Mode.EAGER in modes and not eager_ok:
                self.emit(
                    self._trace(ctx, a.backend, Mode.CUDAGRAPH, Status.SKIPPED,
                                message="eager run failed; graph not attempted")
                )
                return
            self._graph(ctx, a, p, eager_lat, lambda g: a.prepare(p, g), check_ref=True)

    def _eager(self, ctx: Ctx, a: OpAdapter, p: Problem) -> Tuple[bool, Optional[float]]:
        try:
            runner = a.prepare(p, False)
            with torch.no_grad():
                out = runner.fn()
            _sync(self.device)
        except Exception as e:
            self._error(ctx, a.backend, Mode.EAGER, e)
            return False, None
        tol = correctness.tolerance(p, runner, self.cfg.tol)
        status, corr, msg = correctness.check(p, runner, out, tol)
        if status != Status.PASSED:
            self.emit(self._trace(ctx, a.backend, Mode.EAGER, status, correctness=corr,
                                  message=msg, meta={"runner": runner.meta}))
            return False, None
        del out
        try:
            with torch.no_grad():
                lat = gpu_time(runner.fn, self.cfg.timing, self.device)
                host = host_overhead(runner.fn, self.cfg.timing, self.device)
        except Exception as e:
            self._error(ctx, a.backend, Mode.EAGER, e, "timing")
            return False, None
        self.emit(
            self._trace(
                ctx, a.backend, Mode.EAGER, Status.PASSED,
                correctness=corr, latency=lat, host=host,
                metrics=metrics.compute(p, lat.median_ms),
                meta={"runner": runner.meta},
            )
        )
        return True, lat.median_ms

    def _graph(
        self,
        ctx: Ctx,
        a: OpAdapter,
        p: Problem,
        eager_ms: Optional[float],
        make_runner: Callable[[bool], Runner],
        check_ref: bool,
        eager_out: Optional[torch.Tensor] = None,
        extra_meta: Optional[Dict] = None,
    ) -> None:
        mode = Mode.CUDAGRAPH
        try:
            runner = make_runner(True)
        except Exception as e:
            self._error(ctx, a.backend, mode, e, "prepare")
            return
        meta = {"runner": runner.meta, **(extra_meta or {})}
        if not runner.graph_safe:
            self.emit(self._trace(ctx, a.backend, mode, Status.GRAPH_UNSUPPORTED,
                                  message=runner.graph_reason or "not capturable", meta=meta))
            return
        try:
            with torch.no_grad():
                graph, static_out, info = capture_graph(runner.fn, self.cfg.timing, self.device)
                graph.replay()
            _sync(self.device)
        except GraphCaptureError as e:
            self.emit(self._trace(ctx, a.backend, mode, Status.GRAPH_UNSUPPORTED,
                                  message=_short(e), meta=meta))
            return
        except Exception as e:
            self._error(ctx, a.backend, mode, e, "replay")
            return
        corr = None
        if check_ref:
            tol = correctness.tolerance(p, runner, self.cfg.tol)
            status, corr, msg = correctness.check(p, runner, static_out, tol)
        else:
            status, msg = _consistency(static_out, eager_out)
        if status != Status.PASSED:
            self.emit(self._trace(ctx, a.backend, mode, status, correctness=corr,
                                  message=f"after graph replay: {msg}", graph=info, meta=meta))
            return
        try:
            lat = gpu_time(graph.replay, self.cfg.timing, self.device)
            host = host_overhead(graph.replay, self.cfg.timing, self.device)
        except Exception as e:
            self._error(ctx, a.backend, mode, e, "timing")
            return
        m = metrics.compute(p, lat.median_ms) if ctx.op != BLOCK_OP else {}
        if eager_ms and lat.median_ms > 0:
            m["graph_speedup"] = eager_ms / lat.median_ms
        self.emit(
            self._trace(ctx, a.backend, mode, Status.PASSED, correctness=corr, latency=lat,
                        host=host, graph=info, metrics=m, meta=meta)
        )

    # -- block level ----------------------------------------------------------
    def _run_block(self, ctx: Ctx, a: OpAdapter, p: Problem, modes: List[str]) -> None:
        cfg = self.cfg
        make = runner_factory(ctx, a, p, cfg.block_layers, cfg.block_impl, cfg.seed)

        eager_ms = None
        eager_out = None
        if Mode.EAGER in modes:
            try:
                runner = make(False)
                with torch.no_grad():
                    eager_out = runner.fn().clone()
                    _sync(self.device)
                    lat = gpu_time(runner.fn, cfg.timing, self.device)
                    host = host_overhead(runner.fn, cfg.timing, self.device)
            except Exception as e:
                self._error(ctx, a.backend, Mode.EAGER, e)
                return
            eager_ms = lat.median_ms
            self.emit(self._trace(ctx, a.backend, Mode.EAGER, Status.PASSED, latency=lat,
                                  host=host, meta={"runner": runner.meta}))
        if Mode.CUDAGRAPH in modes:
            self._graph(ctx, a, p, eager_ms, make, check_ref=False, eager_out=eager_out)

    def _error(self, ctx: Ctx, backend: str, mode: str, e: Exception, stage: str = "") -> None:
        status = Status.OOM if _is_oom(e) else Status.RUNTIME_ERROR
        tb = traceback.format_exc(limit=6)
        self.emit(self._trace(ctx, backend, mode, status,
                              message=f"{stage + ': ' if stage else ''}{_short(e)}",
                              meta={"traceback": tb}))


def make_ctx(model: ModelSpec, variant: AttnVariant, op: str, wl, device: torch.device,
             moe_expert_cap: int = 0) -> Ctx:
    geom = attn_geom(model, variant, wl.phase) if op in ATTN_LIKE else None
    return Ctx(model=model, variant=variant, op=op, wl=wl, device=device, sm=hardware.sm(),
               geom=geom, options={"moe_expert_cap": moe_expert_cap})


def build_problem(ctx: Ctx, seed: int = 0) -> Problem:
    """Inputs for ``ctx``; for decoder_layer this is the attention (KV cache) problem."""
    if ctx.op == BLOCK_OP:
        actx = Ctx(**{**ctx.__dict__, "op": "attn_decode"})
        return P.build_attention(actx, seed)
    return P.build(ctx, seed)


def runner_factory(ctx: Ctx, a: OpAdapter, p: Problem, block_layers: int = 1,
                   block_impl: Optional[Dict[str, str]] = None,
                   seed: int = 0) -> Callable[[bool], Runner]:
    """``make(graph) -> Runner`` for an op adapter or a decoder-layer block around it."""
    if ctx.op != BLOCK_OP:
        return lambda graph: a.prepare(p, graph)

    def make(graph: bool) -> Runner:
        _, runner, _ = build_block(ctx, a, graph, block_layers, block_impl, seed, attn_problem=p)
        return runner

    return make


def _consistency(out, ref) -> Tuple[str, str]:
    if ref is None:
        return Status.PASSED, ""
    x, y = out.float(), ref.float()
    if x.shape != y.shape:
        return Status.INCORRECT_SHAPE, f"{tuple(x.shape)} vs eager {tuple(y.shape)}"
    if not torch.isfinite(x).all():
        return Status.INCORRECT_NUMERICAL, "NaN/Inf"
    err = ((x - y).abs() / (y.abs() + 1e-3)).max().item()
    if err > 5e-2:
        return Status.INCORRECT_NUMERICAL, f"graph output differs from eager (max rel {err:.3e})"
    return Status.PASSED, ""


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _is_oom(e: BaseException) -> bool:
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom is not None and isinstance(e, oom)) or "out of memory" in str(e).lower()


def _short(e: BaseException) -> str:
    s = f"{type(e).__name__}: {e}".replace("\n", " ")
    return s[:400]


def default_out(tag: str = "") -> Path:
    hw = hardware.probe()
    name = hw.gpu.replace(" ", "_").replace("/", "_")
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("traces") / f"{name}_{tag + '_' if tag else ''}{stamp}.jsonl"


def run(cfg: RunConfig) -> Dict[str, int]:
    m = Matrix(cfg)
    if not cfg.dry_run:
        print(f"hardware: {m.hw.label}  libs: {m.libs}")
        print(f"device: {m.device}  traces -> {cfg.out}")
    return m.run()
