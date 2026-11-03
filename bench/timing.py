"""Eager and CUDA-graph timing.

GPU time: FlashInfer's ``bench_gpu_time_with_cupti`` (CUPTI activity spans
from first kernel start to last kernel end per call, cold L2) when CUPTI >= 13
is installed; otherwise CUDA events around each call with an explicit L2
flush in between. Host overhead is measured separately with ``perf_counter``
because it is exactly what CUDA graphs remove and GPU-side timers hide it.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from bench.schema import GraphInfo, HostOverhead, Latency


@dataclass
class TimingConfig:
    warmup: int = 10
    iters: int = 50
    trials: int = 3
    cold_l2: bool = True
    host_iters: int = 30
    use_cupti: bool = True
    graph_warmup: int = 3


def _cupti_timer() -> Optional[Callable]:
    try:
        from importlib.metadata import version

        from cupti import cupti  # noqa: F401

        if int(version("cupti-python").split(".")[0]) < 13:
            return None
        from flashinfer.testing import bench_gpu_time_with_cupti

        return bench_gpu_time_with_cupti
    except Exception:
        return None


_L2_BUF: dict = {}


def _l2_flush_buffer(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _L2_BUF:
        l2 = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 64 << 20)
        _L2_BUF[key] = torch.empty(max(2 * l2, 8 << 20), dtype=torch.uint8, device=device)
    return _L2_BUF[key]


def _event_times(fn: Callable, cfg: TimingConfig, device: torch.device) -> List[float]:
    flush = _l2_flush_buffer(device) if cfg.cold_l2 else None
    for _ in range(cfg.warmup):
        fn()
    torch.cuda.synchronize(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(cfg.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(cfg.iters)]
    for i in range(cfg.iters):
        if flush is not None:
            flush.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize(device)
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _wall_times(fn: Callable, cfg: TimingConfig) -> List[float]:
    for _ in range(min(cfg.warmup, 3)):
        fn()
    out = []
    for _ in range(cfg.iters):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def gpu_time(fn: Callable, cfg: TimingConfig, device: torch.device) -> Latency:
    if device.type != "cuda":
        samples: List[float] = []
        for _ in range(cfg.trials):
            samples += _wall_times(fn, cfg)
        return Latency.from_samples(samples, "wall(cpu)")
    cupti = _cupti_timer() if cfg.use_cupti else None
    samples = []
    timer = "cuda_event"
    for _ in range(cfg.trials):
        if cupti is not None:
            try:
                samples += list(
                    cupti(
                        fn=fn,
                        dry_run_iters=cfg.warmup,
                        repeat_iters=cfg.iters,
                        cold_l2_cache=cfg.cold_l2,
                        use_cuda_graph=False,
                    )
                )
                timer = "cupti"
                continue
            except Exception:
                cupti = None
        samples += _event_times(fn, cfg, device)
    return Latency.from_samples(samples, timer + ("+cold_l2" if cfg.cold_l2 else ""))


def host_overhead(fn: Callable, cfg: TimingConfig, device: torch.device) -> HostOverhead:
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda *a: None)
    sync()
    enqueue, wall = [], []
    for _ in range(cfg.host_iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        sync()
        t2 = time.perf_counter()
        enqueue.append((t1 - t0) * 1e3)
        wall.append((t2 - t0) * 1e3)
    return HostOverhead(enqueue_ms=statistics.median(enqueue), wall_ms=statistics.median(wall))


class GraphCaptureError(RuntimeError):
    pass


def capture_graph(
    fn: Callable, cfg: TimingConfig, device: torch.device
) -> Tuple["torch.cuda.CUDAGraph", object, GraphInfo]:
    """Warm up on a side stream, capture ``fn`` once, return (graph, static output, info)."""
    if device.type != "cuda":
        raise GraphCaptureError("CUDA graphs need a CUDA device")
    torch.cuda.synchronize(device)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        for _ in range(cfg.graph_warmup):
            fn()
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    reserved0 = torch.cuda.memory_reserved(device)
    free0 = torch.cuda.mem_get_info(device)[0]
    g = torch.cuda.CUDAGraph()
    t0 = time.perf_counter()
    try:
        with torch.cuda.graph(g):
            out = fn()
    except Exception as e:
        torch.cuda.synchronize(device)
        raise GraphCaptureError(f"{type(e).__name__}: {e}") from e
    torch.cuda.synchronize(device)
    capture_ms = (time.perf_counter() - t0) * 1e3
    mem = max(
        torch.cuda.memory_reserved(device) - reserved0,
        free0 - torch.cuda.mem_get_info(device)[0],
        0,
    )
    nodes = 0
    for attr in ("num_nodes",):
        if hasattr(g, attr):
            try:
                nodes = int(getattr(g, attr)())
            except Exception:
                pass
    return g, out, GraphInfo(capture_ms=capture_ms, mem_mb=mem / 2**20, num_nodes=nodes)


def timer_name(cfg: TimingConfig, device: torch.device) -> str:
    if device.type != "cuda":
        return "wall(cpu)"
    return "cupti" if (cfg.use_cupti and _cupti_timer() is not None) else "cuda_event"
